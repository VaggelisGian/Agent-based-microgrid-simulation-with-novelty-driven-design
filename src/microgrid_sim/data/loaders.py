"""Hourly solar and residential demand profile loading.

Strategy (see docs/DECISIONS.md, "Data provenance" for what was actually used):
1. If a cached CSV already exists at the configured path, load it (fast, hermetic,
   used by every test run and by repeated simulation runs).
2. Otherwise, for solar, attempt a bounded real fetch from the PVGIS seriescalc API
   for the configured lat/lon (Thessaloniki by default). Any failure (network,
   parsing, timeout) falls back to a generated sample; nothing here ever raises
   because the network is unavailable.
3. For residential demand, no quick and reliable public hourly API for a
   Thessaloniki-representative residential profile was identified (OPSD's
   household datasets are large multi-year, multi-household files unsuited to a
   bounded per-run fetch), so a physically plausible generated sample is used
   directly by default.
4. An opt-in, real alternative demand series now also exists: config key
   data.demand_series: "opsd" (default "synthetic", unchanged behaviour)
   selects a real OPSD household-load derived series instead, cached at
   data/samples/residential_demand_opsd_hourly.csv; see
   fetch_and_build_opsd_demand and scripts/fetch_opsd_demand.py.

Whichever path is taken, the result is cached to disk with a "# source: ..."
marker line so provenance survives across runs without re-deciding it every time.

Hour-of-day convention: every cached hourly series (solar and both demand
series) is indexed so that index 0 is 00:00 UTC and index % 24 is the hour of
day; MicrogridModel relies on this to key solar generation, demand, and the
fixed evening_reserve_hour dispatch rule off the same clock. The OPSD demand
series is explicitly aligned to this convention at build time (see
_align_to_utc_midnight_multiple_of_24) rather than left at whatever timestamp
its raw coverage window happened to start on.

Agents only ever index these arrays at the current simulation hour and never at
a future hour, so there is no look-ahead leakage.
"""

from __future__ import annotations

import re
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_LAT = 40.64  # Thessaloniki
_DEFAULT_LON = 22.95
_PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
_PVGIS_FETCH_YEAR = 2020  # PVGIS seriescalc only accepts 2005-2020; leap year, tiled/truncated to horizon
_SAMPLE_YEAR_HOURS = 8760

_SOLAR_SEED = 20260101  # fixed seed for the generated sample, independent of simulation seed
_DEMAND_SEED = 20260102

_DEEP_NIGHT_HOURS = (1, 2, 3)

# Sources that are a real fetch (not a synthetic fallback), used to gate the
# F5 "synthetic fallback" warning in _load_or_build below. Any source string
# NOT in this set is treated as synthetic for warning purposes, matching the
# original pvgis_fetch-only check byte for byte for every source that
# existed before this OPSD extension.
_REAL_FETCH_SOURCES = frozenset({"pvgis_fetch", "opsd_household_fetch"})

# --- OPSD household-load demand series (Task: real residential demand data) ---
# See docs/DECISIONS.md, "Data provenance", for why a generated sample was
# used originally, and the config switch below (data.demand_series) for how
# a run opts into this real series instead.
_OPSD_DEMAND_URL = "https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv"
_OPSD_DOWNLOAD_CHUNK_BYTES = 1 << 16  # 64 KiB per streamed write; bounds download memory
_OPSD_PARSE_CHUNK_ROWS = 4000  # pandas read_csv chunksize for the incremental parse passes
_OPSD_MAX_RETRIES = 4  # bounded retries with backoff, never retry forever
_OPSD_BACKOFF_INITIAL_SEC = 2.0
_OPSD_CONNECT_TIMEOUT_SEC = 15.0
_OPSD_READ_TIMEOUT_SEC = 60.0
_OPSD_MIN_COVERAGE_HOURS = _SAMPLE_YEAR_HOURS  # need at least one full year, gap-free
_OPSD_RESIDENTIAL_GRID_IMPORT_RE = re.compile(r"^(DE_KN_residential\d+)_grid_import$")
_ALLOWED_DEMAND_SERIES = ("synthetic", "opsd")
_OPSD_DEMAND_CACHE_DEFAULT_PATH = "data/samples/residential_demand_opsd_hourly.csv"


class DataLoadError(RuntimeError):
    """Raised when a profile series is missing, malformed, or fails validation."""


class OpsdFetchError(DataLoadError):
    """Raised when the real OPSD household-load download or parse pipeline
    fails after bounded, documented retries. Callers must NOT catch this and
    substitute a synthetic series in its place: the whole point of this
    error type is that the real data was genuinely unavailable and nothing
    here is allowed to quietly fabricate a replacement and call it measured.
    """


@dataclass(frozen=True)
class ProfileData:
    solar_kw_per_kwp: np.ndarray
    demand_kwh_reference: np.ndarray
    solar_source: str
    demand_source: str


def validate_series(series: np.ndarray, name: str, allow_negative: bool = True) -> None:
    if series.size == 0:
        raise DataLoadError(f"{name}: series is empty")
    if not np.all(np.isfinite(series)):
        raise DataLoadError(f"{name}: series contains NaN or infinite values")
    if not allow_negative and np.any(series < 0.0):
        raise DataLoadError(f"{name}: series contains negative values")


def tile_to_horizon(series: np.ndarray, horizon_hours: int) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        raise DataLoadError("cannot tile an empty series")
    if series.size >= horizon_hours:
        return series[:horizon_hours].copy()
    repeats = -(-horizon_hours // series.size)  # ceil division
    return np.tile(series, repeats)[:horizon_hours]


def generate_synthetic_solar(hours: int = _SAMPLE_YEAR_HOURS, seed: int = _SOLAR_SEED) -> np.ndarray:
    """Deterministic diurnal plus seasonal solar capacity-factor profile (kW per kWp).

    Sunrise/sunset and midday peak strength vary smoothly with day of year for a
    latitude-40 site; a per-day cloudiness multiplier adds plausible variability.
    Zero outside the daylight window on every day, by construction.
    """
    rng = np.random.default_rng(seed)
    hour_index = np.arange(hours)
    hour_of_day = hour_index % 24
    day_of_year = (hour_index // 24) % 365

    day_length = 12.0 + 2.4 * np.cos(2.0 * np.pi * (day_of_year - 172) / 365.0)
    sunrise = 12.0 - day_length / 2.0
    sunset = 12.0 + day_length / 2.0
    daylight = (hour_of_day >= sunrise) & (hour_of_day <= sunset)
    raw_shape = np.sin(np.pi * (hour_of_day - sunrise) / day_length)
    clear_sky_shape = np.where(daylight, raw_shape, 0.0)

    seasonal_peak_factor = 0.75 + 0.25 * np.cos(2.0 * np.pi * (day_of_year - 172) / 365.0)
    peak_kw_per_kwp = 0.8  # representative clear-sky midday peak per kWp installed

    num_days = -(-hours // 24)
    daily_cloud_factor = rng.uniform(0.55, 1.0, size=num_days)
    cloud_factor = np.repeat(daily_cloud_factor, 24)[:hours]

    solar = peak_kw_per_kwp * seasonal_peak_factor * clear_sky_shape * cloud_factor
    return np.clip(solar, 0.0, None)


def generate_synthetic_demand(
    hours: int = _SAMPLE_YEAR_HOURS,
    seed: int = _DEMAND_SEED,
    annual_kwh_reference: float = 4000.0,
) -> np.ndarray:
    """Deterministic residential demand shape (kWh/h) for one reference household.

    Morning and evening Gaussian bumps, a weekend daytime boost, and a winter-peaked
    seasonal multiplier (heating and lighting load) on top of a constant standby
    base load; small multiplicative noise for realism. Normalized so the implied
    average hourly consumption matches annual_kwh_reference over a full year.
    """
    rng = np.random.default_rng(seed)
    hour_index = np.arange(hours)
    hour_of_day = hour_index % 24
    day_of_year = (hour_index // 24) % 365
    weekday = (hour_index // 24) % 7

    base_load = 0.25
    morning_bump = 0.5 * np.exp(-0.5 * ((hour_of_day - 8) / 1.5) ** 2)
    evening_bump = 0.7 * np.exp(-0.5 * ((hour_of_day - 20) / 2.0) ** 2)
    weekend_multiplier = np.where(weekday >= 5, 1.15, 1.0)
    seasonal_multiplier = 1.0 + 0.25 * np.cos(2.0 * np.pi * (day_of_year - 15) / 365.0)

    shape = (base_load + morning_bump + evening_bump) * weekend_multiplier * seasonal_multiplier
    noise = np.clip(rng.normal(1.0, 0.04, size=hours), 0.85, 1.15)
    shape = shape * noise

    target_hourly_mean = annual_kwh_reference / _SAMPLE_YEAR_HOURS
    shape = shape * (target_hourly_mean / shape.mean())
    return np.clip(shape, 0.02, None)


@dataclass
class _OpsdColumnCoverage:
    """Coverage stats for one cumulative-counter column, gathered in a single
    incremental pass over the raw file: how many rows are non-NaN in total,
    and the [run_start, run_end] (inclusive, 0-based row index) of that
    column's single LONGEST contiguous run of non-NaN values. OPSD household
    meters are commissioned and decommissioned at different real dates but
    were found (this is checked here, not assumed) to not drop out and come
    back mid-life for the columns used in this project, so tracking the
    longest run rather than just the first-to-last span is a genuine check,
    not a formality.
    """

    column: str
    valid_count: int
    run_start: int | None
    run_end: int | None


def _download_opsd_raw_csv(
    dest_path: Path,
    url: str = _OPSD_DEMAND_URL,
    max_retries: int = _OPSD_MAX_RETRIES,
    chunk_bytes: int = _OPSD_DOWNLOAD_CHUNK_BYTES,
) -> dict:
    """Stream-download url to dest_path with chunked writes (bounded memory:
    only one chunk_bytes buffer is held at a time, never the whole response
    body), a bounded number of retries with exponential backoff, and an
    attempt at HTTP range resume on retry.

    Measured against the live OPSD server (2026-07-30): it serves this file
    with Content-Encoding: gzip and, when sent a Range header, still replies
    200 with the full body rather than 206 with the requested slice, so true
    byte-range resume is NOT actually honored by this server for this
    resource. The code below still asks for it (harmless if ignored) and
    only trusts a genuine 206 response as a real resume; anything else
    restarts the destination file from byte 0 rather than risk silently
    duplicating or corrupting data. This is reported honestly rather than
    claimed as working resume.
    """
    import requests

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    backoff = _OPSD_BACKOFF_INITIAL_SEC
    start = time.perf_counter()
    last_exc: Exception | None = None

    while attempt < max_retries:
        attempt += 1
        resume_from = dest_path.stat().st_size if dest_path.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}
        try:
            with requests.get(
                url,
                stream=True,
                headers=headers,
                timeout=(_OPSD_CONNECT_TIMEOUT_SEC, _OPSD_READ_TIMEOUT_SEC),
            ) as response:
                response.raise_for_status()
                resumed = resume_from > 0 and response.status_code == 206
                write_mode = "ab" if resumed else "wb"
                with open(dest_path, write_mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if chunk:
                            handle.write(chunk)
            elapsed = time.perf_counter() - start
            return {
                "bytes_downloaded": dest_path.stat().st_size,
                "elapsed_sec": elapsed,
                "attempts": attempt,
                "resumed_final_attempt": resumed,
            }
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(backoff)
            backoff *= 2.0

    raise OpsdFetchError(f"OPSD download failed after {attempt} attempt(s) from {url}: {last_exc}") from last_exc


def _discover_residential_grid_import_columns(header_cols: list[str]) -> list[str]:
    return [c for c in header_cols if _OPSD_RESIDENTIAL_GRID_IMPORT_RE.match(c)]


def _columns_without_onsite_generation(grid_import_columns: list[str], header_cols: list[str]) -> list[str]:
    """A household's grid_import equals its total consumption only if the
    household has no on-site PV generation to net out first; a household
    with a "<prefix>_pv..." column self-consumes part of its own generation
    before importing from the grid, so its grid_import understates true
    consumption and carries a solar-driven midday dip. This project already
    models solar separately, per prosumer, so folding a real household's own
    solar self-consumption into the demand SHAPE would conflate two
    different things; households with no on-site generation column are
    selected instead, so grid_import is a clean, unconfounded consumption
    series.
    """
    selected = []
    for col in grid_import_columns:
        prefix = _OPSD_RESIDENTIAL_GRID_IMPORT_RE.match(col).group(1)
        if not any(other.startswith(prefix + "_pv") for other in header_cols):
            selected.append(col)
    return selected


def _scan_opsd_coverage(
    raw_csv_path: Path, columns: list[str], chunk_rows: int = _OPSD_PARSE_CHUNK_ROWS
) -> dict[str, _OpsdColumnCoverage]:
    """One incremental pass over raw_csv_path (pandas chunked reader: at most
    chunk_rows rows x len(columns) float values held in memory at any time,
    never the whole file) computing, per column, the total valid (non-NaN)
    count and the [start, end] row-index range of its single longest
    contiguous run of non-NaN values.
    """
    state = {col: {"current_start": None, "best_start": None, "best_end": None, "best_len": 0, "valid_count": 0} for col in columns}
    reader = pd.read_csv(raw_csv_path, usecols=columns, chunksize=chunk_rows)
    row_idx = 0
    for chunk in reader:
        n = len(chunk)
        for col in columns:
            s = state[col]
            values = chunk[col].to_numpy(dtype=float)
            is_valid = ~np.isnan(values)
            for offset in range(n):
                idx = row_idx + offset
                if is_valid[offset]:
                    s["valid_count"] += 1
                    if s["current_start"] is None:
                        s["current_start"] = idx
                elif s["current_start"] is not None:
                    run_len = idx - s["current_start"]
                    if run_len > s["best_len"]:
                        s["best_len"] = run_len
                        s["best_start"] = s["current_start"]
                        s["best_end"] = idx - 1
                    s["current_start"] = None
        row_idx += n

    for col in columns:
        s = state[col]
        if s["current_start"] is not None:
            run_len = row_idx - s["current_start"]
            if run_len > s["best_len"]:
                s["best_len"] = run_len
                s["best_start"] = s["current_start"]
                s["best_end"] = row_idx - 1

    return {
        col: _OpsdColumnCoverage(
            column=col,
            valid_count=state[col]["valid_count"],
            run_start=state[col]["best_start"],
            run_end=state[col]["best_end"],
        )
        for col in columns
    }


def _select_common_coverage_window(
    coverages: dict[str, _OpsdColumnCoverage], min_hours: int
) -> tuple[list[str], int, int]:
    """Starting from every candidate column, find the largest subset whose
    longest contiguous runs share a common [start, end] window of at least
    min_hours rows, by repeatedly dropping the column with the tightest own
    run when the current subset's intersection is too short. This is a
    bounded, greedy heuristic (not an exhaustive search over all subsets),
    documented as such; it reproduces the correct answer whenever the
    columns with the widest individual coverage also share the widest
    overlap, which held for the two-household OPSD case this was built for.
    """
    usable = {col: cov for col, cov in coverages.items() if cov.run_start is not None}
    columns = sorted(usable, key=lambda c: usable[c].run_end - usable[c].run_start, reverse=True)
    while columns:
        start = max(usable[c].run_start for c in columns)
        end = min(usable[c].run_end for c in columns)
        if end - start + 1 >= min_hours:
            return columns, start, end
        columns = columns[:-1]
    raise OpsdFetchError(
        f"no combination of candidate residential columns has a common contiguous coverage "
        f"window of at least {min_hours} hours (columns considered: {sorted(coverages)})"
    )


def _extract_opsd_hourly_consumption(
    raw_csv_path: Path, columns: list[str], start_idx: int, end_idx: int, chunk_rows: int = _OPSD_PARSE_CHUNK_ROWS
) -> tuple[np.ndarray, np.ndarray]:
    """Read utc_timestamp plus the selected cumulative-counter columns for
    rows [start_idx, end_idx] (inclusive) in bounded-size chunks, difference
    each column row-to-row (kWh consumed during that hour) and sum across
    columns into one aggregate hourly consumption series. Bounded memory:
    at most chunk_rows rows are held from the reader at a time; only the
    (end_idx - start_idx)-length result arrays are accumulated, not the
    whole underlying file. Raises OpsdFetchError if a NaN or a counter
    decrease (a reset) is found inside the chosen window, since both would
    mean the "gap-free, monotone" property this window was selected for did
    not actually hold and the series must not be built silently anyway.
    """
    usecols = ["utc_timestamp"] + columns
    reader = pd.read_csv(raw_csv_path, usecols=usecols, chunksize=chunk_rows)
    prev_values: np.ndarray | None = None
    consumption: list[float] = []
    timestamps: list = []
    row_idx = 0
    for chunk in reader:
        n = len(chunk)
        chunk_start, chunk_end = row_idx, row_idx + n - 1
        if chunk_end >= start_idx and chunk_start <= end_idx:
            lo = max(start_idx, chunk_start) - chunk_start
            hi = min(end_idx, chunk_end) - chunk_start
            sub = chunk.iloc[lo : hi + 1]
            values = sub[columns].to_numpy(dtype=float)
            ts = sub["utc_timestamp"].to_numpy()
            for i in range(len(sub)):
                if np.any(np.isnan(values[i])):
                    raise OpsdFetchError(f"unexpected NaN inside the chosen coverage window at row {row_idx + lo + i}")
                if prev_values is not None:
                    diffs = values[i] - prev_values
                    if np.any(diffs < 0.0):
                        raise OpsdFetchError(
                            f"cumulative counter decrease (reset) detected inside the chosen coverage "
                            f"window at row {row_idx + lo + i}"
                        )
                    consumption.append(float(diffs.sum()))
                    timestamps.append(ts[i])
                prev_values = values[i]
        row_idx += n
        if row_idx > end_idx:
            break
    return np.array(consumption, dtype=float), np.array(timestamps)


def _align_to_utc_midnight_multiple_of_24(
    consumption: np.ndarray, timestamps: np.ndarray, min_hours: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Trim (consumption, timestamps) so index 0 lands on a UTC-midnight
    boundary (hour == 0) and the final length is a whole multiple of 24.

    This is REQUIRED, not cosmetic: MicrogridModel indexes every hourly
    profile (solar, demand) by hour_of_day = index % 24, and the cached
    solar series (PVGIS) already uses the convention that index 0 is 00:00
    UTC. Without this trim, the OPSD series' index 0 is whatever timestamp
    the raw coverage window happened to start at, which phase-shifts OPSD
    demand against solar generation and against the fixed
    evening_reserve_hour: 18 dispatch rule by however many hours the window
    start happened to be offset from midnight -- a bug found in review
    before this series was ever used in a real run.

    The offset is DERIVED from the real utc_timestamp column (the first
    hour == 0 occurrence in the window), never assumed or hardcoded. If the
    window is also longer than a whole number of days, the remainder at the
    end is dropped too, so the cached length is always a clean multiple of
    24: tile_to_horizon repeats whole days, never a fractional day that
    would rotate the phase further on every repeat for horizons longer than
    the cached series.

    Returns (aligned_consumption, aligned_timestamps, start_offset_hours),
    where start_offset_hours is how many leading hours were dropped to
    reach the first UTC-midnight boundary.
    """
    hours_of_day = pd.to_datetime(timestamps, utc=True).hour.to_numpy()
    midnight_positions = np.flatnonzero(hours_of_day == 0)
    if midnight_positions.size == 0:
        raise OpsdFetchError("no UTC-midnight (hour==0) timestamp found in the chosen coverage window; cannot align to a day boundary")

    start_offset_hours = int(midnight_positions[0])
    trimmed_consumption = consumption[start_offset_hours:]
    trimmed_timestamps = timestamps[start_offset_hours:]

    usable_length = (trimmed_consumption.size // 24) * 24
    if usable_length < min_hours:
        raise OpsdFetchError(
            f"only {usable_length} UTC-day-aligned hours remain after trimming to a midnight boundary "
            f"(dropped {start_offset_hours} leading hour(s) plus a {trimmed_consumption.size - usable_length}-hour "
            f"trailing remainder); need at least {min_hours}"
        )
    return trimmed_consumption[:usable_length], trimmed_timestamps[:usable_length], start_offset_hours


def _validate_opsd_alignment(series: np.ndarray, timestamps: np.ndarray) -> None:
    """Fail loudly if the build path ever produces a series that is not
    aligned to a UTC-midnight boundary at index 0, or whose length is not a
    whole multiple of 24. Nothing previously checked this invariant, which
    is exactly how a roughly-half-a-day phase shift between the OPSD demand
    series and the solar/dispatch convention went undetected; this
    assertion exists so that specific failure mode cannot recur silently.
    """
    if series.size == 0 or series.size % 24 != 0:
        raise OpsdFetchError(f"OPSD series length {series.size} is not a whole multiple of 24 hours")
    first_hour = pd.Timestamp(str(timestamps[0])).hour
    if first_hour != 0:
        raise OpsdFetchError(f"OPSD series index 0 is not a UTC-midnight boundary (hour={first_hour}, timestamp={timestamps[0]!r})")


def _scale_series_to_annual_kwh(series: np.ndarray, annual_kwh_reference: float) -> np.ndarray:
    """Match generate_synthetic_demand's own normalization convention:
    rescale so the series' mean hourly value implies annual_kwh_reference
    over a _SAMPLE_YEAR_HOURS (8760-hour) year. This makes the OPSD-derived
    series comparable to the synthetic one on SHAPE, not on level, per the
    brief: both sources represent "one reference household" at the same
    annual total.
    """
    target_hourly_mean = annual_kwh_reference / _SAMPLE_YEAR_HOURS
    current_mean = series.mean()
    if not np.isfinite(current_mean) or current_mean <= 0.0:
        raise OpsdFetchError("OPSD-derived consumption series has a non-positive or non-finite mean; cannot normalize")
    return series * (target_hourly_mean / current_mean)


def fetch_and_build_opsd_demand(
    annual_kwh_reference: float = 4000.0,
    raw_download_path: Path | str | None = None,
    url: str = _OPSD_DEMAND_URL,
) -> tuple[np.ndarray, dict]:
    """Real OPSD household-load fetch and derivation pipeline (see
    docs/DECISIONS.md, "Data provenance", and Task brief). Downloads the
    live household_data_60min_singleindex.csv (chunked, bounded memory,
    bounded retries; see _download_opsd_raw_csv), selects the residential
    households with no on-site PV so grid_import equals pure consumption,
    picks the widest common gap-free coverage window across them, DIFFERENCES
    the cumulative kWh counters into hourly consumption (these are running
    meter totals, not per-hour flows; this is checked and enforced, not
    assumed), ALIGNS the result so index 0 is 00:00 UTC and the length is a
    whole multiple of 24 (see _align_to_utc_midnight_multiple_of_24; this is
    the same hour-of-day convention the cached PVGIS solar series already
    uses, and MicrogridModel indexes every profile by hour_of_day = index %
    24), and normalizes the result to annual_kwh_reference so it is
    comparable in SHAPE to the shipped synthetic series.

    Modelling limitation, stated plainly rather than smoothed over: the
    selected households (DE_KN_residential2, DE_KN_residential5) are German,
    metered in local CET/CEST time but published here as UTC timestamps,
    and this project transplants their diurnal SHAPE onto a Thessoloniki
    solar year. Aligning index 0 to UTC midnight (matching the solar
    series' own convention) preserves hour-of-day-of-behaviour in UTC
    terms, not in true local solar time at either location; the households'
    real local evening peak sits roughly one UTC-offset-hour later than
    where it lands here (Germany is UTC+1/+2, Greece is UTC+2/+3, so the
    two are close but not identical, and this project does not attempt a
    further timezone correction on top of the UTC alignment).

    Never fabricates: any genuine download or parsing failure raises
    OpsdFetchError (a DataLoadError subclass) instead of returning a
    synthetic substitute. Returns (hourly_consumption_kwh, diagnostics),
    where diagnostics carries the real byte size, timing, households used,
    and coverage window for honest provenance logging by the caller.
    """
    if raw_download_path is None:
        raw_download_path = Path(tempfile.gettempdir()) / "opsd_household_data_60min_singleindex.csv"
    raw_download_path = Path(raw_download_path)

    download_stats = _download_opsd_raw_csv(raw_download_path, url=url)

    header_cols = list(pd.read_csv(raw_download_path, nrows=0).columns)
    n_columns = len(header_cols)
    grid_import_columns = _discover_residential_grid_import_columns(header_cols)
    if not grid_import_columns:
        raise OpsdFetchError(f"no residential grid_import columns found among the {n_columns} columns in {raw_download_path}")

    coverage_all = _scan_opsd_coverage(raw_download_path, grid_import_columns)
    pv_free_columns = _columns_without_onsite_generation(grid_import_columns, header_cols)
    if not pv_free_columns:
        raise OpsdFetchError(
            "every residential household in the OPSD file has an on-site PV column; none give a "
            "grid_import series that equals pure consumption"
        )

    selected_columns, start_idx, end_idx = _select_common_coverage_window(
        {col: coverage_all[col] for col in pv_free_columns}, _OPSD_MIN_COVERAGE_HOURS
    )

    consumption, timestamps = _extract_opsd_hourly_consumption(raw_download_path, selected_columns, start_idx, end_idx)
    validate_series(consumption, name="opsd_residential_consumption_raw_kwh", allow_negative=False)

    aligned_consumption, aligned_timestamps, start_offset_hours = _align_to_utc_midnight_multiple_of_24(
        consumption, timestamps, _OPSD_MIN_COVERAGE_HOURS
    )
    _validate_opsd_alignment(aligned_consumption, aligned_timestamps)

    normalized = _scale_series_to_annual_kwh(aligned_consumption, annual_kwh_reference)
    validate_series(normalized, name="opsd_residential_consumption_normalized_kwh", allow_negative=False)
    _validate_opsd_alignment(normalized, aligned_timestamps)  # scaling must not change size/order

    diagnostics = {
        "source_url": url,
        "bytes_downloaded": download_stats["bytes_downloaded"],
        "download_elapsed_sec": download_stats["elapsed_sec"],
        "download_attempts": download_stats["attempts"],
        "n_columns_in_source": n_columns,
        "households_selected": selected_columns,
        "households_considered_pv_free": pv_free_columns,
        "per_household_coverage_hours": {
            col: (cov.run_end - cov.run_start + 1 if cov.run_start is not None else 0)
            for col, cov in coverage_all.items()
        },
        "coverage_start_utc_before_alignment": str(timestamps[0]),
        "coverage_end_utc_before_alignment": str(timestamps[-1]),
        "alignment_start_offset_hours": start_offset_hours,
        "coverage_start_utc": str(aligned_timestamps[0]),
        "coverage_end_utc": str(aligned_timestamps[-1]),
        "coverage_hours": int(aligned_consumption.size),
        "counters_were_cumulative": True,
        "annual_kwh_reference": annual_kwh_reference,
        "raw_download_path": str(raw_download_path),
    }
    return normalized, diagnostics


def _fetch_pvgis_hourly(lat: float, lon: float, timeout: float = 10.0) -> np.ndarray | None:
    import requests

    params = {
        "lat": lat,
        "lon": lon,
        "outputformat": "json",
        "startyear": _PVGIS_FETCH_YEAR,
        "endyear": _PVGIS_FETCH_YEAR,
        "pvcalculation": 1,
        "peakpower": 1,
        "loss": 14,
        "angle": 30,
        "aspect": 0,
    }
    try:
        response = requests.get(_PVGIS_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        hourly = payload["outputs"]["hourly"]
        watts_per_kwp = np.array([row["P"] for row in hourly], dtype=float)
    except Exception:
        # any network, HTTP, or payload-shape failure must fall back to the
        # generated sample rather than crash the simulation
        return None
    if watts_per_kwp.size == 0:
        return None
    return watts_per_kwp / 1000.0


def _read_cached_csv(path: Path, column: str) -> tuple[str, np.ndarray]:
    with open(path, "r", encoding="ascii") as handle:
        first_line = handle.readline()
    source = "generated_sample"
    if first_line.startswith("# source:"):
        source = first_line.split(":", 1)[1].strip()

    frame = pd.read_csv(path, comment="#")
    if column not in frame.columns:
        raise DataLoadError(f"cached file {path} is missing expected column '{column}'")
    series = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return source, series


def _write_cached_csv(path: Path, series: np.ndarray, column: str, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="") as handle:
        handle.write(f"# source: {source}\n")
        frame = pd.DataFrame({"hour": np.arange(series.size), column: series})
        frame.to_csv(handle, index=False)


def _load_or_build(path: Path, column: str, build_fn, allow_negative: bool) -> tuple[np.ndarray, str]:
    if path.exists():
        source, series = _read_cached_csv(path, column)
        validate_series(series, name=column, allow_negative=allow_negative)
        return series, source

    series, source = build_fn()
    validate_series(series, name=column, allow_negative=allow_negative)
    if source not in _REAL_FETCH_SOURCES:
        # F5 fix: a cache file rebuilt from a synthetic fallback (rather than
        # loaded from an existing cache or fetched from a live source) silently
        # changes the numerical character of the series (e.g. real Thessaloniki
        # irradiance vs. a smooth synthetic sinusoid) for every subsequent run
        # unless flagged loudly at the point it happens.
        warnings.warn(
            f"{path}: no cached file found; writing a new cache built from a "
            f"synthetic fallback (source={source!r}), not a real fetch. If this "
            f"is unexpected (e.g. the shipped data/samples/ cache was deleted), "
            f"results from this point on use a different '{column}' series than "
            f"prior runs.",
            RuntimeWarning,
            stacklevel=3,
        )
    _write_cached_csv(path, series, column, source)
    return series, source


def _validate_solar_zero_at_night(solar_kw_per_kwp: np.ndarray, tolerance: float = 1e-6) -> None:
    hour_of_day = np.arange(solar_kw_per_kwp.size) % 24
    deep_night = np.isin(hour_of_day, _DEEP_NIGHT_HOURS)
    if np.any(solar_kw_per_kwp[deep_night] > tolerance):
        raise DataLoadError("solar profile has nonzero generation during deep night hours")


def load_profiles(config: dict, horizon_hours: int, allow_network_fetch: bool = True) -> ProfileData:
    data_config = config.get("data", {})
    solar_path = Path(data_config["solar_profile_path"])
    lat = data_config.get("solar_lat", _DEFAULT_LAT)
    lon = data_config.get("solar_lon", _DEFAULT_LON)
    annual_kwh_reference = data_config.get("reference_annual_demand_kwh", 4000.0)

    # Config switch (Task brief): which demand series a run uses. Named
    # data.demand_series in config, deliberately NOT reusing the name
    # MicrogridModel.demand_source (the provenance marker string read back
    # from a cache file's "# source:" line) so the two are never confused:
    # this key SELECTS a data path/behaviour before loading; demand_source
    # REPORTS what was actually loaded, after the fact. Absent key means
    # "synthetic", the current, unchanged default behaviour.
    demand_series_choice = data_config.get("demand_series", "synthetic")
    if demand_series_choice not in _ALLOWED_DEMAND_SERIES:
        raise DataLoadError(
            f"data.demand_series must be one of {_ALLOWED_DEMAND_SERIES!r}, got {demand_series_choice!r}"
        )
    if demand_series_choice == "opsd":
        demand_path = Path(data_config.get("demand_profile_path_opsd", _OPSD_DEMAND_CACHE_DEFAULT_PATH))
    else:
        demand_path = Path(data_config["demand_profile_path"])

    def build_solar():
        if allow_network_fetch:
            fetched = _fetch_pvgis_hourly(lat, lon)
            if fetched is not None:
                return fetched, "pvgis_fetch"
        return generate_synthetic_solar(), "generated_sample"

    def build_demand():
        if demand_series_choice == "opsd":
            if not allow_network_fetch:
                raise DataLoadError(
                    f"data.demand_series is 'opsd' but no cache file exists at {demand_path} and "
                    "allow_network_fetch=False; the real OPSD series is never fabricated as a "
                    "fallback (see docs/DECISIONS.md, 'Data provenance'). Build the cache once "
                    "with scripts/fetch_opsd_demand.py, or call with allow_network_fetch=True."
                )
            series, _diagnostics = fetch_and_build_opsd_demand(annual_kwh_reference=annual_kwh_reference)
            return series, "opsd_household_fetch"
        return generate_synthetic_demand(annual_kwh_reference=annual_kwh_reference), "generated_sample"

    solar_series, solar_source = _load_or_build(solar_path, "solar_kw_per_kwp", build_solar, allow_negative=False)
    demand_series_array, demand_source = _load_or_build(
        demand_path, "demand_kwh_reference", build_demand, allow_negative=False
    )

    solar = tile_to_horizon(solar_series, horizon_hours)
    demand = tile_to_horizon(demand_series_array, horizon_hours)
    validate_series(solar, name="solar_kw_per_kwp", allow_negative=False)
    validate_series(demand, name="demand_kwh_reference", allow_negative=False)
    _validate_solar_zero_at_night(solar)

    return ProfileData(
        solar_kw_per_kwp=solar,
        demand_kwh_reference=demand,
        solar_source=solar_source,
        demand_source=demand_source,
    )
