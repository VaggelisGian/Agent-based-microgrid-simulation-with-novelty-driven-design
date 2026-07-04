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
   directly.

Whichever path is taken, the result is cached to disk with a "# source: ..."
marker line so provenance survives across runs without re-deciding it every time.

Agents only ever index these arrays at the current simulation hour and never at
a future hour, so there is no look-ahead leakage.
"""

from __future__ import annotations

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


class DataLoadError(RuntimeError):
    """Raised when a profile series is missing, malformed, or fails validation."""


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
    if source != "pvgis_fetch":
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
    demand_path = Path(data_config["demand_profile_path"])
    lat = data_config.get("solar_lat", _DEFAULT_LAT)
    lon = data_config.get("solar_lon", _DEFAULT_LON)
    annual_kwh_reference = data_config.get("reference_annual_demand_kwh", 4000.0)

    def build_solar():
        if allow_network_fetch:
            fetched = _fetch_pvgis_hourly(lat, lon)
            if fetched is not None:
                return fetched, "pvgis_fetch"
        return generate_synthetic_solar(), "generated_sample"

    def build_demand():
        return generate_synthetic_demand(annual_kwh_reference=annual_kwh_reference), "generated_sample"

    solar_series, solar_source = _load_or_build(solar_path, "solar_kw_per_kwp", build_solar, allow_negative=False)
    demand_series, demand_source = _load_or_build(
        demand_path, "demand_kwh_reference", build_demand, allow_negative=False
    )

    solar = tile_to_horizon(solar_series, horizon_hours)
    demand = tile_to_horizon(demand_series, horizon_hours)
    validate_series(solar, name="solar_kw_per_kwp", allow_negative=False)
    validate_series(demand, name="demand_kwh_reference", allow_negative=False)
    _validate_solar_zero_at_night(solar)

    return ProfileData(
        solar_kw_per_kwp=solar,
        demand_kwh_reference=demand,
        solar_source=solar_source,
        demand_source=demand_source,
    )
