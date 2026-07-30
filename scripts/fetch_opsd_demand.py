"""One-time real-data fetch: build data/samples/residential_demand_opsd_hourly.csv
from Open Power System Data's household load package.

This is the actual live fetch behind the data.demand_series: opsd config
switch (see src/microgrid_sim/data/loaders.py, fetch_and_build_opsd_demand,
and docs/DECISIONS.md, "Data provenance"). MicrogridModel always loads
profiles with allow_network_fetch=False (simulation runs must be hermetic),
so this script is how the cache file gets built in the first place; once it
exists, every run with data.demand_series: opsd just reads it like any other
cached profile.

What it does, in order:
  1. Stream-downloads household_data_60min_singleindex.csv (chunked writes,
     bounded retries with backoff) to a local raw file.
  2. Scans it incrementally (chunked pandas reads) to find the residential
     households with no on-site PV (so grid_import equals pure consumption,
     not consumption netted against self-consumed solar) and the widest
     common gap-free coverage window across them.
  3. Differences the cumulative kWh counters into hourly consumption,
     verifying monotonicity rather than assuming it.
  4. Normalizes the result to config/default.yaml's
     data.reference_annual_demand_kwh so the series is comparable in SHAPE,
     not level, to the shipped synthetic series.
  5. Writes data/samples/residential_demand_opsd_hourly.csv in the same
     "# source: ..." cache format the existing loader reads, with source
     marker "opsd_household_fetch".
  6. Deletes the raw intermediate download (15+ MB of German multi-year,
     multi-household data has no reason to live in this repo; only the
     small derived hourly series does).

Never fabricates: if the download or parse pipeline fails for real, this
script prints exactly what failed and exits non-zero. It does not write a
synthetic substitute into the cache path.

Run from the repository root:
    python scripts/fetch_opsd_demand.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from microgrid_sim.config.loader import load_config  # noqa: E402
from microgrid_sim.data.loaders import (  # noqa: E402
    OpsdFetchError,
    _write_cached_csv,
    fetch_and_build_opsd_demand,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
CACHE_PATH = REPO_ROOT / "data" / "samples" / "residential_demand_opsd_hourly.csv"


def _peak_rss_mb() -> float:
    """Peak resident-set size of THIS process, in MB (Windows peak_wset
    watermark via psutil when available, current rss otherwise)."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        peak_bytes = getattr(mem_info, "peak_wset", mem_info.rss)
        return peak_bytes / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def main(argv=None) -> int:
    config = load_config(str(DEFAULT_CONFIG_PATH))
    annual_kwh_reference = config["data"]["reference_annual_demand_kwh"]

    print(f"[fetch_opsd_demand] annual_kwh_reference={annual_kwh_reference} (from {DEFAULT_CONFIG_PATH})")
    print(f"[fetch_opsd_demand] peak RSS before fetch: {_peak_rss_mb():.2f} MB")

    start = time.perf_counter()
    try:
        series, diagnostics = fetch_and_build_opsd_demand(annual_kwh_reference=annual_kwh_reference)
    except OpsdFetchError as exc:
        print(f"[fetch_opsd_demand] FAILED: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    peak_rss_after = _peak_rss_mb()
    print(f"[fetch_opsd_demand] peak RSS after fetch+parse: {peak_rss_after:.2f} MB")
    print(f"[fetch_opsd_demand] source_url: {diagnostics['source_url']}")
    print(f"[fetch_opsd_demand] bytes_downloaded: {diagnostics['bytes_downloaded']}")
    print(f"[fetch_opsd_demand] n_columns_in_source: {diagnostics['n_columns_in_source']}")
    print(f"[fetch_opsd_demand] download_elapsed_sec: {diagnostics['download_elapsed_sec']:.2f}")
    print(f"[fetch_opsd_demand] download_attempts: {diagnostics['download_attempts']}")
    print(f"[fetch_opsd_demand] households_considered_pv_free: {diagnostics['households_considered_pv_free']}")
    print(f"[fetch_opsd_demand] households_selected: {diagnostics['households_selected']}")
    print("[fetch_opsd_demand] per_household_coverage_hours (all residential grid_import columns):")
    for household, hours in sorted(diagnostics["per_household_coverage_hours"].items()):
        print(f"    {household}: {hours} hours")
    print(
        f"[fetch_opsd_demand] coverage window BEFORE alignment: "
        f"{diagnostics['coverage_start_utc_before_alignment']} .. {diagnostics['coverage_end_utc_before_alignment']}"
    )
    print(f"[fetch_opsd_demand] alignment_start_offset_hours (derived from utc_timestamp, dropped to reach 00:00 UTC): {diagnostics['alignment_start_offset_hours']}")
    print(f"[fetch_opsd_demand] coverage window AFTER alignment: {diagnostics['coverage_start_utc']} .. {diagnostics['coverage_end_utc']}")
    print(f"[fetch_opsd_demand] coverage_hours (index 0 == 00:00 UTC, whole multiple of 24): {diagnostics['coverage_hours']}")
    print(f"[fetch_opsd_demand] counters_were_cumulative (differencing applied): {diagnostics['counters_were_cumulative']}")
    print(f"[fetch_opsd_demand] total pipeline elapsed_sec: {elapsed:.2f}")
    print(f"[fetch_opsd_demand] series stats: min={series.min():.5f} max={series.max():.5f} mean={series.mean():.5f} n={series.size}")

    _write_cached_csv(CACHE_PATH, series, "demand_kwh_reference", "opsd_household_fetch")
    print(f"[fetch_opsd_demand] wrote {CACHE_PATH} (source marker: opsd_household_fetch)")

    raw_path = Path(diagnostics["raw_download_path"])
    if raw_path.exists():
        raw_bytes = raw_path.stat().st_size
        raw_path.unlink()
        print(f"[fetch_opsd_demand] deleted raw intermediate download ({raw_bytes} bytes): {raw_path}")

    print("[fetch_opsd_demand] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
