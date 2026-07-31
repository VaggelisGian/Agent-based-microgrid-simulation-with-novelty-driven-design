"""Demand-source comparison runner: shipped synthetic demand vs. the real
OPSD household-load derived series, at the headline sweep cell.

Motivation: docs/DECISIONS.md's "Data provenance" section originally used a
generated synthetic demand shape because no quick, reliable OPSD-based
alternative was identified at the time. A real OPSD fetch now exists (see
src/microgrid_sim/data/loaders.py's fetch_and_build_opsd_demand and
scripts/fetch_opsd_demand.py, which built the cached
data/samples/residential_demand_opsd_hourly.csv). This script re-runs the
Phase 5/6 sweep's headline cell (k=1.0, broker_count=3, all four capacity
ablations, the same 30 seeds) with data.demand_series: opsd, and compares
the result against the ALREADY-COMPUTED synthetic-demand rows at that same
cell in results/sweep_raw.parquet (the Phase 5 main sweep), so no
synthetic-demand runs are repeated.

Important, honest scope note: swapping in a single aggregate OPSD series
changes the demand SHAPE (seasonal and day-to-day realism) but does not
change the model's structure, where every one of the 200 agents still reads
this ONE shared series (scaled only by its own per-agent demand_scale).
Any change in the coincidence factor / feeder-peak behaviour this
comparison shows is therefore a shape effect, not new inter-agent timing
diversity; see docs/DECISIONS.md and the loaders.py module docstring.

Grid: k=1.0 x broker_count=3 x ablation in the four D6/D7 ablations x the
main sweep's 30 seeds = 120 runs, all under data.demand_series: opsd, same
168-hour window and 8760-hour horizon as the main sweep. Output:
results/demand_source_comparison.csv (plus the raw 120-row
results/sweep_opsd_headline.parquet backing it).

Design: reuses run_sensitivity_sweep.py's config construction and worker
machinery (build_run_config, _build_row, the atomic parquet/CSV writers),
following the same pattern as scripts/run_monopoly_arm.py, with one addition
each run's config needs that the main sweep's build_run_config does not
apply on its own: data.demand_series is forced to "opsd" after the rest of
the config is built. Resumable: re-running skips any (k, broker_count,
ablation, seed) tuple already present in an existing
parquet/CSV/checkpoint under --results-dir.

CLI:
    --max-tasks N   TESTING ONLY: cap how many of the 120 tasks run, for a
                     quick smoke check. Omit for the real 120-run job.
    --horizon-hours TESTING ONLY override; the real comparison always uses
                     the mandated 8760h horizon.

Run from the repository root (the real, full comparison):
    python scripts/compare_demand_sources.py
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing
import sys
import time
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import run_sensitivity_sweep as sweep  # noqa: E402  (path set above)

HEADLINE_K = 1.0
HEADLINE_BROKER_COUNT = 3
CHECKPOINT_FRACTION = 0.2

OPSD_GRID = tuple(
    (HEADLINE_K, HEADLINE_BROKER_COUNT, ablation, seed, "proportional")
    for ablation in sweep.ABLATIONS
    for seed in sweep.SEEDS
)
TOTAL_RUNS = len(OPSD_GRID)
assert TOTAL_RUNS == 120, f"expected 120 opsd headline-cell runs, got {TOTAL_RUNS}"

METRIC_COLUMNS = (
    "avg_cost_per_agent_eur",
    "avg_cost_per_kwh_eur",
    "load_concentration_hhi",
    "feeder_coefficient_of_variation",
    "feeder_peak_to_average_ratio",
    "feeder_mean_hourly_ramp_kwh",
    "prosumer_self_sufficiency",
    "total_capacity_charge_eur",
    "capacity_fire_rate",
    "total_deferred_kwh",
    "switching_rate",
)


def _result_paths(results_dir: Path) -> dict:
    return {
        "parquet": results_dir / "sweep_opsd_headline.parquet",
        "csv_fallback": results_dir / "sweep_opsd_headline.csv",
        "checkpoint": results_dir / "sweep_opsd_headline_checkpoint.csv",
        "comparison": results_dir / "demand_source_comparison.csv",
    }


def _load_existing_rows(paths: dict) -> list:
    rows_by_key: dict = {}
    for key_name in ("checkpoint", "csv_fallback", "parquet"):
        path = paths[key_name]
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path) if key_name == "parquet" else pd.read_csv(path)
        except Exception as exc:
            print(f"WARNING: failed to read existing results file {path}: {exc}", file=sys.stderr)
            continue
        for record in frame.to_dict(orient="records"):
            # D10: backfill a pre-existing row from before capacity_surcharge_mode
            # existed, so it keys and reproduces exactly as it always did rather
            # than being re-run (same rationale as sweep._load_existing_rows).
            record.setdefault("capacity_surcharge_mode", "proportional")
            try:
                rows_by_key[sweep._row_key(record)] = record
            except (KeyError, TypeError, ValueError) as exc:
                print(f"WARNING: skipping malformed row from {path}: {exc}", file=sys.stderr)
    return list(rows_by_key.values())


def _build_opsd_run_config(base_config: dict, ablation: str, seed: int, horizon_hours: int) -> dict:
    """sweep.build_run_config handles k, broker_count, the ablation flags,
    seed, and horizon; the only thing it does not know about is the demand
    source, so that one override is applied on top here."""
    config = sweep.build_run_config(base_config, HEADLINE_K, HEADLINE_BROKER_COUNT, ablation, seed, horizon_hours)
    config.setdefault("data", {})["demand_series"] = "opsd"
    return config


# ---------------------------------------------------------------------------
# Worker process (multiprocessing.Pool)
# ---------------------------------------------------------------------------

_WORKER_BASE_CONFIG: dict | None = None
_WORKER_HORIZON_HOURS: int = sweep.HORIZON_HOURS


def _init_worker(base_config: dict, horizon_hours: int) -> None:
    global _WORKER_BASE_CONFIG, _WORKER_HORIZON_HOURS
    _WORKER_BASE_CONFIG = base_config
    _WORKER_HORIZON_HOURS = horizon_hours


def _run_one_task(task: tuple) -> dict:
    k, broker_count, ablation, seed, mode = task
    config = _build_opsd_run_config(_WORKER_BASE_CONFIG, ablation, seed, _WORKER_HORIZON_HOURS)

    start = time.perf_counter()
    model = sweep.MicrogridModel(config)
    model.run()
    metrics = model.compute_metrics()
    row = sweep._build_row(k, broker_count, ablation, seed, mode, model, metrics)
    row["run_wallclock_sec"] = time.perf_counter() - start
    row["peak_rss_mb"] = sweep._peak_rss_mb()

    del model, metrics
    gc.collect()
    return row


# ---------------------------------------------------------------------------
# Comparison against the already-computed synthetic-demand headline cell
# ---------------------------------------------------------------------------


def _load_synthetic_headline_rows(results_dir: Path) -> pd.DataFrame:
    """Read the already-computed synthetic-demand rows at (k=1.0, bc=3) from
    the main sweep's results/sweep_raw.parquet (CSV fallback if the parquet
    is absent). Raises FileNotFoundError with a clear message if neither
    exists, rather than silently comparing against nothing."""
    candidates = [results_dir / "sweep_raw.parquet", results_dir / "sweep_raw.csv"]
    for path in candidates:
        if path.exists():
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            subset = frame[(frame["k"] == HEADLINE_K) & (frame["broker_count"] == HEADLINE_BROKER_COUNT)]
            return subset.reset_index(drop=True)
    raise FileNotFoundError(
        f"neither {candidates[0]} nor {candidates[1]} found; run the main sweep "
        "(scripts/run_sensitivity_sweep.py) first so the synthetic-demand headline "
        "cell exists to compare against."
    )


def _write_comparison_csv(opsd_rows: pd.DataFrame, synthetic_rows: pd.DataFrame, path: Path) -> None:
    records = []
    for ablation in sweep.ABLATIONS:
        synth = synthetic_rows[synthetic_rows["ablation"] == ablation]
        opsd = opsd_rows[opsd_rows["ablation"] == ablation]
        for metric in METRIC_COLUMNS:
            synth_vals = pd.to_numeric(synth[metric], errors="coerce") if metric in synth.columns else pd.Series(dtype=float)
            opsd_vals = pd.to_numeric(opsd[metric], errors="coerce") if metric in opsd.columns else pd.Series(dtype=float)
            records.append(
                {
                    "k": HEADLINE_K,
                    "broker_count": HEADLINE_BROKER_COUNT,
                    "ablation": ablation,
                    "metric": metric,
                    "n_seeds_synthetic": int(synth_vals.notna().sum()),
                    "n_seeds_opsd": int(opsd_vals.notna().sum()),
                    "mean_synthetic": synth_vals.mean(),
                    "mean_opsd": opsd_vals.mean(),
                    "std_synthetic": synth_vals.std(ddof=1),
                    "std_opsd": opsd_vals.std(ddof=1),
                    "delta_opsd_minus_synthetic": opsd_vals.mean() - synth_vals.mean(),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame.from_records(records).to_csv(tmp_path, index=False)
    import os

    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Headline-cell (k=1.0, broker_count=3) demand-source comparison: synthetic vs OPSD."
    )
    parser.add_argument("--config", default=str(sweep.DEFAULT_CONFIG_PATH))
    parser.add_argument("--workers", type=int, default=sweep.DEFAULT_WORKERS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=sweep.HORIZON_HOURS,
        help="TESTING ONLY override; the real comparison always uses the mandated 8760h horizon.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="TESTING ONLY: cap how many of the 120 grid tasks run, for a quick smoke check.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Only run the OPSD grid; skip writing demand_source_comparison.csv "
        "(useful when results/sweep_raw.parquet is not available, e.g. a smoke test).",
    )
    args = parser.parse_args(argv)

    base_config = sweep.load_config(args.config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = _result_paths(results_dir)

    existing_rows = _load_existing_rows(paths)
    completed_keys = {sweep._row_key(row) for row in existing_rows}
    remaining_tasks = [task for task in OPSD_GRID if task not in completed_keys]
    if args.max_tasks is not None:
        remaining_tasks = remaining_tasks[: args.max_tasks]
    print(f"[opsd_headline] total={TOTAL_RUNS} done={len(existing_rows)} remaining_this_run={len(remaining_tasks)}")

    all_rows = list(existing_rows)
    if remaining_tasks:
        step = max(1, round(len(remaining_tasks) * CHECKPOINT_FRACTION))
        done_count = len(all_rows)
        last_bucket = done_count // step
        loop_start = time.perf_counter()
        with multiprocessing.Pool(
            processes=args.workers, initializer=_init_worker, initargs=(base_config, args.horizon_hours)
        ) as pool:
            for row in pool.imap_unordered(_run_one_task, remaining_tasks, chunksize=1):
                all_rows.append(row)
                done_count += 1
                bucket = done_count // step
                if bucket > last_bucket or done_count == len(existing_rows) + len(remaining_tasks):
                    elapsed = (time.perf_counter() - loop_start) / 60.0
                    print(
                        f"[opsd_headline] progress {done_count}/{TOTAL_RUNS} "
                        f"({100.0 * done_count / TOTAL_RUNS:.1f}%) elapsed={elapsed:.1f}min"
                    )
                    sweep._atomic_write_csv(pd.DataFrame(all_rows).reindex(columns=sweep.ROW_COLUMNS), paths["checkpoint"])
                    last_bucket = bucket

    opsd_frame = pd.DataFrame(all_rows).reindex(columns=sweep.ROW_COLUMNS)
    opsd_frame = opsd_frame.sort_values(["ablation", "seed"]).reset_index(drop=True)
    try:
        sweep._atomic_write_parquet(opsd_frame, paths["parquet"])
        out_path = paths["parquet"]
    except Exception as exc:
        print(f"WARNING: parquet write failed ({exc}); writing CSV fallback instead", file=sys.stderr)
        sweep._atomic_write_csv(opsd_frame, paths["csv_fallback"])
        out_path = paths["csv_fallback"]
    print(f"[opsd_headline] wrote {len(all_rows)}/{TOTAL_RUNS} rows -> {out_path}")

    if len(all_rows) < TOTAL_RUNS:
        print(
            f"[opsd_headline] grid incomplete ({len(all_rows)}/{TOTAL_RUNS}); "
            "skipping demand_source_comparison.csv until a full run completes."
        )
        return 0
    if args.skip_comparison:
        print("[opsd_headline] --skip-comparison set; not writing demand_source_comparison.csv.")
        return 0

    synthetic_rows = _load_synthetic_headline_rows(results_dir)
    _write_comparison_csv(opsd_frame, synthetic_rows, paths["comparison"])
    print(f"[opsd_headline] wrote comparison -> {paths['comparison']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
