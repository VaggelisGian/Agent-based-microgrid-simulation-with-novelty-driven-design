"""Supplementary monopoly (broker_count = 1) arm for the capacity mechanism.

Motivation: the Phase 8 defense-readiness audit asked for direct evidence that
the metric-3 (feeder stability) improvement is driven by the capacity-charge
pricing/deferral MECHANISM rather than by the NUMBER of competing brokers. The
main D8 sweep varies broker_count over 2, 3, 5 and shows the coefficient-of-
variation gain is broker-count-invariant, which is indirect evidence. This arm
adds the missing single-broker (monopoly) baseline so the attribution can be
made directly.

Grid: capacity_k in {0.5, 1.0, 1.5, 2.0} x broker_count in {1} x ablation in the
four ablations x 30 seeds = 480 runs, with the same 168-hour window and 8760-hour
horizon as the main sweep. Output: results/sweep_monopoly.parquet.

Design: this reuses the main runner's config construction and worker machinery
(scripts/run_sensitivity_sweep.py) unchanged. Only the grid (broker_count fixed
at 1) and the output filename differ, so the main 1440-run artifact
(results/sweep_raw.parquet) is left untouched and D8's design is preserved. The
single broker is the regulated utility (the incumbent monopoly). Each run is
deterministic given (config, seed). Resumable: re-running skips any
(k, 1, ablation, seed) tuple already present in an existing
parquet/CSV/checkpoint under --results-dir.

Run from the repository root:
    python scripts/run_monopoly_arm.py
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import run_sensitivity_sweep as sweep  # noqa: E402  (path set above)

MONOPOLY_BROKER_COUNT = 1
CHECKPOINT_FRACTION = 0.05

MONOPOLY_GRID = tuple(
    (k, MONOPOLY_BROKER_COUNT, ablation, seed, "proportional")
    for k in sweep.K_VALUES
    for ablation in sweep.ABLATIONS
    for seed in sweep.SEEDS
)
TOTAL_RUNS = len(MONOPOLY_GRID)
assert TOTAL_RUNS == 480, f"expected 480 monopoly runs, got {TOTAL_RUNS}"


def _result_paths(results_dir: Path) -> dict:
    return {
        "parquet": results_dir / "sweep_monopoly.parquet",
        "csv_fallback": results_dir / "sweep_monopoly.csv",
        "checkpoint": results_dir / "sweep_monopoly_checkpoint.csv",
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Supplementary bc=1 monopoly arm (audit follow-up).")
    parser.add_argument("--config", default=str(sweep.DEFAULT_CONFIG_PATH))
    parser.add_argument("--workers", type=int, default=sweep.DEFAULT_WORKERS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=sweep.HORIZON_HOURS,
        help="TESTING ONLY override; the real arm uses the mandated 8760h horizon.",
    )
    args = parser.parse_args(argv)

    base_config = sweep.load_config(args.config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = _result_paths(results_dir)

    existing_rows = _load_existing_rows(paths)
    completed_keys = {sweep._row_key(row) for row in existing_rows}
    remaining_tasks = [task for task in MONOPOLY_GRID if task not in completed_keys]
    print(f"[monopoly] total={TOTAL_RUNS} done={len(existing_rows)} remaining={len(remaining_tasks)}")

    all_rows = list(existing_rows)
    if remaining_tasks:
        step = max(1, round(TOTAL_RUNS * CHECKPOINT_FRACTION))
        done_count = len(all_rows)
        last_bucket = done_count // step
        loop_start = time.perf_counter()
        with multiprocessing.Pool(
            processes=args.workers, initializer=sweep._init_worker, initargs=(base_config, args.horizon_hours)
        ) as pool:
            for row in pool.imap_unordered(sweep._run_one_task, remaining_tasks, chunksize=1):
                all_rows.append(row)
                done_count += 1
                bucket = done_count // step
                if bucket > last_bucket or done_count == TOTAL_RUNS:
                    elapsed = (time.perf_counter() - loop_start) / 60.0
                    print(f"[monopoly] progress {done_count}/{TOTAL_RUNS} ({100.0 * done_count / TOTAL_RUNS:.1f}%) elapsed={elapsed:.1f}min")
                    sweep._atomic_write_csv(
                        pd.DataFrame(all_rows).reindex(columns=sweep.ROW_COLUMNS), paths["checkpoint"]
                    )
                    last_bucket = bucket

    frame = pd.DataFrame(all_rows).reindex(columns=sweep.ROW_COLUMNS)
    frame = frame.sort_values(
        ["k", "broker_count", "ablation", "capacity_surcharge_mode", "seed"]
    ).reset_index(drop=True)
    try:
        sweep._atomic_write_parquet(frame, paths["parquet"])
        out_path = paths["parquet"]
    except Exception as exc:
        print(f"WARNING: parquet write failed ({exc}); writing CSV fallback instead", file=sys.stderr)
        sweep._atomic_write_csv(frame, paths["csv_fallback"])
        out_path = paths["csv_fallback"]

    print(f"[monopoly] DONE: {len(all_rows)}/{TOTAL_RUNS} runs recorded -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
