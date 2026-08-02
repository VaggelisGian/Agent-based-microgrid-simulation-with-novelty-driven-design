"""D11 dose-matched monopoly arm: is competition necessary, or only sufficient?

Motivation (docs/DECISIONS.md D11). D10 found the peak-to-average channel is
deferred VOLUME, not desynchronized deferral timing. The shipped monopoly arm
(scripts/run_monopoly_arm.py, results/sweep_monopoly.parquet, broker_count 1)
ran the monopolist at FULL passthrough, because a single broker's contribution
share is 1 by construction, and that arm's failure to shave the peak is
therefore confounded with an overdose: broker_count moves both the number of
suppliers and the per-broker signal strength at once, and the shipped design
cannot hold the second fixed while varying the first. This arm can, via the
capacity_surcharge_divisor config key (docs/DECISIONS.md D11): it holds
broker_count at 1 and evaluates the pricing channel as though the broker count
were the divisor M, so the monopolist's surcharge becomes passthrough / M
under capacity_surcharge_mode "proportional", exactly the per-broker signal
strength a symmetric M-broker market delivers.

THE DIVISOR IS AN ARTIFICIAL ANALYSIS-ONLY CONTROL WITH NO MARKET
INTERPRETATION. Under the shipped allocation rule a monopolist cannot choose
to receive passthrough/M; it receives the full passthrough by construction.
Nothing in this arm describes a market a regulator could build by decree, and
no claim in the thesis may treat divisor M as a policy instrument. Its only
purpose is to break the confound the shipped design bundles, so the arm can
ask whether a broker_count-3 market's peak-to-average result survives at a
dose-matched broker_count-1 monopoly, or whether it needs the desynchronized
multi-broker structure too.

Grid: broker_count fixed at 1 (the regulated utility as incumbent monopoly,
the same single broker scripts/run_monopoly_arm.py uses);
capacity_surcharge_divisor in {2, 3, 5}; capacity_k in {0.5, 1.0}; ablation in
{capacity_disabled, capacity_both}; capacity_surcharge_mode fixed at
"proportional"; the same 30 seeds as every other sweep (sweep.SEEDS); window
sweep.WINDOW_HOURS; horizon sweep.HORIZON_HOURS.
3 x 2 x 2 x 30 = 360 runs.

Why only capacity_disabled and capacity_both: the two single-channel arms
answer which channel does the work, which the main sweep already settled and
this arm does not revisit. The 180 capacity_disabled rows are run rather than
assumed inert, even though the pricing channel is off under them and the
divisor is therefore provably inert there: they cost little and turn a
construction argument into a measurement (see tests/test_surcharge_divisor.py
and docs/DECISIONS.md D11's guardrails).

Reused, not re-run: divisor-equivalent-to-1 is the existing monopoly arm at
full passthrough (results/sweep_monopoly.parquet); the competitive comparators
are the broker_count 2, 3, 5 proportional rows already in
results/sweep_raw.parquet. Neither is re-run by this script.

Design: this reuses the main runner's config construction
(scripts/run_sensitivity_sweep.py's build_run_config, which D11 taught to
accept surcharge_divisor) and its atomic-write / result-loading helpers, the
same pattern scripts/run_monopoly_arm.py established for the bc=1 arm. This
script owns its own row schema (ROW_COLUMNS adds capacity_surcharge_divisor),
its own _row_key, _build_row, _run_one_task and _init_worker, because the D10
monopoly arm's task tuple has no divisor slot and multiprocessing needs the
worker function importable at this module's own top level (Windows' spawn
start method re-imports the module in each child process). Each run is
deterministic given (config, seed). Resumable: re-running skips any
(k, 1, ablation, seed, "proportional", divisor) tuple already present in an
existing parquet/CSV/checkpoint under --results-dir. Output:
results/sweep_dose_matched_monopoly.parquet (CSV fallback and checkpoint
alongside, sharing the stem).

Run from the repository root:
    python scripts/run_dose_matched_monopoly.py --dry-run
    python scripts/run_dose_matched_monopoly.py
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

from microgrid_sim.environment.model import MicrogridModel  # noqa: E402  (src/ on sys.path via sweep's own import)

MONOPOLY_BROKER_COUNT = 1
DOSE_MODE = "proportional"
DIVISORS = (2, 3, 5)
DOSE_K_VALUES = (0.5, 1.0)
DOSE_ABLATIONS = ("capacity_disabled", "capacity_both")
CHECKPOINT_FRACTION = 0.05


def _build_grid(k_values=DOSE_K_VALUES, divisors=DIVISORS, ablations=DOSE_ABLATIONS, seeds=None) -> tuple:
    """Build the task grid. Each task tuple already has exactly the shape
    _row_key produces below (k, broker_count, ablation, seed, mode, divisor),
    the same convention scripts/run_monopoly_arm.py's MONOPOLY_GRID uses, so a
    grid task can be compared directly against a completed-row key without an
    intermediate conversion."""
    seeds = sweep.SEEDS if seeds is None else seeds
    return tuple(
        (k, MONOPOLY_BROKER_COUNT, ablation, seed, DOSE_MODE, divisor)
        for k in k_values
        for divisor in divisors
        for ablation in ablations
        for seed in seeds
    )


DOSE_GRID = _build_grid()
TOTAL_RUNS = len(DOSE_GRID)
assert TOTAL_RUNS == 360, f"expected 360 dose-matched-monopoly grid runs, got {TOTAL_RUNS}"

# ---------------------------------------------------------------------------
# Row schema (sweep.ROW_COLUMNS plus the one extra D11 axis)
# ---------------------------------------------------------------------------

ROW_COLUMNS = sweep.ROW_COLUMNS + ["capacity_surcharge_divisor"]


def _build_row(k, broker_count, ablation, seed, mode, divisor, model, metrics) -> dict:
    row = sweep._build_row(k, broker_count, ablation, seed, mode, model, metrics)
    row["capacity_surcharge_divisor"] = divisor
    return row


def _row_key(record: dict) -> tuple:
    # sweep._row_key already backfills a missing/NaN capacity_surcharge_mode
    # to "proportional"; the divisor is backfilled the same way here (None or
    # NaN both mean "not recorded", which should not arise for this arm's own
    # rows but keeps a malformed/foreign row from crashing resume instead of
    # merely deduping wrong).
    base_key = sweep._row_key(record)
    divisor = record.get("capacity_surcharge_divisor")
    if divisor is None or divisor != divisor:  # divisor != divisor is the NaN check
        divisor_key = None
    else:
        divisor_key = int(divisor)
    return base_key + (divisor_key,)


# ---------------------------------------------------------------------------
# Worker process (multiprocessing.Pool) -- owned here, not in
# run_sensitivity_sweep, because that module's _run_one_task/_init_worker
# close over its own ROW_COLUMNS/task shape, which has no divisor slot.
# ---------------------------------------------------------------------------

_WORKER_BASE_CONFIG: dict | None = None
_WORKER_HORIZON_HOURS: int = sweep.HORIZON_HOURS


def _init_worker(base_config: dict, horizon_hours: int) -> None:
    global _WORKER_BASE_CONFIG, _WORKER_HORIZON_HOURS
    _WORKER_BASE_CONFIG = base_config
    _WORKER_HORIZON_HOURS = horizon_hours


def _run_one_task(task: tuple) -> dict:
    """Run exactly one (k, broker_count, ablation, seed, mode, divisor) to
    completion and return its row dict."""
    k, broker_count, ablation, seed, mode, divisor = task
    config = sweep.build_run_config(
        _WORKER_BASE_CONFIG,
        k,
        broker_count,
        ablation,
        seed,
        _WORKER_HORIZON_HOURS,
        surcharge_mode=mode,
        surcharge_divisor=divisor,
    )

    start = time.perf_counter()
    model = MicrogridModel(config)
    model.run()
    metrics = model.compute_metrics()
    row = _build_row(k, broker_count, ablation, seed, mode, divisor, model, metrics)
    row["run_wallclock_sec"] = time.perf_counter() - start
    row["peak_rss_mb"] = sweep._peak_rss_mb()

    del model, metrics
    import gc

    gc.collect()
    return row


# ---------------------------------------------------------------------------
# Result persistence / resume
# ---------------------------------------------------------------------------


def _result_paths(results_dir: Path) -> dict:
    return {
        "parquet": results_dir / "sweep_dose_matched_monopoly.parquet",
        "csv_fallback": results_dir / "sweep_dose_matched_monopoly.csv",
        "checkpoint": results_dir / "sweep_dose_matched_monopoly_checkpoint.csv",
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
            try:
                rows_by_key[_row_key(record)] = record
            except (KeyError, TypeError, ValueError) as exc:
                print(f"WARNING: skipping malformed row from {path}: {exc}", file=sys.stderr)
    return list(rows_by_key.values())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> tuple:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="D11 dose-matched monopoly arm: broker_count=1 at a matched per-broker dose (docs/DECISIONS.md D11)."
    )
    parser.add_argument("--config", default=str(sweep.DEFAULT_CONFIG_PATH))
    parser.add_argument("--workers", type=int, default=sweep.DEFAULT_WORKERS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=sweep.HORIZON_HOURS,
        help="TESTING ONLY override; the real arm uses the mandated 8760h horizon.",
    )
    parser.add_argument(
        "--divisors",
        default=None,
        help="comma-separated capacity_surcharge_divisor values to sweep (default: 2,3,5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the grid and print its size without running anything",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N remaining tasks (testing only)",
    )
    args = parser.parse_args(argv)

    divisors = _parse_int_list(args.divisors) if args.divisors else DIVISORS
    grid = _build_grid(divisors=divisors)
    total_runs = len(grid)

    if args.dry_run:
        print(
            f"[dose] grid: k={DOSE_K_VALUES} divisors={divisors} ablations={DOSE_ABLATIONS} "
            f"seeds={len(sweep.SEEDS)} mode={DOSE_MODE!r} broker_count={MONOPOLY_BROKER_COUNT} "
            f"-> total={total_runs}"
        )
        if divisors == DIVISORS:
            assert total_runs == 360, f"expected 360 grid runs, got {total_runs}"
            print("[dose] OK: default grid is exactly 360 runs")
        return 0

    base_config = sweep.load_config(args.config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = _result_paths(results_dir)

    existing_rows = _load_existing_rows(paths)
    completed_keys = {_row_key(row) for row in existing_rows}
    remaining_tasks = [task for task in grid if task not in completed_keys]

    if args.limit is not None:
        remaining_tasks = remaining_tasks[: args.limit]
        print(f"[dose] --limit {args.limit}: restricting this invocation to {len(remaining_tasks)} task(s)")

    print(f"[dose] total={total_runs} done={len(existing_rows)} remaining={len(remaining_tasks)}")

    all_rows = list(existing_rows)
    if remaining_tasks:
        step = max(1, round(total_runs * CHECKPOINT_FRACTION))
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
                if bucket > last_bucket or done_count == total_runs:
                    elapsed = (time.perf_counter() - loop_start) / 60.0
                    print(
                        f"[dose] progress {done_count}/{total_runs} "
                        f"({100.0 * done_count / total_runs:.1f}%) elapsed={elapsed:.1f}min"
                    )
                    sweep._atomic_write_csv(pd.DataFrame(all_rows).reindex(columns=ROW_COLUMNS), paths["checkpoint"])
                    last_bucket = bucket

    frame = pd.DataFrame(all_rows).reindex(columns=ROW_COLUMNS)
    frame = frame.sort_values(
        ["k", "broker_count", "ablation", "capacity_surcharge_divisor", "seed"]
    ).reset_index(drop=True)
    try:
        sweep._atomic_write_parquet(frame, paths["parquet"])
        out_path = paths["parquet"]
    except Exception as exc:
        print(f"WARNING: parquet write failed ({exc}); writing CSV fallback instead", file=sys.stderr)
        sweep._atomic_write_csv(frame, paths["csv_fallback"])
        out_path = paths["csv_fallback"]

    print(f"[dose] DONE: {len(all_rows)}/{total_runs} runs recorded -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
