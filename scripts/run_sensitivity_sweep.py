"""Phase 5 sensitivity-sweep runner (docs/DECISIONS.md D8).

Runs the 1440-run grid: capacity_k in {0.5, 1.0, 1.5, 2.0} x broker_count in
{2, 3, 5} x ablation in {capacity_disabled, capacity_pnl_only,
capacity_pricing_only, capacity_both} x 30 seeds, window fixed at 168 hours,
horizon fixed at 8760 hours (one year), num_agents at the config default (200).

Each run is deterministic given (k, broker_count, ablation, seed): config is
built by deep-copying config/default.yaml once loaded and overriding only
simulation.seed, simulation.horizon_hours, capacity_mechanism.{enabled,
feedback_pnl, feedback_pricing, k, window}, and brokers.

Broker-count construction (D8):
- bc=3: the three default brokers (regulated_utility, volatile_low_cost,
  premium_green), unchanged.
- bc=2: regulated_utility and volatile_low_cost only (premium_green dropped).
  Green-preferring agents then have no acceptable broker under their
  categorical filter and fall back to the overall cheapest via the existing
  model logic (MicrogridModel._select_initial_broker / Consumer._reconsider);
  this is exercised and confirmed not to raise by --mode dry-run.
- bc=5: the three defaults PLUS one duplicate of volatile_low_cost and one
  duplicate of premium_green, each with a small deterministic jitter derived
  from the run seed only (see _deterministic_jitter / _build_broker_list) so
  the duplicates are not identical, are reproducible from (config, seed), and
  draw no unseeded randomness. The jitter multiplies volatile_low_cost's
  baseline_eur_per_kwh and premium_green's markup_eur_per_kwh by
  (1 + offset), offset in [-JITTER_MAGNITUDE, +JITTER_MAGNITUDE] (a few
  percent), derived from (seed, a fixed per-field salt) via a fixed
  linear-congruential-style transform -- not Python's salted hash()
  builtin, and not a call into the model's own seeded RNG stream.

One row per run records the config keys, the four thesis metrics, and
audit columns (see ROW_COLUMNS below); results/sweep_raw.parquet is written at
the end (CSV fallback if the parquet writer fails), with an incremental CSV
checkpoint every ~5% of the grid for crash-resume. Re-running this script
skips any (k, broker_count, ablation, seed) tuple already present in an
existing parquet/CSV/checkpoint file under --results-dir.

CLI modes:
    --mode dry-run   build all 12 (broker_count, ablation) configs at one
                     (k, seed) WITHOUT running the horizon; verify broker
                     counts and ablation flags.
    --mode smoke     run a small, real batch of grid runs (see --smoke-size)
                     to measure per-run wallclock/RSS and print the estimated
                     full-sweep wallclock; does NOT continue to the rest of
                     the grid. Rows produced are real, valid grid rows and
                     are persisted (so they count toward the real sweep and
                     are not wasted), scoped to --results-dir.
    --mode full      the actual 1440-run sweep: smoke-estimate first (reusing
                     its 8 runs), then the resumable multiprocessing sweep
                     over whatever remains.

Run from the repository root (data/ and config/ paths are relative), e.g.:
    python scripts/run_sensitivity_sweep.py --mode dry-run
    python scripts/run_sensitivity_sweep.py --mode smoke
    python scripts/run_sensitivity_sweep.py --mode full
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import multiprocessing
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import psutil

from microgrid_sim.config.loader import load_config
from microgrid_sim.environment.model import MicrogridModel

# Mesa 3.5.1 still supports seed= (the model-driver API this script uses) but
# nudges toward rng=; functionally identical per Mesa's own warning message
# (matches scripts/run_baseline.py). This filter is at module level so it is
# also installed in every spawned worker process (module top-level code runs
# again on import in each child under Windows' spawn start method).
warnings.filterwarnings(
    "ignore", message="The use of the `seed` keyword argument is deprecated", category=FutureWarning
)

# ---------------------------------------------------------------------------
# D8 grid constants
# ---------------------------------------------------------------------------

K_VALUES = (0.5, 1.0, 1.5, 2.0)
BROKER_COUNTS = (2, 3, 5)
ABLATION_FLAGS = {
    "capacity_disabled": {"enabled": False, "feedback_pnl": False, "feedback_pricing": False},
    "capacity_pnl_only": {"enabled": True, "feedback_pnl": True, "feedback_pricing": False},
    "capacity_pricing_only": {"enabled": True, "feedback_pnl": False, "feedback_pricing": True},
    "capacity_both": {"enabled": True, "feedback_pnl": True, "feedback_pricing": True},
}
ABLATIONS = tuple(ABLATION_FLAGS.keys())

# Fixed, deterministic, documented seed list (brief: "for example 0..29, or a
# fixed base plus 0..29"): the project's canonical fixed seed (see
# config/default.yaml, simulation.seed) plus a 0..29 offset, so the sweep's
# seed convention is visibly tied to the rest of the project rather than an
# arbitrary small-integer list.
SEED_BASE = 20260704
SEEDS = tuple(SEED_BASE + offset for offset in range(30))

WINDOW_HOURS = 168  # D8: window fixed across the whole sweep
HORIZON_HOURS = 8760  # D8: one year, the mandated sweep horizon

FULL_GRID = tuple(
    (k, bc, ablation, seed)
    for k in K_VALUES
    for bc in BROKER_COUNTS
    for ablation in ABLATIONS
    for seed in SEEDS
)
TOTAL_RUNS = len(FULL_GRID)
assert TOTAL_RUNS == 1440, f"expected 1440 grid runs, got {TOTAL_RUNS}"

# ---------------------------------------------------------------------------
# Hardware policy (docs/HARDWARE.md)
# ---------------------------------------------------------------------------

DEFAULT_WORKERS = 16
RAM_BUDGET_GB = 30.0
WALLCLOCK_WARN_HOURS = 6.0
DEFAULT_SMOKE_SIZE = 8
CHECKPOINT_FRACTION = 0.05  # flush + log progress every ~5% of the grid

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

# ---------------------------------------------------------------------------
# bc=5 duplicate-broker jitter (D8)
# ---------------------------------------------------------------------------

JITTER_MAGNITUDE = 0.03  # +/- 3%, "a few percent" per the brief
_JITTER_SALT_VOLATILE_DUP = 101
_JITTER_SALT_GREEN_DUP = 202


def _deterministic_jitter(seed: int, salt: int, magnitude: float = JITTER_MAGNITUDE) -> float:
    """A pure, deterministic function of (seed, salt) in [-magnitude, +magnitude).

    Used only to perturb the bc=5 duplicate brokers' price parameters so they
    are not identical to the originals. Combines (seed, salt) into one 64-bit
    word and runs the fixed splitmix64 finalizer (Vigna), whose avalanche gives
    every input bit close to a 50 percent chance of flipping every output bit,
    so even the sweep's 30 consecutive seeds map to well-spread jitter values
    that take both signs. This replaces an earlier multiply-and-mod-10000
    transform whose multiplier reduced to 3 mod 10000 and compressed 30
    consecutive seeds into under 1 percent of the intended range (always
    negative), which made the cheaper duplicate win essentially the whole
    customer base of its broker type for every seed. Uses a fixed 64-bit
    integer avalanche, NOT Python's salted hash() builtin and NOT a draw from
    the model's own seeded RNG stream, so it introduces no unseeded randomness
    and never perturbs the RNG sequence MicrogridModel itself consumes.
    """
    mask = (1 << 64) - 1
    state = (seed * 0x9E3779B97F4A7C15 + salt * 0xD1B54A32D192ED03 + 0x2545F4914F6CDD1D) & mask
    state ^= state >> 30
    state = (state * 0xBF58476D1CE4E5B9) & mask
    state ^= state >> 27
    state = (state * 0x94D049BB133111EB) & mask
    state ^= state >> 31
    fraction = state / float(1 << 64)  # deterministic value in [0, 1)
    return magnitude * (2.0 * fraction - 1.0)


def _build_broker_list(base_brokers: list[dict], broker_count: int, seed: int) -> list[dict]:
    """Build the bc-specific broker config list from the default.yaml broker set."""
    by_type = {broker["type"]: broker for broker in base_brokers}
    for required_type in ("regulated_utility", "volatile_low_cost", "premium_green"):
        if required_type not in by_type:
            raise ValueError(f"base config brokers list is missing a '{required_type}' entry")

    regulated = copy.deepcopy(by_type["regulated_utility"])
    volatile = copy.deepcopy(by_type["volatile_low_cost"])
    green = copy.deepcopy(by_type["premium_green"])

    if broker_count == 1:
        # Single-broker monopoly baseline (the abstract's monopolistic broker).
        # The regulated utility is the incumbent monopoly. Used only by the
        # supplementary bc=1 arm (scripts/run_monopoly_arm.py), never by the
        # main D8 grid (BROKER_COUNTS = 2, 3, 5). Every agent is served by this
        # one broker; there is nowhere to switch, so switching_rate is 0.
        return [regulated]
    if broker_count == 2:
        return [regulated, volatile]
    if broker_count == 3:
        return [regulated, volatile, green]
    if broker_count == 5:
        volatile_dup = copy.deepcopy(volatile)
        volatile_dup["id"] = f"{volatile['id']}_dup2"
        volatile_dup["name"] = f"{volatile['name']} (dup 2)"
        jitter_v = _deterministic_jitter(seed, _JITTER_SALT_VOLATILE_DUP)
        volatile_dup["baseline_eur_per_kwh"] = volatile["baseline_eur_per_kwh"] * (1.0 + jitter_v)

        green_dup = copy.deepcopy(green)
        green_dup["id"] = f"{green['id']}_dup2"
        green_dup["name"] = f"{green['name']} (dup 2)"
        jitter_g = _deterministic_jitter(seed, _JITTER_SALT_GREEN_DUP)
        green_dup["markup_eur_per_kwh"] = green["markup_eur_per_kwh"] * (1.0 + jitter_g)

        return [regulated, volatile, green, volatile_dup, green_dup]
    raise ValueError(f"unsupported broker_count: {broker_count} (must be one of {BROKER_COUNTS})")


def build_run_config(
    base_config: dict, k: float, broker_count: int, ablation: str, seed: int, horizon_hours: int = HORIZON_HOURS
) -> dict:
    """Build one run's config: deepcopy the base config, then override only
    simulation.seed, simulation.horizon_hours, capacity_mechanism.{enabled,
    feedback_pnl, feedback_pricing, k, window}, and brokers. Nothing else is
    touched, so num_agents and every structural coefficient stay at the
    config/default.yaml value (D8: "no parameter other than k, broker_count,
    and the ablation flags varies across the sweep").
    """
    if ablation not in ABLATION_FLAGS:
        raise ValueError(f"unknown ablation: {ablation}")

    config = copy.deepcopy(base_config)
    config["simulation"]["seed"] = seed
    config["simulation"]["horizon_hours"] = horizon_hours

    capacity_cfg = config.setdefault("capacity_mechanism", {})
    flags = ABLATION_FLAGS[ablation]
    capacity_cfg["enabled"] = flags["enabled"]
    capacity_cfg["feedback_pnl"] = flags["feedback_pnl"]
    capacity_cfg["feedback_pricing"] = flags["feedback_pricing"]
    capacity_cfg["k"] = k
    capacity_cfg["window"] = WINDOW_HOURS

    config["brokers"] = _build_broker_list(base_config["brokers"], broker_count, seed)
    return config


# ---------------------------------------------------------------------------
# Row schema
# ---------------------------------------------------------------------------

ROW_COLUMNS = [
    "k",
    "broker_count",
    "ablation",
    "seed",
    "avg_cost_per_agent_eur",
    "avg_cost_per_kwh_eur",
    "load_concentration_hhi",
    "broker_load_share_json",
    "feeder_coefficient_of_variation",
    "feeder_peak_to_average_ratio",
    "feeder_mean_hourly_ramp_kwh",
    "prosumer_self_sufficiency",
    "total_capacity_charge_eur",
    "capacity_fire_rate",
    "mean_broker_surcharge_json",
    "broker_load_share_gini",
    "total_deferred_kwh",
    "switching_rate",
    "n_brokers",
    "run_wallclock_sec",
    "peak_rss_mb",
]


def _gini(values: list) -> float:
    """Gini coefficient of a non-negative vector; 0.0 for an empty/degenerate/
    all-zero vector (a single broker, or no energy served, is "perfectly
    equal" by convention here, not undefined)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    total = sum(ordered)
    if n <= 1 or total <= 0.0:
        return 0.0
    cumulative = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * cumulative) / (n * total) - (n + 1) / n


def _build_row(k: float, broker_count: int, ablation: str, seed: int, model, metrics) -> dict:
    broker_load_share = dict(metrics.broker_load_share)
    mean_surcharge = dict(metrics.mean_broker_surcharge_eur_per_kwh)

    agents = list(model.agents)
    num_agents = model.num_agents
    switched = sum(1 for agent in agents if getattr(agent, "switch_count", 0) > 0)
    switching_rate = (switched / num_agents) if num_agents > 0 else float("nan")

    import json

    return {
        "k": k,
        "broker_count": broker_count,
        "ablation": ablation,
        "seed": seed,
        "avg_cost_per_agent_eur": metrics.avg_cost_per_agent_eur,
        "avg_cost_per_kwh_eur": metrics.avg_cost_per_kwh_eur,
        "load_concentration_hhi": metrics.load_concentration_hhi,
        "broker_load_share_json": json.dumps(broker_load_share, sort_keys=True),
        "feeder_coefficient_of_variation": metrics.feeder_coefficient_of_variation,
        "feeder_peak_to_average_ratio": metrics.feeder_peak_to_average_ratio,
        "feeder_mean_hourly_ramp_kwh": metrics.feeder_mean_hourly_ramp_kwh,
        "prosumer_self_sufficiency": metrics.prosumer_self_sufficiency,
        "total_capacity_charge_eur": metrics.total_capacity_charge_eur,
        "capacity_fire_rate": metrics.capacity_fire_rate,
        "mean_broker_surcharge_json": json.dumps(mean_surcharge, sort_keys=True),
        "broker_load_share_gini": _gini(list(broker_load_share.values())),
        "total_deferred_kwh": metrics.total_deferred_kwh,
        "switching_rate": switching_rate,
        "n_brokers": len(broker_load_share),
    }


def _row_key(record: dict) -> tuple:
    return (float(record["k"]), int(record["broker_count"]), str(record["ablation"]), int(record["seed"]))


# ---------------------------------------------------------------------------
# Worker process (multiprocessing.Pool)
# ---------------------------------------------------------------------------

_WORKER_BASE_CONFIG: dict | None = None
_WORKER_HORIZON_HOURS: int = HORIZON_HOURS


def _init_worker(base_config: dict, horizon_hours: int) -> None:
    global _WORKER_BASE_CONFIG, _WORKER_HORIZON_HOURS
    _WORKER_BASE_CONFIG = base_config
    _WORKER_HORIZON_HOURS = horizon_hours


def _peak_rss_mb() -> float:
    """Peak resident-set size of this worker process, in MB. Uses psutil's
    Windows-specific peak_wset field when available (a true watermark across
    the process's lifetime so far), falling back to the current rss on
    platforms where no peak field exists."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        peak_bytes = getattr(mem_info, "peak_wset", mem_info.rss)
        return peak_bytes / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def _run_one_task(task: tuple) -> dict:
    """Run exactly one (k, broker_count, ablation, seed) to completion and
    return its row dict. Releases the model before returning (no accumulation
    of model objects across runs within a persistent worker process)."""
    k, broker_count, ablation, seed = task
    config = build_run_config(_WORKER_BASE_CONFIG, k, broker_count, ablation, seed, _WORKER_HORIZON_HOURS)

    start = time.perf_counter()
    model = MicrogridModel(config)
    model.run()
    metrics = model.compute_metrics()
    row = _build_row(k, broker_count, ablation, seed, model, metrics)
    row["run_wallclock_sec"] = time.perf_counter() - start
    row["peak_rss_mb"] = _peak_rss_mb()

    del model, metrics
    import gc

    gc.collect()
    return row


# ---------------------------------------------------------------------------
# Result persistence / resume
# ---------------------------------------------------------------------------


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _result_paths(results_dir: Path) -> dict:
    return {
        "parquet": results_dir / "sweep_raw.parquet",
        "csv_fallback": results_dir / "sweep_raw.csv",
        "checkpoint": results_dir / "sweep_raw_checkpoint.csv",
    }


def _load_existing_rows(paths: dict) -> list:
    """Read whatever result files already exist (checkpoint, CSV fallback,
    parquet, in that order so a completed parquet -- if present -- has the
    final say for any duplicate key) and return the deduplicated row records.
    """
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


def _write_final_results(rows: list, results_dir: Path) -> Path:
    paths = _result_paths(results_dir)
    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=ROW_COLUMNS)
    frame = frame.sort_values(["k", "broker_count", "ablation", "seed"]).reset_index(drop=True)
    try:
        _atomic_write_parquet(frame, paths["parquet"])
        return paths["parquet"]
    except Exception as exc:
        print(f"WARNING: parquet write failed ({exc}); writing CSV fallback instead", file=sys.stderr)
        _atomic_write_csv(frame, paths["csv_fallback"])
        return paths["csv_fallback"]


# ---------------------------------------------------------------------------
# Smoke batch / estimate
# ---------------------------------------------------------------------------

# Spread across broker_count and ablation (D8's two config axes besides k and
# seed); k cycles across K_VALUES for extra variety. All 8 combos here are
# real grid tasks (at seed == SEEDS[0]) that count toward the real sweep.
_SMOKE_COMBOS = (
    (2, "capacity_disabled"),
    (2, "capacity_both"),
    (3, "capacity_disabled"),
    (3, "capacity_pnl_only"),
    (3, "capacity_pricing_only"),
    (3, "capacity_both"),
    (5, "capacity_disabled"),
    (5, "capacity_both"),
)


def _pick_smoke_tasks(remaining_tasks: list, smoke_size: int) -> list:
    if not remaining_tasks or smoke_size <= 0:
        return []
    remaining_set = set(remaining_tasks)
    picked: list = []
    for index, (broker_count, ablation) in enumerate(_SMOKE_COMBOS):
        if len(picked) >= smoke_size:
            break
        k = K_VALUES[index % len(K_VALUES)]
        seed = SEEDS[0]
        task = (k, broker_count, ablation, seed)
        if task in remaining_set and task not in picked:
            picked.append(task)
    # Pad deterministically from the remaining grid if some preferred combos
    # were already completed by an earlier run (resume case).
    for task in sorted(remaining_tasks):
        if len(picked) >= smoke_size:
            break
        if task not in picked:
            picked.append(task)
    return picked[:smoke_size]


def _run_batch(tasks: list, base_config: dict, workers: int, horizon_hours: int) -> list:
    if not tasks:
        return []
    with multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=(base_config, horizon_hours)) as pool:
        rows = pool.map(_run_one_task, tasks, chunksize=1)
    return rows


def _estimate_via_smoke(
    base_config: dict, workers: int, horizon_hours: int, smoke_size: int, existing_rows: list, remaining_tasks: list
) -> dict:
    """Run a real smoke batch (subset of remaining_tasks), measure per-run
    wallclock/RSS, extrapolate the full 1440-run wallclock at `workers`
    workers, and apply the RAM-budget worker-count adjustment. Returns a dict
    describing what happened plus the updated row/task lists so the caller
    can either stop here (--mode smoke) or continue into the full loop.
    """
    smoke_tasks = _pick_smoke_tasks(remaining_tasks, smoke_size)
    if not smoke_tasks:
        print("[sweep] no remaining tasks available for the smoke batch (grid already complete).")
        return {
            "effective_workers": workers,
            "smoke_rows": [],
            "avg_wallclock_sec": float("nan"),
            "avg_rss_gb": float("nan"),
            "estimated_full_hours": 0.0,
            "all_rows": list(existing_rows),
            "remaining_after_smoke": list(remaining_tasks),
        }

    print(
        f"[sweep] smoke batch: {len(smoke_tasks)} runs at horizon={horizon_hours}h, "
        f"{workers}-worker pool (a smoke batch smaller than `workers` under-uses the "
        f"pool, so measured contention is a lower bound on the full sweep's)..."
    )
    smoke_start = time.perf_counter()
    smoke_rows = _run_batch(smoke_tasks, base_config, workers, horizon_hours)
    smoke_wall = time.perf_counter() - smoke_start
    print(f"[sweep] smoke batch finished in {smoke_wall:.1f}s wall-time ({len(smoke_tasks)} runs).")

    avg_wallclock_sec = statistics.fmean(row["run_wallclock_sec"] for row in smoke_rows)
    rss_values = [row["peak_rss_mb"] for row in smoke_rows if row["peak_rss_mb"] == row["peak_rss_mb"]]  # drop NaN
    avg_rss_gb = (statistics.fmean(rss_values) / 1024.0) if rss_values else float("nan")

    print(f"[sweep] measured: avg per-run wallclock = {avg_wallclock_sec:.2f}s, avg peak RSS/run = {avg_rss_gb:.4f} GB")

    effective_workers = workers
    if avg_rss_gb == avg_rss_gb and avg_rss_gb * workers > RAM_BUDGET_GB:  # avg_rss_gb == avg_rss_gb rules out NaN
        adjusted = max(1, math.floor(RAM_BUDGET_GB / avg_rss_gb))
        print(
            f"[sweep] ADJUSTMENT: {avg_rss_gb:.4f} GB/run * {workers} workers = "
            f"{avg_rss_gb * workers:.2f} GB > {RAM_BUDGET_GB:.0f} GB budget; "
            f"reducing workers {workers} -> {adjusted}"
        )
        effective_workers = adjusted

    estimated_full_hours = (TOTAL_RUNS * avg_wallclock_sec) / effective_workers / 3600.0
    print(
        f"[sweep] estimated full {TOTAL_RUNS}-run sweep wallclock at {effective_workers} workers: "
        f"{estimated_full_hours:.2f} hours"
    )
    if estimated_full_hours > WALLCLOCK_WARN_HOURS:
        print(
            f"WARNING: estimated wallclock {estimated_full_hours:.2f}h exceeds the "
            f"{WALLCLOCK_WARN_HOURS:.0f}-hour budget; proceeding anyway.",
            file=sys.stderr,
        )

    all_rows = list(existing_rows) + smoke_rows
    completed_smoke_keys = {_row_key(row) for row in smoke_rows}
    remaining_after_smoke = [task for task in remaining_tasks if task not in completed_smoke_keys]

    return {
        "effective_workers": effective_workers,
        "smoke_rows": smoke_rows,
        "avg_wallclock_sec": avg_wallclock_sec,
        "avg_rss_gb": avg_rss_gb,
        "estimated_full_hours": estimated_full_hours,
        "all_rows": all_rows,
        "remaining_after_smoke": remaining_after_smoke,
    }


# ---------------------------------------------------------------------------
# Main resumable loop
# ---------------------------------------------------------------------------


def _run_main_loop(
    remaining_tasks: list, base_config: dict, workers: int, horizon_hours: int, all_rows: list, results_dir: Path
) -> list:
    paths = _result_paths(results_dir)
    step = max(1, round(TOTAL_RUNS * CHECKPOINT_FRACTION))
    done_count = len(all_rows)
    last_bucket = done_count // step

    tasks_left = list(remaining_tasks)
    current_workers = workers

    while tasks_left:
        loop_start = time.perf_counter()
        processed_this_attempt = 0
        try:
            with multiprocessing.Pool(
                processes=current_workers, initializer=_init_worker, initargs=(base_config, horizon_hours)
            ) as pool:
                for row in pool.imap_unordered(_run_one_task, tasks_left, chunksize=1):
                    all_rows.append(row)
                    done_count += 1
                    processed_this_attempt += 1
                    bucket = done_count // step
                    if bucket > last_bucket or done_count == TOTAL_RUNS:
                        elapsed = time.perf_counter() - loop_start
                        rate = processed_this_attempt / elapsed if elapsed > 0 else 0.0
                        remaining_n = TOTAL_RUNS - done_count
                        eta_min = (remaining_n / rate / 60.0) if rate > 0 else float("nan")
                        pct = 100.0 * done_count / TOTAL_RUNS
                        print(
                            f"[sweep] progress {done_count}/{TOTAL_RUNS} ({pct:.1f}%) "
                            f"elapsed={elapsed / 60.0:.1f}min ETA={eta_min:.1f}min workers={current_workers}"
                        )
                        _atomic_write_csv(pd.DataFrame(all_rows).reindex(columns=ROW_COLUMNS), paths["checkpoint"])
                        last_bucket = bucket
            tasks_left = []  # normal completion: the pool exhausted tasks_left without error
        except MemoryError as exc:
            new_workers = max(1, current_workers - 1)
            print(
                f"WARNING: MemoryError in the sweep pool ({exc}); reducing workers "
                f"{current_workers} -> {new_workers} and resuming from the last checkpoint.",
                file=sys.stderr,
            )
            current_workers = new_workers
            done_keys = {_row_key(row) for row in all_rows}
            tasks_left = [task for task in tasks_left if task not in done_keys]
            _atomic_write_csv(pd.DataFrame(all_rows).reindex(columns=ROW_COLUMNS), paths["checkpoint"])
            # loop retries with tasks_left and the reduced worker count

    return all_rows


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_dry_run(base_config: dict) -> bool:
    """Build all 12 (broker_count, ablation) configs at one (k, seed) WITHOUT
    running the horizon; confirm broker counts and ablation flags."""
    k = K_VALUES[0]
    seed = SEEDS[0]
    all_ok = True
    print(f"[dry-run] k={k}, seed={seed}; building all {len(BROKER_COUNTS)}x{len(ABLATIONS)} combos...")

    for broker_count, ablation in itertools.product(BROKER_COUNTS, ABLATIONS):
        try:
            config = build_run_config(base_config, k, broker_count, ablation, seed)
            if len(config["brokers"]) != broker_count:
                raise AssertionError(f"config has {len(config['brokers'])} brokers, expected {broker_count}")

            flags = ABLATION_FLAGS[ablation]
            capacity_cfg = config["capacity_mechanism"]
            for flag_name, expected in flags.items():
                if capacity_cfg[flag_name] != expected:
                    raise AssertionError(f"capacity_mechanism.{flag_name} = {capacity_cfg[flag_name]!r}, expected {expected!r}")

            model = MicrogridModel(config)  # construct only: exercises population build, no .run()
            if len(model.brokers) != broker_count:
                raise AssertionError(f"model built {len(model.brokers)} brokers, expected {broker_count}")

            broker_ids = list(model.brokers.keys())
            print(f"[dry-run] OK  bc={broker_count} ablation={ablation:<22} brokers={broker_ids}")
        except Exception as exc:
            all_ok = False
            print(f"[dry-run] FAIL bc={broker_count} ablation={ablation}: {exc}", file=sys.stderr)

    verdict = "ALL PASS" if all_ok else "FAILURES PRESENT"
    print(f"[dry-run] {verdict} ({len(BROKER_COUNTS) * len(ABLATIONS)} combos checked)")
    return all_ok


def cmd_smoke(base_config: dict, workers: int, horizon_hours: int, smoke_size: int, results_dir: Path) -> dict:
    paths = _result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_existing_rows(paths)
    completed_keys = {_row_key(row) for row in existing_rows}
    remaining_tasks = [task for task in FULL_GRID if task not in completed_keys]

    print(f"[sweep] mode=smoke  total_grid={TOTAL_RUNS}  already_completed={len(existing_rows)}  remaining={len(remaining_tasks)}")

    estimate = _estimate_via_smoke(base_config, workers, horizon_hours, smoke_size, existing_rows, remaining_tasks)
    if estimate["smoke_rows"]:
        _atomic_write_csv(pd.DataFrame(estimate["all_rows"]).reindex(columns=ROW_COLUMNS), paths["checkpoint"])
        print(f"[sweep] {len(estimate['smoke_rows'])} smoke rows checkpointed to {paths['checkpoint']}")
    print("[sweep] smoke-only mode: stopping without running the rest of the grid.")
    return estimate


def cmd_full(base_config: dict, workers: int, horizon_hours: int, smoke_size: int, results_dir: Path) -> list:
    if horizon_hours != HORIZON_HOURS:
        print(
            f"WARNING: --mode full with horizon_hours={horizon_hours} (not the mandated {HORIZON_HOURS}); "
            "this is a testing override and should not be used for the real sweep.",
            file=sys.stderr,
        )

    paths = _result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_existing_rows(paths)
    completed_keys = {_row_key(row) for row in existing_rows}
    remaining_tasks = [task for task in FULL_GRID if task not in completed_keys]

    print(f"[sweep] mode=full  total_grid={TOTAL_RUNS}  already_completed={len(existing_rows)}  remaining={len(remaining_tasks)}")

    if not remaining_tasks:
        print("[sweep] nothing to do; all runs already completed.")
        final_path = _write_final_results(existing_rows, results_dir)
        print(f"[sweep] wrote {final_path}")
        return existing_rows

    estimate = _estimate_via_smoke(base_config, workers, horizon_hours, smoke_size, existing_rows, remaining_tasks)
    effective_workers = estimate["effective_workers"]
    all_rows = estimate["all_rows"]
    remaining_after_smoke = estimate["remaining_after_smoke"]

    if remaining_after_smoke:
        all_rows = _run_main_loop(remaining_after_smoke, base_config, effective_workers, horizon_hours, all_rows, results_dir)

    final_path = _write_final_results(all_rows, results_dir)
    print(f"[sweep] DONE: {len(all_rows)}/{TOTAL_RUNS} runs recorded -> {final_path}")
    return all_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 sensitivity-sweep runner (docs/DECISIONS.md D8).")
    parser.add_argument("--mode", choices=("dry-run", "smoke", "full"), default="full")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="base scenario config to sweep from")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--smoke-size", type=int, default=DEFAULT_SMOKE_SIZE)
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=HORIZON_HOURS,
        help="TESTING ONLY override; the real sweep always uses the mandated 8760h horizon (D8).",
    )
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    base_config = load_config(args.config)
    results_dir = Path(args.results_dir)

    if args.mode == "dry-run":
        ok = cmd_dry_run(base_config)
        return 0 if ok else 1

    if args.mode == "smoke":
        cmd_smoke(base_config, args.workers, args.horizon_hours, args.smoke_size, results_dir)
        return 0

    cmd_full(base_config, args.workers, args.horizon_hours, args.smoke_size, results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
