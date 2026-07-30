"""D9 structural-coefficient sensitivity sweep runner (docs/DECISIONS.md D9).

Runs the 2880-row grid: deferrable_fraction in {0.05, 0.10, 0.20, 0.30} x
response_reference_eur_per_kwh in {0.0125, 0.025, 0.10, 0.20} x
payback_cap_fraction in {0.25, 0.5, 1.0} x ablation in {capacity_disabled,
capacity_both} x 30 seeds, at the headline cell of the main sweep
(capacity_k = 1.0, broker_count = 3 fixed), window 168 hours, horizon 8760
hours. The three coefficients are the D7 demand-deferral response's own
structural parameters, held fixed everywhere else in the project; this sweep
is the one place they vary.

Each row is deterministic given (deferrable_fraction, response_reference,
payback_cap_fraction, ablation, seed): config is built by reusing
scripts/run_sensitivity_sweep.py's build_run_config (same k=1.0, bc=3,
ablation-flag, and window construction as the main D8 sweep) and then
overriding only the three capacity_mechanism coefficients D9 sweeps. The seed
list is imported directly from scripts/run_sensitivity_sweep.py (SEEDS), not
re-derived, so it is guaranteed identical to the main sweep's.

Compute note (D9's "why the row count is not the run count"): the
capacity_disabled arm is provably invariant to all three coefficients. With
the master flag off, MicrogridModel never builds a CapacityMechanism
(environment/model.py's _build_capacity_mechanism), broker_surcharges stays
all-zero for the whole run, and Consumer._apply_demand_deferral
(agents/consumer.py) only ever reads deferrable_fraction /
response_reference_eur_per_kwh / payback_cap_fraction inside branches gated
on a positive surcharge -- which never occurs -- so none of the three
coefficients is read at all during a disabled run. Running all 48 coefficient
cells of the disabled arm for real would reproduce the same 30 per-seed
outcomes 48 times over, at real compute cost, for no information. This
runner instead:

  1. Runs the disabled arm ONCE per seed (30 "canonical" runs) at the shipped
     config/default.yaml coefficient values.
  2. Runs a VERIFICATION SAMPLE of real disabled runs at the coefficient
     box's 8 corners (min/max of each of the 3 axes) x the first 5 seeds
     (40 real runs), and asserts each one reproduces its canonical same-seed
     row, to a floating-point noise floor (see _check_invariance), on every
     numeric/JSON-string column that is not a config echo (deferrable_fraction, response_reference_eur_per_kwh,
     payback_cap_fraction, ablation, seed, k, broker_count, row_provenance,
     run_wallclock_sec, peak_rss_mb are the config/timing echoes; everything
     else is checked). The outcome (cells/seeds checked, max abs divergence
     per column) is written to a companion JSON file next to the parquet, so
     the check is auditable after the fact, not just visible in stdout.
  3. If verification passes, the remaining 1400 disabled-arm cells are
     MATERIALIZED (copied) from that seed's canonical row, with their own
     (deferrable_fraction, response_reference, payback_cap_fraction) config
     echo columns, never re-simulated. If verification FAILS on any cell,
     the reuse is abandoned loudly (diagnostic to stderr, report still
     written) and the runner falls back to running the full disabled arm for
     real, cell by cell.
  4. The capacity_both arm (1440 cells) is always run for real regardless of
     the invariance outcome; the two questions are independent.

Every row carries a row_provenance column: "direct" (a real run -- always
true for capacity_both), "verification" (a real disabled run that was also
used to check the invariance), or "materialized_from_canonical_seed_run" (not
simulated, copied from that seed's canonical disabled run). The parquet still
carries all 2880 rows, so downstream analysis is a plain paired comparison
exactly as if every cell had been simulated.

Resumable: re-running skips any (deferrable_fraction, response_reference,
payback_cap_fraction, ablation, seed) tuple already present in the relevant
stage's checkpoint CSV, same idiom as scripts/run_sensitivity_sweep.py (atomic
CSV checkpoint every ~5% of a stage, MemoryError-adaptive worker-count
retry). A fully-complete results/sweep_structural.parquet short-circuits the
whole pipeline.

CLI modes:
    --mode dry-run   build every row spec (2880 for the full grid) WITHOUT
                     simulating; print and cross-check the coefficient axes,
                     the fixed k/broker_count, and the seed list against
                     scripts/run_sensitivity_sweep.py's SEEDS; spot-check real
                     config/model construction at the verification corners.
    --mode smoke     run a small, real, short-horizon (default 336h) version
                     of the whole pipeline (canonical, verification,
                     materialization, capacity_both) end to end, writing to
                     *_smoke-suffixed result files so it never touches the
                     real sweep_structural.parquet.
    --mode full      the actual 2880-row grid (1440 capacity_both direct runs
                     plus the disabled-arm reuse scheme above) at the
                     mandated 8760h horizon.

Run from the repository root (data/ and config/ paths are relative), e.g.:
    python scripts/run_structural_sweep.py --mode dry-run
    python scripts/run_structural_sweep.py --mode smoke
    python scripts/run_structural_sweep.py --mode full --workers 14
"""

from __future__ import annotations

import os

# Pin every BLAS/OpenMP thread pool to 1 thread, BEFORE numpy (or anything
# that imports it) is loaded. Found empirically while verifying the D9
# invariance check: under heavy concurrent system load, two separate process
# launches of the identical (config, seed) task could disagree by about 1 ULP
# on a metric column (confirmed via a direct multiprocessing.Pool
# reproduction; see the task's verification notes). The mechanism is
# OpenBLAS's runtime thread count varying with core contention, which makes a
# reduction's floating-point summation order (hence its last-bit result)
# depend on scheduling, not on (config, seed) alone. This never showed up on
# an idle machine, but this sweep's whole disabled-arm reuse rests on a
# same-seed equality check, so it is pinned out at the source rather than
# left as a rare, unexplained flake. Pinning alone proved insufficient (the
# 1 ULP disagreement still reproduces with the pools pinned), which is why
# _check_invariance also carries an explicit noise-floor tolerance; the two
# measures are complementary, not alternatives. Each worker is a
# single simulation running alone in its own process (no internal
# parallelism to lose), so this also avoids oversubscribing the machine when
# many worker processes each start their own multi-threaded BLAS pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import gc
import json
import multiprocessing
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import pandas as pd

from microgrid_sim.environment.model import MicrogridModel

import run_sensitivity_sweep as sweep  # noqa: E402  (path set above)

# Same rationale as scripts/run_sensitivity_sweep.py: harmless under Mesa
# 3.5.1's seed= driver API, and this filter re-installs in every spawned
# worker process (module top-level code reruns on import under Windows spawn).
warnings.filterwarnings(
    "ignore", message="The use of the `seed` keyword argument is deprecated", category=FutureWarning
)

# ---------------------------------------------------------------------------
# D9 grid constants
# ---------------------------------------------------------------------------

CAPACITY_K = 1.0  # fixed: the headline cell of the main D8 sweep
BROKER_COUNT = 3  # fixed: the headline cell of the main D8 sweep

DEFERRABLE_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
RESPONSE_REFERENCES_EUR_PER_KWH = (0.0125, 0.025, 0.10, 0.20)
PAYBACK_CAP_FRACTIONS = (0.25, 0.5, 1.0)
ABLATIONS = ("capacity_disabled", "capacity_both")

# Imported, not re-derived: guarantees element-for-element identity with the
# main sweep's seed list (docs/DECISIONS.md D9: "30 seeds, same seeds as the
# main sweep, paired by seed").
SEEDS = sweep.SEEDS

WINDOW_HOURS = 168  # D9: same window as the main sweep, unchanged
HORIZON_HOURS = 8760  # D9: the mandated one-year sweep horizon

# The 8 corners of the 3-axis coefficient box (min/max of each axis), used
# for the verification sample. Deterministic: derived from the grid constants
# above, not picked ad hoc.
VERIFICATION_CELLS = tuple(
    (d, r, p)
    for d in (min(DEFERRABLE_FRACTIONS), max(DEFERRABLE_FRACTIONS))
    for r in (min(RESPONSE_REFERENCES_EUR_PER_KWH), max(RESPONSE_REFERENCES_EUR_PER_KWH))
    for p in (min(PAYBACK_CAP_FRACTIONS), max(PAYBACK_CAP_FRACTIONS))
)
# First 5 of the 30 seeds: "at least 5 seeds" per D9's compute note; combined
# with all 8 corners this is 40 real verification runs (>= the 25 required).
VERIFICATION_SEEDS = SEEDS[:5]

FULL_ROW_COUNT = (
    len(DEFERRABLE_FRACTIONS) * len(RESPONSE_REFERENCES_EUR_PER_KWH) * len(PAYBACK_CAP_FRACTIONS) * len(ABLATIONS) * len(SEEDS)
)
assert FULL_ROW_COUNT == 2880, f"expected 2880 grid rows, got {FULL_ROW_COUNT}"

CHECKPOINT_FRACTION = 0.05  # same granularity as scripts/run_sensitivity_sweep.py

DEFAULT_CONFIG_PATH = sweep.DEFAULT_CONFIG_PATH
DEFAULT_WORKERS = sweep.DEFAULT_WORKERS

# ---------------------------------------------------------------------------
# Smoke grid: a small, real, short-horizon version of the same pipeline.
# Deliberately NOT a subset of the full grid's coefficient values (so it never
# collides with real-grid row keys); its own 8 corners are a proper subset of
# its own smaller box, so the smoke run still exercises canonical,
# verification, AND materialization, not just the two edge cases.
# ---------------------------------------------------------------------------

SMOKE_DEFERRABLE_FRACTIONS = (0.05, 0.30)  # both corners of this axis
SMOKE_RESPONSE_REFERENCES_EUR_PER_KWH = (0.0125, 0.20)  # both corners of this axis
SMOKE_PAYBACK_CAP_FRACTIONS = (0.25, 0.5, 1.0)  # 0.5 is NOT a corner: forces a materialized row
SMOKE_SEEDS = SEEDS[:2]
SMOKE_VERIFICATION_CELLS = tuple(
    (d, r, p)
    for d in (min(SMOKE_DEFERRABLE_FRACTIONS), max(SMOKE_DEFERRABLE_FRACTIONS))
    for r in (min(SMOKE_RESPONSE_REFERENCES_EUR_PER_KWH), max(SMOKE_RESPONSE_REFERENCES_EUR_PER_KWH))
    for p in (min(SMOKE_PAYBACK_CAP_FRACTIONS), max(SMOKE_PAYBACK_CAP_FRACTIONS))
)
SMOKE_HORIZON_HOURS = 336  # 2x the 168h window, short but exercises the mechanism


@dataclass(frozen=True)
class StructuralGrid:
    dfracs: tuple
    rrefs: tuple
    pcaps: tuple
    seeds: tuple
    verification_cells: tuple
    verification_seeds: tuple
    horizon_hours: int

    @property
    def row_count(self) -> int:
        return len(self.dfracs) * len(self.rrefs) * len(self.pcaps) * len(ABLATIONS) * len(self.seeds)


FULL_GRID = StructuralGrid(
    DEFERRABLE_FRACTIONS, RESPONSE_REFERENCES_EUR_PER_KWH, PAYBACK_CAP_FRACTIONS, SEEDS,
    VERIFICATION_CELLS, VERIFICATION_SEEDS, HORIZON_HOURS,
)


def _smoke_grid(horizon_hours: int) -> StructuralGrid:
    return StructuralGrid(
        SMOKE_DEFERRABLE_FRACTIONS, SMOKE_RESPONSE_REFERENCES_EUR_PER_KWH, SMOKE_PAYBACK_CAP_FRACTIONS,
        SMOKE_SEEDS, SMOKE_VERIFICATION_CELLS, SMOKE_SEEDS, horizon_hours,
    )


# ---------------------------------------------------------------------------
# Config construction: reuse the main sweep's k/broker_count/ablation/window
# construction unchanged, then override only the three D9 coefficients.
# ---------------------------------------------------------------------------


def build_run_config(
    base_config: dict,
    deferrable_fraction: float,
    response_reference_eur_per_kwh: float,
    payback_cap_fraction: float,
    ablation: str,
    seed: int,
    horizon_hours: int = HORIZON_HOURS,
) -> dict:
    """Build one run's config: scripts/run_sensitivity_sweep.py's
    build_run_config at the fixed (k=1.0, broker_count=3) headline cell,
    with the three D9 coefficients overridden on top. Nothing else is
    touched, so charge_rate_eur_per_kwh and capacity_passthrough stay at the
    config/default.yaml value (D9: "everything else held at the shipped
    configuration")."""
    config = sweep.build_run_config(base_config, CAPACITY_K, BROKER_COUNT, ablation, seed, horizon_hours)
    capacity_cfg = config["capacity_mechanism"]
    capacity_cfg["deferrable_fraction"] = deferrable_fraction
    capacity_cfg["response_reference_eur_per_kwh"] = response_reference_eur_per_kwh
    capacity_cfg["payback_cap_fraction"] = payback_cap_fraction
    return config


# ---------------------------------------------------------------------------
# Row schema
# ---------------------------------------------------------------------------

ROW_COLUMNS = [
    "deferrable_fraction",
    "response_reference_eur_per_kwh",
    "payback_cap_fraction",
    "ablation",
    "seed",
    "k",
    "broker_count",
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
    "row_provenance",
]

# Columns compared for the invariance check: every metric/audit column that is
# not a config echo (deferrable_fraction/response_reference/payback_cap_fraction/
# ablation/seed/k/broker_count), not the provenance tag, and not a timing/
# resource column (run_wallclock_sec, peak_rss_mb legitimately vary run to run
# and are not part of the invariance claim).
NUMERIC_COMPARISON_COLUMNS = (
    "avg_cost_per_agent_eur",
    "avg_cost_per_kwh_eur",
    "load_concentration_hhi",
    "feeder_coefficient_of_variation",
    "feeder_peak_to_average_ratio",
    "feeder_mean_hourly_ramp_kwh",
    "prosumer_self_sufficiency",
    "total_capacity_charge_eur",
    "capacity_fire_rate",
    "broker_load_share_gini",
    "total_deferred_kwh",
    "switching_rate",
    "n_brokers",
)
STRING_COMPARISON_COLUMNS = ("broker_load_share_json", "mean_broker_surcharge_json")

VALID_PROVENANCE = {"direct", "verification", "materialized_from_canonical_seed_run"}


def _build_row(
    deferrable_fraction: float,
    response_reference_eur_per_kwh: float,
    payback_cap_fraction: float,
    ablation: str,
    seed: int,
    model,
    metrics,
) -> dict:
    broker_load_share = dict(metrics.broker_load_share)
    mean_surcharge = dict(metrics.mean_broker_surcharge_eur_per_kwh)

    agents = list(model.agents)
    num_agents = model.num_agents
    switched = sum(1 for agent in agents if getattr(agent, "switch_count", 0) > 0)
    switching_rate = (switched / num_agents) if num_agents > 0 else float("nan")

    return {
        "deferrable_fraction": deferrable_fraction,
        "response_reference_eur_per_kwh": response_reference_eur_per_kwh,
        "payback_cap_fraction": payback_cap_fraction,
        "ablation": ablation,
        "seed": seed,
        "k": CAPACITY_K,
        "broker_count": BROKER_COUNT,
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
        "broker_load_share_gini": sweep._gini(list(broker_load_share.values())),
        "total_deferred_kwh": metrics.total_deferred_kwh,
        "switching_rate": switching_rate,
        "n_brokers": len(broker_load_share),
        "row_provenance": "",  # always overwritten by the caller; see VALID_PROVENANCE
    }


def _ablation_for_kind(kind: str) -> str:
    return "capacity_both" if kind == "both" else "capacity_disabled"


def _task_key(task: tuple) -> tuple:
    kind, dfrac, rref, pcap, seed = task
    return (float(dfrac), float(rref), float(pcap), _ablation_for_kind(kind), int(seed))


def _row_key(record: dict) -> tuple:
    return (
        float(record["deferrable_fraction"]),
        float(record["response_reference_eur_per_kwh"]),
        float(record["payback_cap_fraction"]),
        str(record["ablation"]),
        int(record["seed"]),
    )


# ---------------------------------------------------------------------------
# Worker process (multiprocessing.Pool)
# ---------------------------------------------------------------------------

_WORKER_BASE_CONFIG: dict | None = None
_WORKER_HORIZON_HOURS: int = HORIZON_HOURS


def _init_worker(base_config: dict, horizon_hours: int) -> None:
    global _WORKER_BASE_CONFIG, _WORKER_HORIZON_HOURS
    _WORKER_BASE_CONFIG = base_config
    _WORKER_HORIZON_HOURS = horizon_hours


def _run_one_task(task: tuple) -> dict:
    """Run exactly one (kind, deferrable_fraction, response_reference,
    payback_cap_fraction, seed) task to completion and return its row dict.
    kind is "both" (capacity_both) or "disabled" (capacity_disabled); the same
    function serves canonical runs, verification runs, and capacity_both runs
    alike -- only the caller's bookkeeping differs, not the simulation."""
    kind, dfrac, rref, pcap, seed = task
    ablation = _ablation_for_kind(kind)
    config = build_run_config(_WORKER_BASE_CONFIG, dfrac, rref, pcap, ablation, seed, _WORKER_HORIZON_HOURS)

    start = time.perf_counter()
    model = MicrogridModel(config)
    model.run()
    metrics = model.compute_metrics()
    row = _build_row(dfrac, rref, pcap, ablation, seed, model, metrics)
    row["run_wallclock_sec"] = time.perf_counter() - start
    row["peak_rss_mb"] = sweep._peak_rss_mb()

    del model, metrics
    gc.collect()
    return row


# ---------------------------------------------------------------------------
# Result persistence / resume
# ---------------------------------------------------------------------------


def _result_paths(results_dir: Path, tag: str = "") -> dict:
    """tag distinguishes a smoke run's output files from the real sweep's, so
    a --mode smoke invocation can never write into, or be mistaken for,
    results/sweep_structural.parquet."""
    suffix = f"_{tag}" if tag else ""
    return {
        "parquet": results_dir / f"sweep_structural{suffix}.parquet",
        "csv_fallback": results_dir / f"sweep_structural{suffix}.csv",
        "both_checkpoint": results_dir / f"sweep_structural{suffix}_both_checkpoint.csv",
        "disabled_canonical_checkpoint": results_dir / f"sweep_structural{suffix}_disabled_canonical_checkpoint.csv",
        "disabled_verification_checkpoint": results_dir / f"sweep_structural{suffix}_disabled_verification_checkpoint.csv",
        "disabled_fallback_checkpoint": results_dir / f"sweep_structural{suffix}_disabled_fallback_checkpoint.csv",
        "invariance_report": results_dir / f"sweep_structural{suffix}_invariance_check.json",
    }


def _load_checkpoint_rows(path: Path) -> list:
    """Reload a checkpoint CSV. float_precision='round_trip' is required, not
    optional: pandas' default C float parser can be off by 1 ULP against the
    value that was actually written (confirmed directly: writing
    0.18310360219684363 via to_csv and reading it back with the default
    parser yields 0.1831036021968436, one bit short; float_precision=
    'round_trip' recovers the original value exactly). D9's invariance check
    compares a verification run against its canonical
    same-seed row, so a silent 1-ULP read-back error here would fabricate a
    false divergence between two runs that actually agree, which is exactly
    the kind of false alarm this check exists to rule out, not produce."""
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except Exception as exc:
        print(f"WARNING: failed to read existing checkpoint {path}: {exc}", file=sys.stderr)
        return []
    return frame.to_dict(orient="records")


def _already_complete(results_dir: Path, tag: str, expected_total: int):
    """Returns the existing DataFrame if results_dir already has a complete,
    tag-scoped final parquet (row count matches expected_total), else None."""
    paths = _result_paths(results_dir, tag)
    if not paths["parquet"].exists():
        return None
    try:
        existing = pd.read_parquet(paths["parquet"])
    except Exception as exc:
        print(f"WARNING: failed to read existing {paths['parquet']}: {exc}", file=sys.stderr)
        return None
    return existing if len(existing) == expected_total else None


def _run_tasks_resumable(
    tasks_all: list,
    existing_rows: list,
    base_config: dict,
    workers: int,
    horizon_hours: int,
    checkpoint_path: Path,
    label: str,
    provenance_tag: str | None = None,
) -> list:
    """Run whatever tasks in tasks_all are not yet present in existing_rows
    (keyed by _row_key/_task_key), checkpointing to checkpoint_path every
    ~5% of tasks_all, with the same MemoryError-adaptive worker-count retry
    as scripts/run_sensitivity_sweep.py's _run_main_loop. Returns
    existing_rows plus newly completed rows.

    provenance_tag, when given, is stamped onto each newly-run row's
    row_provenance BEFORE it is checkpointed (not after this function
    returns), so an interrupted-and-resumed run's on-disk checkpoint is
    self-describing too, not just the final assembled parquet."""
    done_keys = {_row_key(row) for row in existing_rows}
    remaining = [task for task in tasks_all if _task_key(task) not in done_keys]
    print(f"[structural] {label}: total={len(tasks_all)} already_done={len(existing_rows)} remaining={len(remaining)}")
    if not remaining:
        return list(existing_rows)

    all_rows = list(existing_rows)
    step = max(1, round(len(tasks_all) * CHECKPOINT_FRACTION))
    done_count = len(all_rows)
    last_bucket = done_count // step
    tasks_left = list(remaining)
    current_workers = workers

    while tasks_left:
        loop_start = time.perf_counter()
        processed_this_attempt = 0
        try:
            with multiprocessing.Pool(
                processes=current_workers, initializer=_init_worker, initargs=(base_config, horizon_hours)
            ) as pool:
                for row in pool.imap_unordered(_run_one_task, tasks_left, chunksize=1):
                    if provenance_tag is not None:
                        row["row_provenance"] = provenance_tag
                    all_rows.append(row)
                    done_count += 1
                    processed_this_attempt += 1
                    bucket = done_count // step
                    if bucket > last_bucket or done_count == len(tasks_all):
                        elapsed = time.perf_counter() - loop_start
                        rate = processed_this_attempt / elapsed if elapsed > 0 else 0.0
                        remaining_n = len(tasks_all) - done_count
                        eta_min = (remaining_n / rate / 60.0) if rate > 0 else float("nan")
                        pct = 100.0 * done_count / len(tasks_all)
                        print(
                            f"[structural] {label} progress {done_count}/{len(tasks_all)} ({pct:.1f}%) "
                            f"elapsed={elapsed / 60.0:.1f}min ETA={eta_min:.1f}min workers={current_workers}"
                        )
                        sweep._atomic_write_csv(pd.DataFrame(all_rows), checkpoint_path)
                        last_bucket = bucket
            tasks_left = []  # normal completion: the pool exhausted tasks_left without error
        except MemoryError as exc:
            new_workers = max(1, current_workers - 1)
            print(
                f"WARNING: MemoryError in {label} pool ({exc}); reducing workers "
                f"{current_workers} -> {new_workers} and resuming from the last checkpoint.",
                file=sys.stderr,
            )
            current_workers = new_workers
            done_row_keys = {_row_key(row) for row in all_rows}
            tasks_left = [task for task in tasks_left if _task_key(task) not in done_row_keys]
            sweep._atomic_write_csv(pd.DataFrame(all_rows), checkpoint_path)

    return all_rows


def _write_final_results(rows: list, results_dir: Path, tag: str) -> Path:
    paths = _result_paths(results_dir, tag)
    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=ROW_COLUMNS)
    frame = frame.sort_values(
        ["ablation", "deferrable_fraction", "response_reference_eur_per_kwh", "payback_cap_fraction", "seed"]
    ).reset_index(drop=True)
    try:
        sweep._atomic_write_parquet(frame, paths["parquet"])
        return paths["parquet"]
    except Exception as exc:
        print(f"WARNING: parquet write failed ({exc}); writing CSV fallback instead", file=sys.stderr)
        sweep._atomic_write_csv(frame, paths["csv_fallback"])
        return paths["csv_fallback"]


# ---------------------------------------------------------------------------
# Disabled-arm invariance check and materialization
# ---------------------------------------------------------------------------

# Floating-point noise floor for the invariance check (see _check_invariance).
# Matches the order of magnitude of tests/test_golden_master.py's own pins
# while staying scale-aware, so it does not become vacuously loose on small
# columns or falsely tight on large ones.
INVARIANCE_ABS_TOL = 1e-9
INVARIANCE_REL_TOL = 1e-9


def _check_invariance(canonical_by_seed: dict, verification_rows: list) -> dict:
    """Compare each verification run to its canonical same-seed row on every
    numeric/JSON-string column that is not a config echo. Returns an auditable
    report dict; never raises by itself (the caller decides what to do with
    report["invariant"]).

    On the tolerance, because this is the check D9's whole disabled-arm reuse
    rests on. The original version required bit-exact equality. That was shown
    to produce FALSE alarms: two separate process launches of the identical
    (config, seed) task can disagree by about 1 ULP on a reduction-derived
    column, reproduced organically three ways (a real kill-and-resume, a
    mixed-provenance resume, and two concurrent launches against one results
    directory), with observed divergences of 5.55e-17 on avg_cost_per_kwh_eur
    and 2.78e-17 on broker_load_share_gini. Pinning the BLAS and OpenMP thread
    pools to one thread (top of this file) reduces the exposure but does NOT
    eliminate it, so a bit-exact check would eventually abandon a sound reuse
    and burn about ten extra machine-hours, or leave a confusing
    "invariant: false" audit artifact next to correct data.

    The tolerance below is therefore a floating-point noise floor, not a
    loosening of the scientific claim. It is scaled (absolute plus relative)
    because the compared columns span very different magnitudes: an absolute
    1e-9 is generous for a Gini around 0.35 but is TIGHTER than accumulated
    float64 noise for a capacity charge in the thousands of EUR, so a
    fixed absolute tolerance would just move the false alarm to a different
    column. A real coefficient leak is not a last-bit effect: the deliberate
    forced-divergence test used to prove this check is not vacuous perturbs a
    column by 12345.678, which exceeds this tolerance by roughly fifteen
    orders of magnitude.
    """
    max_abs_divergence = {column: 0.0 for column in NUMERIC_COMPARISON_COLUMNS}
    string_mismatches = {column: 0 for column in STRING_COMPARISON_COLUMNS}
    per_cell_records = []
    invariant = True

    for row in verification_rows:
        seed = int(row["seed"])
        canonical = canonical_by_seed.get(seed)
        if canonical is None:
            raise RuntimeError(f"missing canonical disabled run for seed {seed}; cannot verify invariance")

        cell_max_divergence = 0.0
        for column in NUMERIC_COMPARISON_COLUMNS:
            actual = float(row[column])
            expected = float(canonical[column])
            divergence = abs(actual - expected)
            if divergence > max_abs_divergence[column]:
                max_abs_divergence[column] = divergence
            cell_max_divergence = max(cell_max_divergence, divergence)
            if divergence > INVARIANCE_ABS_TOL + INVARIANCE_REL_TOL * abs(expected):
                invariant = False

        for column in STRING_COMPARISON_COLUMNS:
            if str(row[column]) != str(canonical[column]):
                string_mismatches[column] += 1
                invariant = False

        per_cell_records.append(
            {
                "deferrable_fraction": row["deferrable_fraction"],
                "response_reference_eur_per_kwh": row["response_reference_eur_per_kwh"],
                "payback_cap_fraction": row["payback_cap_fraction"],
                "seed": seed,
                "max_abs_divergence": cell_max_divergence,
            }
        )

    return {
        "invariant": invariant,
        "n_cells_checked": len({(r["deferrable_fraction"], r["response_reference_eur_per_kwh"], r["payback_cap_fraction"]) for r in verification_rows}),
        "n_seeds_checked": len({int(r["seed"]) for r in verification_rows}),
        "n_verification_runs": len(verification_rows),
        "max_abs_divergence_per_column": max_abs_divergence,
        "string_column_mismatches": string_mismatches,
        "per_cell_max_abs_divergence": per_cell_records,
        "note": (
            "invariant=true means every verification run reproduced its canonical same-seed row "
            "to within a floating-point noise floor (abs 1e-9 plus rel 1e-9; see _check_invariance "
            "for why bit-exact comparison produced false alarms across separate process launches); "
            "this is the empirical check behind D9's disabled-arm reuse (docs/DECISIONS.md)."
        ),
    }


def _write_invariance_report(report: dict, canonical_coefficients: dict, grid: StructuralGrid, path: Path) -> None:
    payload = dict(report)
    payload["canonical_coefficients"] = canonical_coefficients
    payload["verification_cells"] = [list(cell) for cell in grid.verification_cells]
    payload["verification_seeds"] = list(grid.verification_seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _print_invariance_diagnostic(report: dict) -> None:
    print("=" * 78, file=sys.stderr)
    print("STRUCTURAL SWEEP: disabled-arm invariance check FAILED", file=sys.stderr)
    print(f"  cells checked: {report['n_cells_checked']}  seeds checked: {report['n_seeds_checked']}", file=sys.stderr)
    print(f"  max abs divergence per column: {report['max_abs_divergence_per_column']}", file=sys.stderr)
    print(f"  string column mismatches: {report['string_column_mismatches']}", file=sys.stderr)
    print(
        "  ABORTING the canonical-run reuse; falling back to running the FULL "
        "disabled arm for real, cell by cell.",
        file=sys.stderr,
    )
    print("=" * 78, file=sys.stderr)


def _materialize_disabled_rows(grid: StructuralGrid, canonical_by_seed: dict, verification_keys: set) -> list:
    """Copy each seed's canonical disabled row into every (deferrable_fraction,
    response_reference, payback_cap_fraction) cell of the grid that was NOT
    part of the real verification sample. No simulation runs here."""
    rows = []
    for dfrac in grid.dfracs:
        for rref in grid.rrefs:
            for pcap in grid.pcaps:
                for seed in grid.seeds:
                    key = (float(dfrac), float(rref), float(pcap), "capacity_disabled", int(seed))
                    if key in verification_keys:
                        continue
                    canonical = canonical_by_seed[int(seed)]
                    row = dict(canonical)
                    row["deferrable_fraction"] = dfrac
                    row["response_reference_eur_per_kwh"] = rref
                    row["payback_cap_fraction"] = pcap
                    row["row_provenance"] = "materialized_from_canonical_seed_run"
                    # Explicitly 0.0 / NaN, not the canonical run's own wallclock/RSS:
                    # no simulation actually ran for this exact cell, so carrying the
                    # canonical run's timing forward would overstate real compute cost
                    # if this column were ever summed across the parquet.
                    row["run_wallclock_sec"] = 0.0
                    row["peak_rss_mb"] = float("nan")
                    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_structural_grid(grid: StructuralGrid, base_config: dict, workers: int, results_dir: Path, tag: str = "") -> dict:
    paths = _result_paths(results_dir, tag)
    results_dir.mkdir(parents=True, exist_ok=True)

    capacity_cfg = base_config["capacity_mechanism"]
    canonical_dfrac = capacity_cfg["deferrable_fraction"]
    canonical_rref = capacity_cfg["response_reference_eur_per_kwh"]
    canonical_pcap = capacity_cfg["payback_cap_fraction"]
    canonical_coefficients = {
        "deferrable_fraction": canonical_dfrac,
        "response_reference_eur_per_kwh": canonical_rref,
        "payback_cap_fraction": canonical_pcap,
    }

    # Stage 1: canonical disabled runs, one per seed, at the SHIPPED
    # coefficient values. Run and checked FIRST (before the larger
    # capacity_both stage) so a failed invariance check is discovered early.
    canonical_tasks = [("disabled", canonical_dfrac, canonical_rref, canonical_pcap, seed) for seed in grid.seeds]
    canonical_existing = _load_checkpoint_rows(paths["disabled_canonical_checkpoint"])
    canonical_rows = _run_tasks_resumable(
        canonical_tasks, canonical_existing, base_config, workers, grid.horizon_hours,
        paths["disabled_canonical_checkpoint"], "disabled_canonical",
    )
    canonical_by_seed = {int(row["seed"]): row for row in canonical_rows}

    # Stage 2: verification disabled runs at the coefficient box's corners.
    verification_tasks = [("disabled", d, r, p, s) for (d, r, p) in grid.verification_cells for s in grid.verification_seeds]
    verification_existing = _load_checkpoint_rows(paths["disabled_verification_checkpoint"])
    verification_rows = _run_tasks_resumable(
        verification_tasks, verification_existing, base_config, workers, grid.horizon_hours,
        paths["disabled_verification_checkpoint"], "disabled_verification", provenance_tag="verification",
    )
    for row in verification_rows:
        row["row_provenance"] = "verification"  # belt-and-suspenders: also normalizes any stale checkpoint row
    verification_keys = {_row_key(row) for row in verification_rows}

    report = _check_invariance(canonical_by_seed, verification_rows)
    _write_invariance_report(report, canonical_coefficients, grid, paths["invariance_report"])

    # Stage 3: capacity_both, direct runs over the full grid. Independent of
    # the invariance question above; always run for real.
    both_tasks = [("both", d, r, p, s) for d in grid.dfracs for r in grid.rrefs for p in grid.pcaps for s in grid.seeds]
    both_existing = _load_checkpoint_rows(paths["both_checkpoint"])
    both_rows = _run_tasks_resumable(
        both_tasks, both_existing, base_config, workers, grid.horizon_hours, paths["both_checkpoint"], "capacity_both",
        provenance_tag="direct",
    )
    for row in both_rows:
        row["row_provenance"] = "direct"  # belt-and-suspenders: also normalizes any stale checkpoint row

    # Stage 4: materialize the remainder of the disabled arm from the
    # canonical per-seed runs, or fall back to running it for real.
    if report["invariant"]:
        print(
            f"[structural] invariance check PASSED: {report['n_verification_runs']} verification runs "
            f"({report['n_cells_checked']} cells x {report['n_seeds_checked']} seeds), "
            f"max abs divergence per column = {report['max_abs_divergence_per_column']}"
        )
        materialized_rows = _materialize_disabled_rows(grid, canonical_by_seed, verification_keys)
        disabled_rows = verification_rows + materialized_rows
    else:
        _print_invariance_diagnostic(report)
        all_disabled_tasks = [("disabled", d, r, p, s) for d in grid.dfracs for r in grid.rrefs for p in grid.pcaps for s in grid.seeds]
        fallback_existing_disk = _load_checkpoint_rows(paths["disabled_fallback_checkpoint"])
        fallback_existing = verification_rows + [row for row in fallback_existing_disk if _row_key(row) not in verification_keys]
        disabled_rows = _run_tasks_resumable(
            all_disabled_tasks, fallback_existing, base_config, workers, grid.horizon_hours,
            paths["disabled_fallback_checkpoint"], "disabled_full_fallback", provenance_tag="direct",
        )
        for row in disabled_rows:
            if _row_key(row) not in verification_keys:
                row["row_provenance"] = "direct"  # belt-and-suspenders: also normalizes any stale checkpoint row

    all_rows = both_rows + disabled_rows

    for row in all_rows:
        if row["row_provenance"] not in VALID_PROVENANCE:
            raise AssertionError(f"invalid row_provenance: {row['row_provenance']!r}")
    both_provenance = {row["row_provenance"] for row in all_rows if row["ablation"] == "capacity_both"}
    if both_provenance - {"direct"}:
        raise AssertionError(f"capacity_both rows must all be 'direct', found {both_provenance}")

    if len(all_rows) != grid.row_count:
        raise AssertionError(f"expected {grid.row_count} total rows, assembled {len(all_rows)}")

    final_path = _write_final_results(all_rows, results_dir, tag)
    reloaded = (
        pd.read_parquet(final_path)
        if final_path.suffix == ".parquet"
        else pd.read_csv(final_path, float_precision="round_trip")
    )
    print(f"[structural] wrote and reloaded {final_path}: {len(reloaded)} rows, {len(reloaded.columns)} columns")
    if len(reloaded) != grid.row_count:
        print(f"WARNING: reloaded row count {len(reloaded)} != expected {grid.row_count}", file=sys.stderr)

    return {"rows": all_rows, "invariance_report": report, "final_path": final_path}


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_dry_run(base_config: dict) -> bool:
    """Build every row spec for the full D9 grid WITHOUT simulating; confirm
    the coefficient axes, the fixed k/broker_count, and the seed list against
    scripts/run_sensitivity_sweep.py's SEEDS; spot-check real config/model
    construction at the verification corners plus the canonical cell."""
    grid = FULL_GRID
    all_ok = True

    print("[dry-run] D9 grid axes:")
    print(f"  deferrable_fraction = {grid.dfracs}")
    print(f"  response_reference_eur_per_kwh = {grid.rrefs}")
    print(f"  payback_cap_fraction = {grid.pcaps}")
    print(f"  ablation = {ABLATIONS}")
    print(f"  capacity_k = {CAPACITY_K} (fixed), broker_count = {BROKER_COUNT} (fixed)")
    print(f"[dry-run] this runner's seed list (n={len(grid.seeds)}): {grid.seeds}")
    print(f"[dry-run] main-sweep seed list  (n={len(sweep.SEEDS)}): {sweep.SEEDS}")

    if (grid.dfracs, grid.rrefs, grid.pcaps) != (DEFERRABLE_FRACTIONS, RESPONSE_REFERENCES_EUR_PER_KWH, PAYBACK_CAP_FRACTIONS):
        all_ok = False
        print("[dry-run] FAIL: grid axes do not match the expected D9 constants", file=sys.stderr)

    if grid.seeds != sweep.SEEDS:
        all_ok = False
        print("[dry-run] FAIL: seed list does not match scripts/run_sensitivity_sweep.py's SEEDS element-for-element", file=sys.stderr)
    else:
        print("[dry-run] OK: seed list matches scripts/run_sensitivity_sweep.py's SEEDS element-for-element")

    all_specs = [
        (d, r, p, ablation, s)
        for d in grid.dfracs for r in grid.rrefs for p in grid.pcaps for ablation in ABLATIONS for s in grid.seeds
    ]
    print(f"[dry-run] total row specs built: {len(all_specs)} (expected 2880)")
    if len(all_specs) != 2880:
        all_ok = False
        print(f"[dry-run] FAIL: expected 2880 row specs, got {len(all_specs)}", file=sys.stderr)

    canonical_cell = (
        base_config["capacity_mechanism"]["deferrable_fraction"],
        base_config["capacity_mechanism"]["response_reference_eur_per_kwh"],
        base_config["capacity_mechanism"]["payback_cap_fraction"],
    )
    spot_check_cells = list(grid.verification_cells) + [canonical_cell]
    seed = grid.seeds[0]
    for ablation in ABLATIONS:
        for (dfrac, rref, pcap) in spot_check_cells:
            try:
                config = build_run_config(base_config, dfrac, rref, pcap, ablation, seed, horizon_hours=HORIZON_HOURS)
                capacity_config = config["capacity_mechanism"]
                flags = sweep.ABLATION_FLAGS[ablation]
                for flag_name, expected in flags.items():
                    if capacity_config[flag_name] != expected:
                        raise AssertionError(f"capacity_mechanism.{flag_name} = {capacity_config[flag_name]!r}, expected {expected!r}")
                if capacity_config["k"] != CAPACITY_K:
                    raise AssertionError(f"capacity_mechanism.k = {capacity_config['k']!r}, expected {CAPACITY_K!r}")
                if capacity_config["deferrable_fraction"] != dfrac:
                    raise AssertionError("deferrable_fraction override did not apply")
                if capacity_config["response_reference_eur_per_kwh"] != rref:
                    raise AssertionError("response_reference_eur_per_kwh override did not apply")
                if capacity_config["payback_cap_fraction"] != pcap:
                    raise AssertionError("payback_cap_fraction override did not apply")
                if len(config["brokers"]) != BROKER_COUNT:
                    raise AssertionError(f"config has {len(config['brokers'])} brokers, expected {BROKER_COUNT}")

                model = MicrogridModel(config)  # construction only, no .run()
                if len(model.brokers) != BROKER_COUNT:
                    raise AssertionError(f"model built {len(model.brokers)} brokers, expected {BROKER_COUNT}")
                print(
                    f"[dry-run] OK  ablation={ablation:<17} d={dfrac:<6} r={rref:<7} p={pcap:<5} "
                    f"seed={seed} brokers={list(model.brokers.keys())}"
                )
            except Exception as exc:
                all_ok = False
                print(f"[dry-run] FAIL ablation={ablation} d={dfrac} r={rref} p={pcap}: {exc}", file=sys.stderr)

    verdict = "ALL PASS" if all_ok else "FAILURES PRESENT"
    print(f"[dry-run] {verdict} ({len(spot_check_cells) * len(ABLATIONS)} spot-checked combos, {len(all_specs)} total row specs)")
    return all_ok


def cmd_smoke(base_config: dict, workers: int, horizon_hours: int, results_dir: Path) -> dict:
    grid = _smoke_grid(horizon_hours)
    print(
        f"[structural] mode=smoke horizon={horizon_hours}h grid="
        f"{len(grid.dfracs)}x{len(grid.rrefs)}x{len(grid.pcaps)}x{len(ABLATIONS)}x{len(grid.seeds)}={grid.row_count} rows"
    )
    existing = _already_complete(results_dir, "smoke", grid.row_count)
    if existing is not None:
        print(f"[structural] nothing to do; the smoke parquet already has {grid.row_count} rows.")
        return {"rows": existing.to_dict(orient="records"), "invariance_report": None, "final_path": _result_paths(results_dir, "smoke")["parquet"]}
    return run_structural_grid(grid, base_config, workers, results_dir, tag="smoke")


def cmd_full(base_config: dict, workers: int, horizon_hours: int, results_dir: Path) -> dict:
    if horizon_hours != HORIZON_HOURS:
        print(
            f"WARNING: --mode full with horizon_hours={horizon_hours} (not the mandated {HORIZON_HOURS}); "
            "this is a testing override and should not be used for the real sweep.",
            file=sys.stderr,
        )
    grid = StructuralGrid(
        DEFERRABLE_FRACTIONS, RESPONSE_REFERENCES_EUR_PER_KWH, PAYBACK_CAP_FRACTIONS, SEEDS,
        VERIFICATION_CELLS, VERIFICATION_SEEDS, horizon_hours,
    )
    existing = _already_complete(results_dir, "", grid.row_count)
    if existing is not None:
        print(f"[structural] nothing to do; results/sweep_structural.parquet already has {grid.row_count} rows.")
        return {"rows": existing.to_dict(orient="records"), "invariance_report": None, "final_path": _result_paths(results_dir, "")["parquet"]}
    return run_structural_grid(grid, base_config, workers, results_dir, tag="")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D9 structural-coefficient sensitivity sweep runner (docs/DECISIONS.md D9).")
    parser.add_argument("--mode", choices=("dry-run", "smoke", "full"), default="full")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="base scenario config to sweep from")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=None,
        help="TESTING ONLY override; full mode always defaults to the mandated 8760h, smoke mode to 336h.",
    )
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    base_config = sweep.load_config(args.config)
    results_dir = Path(args.results_dir)

    if args.mode == "dry-run":
        ok = cmd_dry_run(base_config)
        return 0 if ok else 1

    if args.mode == "smoke":
        horizon_hours = args.horizon_hours if args.horizon_hours is not None else SMOKE_HORIZON_HOURS
        cmd_smoke(base_config, args.workers, horizon_hours, results_dir)
        return 0

    horizon_hours = args.horizon_hours if args.horizon_hours is not None else HORIZON_HOURS
    cmd_full(base_config, args.workers, horizon_hours, results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
