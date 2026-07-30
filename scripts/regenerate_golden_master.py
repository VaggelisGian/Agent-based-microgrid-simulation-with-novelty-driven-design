"""Deliberate, explicit regenerator for the tests/golden/*.json fixtures used
by tests/test_golden_master.py.

This script is NEVER imported or run by tests/test_golden_master.py, and it
lives outside pyproject.toml's testpaths ("tests"), so pytest's collector
never even looks at it. A plain `pytest` invocation cannot trigger it: the
only way these golden files change is by a human running this script on
purpose, from the repository root:

    python scripts/regenerate_golden_master.py

Run this ONLY after confirming, by hand, that a shift in the pinned numbers
below is the intended consequence of a real change (model, config, or a
rerun of the Phase 5/6 sweep), never to make a failing golden test pass
without understanding why it failed.

What this writes:

  - tests/golden/short_deterministic_run.json: the four thesis metrics from a
    short, fixed-seed, fixed-config run under capacity_both and
    capacity_disabled (see SHORT_RUN_HORIZON_HOURS / SHORT_RUN_NUM_AGENTS /
    SHORT_RUN_SEED below; these must match the same-named constants in
    tests/test_golden_master.py or the pinned values stop corresponding to
    what that test actually runs).
  - tests/golden/summary_stats_pins.json: a field-by-field snapshot of
    results/summary_stats.csv at k=1.0, broker_count=3, across all four
    capacity ablations (the Phase 5/6 sweep's headline configuration). Reads
    that CSV read-only; if it is not present (e.g. the sweep has not been run
    on this machine) this half of the regeneration is skipped with a printed
    note and the short-run golden file is still written.
"""

from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from microgrid_sim.config.loader import load_config  # noqa: E402  (path set above)
from microgrid_sim.environment.model import MicrogridModel  # noqa: E402

warnings.filterwarnings(
    "ignore", message="The use of the `seed` keyword argument is deprecated", category=FutureWarning
)

GOLDEN_DIR = _REPO_ROOT / "tests" / "golden"
SHORT_RUN_GOLDEN_PATH = GOLDEN_DIR / "short_deterministic_run.json"
SUMMARY_STATS_GOLDEN_PATH = GOLDEN_DIR / "summary_stats_pins.json"
SUMMARY_STATS_CSV_PATH = _REPO_ROOT / "results" / "summary_stats.csv"

# Must match tests/test_golden_master.py's SHORT_HORIZON_HOURS / SHORT_NUM_AGENTS
# / SHORT_SEED exactly. Horizon comfortably exceeds the 168h rolling window
# (three window-widths) so the capacity mechanism actually fires; agent count
# and horizon both sit inside the fix brief's suggested envelope.
SHORT_RUN_HORIZON_HOURS = 504
SHORT_RUN_NUM_AGENTS = 50
SHORT_RUN_SEED = 101  # fixed for reproducibility; arbitrary, not tuned

# Must match tests/test_golden_master.py's pinned sweep coordinates.
PINNED_K = 1.0
PINNED_BROKER_COUNT = 3

_REGENERATION_COMMAND = "python scripts/regenerate_golden_master.py"


def _scenario_config(name: str) -> dict:
    config = load_config(f"config/scenarios/{name}.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = SHORT_RUN_HORIZON_HOURS
    config["simulation"]["seed"] = SHORT_RUN_SEED
    config["population"]["num_agents"] = SHORT_RUN_NUM_AGENTS
    return config


def _metrics_dict(result) -> dict:
    return {
        "avg_cost_per_agent_eur": result.avg_cost_per_agent_eur,
        "avg_cost_per_kwh_eur": result.avg_cost_per_kwh_eur,
        "broker_load_share": dict(result.broker_load_share),
        "load_concentration_hhi": result.load_concentration_hhi,
        "feeder_coefficient_of_variation": result.feeder_coefficient_of_variation,
        "feeder_peak_to_average_ratio": result.feeder_peak_to_average_ratio,
        "feeder_mean_hourly_ramp_kwh": result.feeder_mean_hourly_ramp_kwh,
        "prosumer_self_sufficiency": result.prosumer_self_sufficiency,
    }


def _compute_short_run_golden() -> dict:
    golden = {
        "_comment": (
            "Golden-master pins for tests/test_golden_master.py's short deterministic "
            "run (item 1 of the harness). Regenerate DELIBERATELY, never automatically, "
            f"by running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "scenario_horizon_hours": SHORT_RUN_HORIZON_HOURS,
        "scenario_num_agents": SHORT_RUN_NUM_AGENTS,
        "scenario_seed": SHORT_RUN_SEED,
    }
    for name in ("capacity_both", "capacity_disabled"):
        model = MicrogridModel(_scenario_config(name))
        model.run()
        golden[name] = _metrics_dict(model.compute_metrics())
    return golden


def _compute_summary_stats_pins() -> dict | None:
    if not SUMMARY_STATS_CSV_PATH.is_file():
        print(f"note: {SUMMARY_STATS_CSV_PATH} not present; skipping summary_stats_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(SUMMARY_STATS_CSV_PATH)
    subset = df[(df["k"] == PINNED_K) & (df["broker_count"] == PINNED_BROKER_COUNT)]
    if subset.empty:
        print(f"note: no rows at k={PINNED_K}, broker_count={PINNED_BROKER_COUNT} in {SUMMARY_STATS_CSV_PATH}")
        return None

    rows = []
    for _, row in subset.sort_values(["ablation", "metric"]).iterrows():
        rows.append(
            {
                "ablation": row["ablation"],
                "metric": row["metric"],
                "n_seeds": int(row["n_seeds"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
        )

    return {
        "_comment": (
            "Pinned digest of results/summary_stats.csv (the Phase 5/6 capacity-mechanism "
            f"sweep) at k={PINNED_K}, broker_count={PINNED_BROKER_COUNT}, across all four "
            "ablations. results/summary_stats.csv is read-only input here and is never "
            "written by this script; only this pinned snapshot is written. Regenerate "
            "DELIBERATELY, never automatically, only after a verified intentional sweep "
            f"rerun, by running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "k": PINNED_K,
        "broker_count": PINNED_BROKER_COUNT,
        "rows": rows,
    }


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    short_run_golden = _compute_short_run_golden()
    with open(SHORT_RUN_GOLDEN_PATH, "w", encoding="ascii") as handle:
        json.dump(short_run_golden, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {SHORT_RUN_GOLDEN_PATH}")

    summary_stats_golden = _compute_summary_stats_pins()
    if summary_stats_golden is not None:
        with open(SUMMARY_STATS_GOLDEN_PATH, "w", encoding="ascii") as handle:
            json.dump(summary_stats_golden, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {SUMMARY_STATS_GOLDEN_PATH}")


if __name__ == "__main__":
    main()
