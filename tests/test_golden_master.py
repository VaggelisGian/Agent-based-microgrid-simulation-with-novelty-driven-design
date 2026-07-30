"""Golden-master regression harness for the capacity mechanism (D6/D7).

Pins the current headline numbers so a later, unnoticed change to the model
or its calibration fails a test loudly instead of drifting silently. Five
things, matching the five invariants the mechanism's own design log claims:

  1. a short, fixed-seed, fixed-config, bounded-horizon run under
     capacity_both and capacity_disabled, with all four thesis metrics pinned
     to a tight tolerance;
  2. the clean-channel-isolation invariant (D6/D7): capacity_pnl_only equals
     capacity_disabled on every physical metric and on the feeder net import
     series itself, because the P&L channel writes nothing into prices;
  3. the byte-for-byte baseline invariant: with the capacity master flag off,
     the model reproduces the plain baseline exactly, whether the
     capacity_mechanism block is absent or present-but-disabled;
  4. the cross-broker heterogeneity invariant: a strictly larger contribution
     share yields a strictly larger surcharge, both on the mechanism directly
     and in a real short model run;
  5. a pinned, field-by-field digest of results/summary_stats.csv at
     k=1.0, broker_count=3, across all four ablations (the Phase 5/6 sweep's
     headline configuration).

Expected values live in tests/golden/*.json. Regenerating them is a
DELIBERATE, explicit act, never a side effect of running the test suite: a
plain `pytest` invocation must never rewrite these files (see
test_regeneration_script_import_does_not_write_golden_files below, which
proves this structurally rather than just asserting it in prose). To
regenerate, after confirming by hand that a shift in the pinned numbers is an
intended consequence of a real change, run from the repository root:

    python scripts/regenerate_golden_master.py
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from microgrid_sim.config.loader import load_config
from microgrid_sim.environment.capacity import CapacityMechanism
from microgrid_sim.environment.model import MicrogridModel

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
SHORT_RUN_GOLDEN_PATH = GOLDEN_DIR / "short_deterministic_run.json"
SUMMARY_STATS_GOLDEN_PATH = GOLDEN_DIR / "summary_stats_pins.json"
SUMMARY_STATS_CSV_PATH = REPO_ROOT / "results" / "summary_stats.csv"
REGENERATE_SCRIPT_PATH = REPO_ROOT / "scripts" / "regenerate_golden_master.py"

# Must match scripts/regenerate_golden_master.py's SHORT_RUN_* constants
# exactly, or the golden file no longer corresponds to what this test runs.
# 504h = 3x the 168h rolling window (D6), comfortably exceeding it so the
# mechanism actually fires, while staying well under a minute for the whole
# file; 50 agents sits inside the brief's 40-60 envelope.
SHORT_HORIZON_HOURS = 504
SHORT_NUM_AGENTS = 50
SHORT_SEED = 101  # fixed for reproducibility; arbitrary, not tuned

TIGHT_ABS_TOL = 1e-9
CLEAN_CHANNEL_ABS_TOL = 1e-12


def _load_golden(path: Path) -> dict:
    with open(path, "r", encoding="ascii") as handle:
        return json.load(handle)


def _scenario_config(name: str, horizon: int = SHORT_HORIZON_HOURS, num_agents: int = SHORT_NUM_AGENTS, seed: int = SHORT_SEED) -> dict:
    config = load_config(f"config/scenarios/{name}.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = horizon
    config["simulation"]["seed"] = seed
    config["population"]["num_agents"] = num_agents
    return config


def _default_config(horizon: int = SHORT_HORIZON_HOURS, num_agents: int = SHORT_NUM_AGENTS, seed: int = SHORT_SEED) -> dict:
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = horizon
    config["simulation"]["seed"] = seed
    config["population"]["num_agents"] = num_agents
    return config


def _run(config: dict) -> MicrogridModel:
    model = MicrogridModel(copy.deepcopy(config))
    model.run()
    return model


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


def _assert_metrics_match_golden(actual: dict, expected: dict, context: str) -> None:
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            assert set(actual_value.keys()) == set(expected_value.keys()), f"{context}: broker_load_share keys"
            for broker_id, expected_share in expected_value.items():
                assert actual_value[broker_id] == pytest.approx(expected_share, abs=TIGHT_ABS_TOL), (
                    f"{context}: broker_load_share[{broker_id}]"
                )
        else:
            assert actual_value == pytest.approx(expected_value, abs=TIGHT_ABS_TOL), f"{context}: {key}"


# ---------------------------------------------------------------------------
# 1. Short deterministic run: pin the four thesis metrics under capacity_both
#    and capacity_disabled (the mechanism exercised, and off).
# ---------------------------------------------------------------------------


def test_short_deterministic_run_pins_metrics_under_capacity_both():
    if not SHORT_RUN_GOLDEN_PATH.is_file():
        pytest.skip(f"golden file missing: {SHORT_RUN_GOLDEN_PATH}; regenerate deliberately (see module docstring)")
    golden = _load_golden(SHORT_RUN_GOLDEN_PATH)

    model = _run(_scenario_config("capacity_both"))
    assert model.current_hour == SHORT_HORIZON_HOURS
    result = model.compute_metrics()
    assert result.capacity_fire_rate > 0.0, "construction should exercise the mechanism at least once"

    _assert_metrics_match_golden(_metrics_dict(result), golden["capacity_both"], "capacity_both")


def test_short_deterministic_run_pins_metrics_under_capacity_disabled():
    if not SHORT_RUN_GOLDEN_PATH.is_file():
        pytest.skip(f"golden file missing: {SHORT_RUN_GOLDEN_PATH}; regenerate deliberately (see module docstring)")
    golden = _load_golden(SHORT_RUN_GOLDEN_PATH)

    model = _run(_scenario_config("capacity_disabled"))
    assert model.current_hour == SHORT_HORIZON_HOURS
    result = model.compute_metrics()
    assert result.capacity_fire_rate == 0.0  # mechanism off: never fires

    _assert_metrics_match_golden(_metrics_dict(result), golden["capacity_disabled"], "capacity_disabled")


# ---------------------------------------------------------------------------
# 2. Clean-channel-isolation invariant (D6/D7): capacity_pnl_only debits
#    broker ledgers but writes nothing into prices, so it must equal
#    capacity_disabled on every physical metric and on the feeder net import
#    series itself, to a much tighter tolerance than the pinned-metric checks
#    above (this is an identity the mechanism must satisfy exactly, not a
#    calibration-sensitive headline number).
# ---------------------------------------------------------------------------


def test_pnl_only_equals_disabled_on_physical_metrics_and_feeder_series():
    disabled = _run(_scenario_config("capacity_disabled"))
    pnl_only = _run(_scenario_config("capacity_pnl_only"))

    assert pnl_only.feeder_net_import_history == disabled.feeder_net_import_history  # exact, not approx

    disabled_metrics = disabled.compute_metrics()
    pnl_metrics = pnl_only.compute_metrics()
    physical_fields = (
        "feeder_coefficient_of_variation",
        "feeder_peak_to_average_ratio",
        "feeder_mean_hourly_ramp_kwh",
        "prosumer_self_sufficiency",
    )
    for field in physical_fields:
        assert getattr(pnl_metrics, field) == pytest.approx(
            getattr(disabled_metrics, field), abs=CLEAN_CHANNEL_ABS_TOL
        ), field

    # The mechanism did levy a nonzero ledger charge (otherwise this isolation
    # check would be vacuous: nothing fired, so of course nothing diverged).
    total_charge = sum(broker.cumulative_capacity_charge_eur for broker in pnl_only.brokers.values())
    assert total_charge > 0.0


# ---------------------------------------------------------------------------
# 3. Byte-for-byte baseline invariant: with the capacity master flag off, the
#    model reproduces the plain baseline exactly, whether the
#    capacity_mechanism block is absent entirely or present-but-disabled, and
#    whether the scenario file is capacity_disabled.yaml or config/default.yaml
#    itself.
# ---------------------------------------------------------------------------


def test_capacity_block_absent_equals_present_but_disabled_byte_for_byte():
    with_block = _scenario_config("capacity_disabled")
    assert with_block["capacity_mechanism"]["enabled"] is False
    without_block = copy.deepcopy(with_block)
    del without_block["capacity_mechanism"]

    model_with_block = _run(with_block)
    model_without_block = _run(without_block)

    assert model_with_block.feeder_net_import_history == model_without_block.feeder_net_import_history
    assert model_with_block.compute_metrics() == model_without_block.compute_metrics()


def test_capacity_disabled_scenario_equals_plain_default_byte_for_byte():
    disabled_model = _run(_scenario_config("capacity_disabled"))
    plain_default_model = _run(_default_config())

    assert disabled_model.feeder_net_import_history == plain_default_model.feeder_net_import_history
    assert disabled_model.compute_metrics() == plain_default_model.compute_metrics()


# ---------------------------------------------------------------------------
# 4. Cross-broker heterogeneity invariant: a strictly larger contribution
#    share yields a strictly larger surcharge. Checked directly on
#    environment/capacity.py with constructed contributions, and again inside
#    a real short model run at the steps where the charge actually fires.
# ---------------------------------------------------------------------------


def test_surcharge_strictly_monotone_in_contribution_share_direct():
    mechanism = CapacityMechanism(window=2, k=0.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.3)
    mechanism.step(10.0, {"alpha": 6.0, "beta": 3.0, "gamma": 1.0})
    result = mechanism.step(120.0, {"alpha": 60.0, "beta": 40.0, "gamma": 20.0})

    assert result.total_charge_eur > 0.0
    assert result.surcharges_eur_per_kwh["alpha"] > result.surcharges_eur_per_kwh["beta"] > result.surcharges_eur_per_kwh["gamma"] > 0.0

    contributions = {"alpha": 60.0, "beta": 40.0, "gamma": 20.0}
    for broker_id, contribution in contributions.items():
        expected_surcharge = 0.3 * (contribution / 120.0)
        assert result.surcharges_eur_per_kwh[broker_id] == pytest.approx(expected_surcharge, abs=1e-12)


def test_surcharge_strictly_monotone_in_contribution_share_real_model_run():
    """Same invariant, exercised in a real short model run. At every hour
    where the pricing channel actually writes a positive surcharge, brokers
    whose customers contributed strictly different positive amounts to that
    step's feeder load must end up with strictly, correspondingly ordered
    surcharges (see environment/capacity.py's allocation formula: surcharge is
    capacity_passthrough times a broker's SHARE of total positive
    contribution, a monotone function of the raw contribution for a fixed
    denominator)."""
    config = _scenario_config("capacity_pricing_only")
    model = MicrogridModel(copy.deepcopy(config))

    checked_a_firing_hour = False
    for _ in range(SHORT_HORIZON_HOURS):
        model.step()
        contributions = {broker_id: 0.0 for broker_id in model.brokers}
        for agent in model.agents:
            contributions[agent.last_broker_id] += agent.last_net_import_kwh
        surcharges = model.broker_surcharges  # computed FROM this step's contributions

        positive = {broker_id: value for broker_id, value in contributions.items() if value > 0.0}
        if len(positive) < 2 or not any(surcharge > 0.0 for surcharge in surcharges.values()):
            continue

        checked_a_firing_hour = True
        broker_ids = list(positive.keys())
        for i, broker_a in enumerate(broker_ids):
            for broker_b in broker_ids[i + 1 :]:
                if positive[broker_a] > positive[broker_b] + 1e-9:
                    assert surcharges[broker_a] > surcharges[broker_b], (
                        f"hour {model.current_hour}: {broker_a} contributed more than {broker_b} "
                        "but did not get a strictly larger surcharge"
                    )
                elif positive[broker_b] > positive[broker_a] + 1e-9:
                    assert surcharges[broker_b] > surcharges[broker_a], (
                        f"hour {model.current_hour}: {broker_b} contributed more than {broker_a} "
                        "but did not get a strictly larger surcharge"
                    )

    assert checked_a_firing_hour, "construction should produce at least one hour with a positive surcharge"


# ---------------------------------------------------------------------------
# 5. Pinned digest of results/summary_stats.csv at k=1.0, broker_count=3,
#    across all four ablations. Read-only: this test never writes the CSV,
#    and skips (does not fail) if the sweep output is absent on this machine.
# ---------------------------------------------------------------------------


def test_summary_stats_csv_matches_pinned_digest_k1_bc3():
    if not SUMMARY_STATS_CSV_PATH.is_file():
        pytest.skip(f"results/summary_stats.csv not present (sweep output, not regenerated by tests): {SUMMARY_STATS_CSV_PATH}")
    if not SUMMARY_STATS_GOLDEN_PATH.is_file():
        pytest.skip(f"golden file missing: {SUMMARY_STATS_GOLDEN_PATH}; regenerate deliberately (see module docstring)")

    import pandas as pd

    golden = _load_golden(SUMMARY_STATS_GOLDEN_PATH)
    df = pd.read_csv(SUMMARY_STATS_CSV_PATH)
    subset = df[(df["k"] == golden["k"]) & (df["broker_count"] == golden["broker_count"])]
    actual_rows = {(row["ablation"], row["metric"]): row for _, row in subset.iterrows()}

    assert len(golden["rows"]) > 0
    for expected_row in golden["rows"]:
        key = (expected_row["ablation"], expected_row["metric"])
        assert key in actual_rows, f"missing row in results/summary_stats.csv for {key}"
        actual_row = actual_rows[key]
        assert int(actual_row["n_seeds"]) == expected_row["n_seeds"], f"{key}: n_seeds"
        assert float(actual_row["mean"]) == pytest.approx(expected_row["mean"], rel=1e-9, abs=1e-9), f"{key}: mean"
        assert float(actual_row["std"]) == pytest.approx(expected_row["std"], rel=1e-9, abs=1e-9), f"{key}: std"


# ---------------------------------------------------------------------------
# Regeneration guard: a plain pytest run must never rewrite the golden files.
# scripts/regenerate_golden_master.py sits outside pyproject.toml's testpaths
# ("tests"), so pytest's collector never looks at it and this module never
# imports it either; the check below goes one step further and proves that
# even DIRECTLY importing that script's code (as this test does, to reach its
# module-level constants and functions) does not write anything, because the
# write logic is gated behind `if __name__ == "__main__":`, which is false
# for any import.
# ---------------------------------------------------------------------------


def test_regeneration_script_import_does_not_write_golden_files():
    assert REGENERATE_SCRIPT_PATH.is_file()

    golden_paths = [path for path in (SHORT_RUN_GOLDEN_PATH, SUMMARY_STATS_GOLDEN_PATH) if path.is_file()]
    assert golden_paths, "expected at least one golden file to already exist"
    mtimes_before = {path: path.stat().st_mtime_ns for path in golden_paths}

    spec = importlib.util.spec_from_file_location("_golden_regen_import_probe", REGENERATE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # executes top-level code; module.__name__ != "__main__" here

    assert module.__name__ != "__main__"
    assert hasattr(module, "main"), "regeneration entry point must be a function, not top-level code"

    mtimes_after = {path: path.stat().st_mtime_ns for path in golden_paths}
    assert mtimes_after == mtimes_before, "importing the regeneration script rewrote a golden file"
