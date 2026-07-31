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
     capacity_mechanism block is absent or present-but-disabled (these two
     tests compare the whole MetricsResult, so they pin the Phase 5/6
     audit-only fields such as total_capacity_charge_eur and
     capacity_fire_rate as well, not just the four thesis metrics);
  4. the cross-broker heterogeneity invariant: a strictly larger contribution
     share yields a strictly larger surcharge, both on the mechanism directly
     and in a real short model run;
  5. a pinned, field-by-field digest of results/summary_stats.csv (the Phase
     5/6 capacity-mechanism sweep), every data row and every column.

Three further pinned digests cover the CSV artifacts that back the thesis's
most fragile claims and that previously had no regression guard at all, so an
unnoticed regeneration that silently shifted a number fails loudly here:

  6. results/structural_sensitivity.csv, every data row (the D9 structural
     coefficient sensitivity analysis);
  7. results/demand_source_comparison.csv, every data row (the Phase 12
     synthetic-versus-OPSD headline comparison);
  8. results/monopoly_comparison.csv, every data row (the broker_count=1
     monopoly arm).

Items 5 to 8 pin every row keyed by that row's own identity columns, plus the
CSV's row count, so neither a reordered file nor a vanished or appended row
can pass silently. Their comparison logic is a pure helper over an
already-loaded DataFrame, which lets a shipped test point the same code at a
deliberately perturbed in-memory copy and prove the pin bites, all without
ever writing the CSV on disk (these CSVs are read-only sweep outputs).

A column that is constant across a whole file gets pinned once, at the top
level of that file's golden snapshot, and is then checked against every row.
Both such columns today are metric3_sign_convention, in
monopoly_comparison.csv and in structural_sensitivity.csv: rewording it
reverses how every metric-3 number in either file is read without changing a
single number.

The p-value columns inside those digests are compared by relative tolerance
alone, on a field kind of their own (see _assert_pvalue_close). An absolute
tolerance of 1e-9 cannot tell 1e-55 from 1e-20, and those columns are exactly
where the thesis's significance claims live, so pinning them by magnitude is
the only pin that means anything.

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
import hashlib
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import pytest

from microgrid_sim.config.loader import load_config
from microgrid_sim.environment.capacity import CapacityMechanism
from microgrid_sim.environment.model import MicrogridModel

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
RESULTS_DIR = REPO_ROOT / "results"
SHORT_RUN_GOLDEN_PATH = GOLDEN_DIR / "short_deterministic_run.json"
SUMMARY_STATS_GOLDEN_PATH = GOLDEN_DIR / "summary_stats_pins.json"
SUMMARY_STATS_CSV_PATH = RESULTS_DIR / "summary_stats.csv"
STRUCTURAL_SENSITIVITY_GOLDEN_PATH = GOLDEN_DIR / "structural_sensitivity_pins.json"
STRUCTURAL_SENSITIVITY_CSV_PATH = RESULTS_DIR / "structural_sensitivity.csv"
DEMAND_SOURCE_GOLDEN_PATH = GOLDEN_DIR / "demand_source_comparison_pins.json"
DEMAND_SOURCE_CSV_PATH = RESULTS_DIR / "demand_source_comparison.csv"
MONOPOLY_GOLDEN_PATH = GOLDEN_DIR / "monopoly_comparison_pins.json"
MONOPOLY_CSV_PATH = RESULTS_DIR / "monopoly_comparison.csv"
SWEEP_RAW_PARQUET_PATH = RESULTS_DIR / "sweep_raw.parquet"
SWEEP_MONOPOLY_PARQUET_PATH = RESULTS_DIR / "sweep_monopoly.parquet"
SWEEP_OPSD_HEADLINE_PARQUET_PATH = RESULTS_DIR / "sweep_opsd_headline.parquet"
SWEEP_STRUCTURAL_PARQUET_PATH = RESULTS_DIR / "sweep_structural.parquet"
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

# Tolerance for the pinned CSV digests (items 5 to 8): these compare numbers
# that were written to a text file and read back, not recomputed, so anything
# looser would let a real shift through. Correct for the O(1) quantities in
# these files (means, effect sizes, percent changes), where an absolute floor
# of 1e-9 is far below anything that could matter.
CSV_PIN_REL_TOL = 1e-9
CSV_PIN_ABS_TOL = 1e-9

# Tolerance for the p-value columns, which get their own kind (see
# _assert_pvalue_close). Relative only, deliberately: no absolute floor.
CSV_PIN_PVALUE_REL_TOL = 1e-9

# Tolerance for the parquet-derived deferred-energy aggregates (items 10 and
# 11 below). A parquet column round-trips its float64 bit pattern exactly, so
# a groupby mean recomputed from it is deterministic to the last bit, unlike a
# CSV value that has already lost precision to text formatting; that is why
# this stays as tight as CSV_PIN_REL_TOL rather than being loosened. What is
# genuinely separate is the second check each aggregate carries: the thesis
# text quotes every one of these figures to the nearest whole kWh (for
# example "38822 kWh deferred"), so each aggregate is pinned twice, once as an
# exact float against any silent drift in the underlying data, and once as
# the rounded integer against the specific figure the thesis prints.
DEFERRED_ENERGY_REL_TOL = 1e-9
DEFERRED_ENERGY_ABS_TOL = 1e-9


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
# Shared machinery for the row-keyed CSV digests (items 5 to 8). Each digest
# is a PURE comparison over an already-loaded DataFrame plus an already-loaded
# golden dict, raising AssertionError on any mismatch. That shape is what lets
# the bite tests below point the very same code at a deliberately perturbed
# in-memory copy, proving each pin actually fails on a plausible regression,
# without ever writing the CSV on disk.
#
# Each field list below must match the same-named list in
# scripts/regenerate_golden_master.py exactly, or the pinned snapshot stops
# describing what these tests check.
# ---------------------------------------------------------------------------

_MISSING_TOKEN = "__NA__"

SUMMARY_STATS_IDENTITY = ("k", "broker_count", "ablation", "metric")
# Every non-identity column results/summary_stats.csv carries. The four
# identity columns plus these fifteen are the whole header, so no column of
# that file is left unpinned. (k, broker_count, ablation, metric) is unique
# over the whole file because the sweep emits exactly one row per combination
# of those four; _rows_by_identity asserts that rather than assuming it.
SUMMARY_STATS_PINNED_FIELDS = {
    "metric_label": "str",
    "n_seeds": "int",
    "mean": "float",
    "std": "float",
    "ci95_lo": "float",
    "ci95_hi": "float",
    "ci95_halfwidth": "float",
    "vs_disabled_cohens_d_independent": "float",
    "vs_disabled_paired_dz": "float",
    "vs_disabled_paired_p": "pvalue",
    "vs_disabled_paired_mean_diff": "float",
    "vs_disabled_paired_mean_pct_change": "float",
    "vs_disabled_degenerate_paired_diff": "bool",
    "metric3_sign_convention": "str",
    "notes": "str",
}

STRUCTURAL_SENSITIVITY_IDENTITY = (
    "scope",
    "deferrable_fraction",
    "response_reference_eur_per_kwh",
    "payback_cap_fraction",
    "coefficient",
    "coefficient_level",
    "metric",
)
# Field -> kind. "float" and "pvalue" compare with pytest.approx under
# different tolerances (see _assert_pinned_field), everything else exactly.
STRUCTURAL_SENSITIVITY_PINNED_FIELDS = {
    "n_seeds": "int",
    "mean_disabled": "float",
    "mean_both": "float",
    "paired_mean_diff": "float",
    "paired_mean_pct_change": "float",
    "paired_dz": "float",
    "p_uncorrected": "pvalue",
    "p_holm": "pvalue",
    "p_bh": "pvalue",
    "survives_holm_alpha05": "bool",
    "survives_bh_alpha05": "bool",
    "worsened_vs_disabled": "bool",
    # The two confidence intervals and the flag that fires when they disagree.
    # D9 is a FRAGILITY claim, and interval disagreement between the t and
    # bootstrap intervals is exactly the shape a problem with it would take,
    # so leaving the flag (False on all 118 non-verdict rows today) unpinned
    # would leave the one column that announces trouble free to stop
    # announcing it.
    "ci95_t_lo": "float",
    "ci95_t_hi": "float",
    "ci95_boot_lo": "float",
    "ci95_boot_hi": "float",
    "ci_disagreement_flag": "bool",
    "correction_family": "str",
    "correction_family_size": "int",
    # The six coefficient_verdict rows' supporting evidence. The verdict string
    # alone does not carry it: the sign flip the FRAGILE verdict rests on lives
    # in min/max_pct_change_across_levels (the -5.919773 to +8.382572 range),
    # and no verdict string mentions significance at all, so
    # significant_at_every_level_holm could flip from True without a word of
    # any verdict changing.
    "min_pct_change_across_levels": "float",
    "max_pct_change_across_levels": "float",
    "sign_consistent_across_marginal_levels": "bool",
    "significant_at_every_level_holm": "bool",
    "cell_level_sign_consistent_within_every_level": "bool",
    "levels_with_within_level_cell_sign_disagreement": "str",
    "verdict": "str",
}
# Constant across every row of structural_sensitivity.csv, so pinned once at
# the top level of the golden file and then checked on every row. Same column
# and same reason as MONOPOLY_CONSTANT_FIELDS below.
STRUCTURAL_SENSITIVITY_CONSTANT_FIELDS = ("metric3_sign_convention",)

DEMAND_SOURCE_IDENTITY = ("k", "broker_count", "ablation", "metric")
DEMAND_SOURCE_PINNED_FIELDS = {
    "n_seeds_synthetic": "int",
    "n_seeds_opsd": "int",
    "mean_synthetic": "float",
    "mean_opsd": "float",
    "std_synthetic": "float",
    "std_opsd": "float",
    "delta_opsd_minus_synthetic": "float",
}

MONOPOLY_IDENTITY = ("broker_count", "k")
MONOPOLY_PINNED_FIELDS = {
    "is_monopoly": "bool",
    "n_seeds": "int",
    "feeder_peak_to_average_ratio_paired_mean_pct_change": "float",
    "feeder_peak_to_average_ratio_paired_dz": "float",
    "feeder_peak_to_average_ratio_paired_p": "pvalue",
    "feeder_peak_to_average_ratio_n_improved_of_30": "int",
    "feeder_coefficient_of_variation_paired_mean_pct_change": "float",
    "feeder_coefficient_of_variation_paired_dz": "float",
    "feeder_coefficient_of_variation_paired_p": "pvalue",
    "feeder_coefficient_of_variation_n_improved_of_30": "int",
    "capacity_both_mean_fire_rate": "float",
    "capacity_both_mean_total_deferred_kwh": "float",
    "avg_cost_per_agent_eur_paired_mean_pct_change": "float",
}
# Constant across every row of monopoly_comparison.csv, so pinned once at the
# top level of the golden file and then checked on every row. The sign
# convention in particular decides how the metric-3 numbers are READ, so a
# silent edit to it would change the claim without changing any number.
MONOPOLY_CONSTANT_FIELDS = ("metric3_sign_convention", "note")

# The four row-keyed pinned CSVs, each with the number of p-value columns its
# header actually carries. Pinned as a count rather than a list so that a
# p-value column added to one of these files later cannot quietly land back on
# the "float" kind, and so the claim that demand_source_comparison.csv has no
# p-value column at all is checked against the real header rather than assumed.
PINNED_CSV_SPECS = (
    ("results/summary_stats.csv", SUMMARY_STATS_CSV_PATH, SUMMARY_STATS_PINNED_FIELDS, 1),
    ("results/structural_sensitivity.csv", STRUCTURAL_SENSITIVITY_CSV_PATH, STRUCTURAL_SENSITIVITY_PINNED_FIELDS, 3),
    ("results/demand_source_comparison.csv", DEMAND_SOURCE_CSV_PATH, DEMAND_SOURCE_PINNED_FIELDS, 0),
    ("results/monopoly_comparison.csv", MONOPOLY_CSV_PATH, MONOPOLY_PINNED_FIELDS, 2),
)


def _looks_like_a_pvalue_column(name: str) -> bool:
    return name == "p" or name.startswith("p_") or name.endswith("_p")


def _plain(value):
    """A numpy/pandas cell as a plain Python object, with missing as None.

    Missing has to collapse to a single representation before anything is
    compared or keyed on: an empty CSV cell reads back as float NaN, and
    NaN != NaN, so a raw cell value would not even compare equal to itself.
    """
    if value is None:
        return None
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _identity_token(value) -> str:
    """One deterministic, collision-free string for one identity-column cell.

    Several structural_sensitivity.csv identity columns are empty for some
    scopes (a coefficient_verdict row has no deferrable_fraction; a
    cell_full_factorial row has no coefficient), so missing needs an explicit
    token of its own. It must stay distinct from the literal string "ALL",
    which those same columns use for a marginal row that genuinely aggregates
    over every level: "empty" and "ALL" are different statements and must not
    collide into one key.
    """
    plain = _plain(value)
    if plain is None:
        return _MISSING_TOKEN
    if isinstance(plain, str):
        assert plain != _MISSING_TOKEN, f"identity value collides with the missing token: {plain!r}"
        return plain
    return repr(plain)


def _as_bool(value) -> bool:
    """Twin of the same-named helper in scripts/regenerate_golden_master.py.

    ValueError, not AssertionError, deliberately: the bite tests below assert
    that a perturbed copy raises AssertionError, and a malformed boolean cell
    is a different failure from a pinned value that moved. Raising
    AssertionError here would let a bite test that happened to write a
    non-literal into a boolean column pass for entirely the wrong reason.
    """
    if isinstance(value, bool):
        return value
    if value in ("True", "False"):
        return value == "True"
    raise ValueError(f"not a boolean literal: {value!r}")


def _identity_key(source, identity_columns) -> tuple:
    return tuple(_identity_token(source[column]) for column in identity_columns)


def _rows_by_identity(df, identity_columns) -> dict:
    rows = {}
    for _, row in df.iterrows():
        key = _identity_key(row, identity_columns)
        assert key not in rows, f"duplicate identity key in CSV: {key}"
        rows[key] = row
    return rows


def _assert_pvalue_close(actual: float, expected: float, message: str) -> None:
    """Compare one p-value by RELATIVE tolerance only, with no absolute floor.

    The p-values in these files span roughly 1e-55 to 0.5, so the abs=1e-9 that
    is right for the O(1) columns would call every significant p-value equal to
    every other one: 1e-55, 1e-20 and 1e-12 all sit below that floor, and
    pytest.approx passes on EITHER tolerance. A relative-only comparison pins
    the magnitude itself, which is what the significance claims actually rest on.

    Exact zero gets its own branch: a relative tolerance around zero is zero, so
    letting pytest.approx handle it would demand exact equality by accident
    rather than by decision. Demand it deliberately instead. Every other value
    goes through pytest.approx with abs=0.0, which multiplies rather than
    divides, so there is nothing to divide by zero; a denormal expected value
    simply underflows its own tolerance to zero and therefore also requires
    exact equality, which is the strict answer rather than an error.
    """
    if expected == 0.0:
        assert actual == 0.0, f"{message}: pinned exactly 0.0, found {actual!r}"
        return
    assert actual == pytest.approx(expected, rel=CSV_PIN_PVALUE_REL_TOL, abs=0.0), (
        f"{message}: pinned {expected!r}, found {actual!r}"
    )


def _assert_pinned_field(actual_row, expected_row, field: str, kind: str, context: str) -> None:
    expected = expected_row[field]
    actual = _plain(actual_row[field])
    if expected is None:
        assert actual is None, f"{context}: {field} expected empty, found {actual!r}"
        return
    assert actual is not None, f"{context}: {field} expected {expected!r}, found empty"
    if kind == "float":
        assert float(actual) == pytest.approx(expected, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), f"{context}: {field}"
    elif kind == "pvalue":
        _assert_pvalue_close(float(actual), expected, f"{context}: {field}")
    elif kind == "int":
        assert int(actual) == expected, f"{context}: {field}"
    elif kind == "bool":
        assert _as_bool(actual) is expected, f"{context}: {field}"
    else:
        assert str(actual) == expected, f"{context}: {field}"


def _assert_csv_matches_row_keyed_golden(
    df, golden: dict, identity_columns, pinned_fields, label: str, golden_label: str
) -> None:
    """Compare a whole CSV against a row-keyed golden snapshot.

    Row count first (so a vanished or appended row reports as exactly that),
    then every pinned row looked up by its own identity key rather than by
    position, so a reordered file still matches while a changed value in a
    reordered file still fails.

    Together with _rows_by_identity's own no-duplicates assertion on the CSV
    side, the golden-side uniqueness check below is what makes "every CSV row
    is checked" a fact rather than a hope. The argument is a pigeonhole:
    n_rows distinct CSV keys, n_rows distinct golden keys, and every golden
    key present in the CSV, therefore the two key sets are equal and no CSV
    row escapes. Drop golden-side uniqueness and a hand-edited golden file
    carrying one key twice would leave one CSV row entirely unchecked while
    every other assertion here still passed.
    """
    assert len(df) == golden["n_rows"], f"{label}: row count changed ({len(df)} rows, pinned {golden['n_rows']})"
    assert len(golden["rows"]) == golden["n_rows"], f"{label}: golden file is internally inconsistent"

    golden_keys = set()
    for expected_row in golden["rows"]:
        key = _identity_key(expected_row, identity_columns)
        assert key not in golden_keys, f"{golden_label}: duplicate identity key in the golden file: {key}"
        golden_keys.add(key)
    assert len(golden_keys) == golden["n_rows"], (
        f"{golden_label}: golden file holds {len(golden_keys)} distinct identity keys, pinned {golden['n_rows']} rows"
    )

    actual_rows = _rows_by_identity(df, identity_columns)
    for expected_row in golden["rows"]:
        key = _identity_key(expected_row, identity_columns)
        assert key in actual_rows, f"{label}: missing row for {key}"
        actual_row = actual_rows[key]
        context = f"{label} {key}"
        for field, kind in pinned_fields.items():
            _assert_pinned_field(actual_row, expected_row, field, kind, context)


def _assert_constant_fields(df, golden: dict, constant_fields, label: str) -> None:
    """A column pinned once for the whole file, re-checked on every row."""
    for _, row in df.iterrows():
        for field in constant_fields:
            assert str(_plain(row[field])) == golden[field], f"{label}: {field}"


def _assert_summary_stats_matches_golden(df, golden: dict) -> None:
    _assert_csv_matches_row_keyed_golden(
        df,
        golden,
        SUMMARY_STATS_IDENTITY,
        SUMMARY_STATS_PINNED_FIELDS,
        "results/summary_stats.csv",
        SUMMARY_STATS_GOLDEN_PATH.name,
    )


def _assert_structural_sensitivity_matches_golden(df, golden: dict) -> None:
    label = "results/structural_sensitivity.csv"
    _assert_csv_matches_row_keyed_golden(
        df,
        golden,
        STRUCTURAL_SENSITIVITY_IDENTITY,
        STRUCTURAL_SENSITIVITY_PINNED_FIELDS,
        label,
        STRUCTURAL_SENSITIVITY_GOLDEN_PATH.name,
    )
    _assert_constant_fields(df, golden, STRUCTURAL_SENSITIVITY_CONSTANT_FIELDS, label)


def _assert_demand_source_comparison_matches_golden(df, golden: dict) -> None:
    _assert_csv_matches_row_keyed_golden(
        df,
        golden,
        DEMAND_SOURCE_IDENTITY,
        DEMAND_SOURCE_PINNED_FIELDS,
        "results/demand_source_comparison.csv",
        DEMAND_SOURCE_GOLDEN_PATH.name,
    )


def _assert_monopoly_comparison_matches_golden(df, golden: dict) -> None:
    label = "results/monopoly_comparison.csv"
    _assert_csv_matches_row_keyed_golden(
        df, golden, MONOPOLY_IDENTITY, MONOPOLY_PINNED_FIELDS, label, MONOPOLY_GOLDEN_PATH.name
    )
    _assert_constant_fields(df, golden, MONOPOLY_CONSTANT_FIELDS, label)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _skip_unless_pinned_csv_available(csv_path: Path, golden_path: Path) -> None:
    if not csv_path.is_file():
        pytest.skip(f"{csv_path.name} not present (sweep output, not regenerated by tests): {csv_path}")
    if not golden_path.is_file():
        pytest.skip(f"golden file missing: {golden_path}; regenerate deliberately (see module docstring)")


def test_pvalue_comparison_pins_magnitude_and_handles_zero_and_denormals():
    """The "pvalue" kind, checked directly on its comparison helper.

    Every mismatched pair below sits entirely under the 1e-9 absolute floor the
    "float" kind uses, so under that floor all of them compared EQUAL, which is
    asserted here pair by pair before the new helper is asked to reject them.
    """
    _assert_pvalue_close(8.930161324964124e-56, 8.930161324964124e-56, "identical")
    _assert_pvalue_close(0.0, 0.0, "exact zero, the case a relative tolerance cannot express")

    for actual, expected in ((1e-20, 1e-55), (1e-12, 1e-20), (5e-324, 1e-300), (0.0, 1e-55), (1e-55, 0.0)):
        assert actual == pytest.approx(expected, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), (
            f"the old rule really did call {actual!r} and {expected!r} equal"
        )
        with pytest.raises(AssertionError):
            _assert_pvalue_close(actual, expected, "magnitude moved")

    # Denormals: equal to themselves, not to the next representable value up,
    # and neither comparison overflows or divides by zero.
    denormal = 5e-324
    _assert_pvalue_close(denormal, denormal, "denormal identity")
    with pytest.raises(AssertionError):
        _assert_pvalue_close(2.0 * denormal, denormal, "denormal neighbour")

    # An O(1) p-value keeps behaving sensibly: identical passes, a 1e-6 drift
    # (far larger than the 1e-9 relative tolerance at this magnitude) fails.
    _assert_pvalue_close(0.4780556677281509, 0.4780556677281509, "O(1) p-value")
    with pytest.raises(AssertionError):
        _assert_pvalue_close(0.4780556677281509 + 1e-6, 0.4780556677281509, "O(1) p-value drifted")


def test_every_pvalue_column_in_the_pinned_csvs_uses_the_pvalue_kind():
    """No pinned p-value column may sit on the "float" kind.

    Both directions: every p-value column that is pinned must carry the
    "pvalue" kind, and nothing else may carry it. The per-file counts also make
    a later schema change visible, including the zero that records the checked
    fact that demand_source_comparison.csv carries no p-value column at all.

    A file missing on this machine is skipped INDIVIDUALLY, not by skipping the
    whole test: skipping on the first absent CSV would silently stop checking
    every later file in the list too, which is the sort of coverage hole this
    test exists to prevent.
    """
    import pandas as pd

    unreachable = []
    for label, csv_path, pinned_fields, expected_pvalue_column_count in PINNED_CSV_SPECS:
        if not csv_path.is_file():
            unreachable.append(label)
            continue
        columns = list(pd.read_csv(csv_path, nrows=0).columns)
        pvalue_columns = [name for name in columns if _looks_like_a_pvalue_column(name)]
        assert len(pvalue_columns) == expected_pvalue_column_count, (
            f"{label}: p-value columns changed, found {pvalue_columns}; classify any new one as \"pvalue\""
        )
        pinned_as_pvalue = {name for name, kind in pinned_fields.items() if kind == "pvalue"}
        assert pinned_as_pvalue == {name for name in pvalue_columns if name in pinned_fields}, (
            f"{label}: the \"pvalue\" kind and this file's p-value columns disagree"
        )

    if len(unreachable) == len(PINNED_CSV_SPECS):
        pytest.skip(f"none of the pinned CSVs are present (sweep outputs, not regenerated by tests): {unreachable}")


# ---------------------------------------------------------------------------
# 5. Pinned digest of results/summary_stats.csv, the Phase 5/6
#    capacity-mechanism sweep, every data row and every column. Successor to
#    the earlier k=1.0, broker_count=3 pin, which covered 20 of the file's 240
#    rows, pinned no row count (so an appended row was invisible), and left
#    twelve columns unpinned. Read-only, and skips (does not fail) if the sweep
#    output is absent on this machine, exactly as before.
# ---------------------------------------------------------------------------


def test_summary_stats_csv_matches_pinned_digest():
    _skip_unless_pinned_csv_available(SUMMARY_STATS_CSV_PATH, SUMMARY_STATS_GOLDEN_PATH)

    import pandas as pd

    golden = _load_golden(SUMMARY_STATS_GOLDEN_PATH)
    df = pd.read_csv(SUMMARY_STATS_CSV_PATH)
    _assert_summary_stats_matches_golden(df, golden)


def test_summary_stats_pin_bites_on_plausible_regressions():
    """Prove the pin above is not vacuous, on in-memory copies only.

    Four regressions specific to this file, none of them shared with the other
    bite tests: a confidence interval reported the wrong way round, a metric's
    human-readable label detached from its metric key, the reference arm's
    deliberately empty vs_disabled_* cells filled in with zeros (an empty cell
    and a computed 0.0 are different statements, and only the reference arm is
    entitled to the empty one), and the paired p-value moving far in magnitude
    while staying under the 1e-9 absolute floor, which is what routing that
    column through the "pvalue" kind is for. The CSV's own digest is captured
    before and re-checked after, so the file on disk is provably untouched.
    """
    _skip_unless_pinned_csv_available(SUMMARY_STATS_CSV_PATH, SUMMARY_STATS_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(SUMMARY_STATS_CSV_PATH)
    golden = _load_golden(SUMMARY_STATS_GOLDEN_PATH)
    df = pd.read_csv(SUMMARY_STATS_CSV_PATH)

    _assert_summary_stats_matches_golden(df, golden)  # unperturbed: passes

    # (a) one confidence interval reported low-for-high.
    inverted = df.copy(deep=True)
    wide = inverted.index[(inverted["ci95_hi"] - inverted["ci95_lo"]).abs() > 1e-6]
    assert len(wide) > 0, "expected at least one row with a non-degenerate confidence interval"
    row = wide[0]
    inverted.loc[row, ["ci95_lo", "ci95_hi"]] = [inverted.loc[row, "ci95_hi"], inverted.loc[row, "ci95_lo"]]
    with pytest.raises(AssertionError):
        _assert_summary_stats_matches_golden(inverted, golden)

    # (b) a metric's label detached from its metric key.
    relabelled = df.copy(deep=True)
    cost_rows = relabelled.index[relabelled["metric"] == "avg_cost_per_agent_eur"]
    assert len(cost_rows) > 0
    relabelled.loc[cost_rows[0], "metric_label"] = "Avg cost per agent (EUR/month)"
    with pytest.raises(AssertionError):
        _assert_summary_stats_matches_golden(relabelled, golden)

    # (c) the reference arm's empty vs_disabled_* cells filled in with zeros.
    zero_filled = df.copy(deep=True)
    reference_rows = zero_filled.index[zero_filled["vs_disabled_paired_dz"].isna()]
    assert len(reference_rows) > 0, "expected the capacity_disabled reference arm to leave vs_disabled_* empty"
    zero_filled.loc[reference_rows[0], "vs_disabled_paired_dz"] = 0.0
    with pytest.raises(AssertionError):
        _assert_summary_stats_matches_golden(zero_filled, golden)

    # (d) the paired p-value moves 35 orders of magnitude, still far below the
    #     1e-9 absolute floor the "float" kind would have compared it under.
    moved = df.copy(deep=True)
    smallest = moved["vs_disabled_paired_p"].idxmin()
    original = float(moved.loc[smallest, "vs_disabled_paired_p"])
    assert original < 1e-30, "expected a real p-value far below the old absolute floor"
    perturbed_to = 1e-20
    assert perturbed_to == pytest.approx(original, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), (
        f"the float kind really would call {original!r} and {perturbed_to!r} equal"
    )
    moved.loc[smallest, "vs_disabled_paired_p"] = perturbed_to
    with pytest.raises(AssertionError):
        _assert_summary_stats_matches_golden(moved, golden)

    assert _sha256(SUMMARY_STATS_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


# ---------------------------------------------------------------------------
# 6. Pinned digest of results/structural_sensitivity.csv, the D9 structural
#    coefficient sensitivity analysis, every data row across all three scopes
#    (cell_full_factorial, marginal_per_coefficient, coefficient_verdict).
#    Read-only: this test never writes the CSV, and skips (does not fail) if
#    the analysis output is absent on this machine.
# ---------------------------------------------------------------------------


def test_structural_sensitivity_csv_matches_pinned_digest():
    _skip_unless_pinned_csv_available(STRUCTURAL_SENSITIVITY_CSV_PATH, STRUCTURAL_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    golden = _load_golden(STRUCTURAL_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(STRUCTURAL_SENSITIVITY_CSV_PATH)
    _assert_structural_sensitivity_matches_golden(df, golden)


def test_structural_sensitivity_pin_bites_on_plausible_regressions():
    """Prove the pin above is not vacuous, on in-memory copies only.

    Three regressions this analysis could plausibly suffer, one per copy: the
    paired statistic silently replaced by the naive change of means (a real
    hazard here, since both columns sit side by side in this CSV and only the
    paired one respects the seed pairing), a multiple-comparison survival flag
    flipping, and a row disappearing. The CSV's own digest is captured before
    and re-checked after, so the file on disk is provably untouched.
    """
    _skip_unless_pinned_csv_available(STRUCTURAL_SENSITIVITY_CSV_PATH, STRUCTURAL_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(STRUCTURAL_SENSITIVITY_CSV_PATH)
    golden = _load_golden(STRUCTURAL_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(STRUCTURAL_SENSITIVITY_CSV_PATH)

    _assert_structural_sensitivity_matches_golden(df, golden)  # unperturbed: passes

    # (a) paired_mean_pct_change silently swapped for the naive change of means.
    naive_swap = df.copy(deep=True)
    differ = naive_swap.index[(naive_swap["paired_mean_pct_change"] - naive_swap["naive_pct_change_of_means"]).abs() > 1e-6]
    assert len(differ) > 0, "expected at least one row where the paired and naive statistics differ"
    naive_swap.loc[differ[0], "paired_mean_pct_change"] = naive_swap.loc[differ[0], "naive_pct_change_of_means"]
    with pytest.raises(AssertionError):
        _assert_structural_sensitivity_matches_golden(naive_swap, golden)

    # (b) a Holm survival flag flips.
    flipped = df.copy(deep=True)
    surviving = flipped.index[flipped["survives_holm_alpha05"].eq(True)]
    assert len(surviving) > 0, "expected at least one row surviving Holm correction"
    flipped.loc[surviving[0], "survives_holm_alpha05"] = False
    with pytest.raises(AssertionError):
        _assert_structural_sensitivity_matches_golden(flipped, golden)

    # (c) a row disappears.
    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_structural_sensitivity_matches_golden(dropped, golden)

    assert _sha256(STRUCTURAL_SENSITIVITY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_structural_sensitivity_pin_bites_on_a_p_value_magnitude_shift():
    """A p-value moving 35 orders of magnitude, while staying far below 1e-9.

    Under the previous comparison (rel=1e-9 with abs=1e-9, and pytest.approx
    passes on EITHER tolerance) this exact perturbation PASSED: both the real
    value and the perturbed one sit under the absolute floor, so the floor alone
    satisfied the comparison and the magnitude was never pinned at all. The
    assertion inside the loop reproduces that old rule and shows it calling the
    two values equal, immediately before the current rule rejects them. Done for
    all three correction columns, on in-memory copies only.
    """
    _skip_unless_pinned_csv_available(STRUCTURAL_SENSITIVITY_CSV_PATH, STRUCTURAL_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(STRUCTURAL_SENSITIVITY_CSV_PATH)
    golden = _load_golden(STRUCTURAL_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(STRUCTURAL_SENSITIVITY_CSV_PATH)

    _assert_structural_sensitivity_matches_golden(df, golden)  # unperturbed: passes

    perturbed_to = 1e-20  # still hopelessly significant, still far below 1e-9
    for column in ("p_uncorrected", "p_holm", "p_bh"):
        moved = df.copy(deep=True)
        row = moved[column].idxmin()
        original = float(moved.loc[row, column])
        assert original < 1e-30, f"{column}: expected a real p-value far below the old absolute floor"
        assert perturbed_to == pytest.approx(original, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), (
            f"{column}: the old rule really did call {original!r} and {perturbed_to!r} equal"
        )
        moved.loc[row, column] = perturbed_to
        with pytest.raises(AssertionError):
            _assert_structural_sensitivity_matches_golden(moved, golden)

    assert _sha256(STRUCTURAL_SENSITIVITY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_structural_sensitivity_pin_bites_on_the_verdict_supporting_evidence():
    """The D9 fragility verdict rests on more than the verdict string.

    Every perturbation below leaves all six verdict strings untouched and
    still fails, which is the whole point: the strings do not quote the
    percent-change range they were derived from and do not mention
    significance at all, so pinning them alone would let the evidence move
    while the conclusion stood. In-memory copies only.
    """
    _skip_unless_pinned_csv_available(STRUCTURAL_SENSITIVITY_CSV_PATH, STRUCTURAL_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(STRUCTURAL_SENSITIVITY_CSV_PATH)
    golden = _load_golden(STRUCTURAL_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(STRUCTURAL_SENSITIVITY_CSV_PATH)

    _assert_structural_sensitivity_matches_golden(df, golden)  # unperturbed: passes

    verdict_rows = df.index[df["scope"] == "coefficient_verdict"]
    assert len(verdict_rows) > 0, "expected the coefficient_verdict scope to be present"

    # (a) the peak-to-average sign flip flattened: the positive end of the
    #     range pulled back below zero, so the range no longer straddles zero
    #     at all, while the FRAGILE verdict string still says it does.
    flattened = df.copy(deep=True)
    straddling = [row for row in verdict_rows if float(flattened.loc[row, "max_pct_change_across_levels"]) > 0.0]
    assert straddling, "expected at least one verdict row whose percent-change range crosses zero"
    flattened.loc[straddling[0], "max_pct_change_across_levels"] = -0.5
    assert "FRAGILE" in str(flattened.loc[straddling[0], "verdict"]), "the verdict string is deliberately left alone"
    with pytest.raises(AssertionError):
        _assert_structural_sensitivity_matches_golden(flattened, golden)

    # (b) significance at every level quietly lost. No verdict string mentions
    #     significance, so nothing else in the file records this.
    designificant = df.copy(deep=True)
    significant = [row for row in verdict_rows if _as_bool(designificant.loc[row, "significant_at_every_level_holm"])]
    assert significant, "expected every verdict row to be significant at every level today"
    verdict_before = str(designificant.loc[significant[0], "verdict"])
    designificant.loc[significant[0], "significant_at_every_level_holm"] = False
    assert str(designificant.loc[significant[0], "verdict"]) == verdict_before
    with pytest.raises(AssertionError):
        _assert_structural_sensitivity_matches_golden(designificant, golden)

    # (c) a confidence-interval disagreement appears and goes unannounced.
    disagreeing = df.copy(deep=True)
    agreeing = disagreeing.index[disagreeing["ci_disagreement_flag"].eq(False)]
    assert len(agreeing) > 0, "expected the t and bootstrap intervals to agree everywhere today"
    disagreeing.loc[agreeing[0], "ci_disagreement_flag"] = True
    with pytest.raises(AssertionError):
        _assert_structural_sensitivity_matches_golden(disagreeing, golden)

    # (d) the metric-3 sign convention reworded, reversing how every
    #     percent-change number in the file is read without moving one of them.
    reworded = df.copy(deep=True)
    reworded["metric3_sign_convention"] = (
        "metric3 sign convention: positive (ablation - disabled) diff means IMPROVED stability."
    )
    with pytest.raises(AssertionError):
        _assert_structural_sensitivity_matches_golden(reworded, golden)

    assert _sha256(STRUCTURAL_SENSITIVITY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


# ---------------------------------------------------------------------------
# 7. Pinned digest of results/demand_source_comparison.csv, the Phase 12
#    synthetic-versus-OPSD headline comparison, every data row. Read-only and
#    skip-if-absent, same as item 6.
# ---------------------------------------------------------------------------


def test_demand_source_comparison_csv_matches_pinned_digest():
    _skip_unless_pinned_csv_available(DEMAND_SOURCE_CSV_PATH, DEMAND_SOURCE_GOLDEN_PATH)

    import pandas as pd

    golden = _load_golden(DEMAND_SOURCE_GOLDEN_PATH)
    df = pd.read_csv(DEMAND_SOURCE_CSV_PATH)
    _assert_demand_source_comparison_matches_golden(df, golden)


def test_demand_source_comparison_pin_bites_on_plausible_regressions():
    """Prove the pin above is not vacuous, on in-memory copies only.

    Regressions specific to a two-source comparison: a rounding-level drift in
    one mean, the two demand sources swapped in one row (the columns are
    adjacent and interchangeable in shape, so a mis-ordered merge looks
    entirely plausible), and a reordered file. A pure reorder must still PASS,
    because rows are keyed by identity rather than by position; a reorder that
    also changes a number must still fail.
    """
    _skip_unless_pinned_csv_available(DEMAND_SOURCE_CSV_PATH, DEMAND_SOURCE_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(DEMAND_SOURCE_CSV_PATH)
    golden = _load_golden(DEMAND_SOURCE_GOLDEN_PATH)
    df = pd.read_csv(DEMAND_SOURCE_CSV_PATH)

    _assert_demand_source_comparison_matches_golden(df, golden)  # unperturbed: passes

    # (a) one mean drifts by 1e-6, far too small to notice by eye.
    drifted = df.copy(deep=True)
    small_mean = drifted.index[drifted["metric"] == "avg_cost_per_kwh_eur"]
    assert len(small_mean) > 0
    drifted.loc[small_mean[0], "mean_opsd"] = drifted.loc[small_mean[0], "mean_opsd"] + 1e-6
    with pytest.raises(AssertionError):
        _assert_demand_source_comparison_matches_golden(drifted, golden)

    # (b) the synthetic and OPSD columns swapped in one row.
    swapped = df.copy(deep=True)
    differ = swapped.index[(swapped["mean_synthetic"] - swapped["mean_opsd"]).abs() > 1e-6]
    assert len(differ) > 0, "expected at least one row where the two demand sources differ"
    row = differ[0]
    swapped.loc[row, ["mean_synthetic", "mean_opsd"]] = [swapped.loc[row, "mean_opsd"], swapped.loc[row, "mean_synthetic"]]
    with pytest.raises(AssertionError):
        _assert_demand_source_comparison_matches_golden(swapped, golden)

    # (c) rows reordered: fine on its own, fatal once a number moves with it.
    reordered = df.iloc[::-1].reset_index(drop=True)
    _assert_demand_source_comparison_matches_golden(reordered, golden)
    reordered_and_changed = reordered.copy(deep=True)
    nonzero_delta = reordered_and_changed.index[reordered_and_changed["delta_opsd_minus_synthetic"].abs() > 1e-6]
    assert len(nonzero_delta) > 0
    reordered_and_changed.loc[nonzero_delta[0], "delta_opsd_minus_synthetic"] *= -1.0
    with pytest.raises(AssertionError):
        _assert_demand_source_comparison_matches_golden(reordered_and_changed, golden)

    assert _sha256(DEMAND_SOURCE_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


# ---------------------------------------------------------------------------
# 8. Pinned digest of results/monopoly_comparison.csv, the broker_count=1
#    monopoly arm, every data row and every numeric column. Read-only and
#    skip-if-absent, same as items 6 and 7.
# ---------------------------------------------------------------------------


def test_monopoly_comparison_csv_matches_pinned_digest():
    _skip_unless_pinned_csv_available(MONOPOLY_CSV_PATH, MONOPOLY_GOLDEN_PATH)

    import pandas as pd

    golden = _load_golden(MONOPOLY_GOLDEN_PATH)
    df = pd.read_csv(MONOPOLY_CSV_PATH)
    _assert_monopoly_comparison_matches_golden(df, golden)


def test_monopoly_comparison_pin_bites_on_plausible_regressions():
    """Prove the pin above is not vacuous, on in-memory copies only.

    Regressions specific to this arm: a seed-count tally off by one, a fire
    rate reported as a percentage instead of a fraction (this column is a
    fraction of steps, and the percent/fraction confusion is the classic way
    it goes wrong), the monopoly flag itself flipping, and a duplicated row
    appended. The sign-convention note is pinned too, because editing it
    silently reverses how every metric-3 number in the file is read.
    """
    _skip_unless_pinned_csv_available(MONOPOLY_CSV_PATH, MONOPOLY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(MONOPOLY_CSV_PATH)
    golden = _load_golden(MONOPOLY_GOLDEN_PATH)
    df = pd.read_csv(MONOPOLY_CSV_PATH)

    _assert_monopoly_comparison_matches_golden(df, golden)  # unperturbed: passes

    # (a) an improved-seed tally off by one.
    off_by_one = df.copy(deep=True)
    column = "feeder_peak_to_average_ratio_n_improved_of_30"
    off_by_one.loc[off_by_one.index[0], column] = int(off_by_one.loc[off_by_one.index[0], column]) - 1
    with pytest.raises(AssertionError):
        _assert_monopoly_comparison_matches_golden(off_by_one, golden)

    # (b) the fire rate reported as a percentage rather than a fraction.
    as_percent = df.copy(deep=True)
    as_percent["capacity_both_mean_fire_rate"] = as_percent["capacity_both_mean_fire_rate"] * 100.0
    with pytest.raises(AssertionError):
        _assert_monopoly_comparison_matches_golden(as_percent, golden)

    # (c) the monopoly flag flips on the broker_count=1 arm.
    misflagged = df.copy(deep=True)
    monopoly_rows = misflagged.index[misflagged["broker_count"] == 1]
    assert len(monopoly_rows) > 0, "expected a broker_count=1 arm"
    misflagged.loc[monopoly_rows[0], "is_monopoly"] = False
    with pytest.raises(AssertionError):
        _assert_monopoly_comparison_matches_golden(misflagged, golden)

    # (d) a duplicated row appended.
    appended = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
    with pytest.raises(AssertionError):
        _assert_monopoly_comparison_matches_golden(appended, golden)

    # (e) the metric-3 sign convention silently reworded.
    reworded = df.copy(deep=True)
    reworded["metric3_sign_convention"] = "metric3 sign convention: positive diff means IMPROVED stability."
    with pytest.raises(AssertionError):
        _assert_monopoly_comparison_matches_golden(reworded, golden)

    assert _sha256(MONOPOLY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_monopoly_comparison_pin_bites_on_a_p_value_magnitude_shift():
    """The same magnitude shift on the monopoly arm's two paired p-values.

    These are the columns the metric-3 significance claim rests on, and under
    the previous comparison (rel=1e-9 with abs=1e-9, either sufficing) this
    exact perturbation PASSED, because both values sit under the absolute floor.
    The assertion inside the loop reproduces that old rule and shows it calling
    the two values equal, immediately before the current rule rejects them.
    In-memory copies only.
    """
    _skip_unless_pinned_csv_available(MONOPOLY_CSV_PATH, MONOPOLY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(MONOPOLY_CSV_PATH)
    golden = _load_golden(MONOPOLY_GOLDEN_PATH)
    df = pd.read_csv(MONOPOLY_CSV_PATH)

    _assert_monopoly_comparison_matches_golden(df, golden)  # unperturbed: passes

    perturbed_to = 1e-20  # still hopelessly significant, still far below 1e-9
    for column in ("feeder_peak_to_average_ratio_paired_p", "feeder_coefficient_of_variation_paired_p"):
        moved = df.copy(deep=True)
        row = moved[column].idxmin()
        original = float(moved.loc[row, column])
        assert original < 1e-30, f"{column}: expected a real p-value far below the old absolute floor"
        assert perturbed_to == pytest.approx(original, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), (
            f"{column}: the old rule really did call {original!r} and {perturbed_to!r} equal"
        )
        moved.loc[row, column] = perturbed_to
        with pytest.raises(AssertionError):
            _assert_monopoly_comparison_matches_golden(moved, golden)

    assert _sha256(MONOPOLY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


# ---------------------------------------------------------------------------
# 9. Column-set pins for the four CSVs already digested above (items 5 to 8).
#    Those digests check every PINNED field's value, but none of them check
#    that the column LIST has not grown: a new column can land on a shipped
#    sweep output and every existing assertion still passes, because they all
#    look fields up by name rather than counting them. Pinned here as an exact
#    set against the real header read straight off disk, so both an added and
#    a removed column fail. structural_sensitivity.csv's twelve config-echo,
#    label and diagnostic columns (see module docstring) are not individually
#    value-pinned, but they are covered here: this is a growth check, not a
#    value check, and unnoticed growth is exactly what an unpinned column
#    would otherwise get away with.
# ---------------------------------------------------------------------------


def _skip_unless_results_artifact_available(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"{path.name} not present (sweep output, not regenerated by tests): {path}")


def _assert_exact_column_set(columns, expected_columns, label: str) -> None:
    actual = set(columns)
    expected = set(expected_columns)
    assert actual == expected, (
        f"{label}: column set changed (added {sorted(actual - expected)}, removed {sorted(expected - actual)})"
    )


SUMMARY_STATS_COLUMNS = (
    "k",
    "broker_count",
    "ablation",
    "metric",
    "metric_label",
    "n_seeds",
    "mean",
    "std",
    "ci95_lo",
    "ci95_hi",
    "ci95_halfwidth",
    "vs_disabled_cohens_d_independent",
    "vs_disabled_paired_dz",
    "vs_disabled_paired_p",
    "vs_disabled_paired_mean_diff",
    "vs_disabled_paired_mean_pct_change",
    "vs_disabled_degenerate_paired_diff",
    "metric3_sign_convention",
    "notes",
)

STRUCTURAL_SENSITIVITY_COLUMNS = (
    "scope",
    "deferrable_fraction",
    "response_reference_eur_per_kwh",
    "payback_cap_fraction",
    "coefficient",
    "coefficient_level",
    "metric",
    "metric_label",
    "n_seeds",
    "mean_disabled",
    "mean_both",
    "paired_mean_diff",
    "paired_mean_pct_change",
    "naive_pct_change_of_means",
    "paired_dz",
    "cohens_d_independent",
    "p_uncorrected",
    "p_holm",
    "p_bh",
    "survives_holm_alpha05",
    "survives_bh_alpha05",
    "ci95_t_lo",
    "ci95_t_hi",
    "ci95_boot_lo",
    "ci95_boot_hi",
    "ci_disagreement_flag",
    "ci_disagreement_detail",
    "ci_width_ratio",
    "degenerate",
    "correction_family",
    "correction_family_size",
    "is_flagged_d9_corner",
    "worsened_vs_disabled",
    "is_shipped_axis_level",
    "min_pct_change_across_levels",
    "max_pct_change_across_levels",
    "sign_consistent_across_marginal_levels",
    "significant_at_every_level_holm",
    "cell_level_sign_consistent_within_every_level",
    "levels_with_within_level_cell_sign_disagreement",
    "verdict",
    "bootstrap_seed",
    "n_bootstrap",
    "alpha",
    "metric3_sign_convention",
    "notes",
)

DEMAND_SOURCE_COLUMNS = (
    "k",
    "broker_count",
    "ablation",
    "metric",
    "n_seeds_synthetic",
    "n_seeds_opsd",
    "mean_synthetic",
    "mean_opsd",
    "std_synthetic",
    "std_opsd",
    "delta_opsd_minus_synthetic",
)

MONOPOLY_COLUMNS = (
    "broker_count",
    "is_monopoly",
    "k",
    "n_seeds",
    "feeder_peak_to_average_ratio_paired_mean_pct_change",
    "feeder_peak_to_average_ratio_paired_dz",
    "feeder_peak_to_average_ratio_paired_p",
    "feeder_peak_to_average_ratio_n_improved_of_30",
    "feeder_coefficient_of_variation_paired_mean_pct_change",
    "feeder_coefficient_of_variation_paired_dz",
    "feeder_coefficient_of_variation_paired_p",
    "feeder_coefficient_of_variation_n_improved_of_30",
    "capacity_both_mean_fire_rate",
    "capacity_both_mean_total_deferred_kwh",
    "avg_cost_per_agent_eur_paired_mean_pct_change",
    "metric3_sign_convention",
    "note",
)


def test_summary_stats_csv_column_set_matches_pinned():
    _skip_unless_results_artifact_available(SUMMARY_STATS_CSV_PATH)
    import pandas as pd

    header = pd.read_csv(SUMMARY_STATS_CSV_PATH, nrows=0).columns
    _assert_exact_column_set(header, SUMMARY_STATS_COLUMNS, "results/summary_stats.csv")


def test_summary_stats_csv_column_set_pin_bites_on_growth_and_shrinkage():
    _skip_unless_results_artifact_available(SUMMARY_STATS_CSV_PATH)
    import pandas as pd

    header = list(pd.read_csv(SUMMARY_STATS_CSV_PATH, nrows=0).columns)
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header + ["a_new_column"], SUMMARY_STATS_COLUMNS, "results/summary_stats.csv")
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header[:-1], SUMMARY_STATS_COLUMNS, "results/summary_stats.csv")


def test_structural_sensitivity_csv_column_set_matches_pinned():
    _skip_unless_results_artifact_available(STRUCTURAL_SENSITIVITY_CSV_PATH)
    import pandas as pd

    header = pd.read_csv(STRUCTURAL_SENSITIVITY_CSV_PATH, nrows=0).columns
    _assert_exact_column_set(header, STRUCTURAL_SENSITIVITY_COLUMNS, "results/structural_sensitivity.csv")


def test_structural_sensitivity_csv_column_set_pin_bites_on_growth_and_shrinkage():
    _skip_unless_results_artifact_available(STRUCTURAL_SENSITIVITY_CSV_PATH)
    import pandas as pd

    header = list(pd.read_csv(STRUCTURAL_SENSITIVITY_CSV_PATH, nrows=0).columns)
    with pytest.raises(AssertionError):
        _assert_exact_column_set(
            header + ["a_new_column"], STRUCTURAL_SENSITIVITY_COLUMNS, "results/structural_sensitivity.csv"
        )
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header[:-1], STRUCTURAL_SENSITIVITY_COLUMNS, "results/structural_sensitivity.csv")


def test_demand_source_comparison_csv_column_set_matches_pinned():
    _skip_unless_results_artifact_available(DEMAND_SOURCE_CSV_PATH)
    import pandas as pd

    header = pd.read_csv(DEMAND_SOURCE_CSV_PATH, nrows=0).columns
    _assert_exact_column_set(header, DEMAND_SOURCE_COLUMNS, "results/demand_source_comparison.csv")


def test_demand_source_comparison_csv_column_set_pin_bites_on_growth_and_shrinkage():
    _skip_unless_results_artifact_available(DEMAND_SOURCE_CSV_PATH)
    import pandas as pd

    header = list(pd.read_csv(DEMAND_SOURCE_CSV_PATH, nrows=0).columns)
    with pytest.raises(AssertionError):
        _assert_exact_column_set(
            header + ["a_new_column"], DEMAND_SOURCE_COLUMNS, "results/demand_source_comparison.csv"
        )
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header[:-1], DEMAND_SOURCE_COLUMNS, "results/demand_source_comparison.csv")


def test_monopoly_comparison_csv_column_set_matches_pinned():
    _skip_unless_results_artifact_available(MONOPOLY_CSV_PATH)
    import pandas as pd

    header = pd.read_csv(MONOPOLY_CSV_PATH, nrows=0).columns
    _assert_exact_column_set(header, MONOPOLY_COLUMNS, "results/monopoly_comparison.csv")


def test_monopoly_comparison_csv_column_set_pin_bites_on_growth_and_shrinkage():
    _skip_unless_results_artifact_available(MONOPOLY_CSV_PATH)
    import pandas as pd

    header = list(pd.read_csv(MONOPOLY_CSV_PATH, nrows=0).columns)
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header + ["a_new_column"], MONOPOLY_COLUMNS, "results/monopoly_comparison.csv")
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header[:-1], MONOPOLY_COLUMNS, "results/monopoly_comparison.csv")


# ---------------------------------------------------------------------------
# 10. Shape pins (row count and exact column set) for the four raw sweep
#     parquets under results/, none of which had any regression guard at all
#     before this. Each one feeds a CSV digest already pinned above
#     (summary_stats.csv, monopoly_comparison.csv and demand_source_comparison.csv
#     all derive from one of sweep_raw/sweep_monopoly/sweep_opsd_headline, and
#     structural_sensitivity.csv derives from sweep_structural), but a change
#     to the parquet itself, upstream of those derived CSVs, would only be
#     caught here: a stale derived CSV checked in against a regenerated
#     parquet would still pass every digest test above.
# ---------------------------------------------------------------------------

SWEEP_RAW_COLUMNS = (
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
)
SWEEP_RAW_ROW_COUNT = 1440  # 4 k levels x 3 broker_count levels x 4 ablations x 30 seeds

# sweep_monopoly.parquet and sweep_opsd_headline.parquet are written by the
# same per-run metrics writer as sweep_raw.parquet, so they share its exact
# schema; only the swept parameters and the row count differ.
SWEEP_MONOPOLY_COLUMNS = SWEEP_RAW_COLUMNS
SWEEP_MONOPOLY_ROW_COUNT = 480  # 4 k levels x 1 broker_count level (monopoly) x 4 ablations x 30 seeds
SWEEP_OPSD_HEADLINE_COLUMNS = SWEEP_RAW_COLUMNS
SWEEP_OPSD_HEADLINE_ROW_COUNT = 120  # 1 k level (headline cell) x 1 broker_count level x 4 ablations x 30 seeds

SWEEP_STRUCTURAL_COLUMNS = (
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
)
SWEEP_STRUCTURAL_ROW_COUNT = 2880  # 16 (deferrable_fraction, response_reference) pairs x 3 payback_cap_fraction x 2 ablations x 30 seeds


def _assert_parquet_shape(df, expected_row_count: int, expected_columns, label: str) -> None:
    assert len(df) == expected_row_count, f"{label}: row count changed ({len(df)} rows, pinned {expected_row_count})"
    _assert_exact_column_set(df.columns, expected_columns, label)


def test_sweep_raw_parquet_shape_matches_pinned():
    _skip_unless_results_artifact_available(SWEEP_RAW_PARQUET_PATH)
    import pandas as pd

    df = pd.read_parquet(SWEEP_RAW_PARQUET_PATH)
    _assert_parquet_shape(df, SWEEP_RAW_ROW_COUNT, SWEEP_RAW_COLUMNS, "results/sweep_raw.parquet")


def test_sweep_raw_parquet_shape_pin_bites_on_growth_shrinkage_and_row_count_drift():
    _skip_unless_results_artifact_available(SWEEP_RAW_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_RAW_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_RAW_PARQUET_PATH)

    grown = df.copy(deep=True)
    grown["a_new_column"] = 0.0
    with pytest.raises(AssertionError):
        _assert_parquet_shape(grown, SWEEP_RAW_ROW_COUNT, SWEEP_RAW_COLUMNS, "results/sweep_raw.parquet")

    shrunk = df.drop(columns=["peak_rss_mb"])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(shrunk, SWEEP_RAW_ROW_COUNT, SWEEP_RAW_COLUMNS, "results/sweep_raw.parquet")

    appended = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
    with pytest.raises(AssertionError):
        _assert_parquet_shape(appended, SWEEP_RAW_ROW_COUNT, SWEEP_RAW_COLUMNS, "results/sweep_raw.parquet")

    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(dropped, SWEEP_RAW_ROW_COUNT, SWEEP_RAW_COLUMNS, "results/sweep_raw.parquet")

    assert _sha256(SWEEP_RAW_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


def test_sweep_monopoly_parquet_shape_matches_pinned():
    _skip_unless_results_artifact_available(SWEEP_MONOPOLY_PARQUET_PATH)
    import pandas as pd

    df = pd.read_parquet(SWEEP_MONOPOLY_PARQUET_PATH)
    _assert_parquet_shape(df, SWEEP_MONOPOLY_ROW_COUNT, SWEEP_MONOPOLY_COLUMNS, "results/sweep_monopoly.parquet")


def test_sweep_monopoly_parquet_shape_pin_bites_on_growth_shrinkage_and_row_count_drift():
    _skip_unless_results_artifact_available(SWEEP_MONOPOLY_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_MONOPOLY_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_MONOPOLY_PARQUET_PATH)

    grown = df.copy(deep=True)
    grown["a_new_column"] = 0.0
    with pytest.raises(AssertionError):
        _assert_parquet_shape(grown, SWEEP_MONOPOLY_ROW_COUNT, SWEEP_MONOPOLY_COLUMNS, "results/sweep_monopoly.parquet")

    shrunk = df.drop(columns=["peak_rss_mb"])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            shrunk, SWEEP_MONOPOLY_ROW_COUNT, SWEEP_MONOPOLY_COLUMNS, "results/sweep_monopoly.parquet"
        )

    appended = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            appended, SWEEP_MONOPOLY_ROW_COUNT, SWEEP_MONOPOLY_COLUMNS, "results/sweep_monopoly.parquet"
        )

    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            dropped, SWEEP_MONOPOLY_ROW_COUNT, SWEEP_MONOPOLY_COLUMNS, "results/sweep_monopoly.parquet"
        )

    assert _sha256(SWEEP_MONOPOLY_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


def test_sweep_opsd_headline_parquet_shape_matches_pinned():
    _skip_unless_results_artifact_available(SWEEP_OPSD_HEADLINE_PARQUET_PATH)
    import pandas as pd

    df = pd.read_parquet(SWEEP_OPSD_HEADLINE_PARQUET_PATH)
    _assert_parquet_shape(
        df, SWEEP_OPSD_HEADLINE_ROW_COUNT, SWEEP_OPSD_HEADLINE_COLUMNS, "results/sweep_opsd_headline.parquet"
    )


def test_sweep_opsd_headline_parquet_shape_pin_bites_on_growth_shrinkage_and_row_count_drift():
    _skip_unless_results_artifact_available(SWEEP_OPSD_HEADLINE_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_OPSD_HEADLINE_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_OPSD_HEADLINE_PARQUET_PATH)

    grown = df.copy(deep=True)
    grown["a_new_column"] = 0.0
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            grown, SWEEP_OPSD_HEADLINE_ROW_COUNT, SWEEP_OPSD_HEADLINE_COLUMNS, "results/sweep_opsd_headline.parquet"
        )

    shrunk = df.drop(columns=["peak_rss_mb"])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            shrunk, SWEEP_OPSD_HEADLINE_ROW_COUNT, SWEEP_OPSD_HEADLINE_COLUMNS, "results/sweep_opsd_headline.parquet"
        )

    appended = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            appended, SWEEP_OPSD_HEADLINE_ROW_COUNT, SWEEP_OPSD_HEADLINE_COLUMNS, "results/sweep_opsd_headline.parquet"
        )

    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            dropped, SWEEP_OPSD_HEADLINE_ROW_COUNT, SWEEP_OPSD_HEADLINE_COLUMNS, "results/sweep_opsd_headline.parquet"
        )

    assert _sha256(SWEEP_OPSD_HEADLINE_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


def test_sweep_structural_parquet_shape_matches_pinned():
    _skip_unless_results_artifact_available(SWEEP_STRUCTURAL_PARQUET_PATH)
    import pandas as pd

    df = pd.read_parquet(SWEEP_STRUCTURAL_PARQUET_PATH)
    _assert_parquet_shape(df, SWEEP_STRUCTURAL_ROW_COUNT, SWEEP_STRUCTURAL_COLUMNS, "results/sweep_structural.parquet")


def test_sweep_structural_parquet_shape_pin_bites_on_growth_shrinkage_and_row_count_drift():
    _skip_unless_results_artifact_available(SWEEP_STRUCTURAL_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_STRUCTURAL_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_STRUCTURAL_PARQUET_PATH)

    grown = df.copy(deep=True)
    grown["a_new_column"] = 0.0
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            grown, SWEEP_STRUCTURAL_ROW_COUNT, SWEEP_STRUCTURAL_COLUMNS, "results/sweep_structural.parquet"
        )

    shrunk = df.drop(columns=["row_provenance"])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            shrunk, SWEEP_STRUCTURAL_ROW_COUNT, SWEEP_STRUCTURAL_COLUMNS, "results/sweep_structural.parquet"
        )

    appended = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            appended, SWEEP_STRUCTURAL_ROW_COUNT, SWEEP_STRUCTURAL_COLUMNS, "results/sweep_structural.parquet"
        )

    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            dropped, SWEEP_STRUCTURAL_ROW_COUNT, SWEEP_STRUCTURAL_COLUMNS, "results/sweep_structural.parquet"
        )

    assert _sha256(SWEEP_STRUCTURAL_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


# ---------------------------------------------------------------------------
# 11. The five deferred-energy figures docs/PROGRESS.md calls out by name as
#     having no regression guard (38822, 58602, 5717, 49941 and 25990 kWh/year).
#     Each is the mean of total_deferred_kwh, filtered to one (k, broker_count,
#     ablation) cell over that cell's 30 seeds, in whichever of the raw sweep
#     parquets holds the cell. sweep_structural.parquet backs none of these
#     five directly (Section 6.11's own numbers are the structural-coefficient
#     sweep pinned already via structural_sensitivity.csv's digest above, and
#     the text is explicit that 38822 kWh/year "come[s] from the main D8 sweep
#     ... not from a re-run inside this sweep"), so it carries only the shape
#     pin above, not an aggregate pin here.
# ---------------------------------------------------------------------------


def _assert_deferred_energy_aggregate(
    df, k: float, broker_count: int, ablation: str, expected_mean_kwh: float, expected_rounded_kwh: int, label: str
) -> None:
    subset = df[(df["k"] == k) & (df["broker_count"] == broker_count) & (df["ablation"] == ablation)]
    assert len(subset) > 0, f"{label}: no rows for k={k}, broker_count={broker_count}, ablation={ablation!r}"
    mean_kwh = subset["total_deferred_kwh"].mean()
    assert mean_kwh == pytest.approx(expected_mean_kwh, rel=DEFERRED_ENERGY_REL_TOL, abs=DEFERRED_ENERGY_ABS_TOL), label
    assert round(mean_kwh) == expected_rounded_kwh, f"{label}: thesis-quoted rounded kWh figure changed"


def test_sweep_raw_parquet_pins_headline_deferred_energy_at_k1_bc3():
    """results/sweep_raw.parquet, filtered to k=1.0, broker_count=3,
    ablation="capacity_both", mean of total_deferred_kwh over the cell's 30
    seeds: the shipped calibration's headline deferred-energy figure, the same
    cell whose -9.84 percent peak-to-average / -7.22 percent coefficient-of-
    variation result Section 6.2 reports as the primary result. Quoted as
    "38822 kWh deferred" in Section 6.9's monopoly comparison and again in
    Section 6.11 as "the shipped calibration ... defers 38822 kWh/year".
    """
    _skip_unless_results_artifact_available(SWEEP_RAW_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_RAW_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_RAW_PARQUET_PATH)
    label = "results/sweep_raw.parquet k=1.0 bc=3 capacity_both total_deferred_kwh"

    _assert_deferred_energy_aggregate(
        df, k=1.0, broker_count=3, ablation="capacity_both",
        expected_mean_kwh=38821.852828800955, expected_rounded_kwh=38822, label=label,
    )

    nudged = df.copy(deep=True)
    target = nudged.index[(nudged["k"] == 1.0) & (nudged["broker_count"] == 3) & (nudged["ablation"] == "capacity_both")]
    assert len(target) > 0
    nudged.loc[target[0], "total_deferred_kwh"] += 100.0
    with pytest.raises(AssertionError):
        _assert_deferred_energy_aggregate(
            nudged, k=1.0, broker_count=3, ablation="capacity_both",
            expected_mean_kwh=38821.852828800955, expected_rounded_kwh=38822, label=label,
        )

    assert _sha256(SWEEP_RAW_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


def test_sweep_raw_parquet_pins_deferred_energy_at_k0p5_bc3():
    """results/sweep_raw.parquet, filtered to k=0.5, broker_count=3,
    ablation="capacity_both", mean of total_deferred_kwh over the cell's 30
    seeds: Section 6.2's low-k end of the deferred-energy-versus-k mechanism
    explanation, prose-rounded there to "about 58600 kWh at k=0.5".
    """
    _skip_unless_results_artifact_available(SWEEP_RAW_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_RAW_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_RAW_PARQUET_PATH)
    label = "results/sweep_raw.parquet k=0.5 bc=3 capacity_both total_deferred_kwh"

    _assert_deferred_energy_aggregate(
        df, k=0.5, broker_count=3, ablation="capacity_both",
        expected_mean_kwh=58602.22566287547, expected_rounded_kwh=58602, label=label,
    )

    nudged = df.copy(deep=True)
    target = nudged.index[(nudged["k"] == 0.5) & (nudged["broker_count"] == 3) & (nudged["ablation"] == "capacity_both")]
    assert len(target) > 0
    nudged.loc[target[0], "total_deferred_kwh"] += 100.0
    with pytest.raises(AssertionError):
        _assert_deferred_energy_aggregate(
            nudged, k=0.5, broker_count=3, ablation="capacity_both",
            expected_mean_kwh=58602.22566287547, expected_rounded_kwh=58602, label=label,
        )

    assert _sha256(SWEEP_RAW_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


def test_sweep_raw_parquet_pins_deferred_energy_at_k2_bc3():
    """results/sweep_raw.parquet, filtered to k=2.0, broker_count=3,
    ablation="capacity_both", mean of total_deferred_kwh over the cell's 30
    seeds: Section 6.2's high-k end of the same explanation, prose-rounded
    there to "about 5700 kWh over the year at k=2.0", the small deferred
    volume that the section uses to explain why the peak-to-average ratio
    reverses to worsening at high k while the coefficient of variation does not.
    """
    _skip_unless_results_artifact_available(SWEEP_RAW_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_RAW_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_RAW_PARQUET_PATH)
    label = "results/sweep_raw.parquet k=2.0 bc=3 capacity_both total_deferred_kwh"

    _assert_deferred_energy_aggregate(
        df, k=2.0, broker_count=3, ablation="capacity_both",
        expected_mean_kwh=5717.030428500567, expected_rounded_kwh=5717, label=label,
    )

    nudged = df.copy(deep=True)
    target = nudged.index[(nudged["k"] == 2.0) & (nudged["broker_count"] == 3) & (nudged["ablation"] == "capacity_both")]
    assert len(target) > 0
    nudged.loc[target[0], "total_deferred_kwh"] += 100.0
    with pytest.raises(AssertionError):
        _assert_deferred_energy_aggregate(
            nudged, k=2.0, broker_count=3, ablation="capacity_both",
            expected_mean_kwh=5717.030428500567, expected_rounded_kwh=5717, label=label,
        )

    assert _sha256(SWEEP_RAW_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


def test_sweep_monopoly_parquet_pins_deferred_energy_at_bc1():
    """results/sweep_monopoly.parquet, filtered to k=1.0, broker_count=1,
    ablation="capacity_both", mean of total_deferred_kwh over the cell's 30
    seeds: Section 6.9's monopoly-arm figure, "bc=1 defers 49941 kWh over the
    year", contrasted there against bc=3's 38822 kWh to argue monopoly defers
    more energy for a larger consumer-cost penalty and no peak-to-average
    benefit. The same value already sits, unrounded, in
    results/monopoly_comparison.csv's capacity_both_mean_total_deferred_kwh
    column for this row, and is pinned there too (item 8 above); this test
    recomputes it straight from the raw parquet instead, so a change to the
    parquet that a stale, un-regenerated monopoly_comparison.csv failed to
    pick up would still be caught here.
    """
    _skip_unless_results_artifact_available(SWEEP_MONOPOLY_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_MONOPOLY_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_MONOPOLY_PARQUET_PATH)
    label = "results/sweep_monopoly.parquet k=1.0 bc=1 capacity_both total_deferred_kwh"

    _assert_deferred_energy_aggregate(
        df, k=1.0, broker_count=1, ablation="capacity_both",
        expected_mean_kwh=49941.44911896594, expected_rounded_kwh=49941, label=label,
    )

    nudged = df.copy(deep=True)
    target = nudged.index[(nudged["k"] == 1.0) & (nudged["broker_count"] == 1) & (nudged["ablation"] == "capacity_both")]
    assert len(target) > 0
    nudged.loc[target[0], "total_deferred_kwh"] += 100.0
    with pytest.raises(AssertionError):
        _assert_deferred_energy_aggregate(
            nudged, k=1.0, broker_count=1, ablation="capacity_both",
            expected_mean_kwh=49941.44911896594, expected_rounded_kwh=49941, label=label,
        )

    assert _sha256(SWEEP_MONOPOLY_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


def test_sweep_opsd_headline_parquet_pins_deferred_energy():
    """results/sweep_opsd_headline.parquet, filtered to k=1.0, broker_count=3,
    ablation="capacity_both" (the sole headline cell this file holds, re-run
    on measured OPSD demand instead of the synthetic series), mean of
    total_deferred_kwh over the cell's 30 seeds: Section 6.12's cross-check
    figure, "The OPSD run defers 25990 kWh/year", used there to argue the
    measured-demand run sits inside D9's own benefit region rather than near
    its fragile high-volume corner.
    """
    _skip_unless_results_artifact_available(SWEEP_OPSD_HEADLINE_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_OPSD_HEADLINE_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_OPSD_HEADLINE_PARQUET_PATH)
    label = "results/sweep_opsd_headline.parquet k=1.0 bc=3 capacity_both total_deferred_kwh"

    _assert_deferred_energy_aggregate(
        df, k=1.0, broker_count=3, ablation="capacity_both",
        expected_mean_kwh=25989.51152211962, expected_rounded_kwh=25990, label=label,
    )

    nudged = df.copy(deep=True)
    target = nudged.index[(nudged["k"] == 1.0) & (nudged["broker_count"] == 3) & (nudged["ablation"] == "capacity_both")]
    assert len(target) > 0
    nudged.loc[target[0], "total_deferred_kwh"] += 100.0
    with pytest.raises(AssertionError):
        _assert_deferred_energy_aggregate(
            nudged, k=1.0, broker_count=3, ablation="capacity_both",
            expected_mean_kwh=25989.51152211962, expected_rounded_kwh=25990, label=label,
        )

    assert _sha256(SWEEP_OPSD_HEADLINE_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


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

    # Every golden file, found by glob rather than by name, so a pin added
    # later is covered by this guarantee automatically. The named paths below
    # are then checked to be inside that snapshot, so a rename cannot quietly
    # drop one of the known pins out of the guard.
    golden_paths = sorted(GOLDEN_DIR.glob("*.json"))
    assert golden_paths, "expected at least one golden file to already exist"
    for known_path in (
        SHORT_RUN_GOLDEN_PATH,
        SUMMARY_STATS_GOLDEN_PATH,
        STRUCTURAL_SENSITIVITY_GOLDEN_PATH,
        DEMAND_SOURCE_GOLDEN_PATH,
        MONOPOLY_GOLDEN_PATH,
    ):
        if known_path.is_file():
            assert known_path in golden_paths, f"golden file missing from the no-rewrite snapshot: {known_path}"
    mtimes_before = {path: path.stat().st_mtime_ns for path in golden_paths}

    spec = importlib.util.spec_from_file_location("_golden_regen_import_probe", REGENERATE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # That script's top level inserts into sys.path and installs a warnings
    # filter, both process-global and both otherwise outliving this test for
    # the rest of the pytest session. Undo both: a test that checks for
    # side effects should not leave any of its own.
    sys_path_before = list(sys.path)
    try:
        with warnings.catch_warnings():
            spec.loader.exec_module(module)  # executes top-level code; module.__name__ != "__main__" here
    finally:
        sys.path[:] = sys_path_before

    assert module.__name__ != "__main__"
    assert hasattr(module, "main"), "regeneration entry point must be a function, not top-level code"

    mtimes_after = {path: path.stat().st_mtime_ns for path in golden_paths}
    assert mtimes_after == mtimes_before, "importing the regeneration script rewrote a golden file"
