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

Three further pinned digests (items 12 to 14) close the gap left open at the
end of Phase 17: results/effect_sizes.csv (185 rows, the file Chapter 1's own
headline figures are drawn from) and results/summary_stats_corrected.csv (188
rows, Section 6.10's per-comparison correction and bootstrap-CI detail) are
each pinned WHOLE, the same way Phase 16 converted summary_stats.csv, since
both are small enough that a claim-bearing subset would save little.
results/family_sensitivity.csv is different: at 1227 rows, a whole-file
digest in the same style would run to roughly 1.6 MB, so it is pinned as a
named CLAIM-BEARING SUBSET instead (the seven family_summary rows plus one
real per-test row), alongside the file's total row count as a shape guard
covering the rows that subset does not otherwise look at. See
FAMILY_SENSITIVITY_IDENTITY's comment below for the full reasoning.

Phase 18.3 adds three more items, pinning the D11 dose-matched-monopoly arm
(docs/DECISIONS.md's "## D11." and "## D11 result:" sections) now that its
artifacts exist:

  15. results/dose_matched_monopoly.csv (46 rows), pinned WHOLE the same way
      effect_sizes.csv and summary_stats_corrected.csv are: small enough that
      a claim-bearing subset would save nothing;
  16. results/sweep_dose_matched_monopoly.parquet: a shape pin (row count and
      exact column set, matching items 10/11's style for the other four raw
      sweep parquets) plus a per-(k, capacity_surcharge_divisor, ablation)
      cell aggregate pin (row count and six metric means), reusing
      _assert_csv_matches_row_keyed_golden/_row_keyed_rows over a derived,
      groupby-aggregated DataFrame rather than inventing a second comparison
      helper for aggregates that no CSV happens to cover directly;
  17. the two guardrail invariants D11 result measures rather than assumes:
      capacity_disabled rows are bit-identical across divisor and against the
      shipped monopoly arm (exact equality, not a tolerance: this was
      measured at exactly 0.0), and divisor 2 matches divisor 1 exactly on
      the physical metrics while differing on cost (the vacuity-trap check:
      a divisor that silently did nothing would still pass the first half).

A later mutation-testing pass over this whole suite (an external examiner
running deliberate, plausible mutations of the model code against every test
here) found that results/surcharge_mode_comparison.csv, D10's own analysis
artifact and the file docs/thesis/06_results.md Section 6.9 leans on hardest,
was referenced by no test at all, unlike every other CSV artifact above. Item
18 closes that gap:

  18. results/surcharge_mode_comparison.csv, pinned WHOLE (24 rows: the 18
      scope="mode_cell" per-cell rows plus the 6 scope="gradient_bc2_to_bc5"
      rows scripts/analyze_surcharge_mode.py added at the same time this pin
      was added; see that script's own "Correction lifted" docstring
      paragraph), keyed by (scope, capacity_surcharge_mode, k, broker_count,
      metric). broker_count is empty on gradient_bc2_to_bc5 rows and metric is
      empty on mode_cell rows, the same empty-cell-as-null handling
      structural_sensitivity.csv's identity already relies on for its own
      per-scope varying columns. Its expected values live in
      tests/golden/surcharge_mode_comparison_pins.json like every other
      item's. That was not true when item 18 was first added: the fix that
      added it was allowed to edit this module but not to add a file under
      tests/golden/, so the snapshot shipped as an INLINE JSON literal of
      about a thousand lines near the bottom of this module. What it pinned
      was right; where it lived was not, because it made this the one pin here
      that the regeneration command below could not produce, leaving a
      legitimate future change to that CSV to be hand-edited as JSON inside a
      test module. Phase 19 moved it into tests/golden/ and taught
      scripts/regenerate_golden_master.py to compute it, changing nothing
      about which columns are pinned or how they are compared.

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

Expected values live in tests/golden/*.json, every one of them, with no
exceptions left. Regenerating those files is a DELIBERATE, explicit act, never
a side effect of running the test suite: a plain `pytest` invocation must
never rewrite these files (see
test_regeneration_script_import_does_not_write_golden_files below, which
proves this structurally rather than just asserting it in prose). To
regenerate, after confirming by hand that a shift in the pinned numbers is an
intended consequence of a real change, run from the repository root:

    python scripts/regenerate_golden_master.py

That command is the ONLY way any pin in this module is written, which is a
stronger claim than it sounds: a regeneration that quietly writes LESS than
the suite checks is the failure mode this harness actually hit in Phase 19
(see _assert_metrics_match_golden's key-set assertion and
test_the_two_metrics_dict_twins_emit_the_same_keys at the bottom of this
module for what went wrong and what now stops it recurring).
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
from microgrid_sim.environment.metrics import MetricsResult
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
EFFECT_SIZES_GOLDEN_PATH = GOLDEN_DIR / "effect_sizes_pins.json"
EFFECT_SIZES_CSV_PATH = RESULTS_DIR / "effect_sizes.csv"
FAMILY_SENSITIVITY_GOLDEN_PATH = GOLDEN_DIR / "family_sensitivity_pins.json"
FAMILY_SENSITIVITY_CSV_PATH = RESULTS_DIR / "family_sensitivity.csv"
SUMMARY_STATS_CORRECTED_GOLDEN_PATH = GOLDEN_DIR / "summary_stats_corrected_pins.json"
SUMMARY_STATS_CORRECTED_CSV_PATH = RESULTS_DIR / "summary_stats_corrected.csv"
DOSE_MATCHED_MONOPOLY_GOLDEN_PATH = GOLDEN_DIR / "dose_matched_monopoly_pins.json"
DOSE_MATCHED_MONOPOLY_CSV_PATH = RESULTS_DIR / "dose_matched_monopoly.csv"
SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_CELL_GOLDEN_PATH = (
    GOLDEN_DIR / "sweep_dose_matched_monopoly_parquet_cell_pins.json"
)
SWEEP_RAW_PARQUET_PATH = RESULTS_DIR / "sweep_raw.parquet"
SWEEP_MONOPOLY_PARQUET_PATH = RESULTS_DIR / "sweep_monopoly.parquet"
SWEEP_OPSD_HEADLINE_PARQUET_PATH = RESULTS_DIR / "sweep_opsd_headline.parquet"
SWEEP_STRUCTURAL_PARQUET_PATH = RESULTS_DIR / "sweep_structural.parquet"
SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH = RESULTS_DIR / "sweep_dose_matched_monopoly.parquet"
REGENERATE_SCRIPT_PATH = REPO_ROOT / "scripts" / "regenerate_golden_master.py"
# Item 18 (see the module docstring's own paragraph below): D10's own analysis
# artifact. Its snapshot used to be an inline JSON literal in this module
# rather than a file under tests/golden/, because the fix that added the pin
# could not write there; that made it the ONE pin here the documented
# regeneration command could not produce, so changing it legitimately meant
# hand-editing about a thousand lines of JSON inside a test module. It is an
# ordinary golden file now, written by that command like every other.
SURCHARGE_MODE_COMPARISON_GOLDEN_PATH = GOLDEN_DIR / "surcharge_mode_comparison_pins.json"
SURCHARGE_MODE_COMPARISON_CSV_PATH = RESULTS_DIR / "surcharge_mode_comparison.csv"

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
        # Phase 3/3b (D6/D7) audit fields, pinned here for the first time: every
        # OTHER place these three are checked against a golden value runs the
        # mechanism OFF (test_master_flag_off_reproduces_baseline_byte_for_byte
        # and friends), so those pins only ever caught a regression that broke
        # the all-zero case. The capacity_both branch below is the one run in
        # this whole harness where the mechanism actually fires
        # (capacity_fire_rate > 0.0 is asserted separately above), so this is
        # the only golden pin in the suite that can catch a regression in the
        # NONZERO arithmetic itself, e.g. CapacityMechanism.fire_rate() using
        # the wrong denominator (hours_observed - window + 1 instead of
        # hours_observed): a plausible "exclude the fill window" mistake that
        # shifts every reported fire rate by a couple of percent and that no
        # other assertion in this suite (all of which only check > 0.0 or
        # == 0.0) would catch.
        "capacity_fire_rate": result.capacity_fire_rate,
        "total_capacity_charge_eur": result.total_capacity_charge_eur,
        "total_deferred_kwh": result.total_deferred_kwh,
    }


def _assert_metrics_match_golden(actual: dict, expected: dict, context: str) -> None:
    # Key set first, in both directions, BEFORE any value is compared. The
    # loop below iterates the GOLDEN file's keys, so on its own a key that had
    # vanished from the golden file was never looked for: the pin quietly
    # weakened to whatever the file still happened to carry, and the suite
    # still passed. That is not a hypothetical either. Running the documented
    # regeneration command really did strip capacity_fire_rate,
    # total_capacity_charge_eur and total_deferred_kwh out of
    # tests/golden/short_deterministic_run.json (the regeneration script's
    # _metrics_dict twin had not been given them), and nothing failed. The
    # assertion below is what makes a vanished pin loud; the twin-agreement
    # test at the bottom of this module is what stops the twins drifting in
    # the first place.
    assert set(actual.keys()) == set(expected.keys()), (
        f"{context}: pinned key set does not match the key set the code produces "
        f"(missing from the golden file: {sorted(set(actual) - set(expected))}; "
        f"in the golden file but not produced: {sorted(set(expected) - set(actual))}). "
        f"A key missing from the golden file is an UNPINNED metric, not a passing test; "
        f"regenerate deliberately (see the module docstring) if the change is intended."
    )
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

EFFECT_SIZES_IDENTITY = ("block", "k", "broker_count", "ablation", "metric")
# Every non-identity column results/effect_sizes.csv carries (Chapter 1's own
# headline figures and Section 6.1's clean-channel-isolation pnl_sanity_check
# block both live in this file). The five identity columns plus these fifteen
# are the whole header, so no column of this file is left unpinned.
# (block, k, broker_count, ablation, metric) is unique over the whole file
# because the analysis emits exactly one row per combination of those five
# (three blocks, each swept over its own combination of k, broker_count and
# ablation, with the "ALL" grand-total rows using the literal string "ALL"
# for k and broker_count rather than colliding with a real level);
# _rows_by_identity asserts that rather than assuming it.
EFFECT_SIZES_PINNED_FIELDS = {
    "metric_label": "str",
    "n_seeds": "int",
    "mean_disabled": "float",
    "mean_ablation": "float",
    "paired_mean_diff": "float",
    "paired_mean_pct_change": "float",
    "naive_pct_change_of_means": "float",
    "paired_dz": "float",
    "paired_p": "pvalue",
    "cohens_d_independent": "float",
    "degenerate_paired_diff": "bool",
    "max_abs_paired_diff": "float",
    "pnl_physical_leak": "str",
    "metric3_sign_convention": "str",
    "note": "str",
}

SUMMARY_STATS_CORRECTED_IDENTITY = ("scope", "k", "broker_count", "ablation", "metric")
# Every non-identity column results/summary_stats_corrected.csv carries: the
# per-comparison detail Section 6.10 promises ("one row per comparison") for
# the multiple-comparison correction and bootstrap-CI story. The five
# identity columns plus these twenty-eight are the whole header.
# (scope, k, broker_count, ablation, metric) is unique over the whole file:
# scope alone separates the 180 main-sweep rows (120 real comparisons plus the
# 60 excluded capacity_pnl_only deterministic identities) from the 8
# monopoly-supplement rows, and within each scope the remaining four columns
# are exactly that scope's own grid, with no repeated cell.
SUMMARY_STATS_CORRECTED_PINNED_FIELDS = {
    "metric_label": "str",
    "n_seeds": "int",
    "paired_mean_diff": "float",
    "paired_mean_pct_change": "float",
    "paired_dz": "float",
    "p_uncorrected": "pvalue",
    "p_holm": "pvalue",
    "p_bh": "pvalue",
    "survives_holm_alpha05": "bool",
    "survives_bh_alpha05": "bool",
    "ci95_t_lo": "float",
    "ci95_t_hi": "float",
    "ci95_boot_lo": "float",
    "ci95_boot_hi": "float",
    "ci_disagreement_flag": "bool",
    "ci_disagreement_detail": "str",
    "ci_width_ratio": "float",
    "degenerate": "bool",
    "correction_family": "str",
    "correction_family_size": "int",
    "family_size_primary": "int",
    "family_size_alternative": "int",
    "family_size_monopoly_secondary": "int",
    "bootstrap_seed": "int",
    "n_bootstrap": "int",
    "alpha": "float",
    "metric3_sign_convention": "str",
    "notes": "str",
}

# results/dose_matched_monopoly.csv (D11's dose-matched-monopoly arm,
# Phase 18.1) is 46 rows: pinned WHOLE, the same way effect_sizes.csv and
# summary_stats_corrected.csv are, since a claim-bearing subset would save
# nothing at this size. (scope, arm, k, metric) is unique over the whole
# file: each of the seven arms (monopoly at divisor 1/2/3/5, broker_count
# 2/3/5) reports at most one row per (scope, k, metric) combination it
# carries, the decisive_contrast scope's one arm
# (mono_divisor_3_minus_bc_3) does the same, and no two arms share a name;
# _rows_by_identity asserts this rather than assuming it.
DOSE_MATCHED_MONOPOLY_IDENTITY = ("scope", "arm", "k", "metric")
# Every non-identity column results/dose_matched_monopoly.csv carries. The
# four identity columns plus these twenty-eight are the whole header, so no
# column of that file is left unpinned. Several of these are legitimately
# empty on some rows (capacity_surcharge_divisor and comparator_arm are only
# populated for the monopoly and decisive_contrast arms respectively;
# paired_mean_pct_change/paired_mean_diff_raw are empty on the
# decisive_contrast rows, which report pct_point_diff_mono_minus_comparator
# instead), which _assert_pinned_field's None-handling already covers.
DOSE_MATCHED_MONOPOLY_PINNED_FIELDS = {
    "arm_type": "str",
    "capacity_surcharge_divisor": "float",
    "broker_count": "int",
    "comparator_arm": "str",
    "data_source": "str",
    "n_seeds": "int",
    "paired_mean_pct_change": "float",
    "pct_point_diff_mono_minus_comparator": "float",
    "paired_mean_diff_raw": "float",
    "paired_dz": "float",
    "p_uncorrected": "pvalue",
    "p_holm": "pvalue",
    "p_bh": "pvalue",
    "survives_holm_alpha05": "bool",
    "survives_bh_alpha05": "bool",
    "ci95_t_lo": "float",
    "ci95_t_hi": "float",
    "ci95_boot_lo": "float",
    "ci95_boot_hi": "float",
    "n_favorable_of_n_seeds": "int",
    "total_deferred_kwh_capacity_both_mean": "float",
    "total_deferred_kwh_capacity_disabled_mean": "float",
    "comparator_total_deferred_kwh_capacity_both_mean": "float",
    "deferred_volume_ratio_mono_over_comparator": "float",
    "correction_family": "str",
    "correction_family_size": "int",
    "metric3_sign_convention": "str",
    "notes": "str",
}

# results/surcharge_mode_comparison.csv (D10's surcharge-distribution control
# arm, scripts/analyze_surcharge_mode.py): item 18, see the module docstring's
# own paragraph above for the mutation-testing gap this closes. Pinned WHOLE:
# 24 rows is the same small scale as dose_matched_monopoly.csv's 46. Must
# match the same-named constants in scripts/regenerate_golden_master.py
# exactly, which
# test_the_twinned_pin_shape_constants_agree_between_this_module_and_the_script
# checks rather than leaves to this comment.
#
# broker_count is real on the 18 scope="mode_cell" rows and empty on the 6
# scope="gradient_bc2_to_bc5" rows (which span broker_count 2 to 5 by
# construction, not any single value); metric is the mirror image, empty on
# mode_cell rows (which carry both metric3 sub-measures side by side as
# separate columns) and real on gradient_bc2_to_bc5 rows (always
# "feeder_peak_to_average_ratio", the one metric D10's gradient test is
# about). (scope, capacity_surcharge_mode, k, broker_count, metric) is unique
# over the whole file for the same reason structural_sensitivity.csv's own
# per-scope-varying identity is: each scope's own rows are unique on the
# columns that scope actually populates, and scope itself keeps the two
# scopes' key spaces from colliding; _rows_by_identity asserts this rather
# than assuming it, exactly as it does for every other pinned CSV here.
SURCHARGE_MODE_COMPARISON_IDENTITY = ("scope", "capacity_surcharge_mode", "k", "broker_count", "metric")
# Every non-identity column results/surcharge_mode_comparison.csv carries. The
# five identity columns plus these twenty-eight are the whole 33-column
# header, so no column of that file is left unpinned. A field that does not
# apply to a row's own scope (the twelve feeder_*_paired_*/p_holm/p_bh/
# survives_holm_alpha05 columns and the two total_deferred_kwh_* columns on a
# gradient_bc2_to_bc5 row; the ten gradient_* columns on a mode_cell row) is
# empty on that row, which _assert_pinned_field's None-handling already
# covers, the same way DOSE_MATCHED_MONOPOLY_PINNED_FIELDS's comment describes
# for that file's own per-arm-type empty columns.
SURCHARGE_MODE_COMPARISON_PINNED_FIELDS = {
    "n_seeds": "int",
    "feeder_peak_to_average_ratio_paired_mean_pct_change": "float",
    "feeder_peak_to_average_ratio_paired_dz": "float",
    "feeder_peak_to_average_ratio_paired_p": "pvalue",
    "feeder_peak_to_average_ratio_p_holm": "pvalue",
    "feeder_peak_to_average_ratio_p_bh": "pvalue",
    "feeder_peak_to_average_ratio_survives_holm_alpha05": "bool",
    "feeder_coefficient_of_variation_paired_mean_pct_change": "float",
    "feeder_coefficient_of_variation_paired_dz": "float",
    "feeder_coefficient_of_variation_paired_p": "pvalue",
    "feeder_coefficient_of_variation_p_holm": "pvalue",
    "feeder_coefficient_of_variation_p_bh": "pvalue",
    "feeder_coefficient_of_variation_survives_holm_alpha05": "bool",
    "total_deferred_kwh_capacity_both_mean": "float",
    "total_deferred_kwh_capacity_disabled_mean": "float",
    "correction_family": "str",
    "metric3_sign_convention": "str",
    "notes": "str",
    "gradient_pct_at_bc2": "float",
    "gradient_pct_at_bc5": "float",
    "gradient_pct_points_bc2_to_bc5": "float",
    "gradient_dz": "float",
    "gradient_p_uncorrected": "pvalue",
    "gradient_p_holm": "pvalue",
    "gradient_p_bh": "pvalue",
    "gradient_survives_holm_alpha05": "bool",
    "gradient_survives_bh_alpha05": "bool",
    "gradient_degenerate": "bool",
}

# Per-(k, capacity_surcharge_divisor, ablation) cell aggregates over
# results/sweep_dose_matched_monopoly.parquet: the row count (30 in every one
# of the 12 cells) and the mean of the six metrics docs/DECISIONS.md's "D11
# result" section quotes directly from this parquet (the peak-to-average
# table and the guardrail paragraph), none of which any pinned CSV covers on
# its own. Must match the same-named constants in
# scripts/regenerate_golden_master.py exactly.
DOSE_MATCHED_MONOPOLY_PARQUET_CELL_IDENTITY = ("k", "capacity_surcharge_divisor", "ablation")
DOSE_MATCHED_MONOPOLY_PARQUET_CELL_FIELDS = {
    "n_rows": "int",
    "mean_total_deferred_kwh": "float",
    "mean_feeder_peak_to_average_ratio": "float",
    "mean_feeder_coefficient_of_variation": "float",
    "mean_avg_cost_per_agent_eur": "float",
    "mean_capacity_fire_rate": "float",
    "mean_total_capacity_charge_eur": "float",
}


def _dose_matched_monopoly_parquet_cell_aggregates(df):
    """One row per (k, capacity_surcharge_divisor, ablation) cell of
    results/sweep_dose_matched_monopoly.parquet (12 cells, 30 seeds each).

    Shaped exactly like any other pinned CSV in this module (identity columns
    plus pinned fields), on purpose: the parquet's own rows are one per
    (cell, seed), not one per cell, so this aggregates first and then hands
    the result to the SAME _assert_csv_matches_row_keyed_golden /
    _row_keyed_rows machinery every whole-file CSV digest already uses,
    rather than inventing a second comparison helper for a second snapshot
    shape. Twin of the same-named function in
    scripts/regenerate_golden_master.py; must compute identically or the
    golden snapshot stops describing what this checks.
    """
    return df.groupby(list(DOSE_MATCHED_MONOPOLY_PARQUET_CELL_IDENTITY), as_index=False).agg(
        n_rows=("seed", "size"),
        mean_total_deferred_kwh=("total_deferred_kwh", "mean"),
        mean_feeder_peak_to_average_ratio=("feeder_peak_to_average_ratio", "mean"),
        mean_feeder_coefficient_of_variation=("feeder_coefficient_of_variation", "mean"),
        mean_avg_cost_per_agent_eur=("avg_cost_per_agent_eur", "mean"),
        mean_capacity_fire_rate=("capacity_fire_rate", "mean"),
        mean_total_capacity_charge_eur=("total_capacity_charge_eur", "mean"),
    )


# results/family_sensitivity.csv is 1227 rows, about ten times
# structural_sensitivity.csv's 124 (the largest whole-file digest pinned so
# far, at 177891 bytes on disk). A whole-file digest built the same way here
# would run to roughly 1.6 MB, nearly as large as the rest of tests/golden/
# combined, for a return that a claim-bearing subset gets far more cheaply:
# almost every one of the 1227 rows is intermediate arithmetic feeding a
# handful of quoted aggregates, not a number this package ever quotes on its
# own. So this file is pinned as a CLAIM-BEARING SUBSET rather than whole,
# plus the file's total row count as a shape guard, so a row vanishing or
# appearing ANYWHERE in the file, including the 1219 rows this subset does
# not otherwise look at, still fails loudly (see
# _assert_csv_matches_claim_bearing_subset_golden below).
#
# The seven family_summary rows carry every number Section 6.10, Slide 13a
# and 14, DECISIONS and qa_prep's multiple-comparison-family question quote:
# family_size, the "72 of 72 survive" tally under both corrections at every
# one of the seven family definitions, and the worst corrected p-value per
# definition (4.9e-5 at N=120 up to 2.34e-4 at N=306, deliberately
# non-monotone in family size, for the reason Section 6.10 states). The
# eighth row, the one real per-test comparison for k=1.0/broker_count=3
# /capacity_both on the coefficient of variation under the primary family
# (F1), is pinned for two further reasons: it backs Section 6.2's own quoted
# "d_z = -22.2 (p = 7.4e-41)" figure directly, and its p_holm is the specific
# number Section 6.10 asserts "reproduces the shipped p_holm column of
# results/summary_stats_corrected.csv exactly". It is also the only row in
# this subset whose p-values sit far enough below CSV_PIN_ABS_TOL to prove
# the relative-only p-value rule earns its keep on this file too: the seven
# family_summary rows' own worst-p figures are all no smaller than about
# 5e-6, comfortably above that floor on their own.
FAMILY_SENSITIVITY_IDENTITY = ("family_definition", "test_id")
FAMILY_SENSITIVITY_PINNED_FIELDS = {
    "family_size": "int",
    "n_families_in_definition": "int",
    "ablation": "str",
    "metric": "str",
    "is_headline_72": "bool",
    "degenerate": "bool",
    "p_uncorrected": "pvalue",
    "p_holm": "pvalue",
    "p_bh": "pvalue",
    "survives_holm_alpha05": "bool",
    "survives_bh_alpha05": "bool",
    "alpha": "float",
    "n_headline_tests": "int",
    "n_headline_surviving_holm": "int",
    "n_headline_surviving_bh": "int",
    "worst_headline_p_uncorrected": "pvalue",
    "worst_headline_p_holm": "pvalue",
    "worst_headline_p_bh": "pvalue",
    "bonferroni_alpha_over_family_size": "float",
    "notes": "str",
}
# The explicit, named claim-bearing subset described above: the seven
# family_summary rows (one per family definition) plus the one real per-test
# row. Must match the same-named constant in
# scripts/regenerate_golden_master.py exactly, or the golden file stops
# describing the selection that was deliberately made.
FAMILY_SENSITIVITY_PINNED_ROW_KEYS = (
    ("F1_main_sweep_primary_120", "family_summary|F1_main_sweep_primary_120"),
    ("F2_main_sweep_with_degenerates_180", "family_summary|F2_main_sweep_with_degenerates_180"),
    ("F3_per_metric_24", "family_summary|F3_per_metric_24"),
    ("F4_pooled_with_monopoly", "family_summary|F4_pooled_with_monopoly"),
    ("F5_everything_pooled", "family_summary|F5_everything_pooled"),
    ("F6_per_cell_k_by_broker_count", "family_summary|F6_per_cell_k_by_broker_count"),
    ("F7_everything_pooled_with_degenerates", "family_summary|F7_everything_pooled_with_degenerates"),
    ("F1_main_sweep_primary_120", "main|k=1|bc=3|capacity_both|feeder_coefficient_of_variation"),
)

# The eight pinned rows above cover exactly eight of family_sensitivity.csv's
# 1227 rows; the other 1219 were, until this section, guarded only by the
# total-row-count shape guard and the exact-column-set pin, neither of which
# would catch a single unpinned p-value shifting, a single unpinned
# survives_holm_alpha05 flag flipping, or a single unpinned row's metric,
# ablation or is_headline_72 label being corrupted. This closes that gap with
# WHOLE-FILE AGGREGATE GUARDS, in three parts, computed over every row:
#
#   1. counts and p-value extrema on the six VALUE columns (p_uncorrected,
#      p_holm, p_bh, survives_holm_alpha05, survives_bh_alpha05, degenerate),
#      pinned BOTH per family_definition AND over the whole file;
#   2. log-magnitude sums on the three p-value columns (see
#      P_VALUE_LOG_FLOOR's comment below), also pinned at both levels,
#      closing the blind spot part 1 leaves for a p-value that moves in the
#      MIDDLE of the distribution;
#   3. an exact multiset over EVERY identifier and label column the file has
#      (see LABEL_COMBO_COLUMNS's comment below for the full list and why
#      each one belongs there), pinned PER family_definition ONLY, not also
#      at the overall level (see FAMILY_SENSITIVITY_OVERALL_AGGREGATE_FIELDS's
#      comment below for why the overall figure would be a pure duplicate).
#      This closes a DIFFERENT blind spot from parts 1 and 2: they only ever
#      read the six value columns, so a bug that mislabels which row a
#      p-value belongs to (a wrong metric, a swapped ablation, a flipped
#      is_headline_72, a mislabelled source, and so on) moves no count, no
#      extremum and no log-sum at all, and would otherwise pass silently on
#      any of the 1219 rows outside the eight individually pinned above. An
#      adversarial review reproduced exactly this, three separate ways,
#      against an earlier version of this section that had only parts 1 and
#      2, and a follow-up review then found that the first fix for it was
#      itself a hand-picked subset of label columns rather than all of them;
#      it is not a hypothetical, and this file's own docs/DECISIONS.md keeps
#      a correction section for exactly this failure mode.
#
# Deliberately NOT a bit-exact whole-file digest or a checksum: Phase 11's
# structural-sensitivity work already found that a bit-exact equality check
# on a large grid false-alarms on cross-process 1-ULP floating-point drift
# (reproduced three ways at the time, see docs/PROGRESS.md), which is exactly
# why every pin in this module compares by tolerance instead of by hash.
#
# What is and is not covered, stated precisely rather than left to be
# inferred: row count, column set, the six value columns' counts and
# extremes, the log-magnitude sums, and the identity-label multisets are
# pinned across all 1227 rows. What remains unpinned is any change to an
# unpinned row's individual p-value that preserves its family's log-sum,
# which requires two compensating shifts in exact reciprocal proportion (for
# example row A's p_holm multiplied by some factor while row B's is divided
# by that same factor, holding the sum of logs fixed): a coincidence, not a
# realistic regression, but a real and disclosed limit of a sum rather than a
# per-row pin, and stated here rather than left for the aggregate to imply
# more than it delivers.
#
# Counts and min/max alone still have a blind spot: a p-value that moves in
# the MIDDLE of the distribution (neither the smallest nor the largest, and
# not crossing the alpha=0.05 survives_holm_alpha05/survives_bh_alpha05
# boundary) flips no flag, sets no new min, sets no new max, and would pass
# every guard above. That is precisely the regression class this file exists
# to catch, since its whole job is recomputing Holm and BH under seven family
# definitions. sum_log10_p_uncorrected / sum_log10_p_holm / sum_log10_p_bh
# close that: the sum of log10(p) over a group's rows is sensitive to a
# relative change in ANY one p, at ANY magnitude, which is exactly the
# property min and max lack (one p moving from 1e-30 to 1e-25 shifts the
# log-sum by 5, enormous against a relative tolerance, while leaving every
# count and every extremum in this dict untouched). A plain sum of the p
# values themselves would not do this: it is dominated by the handful of
# large (near-1) p's in any group and is numerically blind to a tiny p moving
# by many orders of magnitude, which is the exact failure mode being closed.
#
# log10(0.0) is undefined and a p-value of exactly 0.0 does occur in this
# codebase's p-value machinery (see
# test_pvalue_comparison_pins_magnitude_and_handles_zero_and_denormals's
# "exact zero" case), so every p is floored at P_VALUE_LOG_FLOOR before its
# log is taken. 1e-300 was chosen deliberately: it sits comfortably above
# float64's smallest normal number (about 2.2e-308) and smallest denormal
# (about 4.9e-324), so the floor itself is never close to a numerical edge,
# and it sits about 244 orders of magnitude below the smallest real p-value
# in results/family_sensitivity.csv today (about 8.9e-56), so no row is
# actually floored by it now. The cost is explicit and pinned, not hidden:
# any real p at or under the floor is compressed to log10(P_VALUE_LOG_FLOOR)
# exactly, so the log-sum can no longer distinguish how far under the floor
# such a p went; n_at_floor_p_uncorrected / n_at_floor_p_holm / n_at_floor_p_bh
# pin how many rows hit it (0 everywhere today), so a regression that pushes
# more rows to zero or below the floor is itself visible as a changed count,
# rather than silently absorbed into the sum.
P_VALUE_LOG_FLOOR = 1e-300


def _log10_with_floor(value: float) -> float:
    return math.log10(value) if value > P_VALUE_LOG_FLOOR else math.log10(P_VALUE_LOG_FLOOR)


FAMILY_SENSITIVITY_AGGREGATE_FIELDS = {
    "n_rows": "int",
    "n_survives_holm_true": "int",
    "n_survives_holm_false": "int",
    "n_survives_bh_true": "int",
    "n_survives_bh_false": "int",
    "n_degenerate_true": "int",
    "p_uncorrected_min": "pvalue",
    "p_uncorrected_max": "pvalue",
    "p_holm_min": "pvalue",
    "p_holm_max": "pvalue",
    "p_bh_min": "pvalue",
    "p_bh_max": "pvalue",
    "n_rows_contributing": "int",
    "sum_log10_p_uncorrected": "logsum",
    "sum_log10_p_holm": "logsum",
    "sum_log10_p_bh": "logsum",
    "n_at_floor_p_uncorrected": "int",
    "n_at_floor_p_holm": "int",
    "n_at_floor_p_bh": "int",
}
# The whole-file aggregate carries two further fields the per-family-
# definition breakdown does not: how many distinct family_definition and
# test_id values the WHOLE file holds, so a family_definition silently
# renamed, or a test_id silently duplicated anywhere in the file (not just
# within one family definition's own rows), still fails here.
FAMILY_SENSITIVITY_OVERALL_AGGREGATE_FIELDS = dict(
    FAMILY_SENSITIVITY_AGGREGATE_FIELDS,
    n_distinct_family_definitions="int",
    n_distinct_test_ids="int",
)
# The label multiset itself (see LABEL_COMBO_COLUMNS below) is pinned PER
# family_definition ONLY, not at the overall level too, unlike every other
# field above. Reason, not a size shortcut: with test_id in the combination,
# every row's combination is already unique within its own family_definition
# (each family_definition's own test_id values never repeat), so the overall
# multiset would hold the exact same 1227 entries the seven per-family dicts
# already hold between them, just merged into one dict, which is the OVERALL
# figure summing the per-family ones (1227 total combinations either way).
# Storing that sum a second time at the overall level is a pure duplicate: it
# carries strictly less information than the per-family breakdown, since it
# cannot say WHICH family_definition a mislabel landed in, only that
# something moved, and the per-family breakdown is what makes this pin
# diagnostic rather than merely a tripwire. Every one of the label-multiset
# bite tests below still fails via the per-family_definition check alone;
# none of them relied on an overall-level label check.

# Columns whose per-row VALUE this pin protects by counting how often each
# combination occurs, per family_definition and over the whole file, closing
# the gap the value-column aggregates above cannot: they never read these
# columns at all, so a mislabel on any of them moves nothing above.
#
# The rule is simple rather than a hand-picked subset, because a hand-picked
# subset is exactly the wrong shape here: EVERY column of
# results/family_sensitivity.csv that is an identifier or a label, rather
# than a computed statistical output, is included. That is family_key, scope,
# source, k, broker_count, ablation, cell_is_stamped_not_native, metric,
# test_id and is_headline_72; every other column (family_size,
# n_families_in_definition, degenerate, p_uncorrected, p_holm, p_bh,
# survives_holm_alpha05, survives_bh_alpha05, alpha, n_headline_tests,
# n_headline_surviving_holm, n_headline_surviving_bh,
# worst_headline_p_uncorrected, worst_headline_p_holm, worst_headline_p_bh,
# bonferroni_alpha_over_family_size, notes) is either a computed statistic
# already covered above (degenerate and the three p-value columns, via the
# counts/extremes/log-sums), a family_summary-only aggregate already covered
# by the claim-bearing subset's own pinned fields (n_headline_tests and its
# neighbours), a constant setting (alpha), free text (notes), or itself the
# grouping key (family_definition; a row's family_definition changing to
# another EXISTING family_definition already shows up as a changed n_rows on
# both the losing and the gaining group, so it needs no separate label
# check). None of the ten included columns turned out, on inspection, to be
# a computed output masquerading as a label; all ten are genuine identifiers
# or labels (verified: none is constant, all vary per row where the
# underlying design varies, k and broker_count and ablation and metric and
# is_headline_72 are the sweep's own design coordinates, family_key/scope/
# source/cell_is_stamped_not_native are provenance labels, and test_id is the
# row's own name).
#
# Including test_id pushes this multiset close to one entry per row (313
# distinct values across 1227 rows): that is deliberate, not an accident to
# be trimmed back. With test_id included, the multiset becomes a complete
# row-identity pin on every label column, with the six value columns already
# pinned in aggregate (counts and extremes) and by log-sum, so the combined
# rule an examiner can check in one sentence is: every label is pinned
# exactly, every value column is pinned in aggregate, and the only remaining
# gap is a value change that exactly preserves its family's log-sum (see the
# module docstring's own statement of that residual limit).
#
# This multiset is computed and pinned PER family_definition ONLY, not also
# at the overall (whole-file) level; see the comment above
# FAMILY_SENSITIVITY_OVERALL_AGGREGATE_FIELDS for why (the overall figure
# would be the identical 1227 entries a second time, derivable by summing the
# seven per-family dicts, and strictly less diagnostic than they are since it
# cannot say which family a mislabel landed in).
LABEL_COMBO_COLUMNS = (
    "family_key",
    "scope",
    "source",
    "k",
    "broker_count",
    "ablation",
    "cell_is_stamped_not_native",
    "metric",
    "test_id",
    "is_headline_72",
)


def _label_combo_key(row) -> str:
    """One canonical, collision-free string key for a row's combination of
    LABEL_COMBO_COLUMNS values. Reuses _identity_token so a missing value
    (metric/ablation/is_headline_72 on a family_summary row) collapses to the
    same _MISSING_TOKEN used everywhere else in this module, rather than a
    third, ad hoc representation of "empty".
    """
    return "|".join(f"{column}={_identity_token(row[column])}" for column in LABEL_COMBO_COLUMNS)


def _label_combo_counts(group) -> dict:
    """value_counts over LABEL_COMBO_COLUMNS, as a plain dict: key -> count.

    Twin of the same-named function in scripts/regenerate_golden_master.py;
    must compute identically or the golden snapshot stops describing what
    this checks. An exact dict of ints, not a tolerance-based comparison:
    there is no floating-point noise in a row count, so nothing is gained by
    loosening it, and pytest.approx would not even apply to a dict.
    """
    counts: dict = {}
    for _, row in group.iterrows():
        key = _label_combo_key(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _assert_label_counts_match_golden(actual_counts: dict, expected_counts: dict, context: str) -> None:
    if actual_counts == expected_counts:
        return
    added = sorted(set(actual_counts) - set(expected_counts))
    removed = sorted(set(expected_counts) - set(actual_counts))
    changed = {
        key: {"expected": expected_counts[key], "actual": actual_counts[key]}
        for key in sorted(set(actual_counts) & set(expected_counts))
        if actual_counts[key] != expected_counts[key]
    }
    raise AssertionError(
        f"{context}: label combination counts changed (added={added}, removed={removed}, changed={changed})"
    )


# The row-keyed pinned CSVs that are pinned WHOLE, each with the number of
# p-value columns its header actually carries. Pinned as a count rather than a
# list so that a p-value column added to one of these files later cannot
# quietly land back on the "float" kind, and so the claim that
# demand_source_comparison.csv has no p-value column at all is checked against
# the real header rather than assumed. results/family_sensitivity.csv is
# deliberately NOT in this list: it is pinned as a claim-bearing SUBSET (see
# FAMILY_SENSITIVITY_IDENTITY's comment above), so its real per-test p-value
# columns (p_uncorrected, p_holm, p_bh) are not all individually value-pinned,
# and folding it into this shared, whole-header check would require pretending
# otherwise.
PINNED_CSV_SPECS = (
    ("results/summary_stats.csv", SUMMARY_STATS_CSV_PATH, SUMMARY_STATS_PINNED_FIELDS, 1),
    ("results/structural_sensitivity.csv", STRUCTURAL_SENSITIVITY_CSV_PATH, STRUCTURAL_SENSITIVITY_PINNED_FIELDS, 3),
    ("results/demand_source_comparison.csv", DEMAND_SOURCE_CSV_PATH, DEMAND_SOURCE_PINNED_FIELDS, 0),
    ("results/monopoly_comparison.csv", MONOPOLY_CSV_PATH, MONOPOLY_PINNED_FIELDS, 2),
    ("results/effect_sizes.csv", EFFECT_SIZES_CSV_PATH, EFFECT_SIZES_PINNED_FIELDS, 1),
    ("results/summary_stats_corrected.csv", SUMMARY_STATS_CORRECTED_CSV_PATH, SUMMARY_STATS_CORRECTED_PINNED_FIELDS, 3),
    ("results/dose_matched_monopoly.csv", DOSE_MATCHED_MONOPOLY_CSV_PATH, DOSE_MATCHED_MONOPOLY_PINNED_FIELDS, 3),
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
    elif kind in ("pvalue", "logsum"):
        # "logsum" (a sum of log10(p) over a group of rows, possibly large and
        # negative) wants the exact same rule as "pvalue": relative tolerance
        # only, abs=0, with the same zero-special-case. _assert_pvalue_close's
        # own zero handling is unreachable here in practice (a log-sum is a
        # sum of many negative terms, not a single p-value, so it is never
        # exactly 0.0 for real data), but reusing the one helper rather than
        # writing a second is deliberate: the tolerance RULE is identical, only
        # the quantity it is applied to differs.
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


def _assert_effect_sizes_matches_golden(df, golden: dict) -> None:
    _assert_csv_matches_row_keyed_golden(
        df, golden, EFFECT_SIZES_IDENTITY, EFFECT_SIZES_PINNED_FIELDS,
        "results/effect_sizes.csv", EFFECT_SIZES_GOLDEN_PATH.name,
    )


def _assert_summary_stats_corrected_matches_golden(df, golden: dict) -> None:
    _assert_csv_matches_row_keyed_golden(
        df, golden, SUMMARY_STATS_CORRECTED_IDENTITY, SUMMARY_STATS_CORRECTED_PINNED_FIELDS,
        "results/summary_stats_corrected.csv", SUMMARY_STATS_CORRECTED_GOLDEN_PATH.name,
    )


def _assert_dose_matched_monopoly_matches_golden(df, golden: dict) -> None:
    _assert_csv_matches_row_keyed_golden(
        df, golden, DOSE_MATCHED_MONOPOLY_IDENTITY, DOSE_MATCHED_MONOPOLY_PINNED_FIELDS,
        "results/dose_matched_monopoly.csv", DOSE_MATCHED_MONOPOLY_GOLDEN_PATH.name,
    )


def _assert_surcharge_mode_comparison_matches_golden(df, golden: dict) -> None:
    _assert_csv_matches_row_keyed_golden(
        df, golden, SURCHARGE_MODE_COMPARISON_IDENTITY, SURCHARGE_MODE_COMPARISON_PINNED_FIELDS,
        "results/surcharge_mode_comparison.csv", SURCHARGE_MODE_COMPARISON_GOLDEN_PATH.name,
    )


def _assert_dose_matched_monopoly_parquet_cells_match_golden(df, golden: dict) -> None:
    """df is the RAW sweep_dose_matched_monopoly.parquet (one row per (cell,
    seed)); this aggregates it to one row per cell before reusing the same
    row-keyed comparison every whole-file CSV digest uses. See
    _dose_matched_monopoly_parquet_cell_aggregates's own docstring for why
    aggregating first, rather than inventing a second comparison helper, is
    the right shape here.
    """
    grouped = _dose_matched_monopoly_parquet_cell_aggregates(df)
    _assert_csv_matches_row_keyed_golden(
        grouped, golden, DOSE_MATCHED_MONOPOLY_PARQUET_CELL_IDENTITY, DOSE_MATCHED_MONOPOLY_PARQUET_CELL_FIELDS,
        "results/sweep_dose_matched_monopoly.parquet (per-cell aggregates)",
        SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_CELL_GOLDEN_PATH.name,
    )


def _rows_by_identity_subset(df, identity_columns, wanted_keys) -> dict:
    """Twin of _rows_by_identity, restricted to a named subset of identity keys.

    Used for a CSV whose identity key legitimately repeats across rows this
    pin does not cover (family_sensitivity.csv's test_id repeats once per
    family_definition, since the same underlying test is pooled into several
    family definitions), so asserting no duplicates over the WHOLE file, the
    way _rows_by_identity does, would fail for a reason that has nothing to do
    with the rows actually being pinned. Duplicate detection still applies,
    just scoped to the keys this call was asked to look for.
    """
    rows = {}
    for _, row in df.iterrows():
        key = _identity_key(row, identity_columns)
        if key not in wanted_keys:
            continue
        assert key not in rows, f"duplicate identity key among the pinned subset: {key}"
        rows[key] = row
    return rows


def _assert_csv_matches_claim_bearing_subset_golden(
    df, golden: dict, identity_columns, pinned_fields, label: str, golden_label: str
) -> None:
    """Compare a CSV against a golden snapshot of a CLAIM-BEARING SUBSET of its rows.

    For a CSV too large to pin whole without absurdity (see
    FAMILY_SENSITIVITY_IDENTITY's comment above for the concrete reasoning),
    this pins the file's total row count as a shape guard, so a row vanishing
    or appearing ANYWHERE in the file still fails here even though most of the
    file is not individually value-checked, plus the named subset of rows that
    actually carry a claim, looked up by identity exactly as
    _assert_csv_matches_row_keyed_golden does for a whole-file digest. Written
    generically over identity columns and pinned fields (both parameters, not
    hardcoded), so a future large CSV needing the same treatment (see the
    Phase 18.1 follow-up note in PROGRESS.md) can call this directly rather
    than copying it.
    """
    assert len(df) == golden["n_total_rows"], (
        f"{label}: total row count changed ({len(df)} rows, pinned {golden['n_total_rows']})"
    )
    assert len(golden["rows"]) == golden["n_rows"], f"{golden_label}: golden file is internally inconsistent"

    golden_keys = set()
    for expected_row in golden["rows"]:
        key = _identity_key(expected_row, identity_columns)
        assert key not in golden_keys, f"{golden_label}: duplicate identity key in the golden file: {key}"
        golden_keys.add(key)
    assert len(golden_keys) == golden["n_rows"], (
        f"{golden_label}: golden file holds {len(golden_keys)} distinct identity keys, pinned {golden['n_rows']} rows"
    )

    actual_rows = _rows_by_identity_subset(df, identity_columns, golden_keys)
    for expected_row in golden["rows"]:
        key = _identity_key(expected_row, identity_columns)
        assert key in actual_rows, f"{label}: missing pinned row for {key}"
        actual_row = actual_rows[key]
        context = f"{label} {key}"
        for field, kind in pinned_fields.items():
            _assert_pinned_field(actual_row, expected_row, field, kind, context)


def _family_sensitivity_aggregate_row(group) -> dict:
    """Whole-group counts, p-value extrema, and log-magnitude sums, over
    EVERY row in the group, not just the eight individually pinned above.
    Twin of the same-named function in scripts/regenerate_golden_master.py;
    must compute identically or the golden snapshot stops describing what
    this checks.

    survives_holm_alpha05, survives_bh_alpha05 and degenerate are all empty on
    the family_summary rows (see the null counts in the module's own
    exploration), so .eq(True) / .eq(False) on those rows both correctly
    return False rather than raising, and the two counts do not have to sum to
    the group's row count; the difference is exactly the group's summary rows.
    p_uncorrected/p_holm/p_bh are likewise empty only on those same rows
    (verified identical across all three columns), and pandas' min/max skip
    missing values by construction, so no special casing is needed for that
    either; n_rows_contributing (one shared count, not one per column, since
    the three columns are empty on exactly the same rows) makes that
    explicit rather than leaving a reader to infer it from n_rows.
    """
    p_columns = ("p_uncorrected", "p_holm", "p_bh")
    row = {
        "n_rows": int(len(group)),
        "n_survives_holm_true": int(group["survives_holm_alpha05"].eq(True).sum()),
        "n_survives_holm_false": int(group["survives_holm_alpha05"].eq(False).sum()),
        "n_survives_bh_true": int(group["survives_bh_alpha05"].eq(True).sum()),
        "n_survives_bh_false": int(group["survives_bh_alpha05"].eq(False).sum()),
        "n_degenerate_true": int(group["degenerate"].eq(True).sum()),
        "p_uncorrected_min": float(group["p_uncorrected"].min()),
        "p_uncorrected_max": float(group["p_uncorrected"].max()),
        "p_holm_min": float(group["p_holm"].min()),
        "p_holm_max": float(group["p_holm"].max()),
        "p_bh_min": float(group["p_bh"].min()),
        "p_bh_max": float(group["p_bh"].max()),
        "n_rows_contributing": int(group["p_uncorrected"].notna().sum()),
    }
    for column in p_columns:
        values = group[column].dropna()
        row[f"sum_log10_{column}"] = float(sum(_log10_with_floor(float(value)) for value in values))
        row[f"n_at_floor_{column}"] = int((values <= P_VALUE_LOG_FLOOR).sum())
    row["label_counts"] = _label_combo_counts(group)
    return row


def _assert_family_sensitivity_aggregates_match_golden(df, golden: dict) -> None:
    """Whole-file aggregate guard over all 1227 rows (see
    LABEL_COMBO_COLUMNS's comment above for exactly what is and is not
    covered): counts and p-value extrema on the six value columns and the
    log-magnitude sums, at BOTH the overall level and per family_definition;
    the identity-adjacent label multiset, PER family_definition ONLY (see
    FAMILY_SENSITIVITY_OVERALL_AGGREGATE_FIELDS's comment above for why the
    overall level would be a pure duplicate of the per-family breakdown). A
    regression anywhere outside the eight individually pinned rows still
    fails here even though it is not itself value-pinned.
    """
    label = "results/family_sensitivity.csv aggregates"
    aggregates = golden["aggregates"]

    overall_actual = _family_sensitivity_aggregate_row(df)
    overall_actual["n_distinct_family_definitions"] = int(df["family_definition"].nunique())
    overall_actual["n_distinct_test_ids"] = int(df["test_id"].nunique())
    for field, kind in FAMILY_SENSITIVITY_OVERALL_AGGREGATE_FIELDS.items():
        _assert_pinned_field(overall_actual, aggregates["overall"], field, kind, f"{label}: overall")
    # No overall-level label_counts check: deliberately not stored (see the
    # comment above FAMILY_SENSITIVITY_OVERALL_AGGREGATE_FIELDS). The
    # per-family_definition check below is where every label-multiset
    # regression is actually caught.

    expected_by_family = aggregates["by_family_definition"]
    assert set(df["family_definition"].unique()) == set(expected_by_family.keys()), (
        f"{label}: set of family_definition values changed"
    )
    for family_definition, expected_group in expected_by_family.items():
        group = df[df["family_definition"] == family_definition]
        actual_group = _family_sensitivity_aggregate_row(group)
        context = f"{label}: {family_definition}"
        for field, kind in FAMILY_SENSITIVITY_AGGREGATE_FIELDS.items():
            _assert_pinned_field(actual_group, expected_group, field, kind, context)
        _assert_label_counts_match_golden(actual_group["label_counts"], expected_group["label_counts"], context)


def _assert_family_sensitivity_matches_golden(df, golden: dict) -> None:
    # Guards the SELECTION itself, not just its internal consistency: a
    # regenerate-script bug that quietly pinned the wrong eight rows would
    # still produce a golden file that passes every check inside
    # _assert_csv_matches_claim_bearing_subset_golden, since that helper only
    # checks the golden file against itself and then against whatever rows it
    # names. Comparing against the named constant catches that class of bug.
    actual_keys = {_identity_key(row, FAMILY_SENSITIVITY_IDENTITY) for row in golden["rows"]}
    assert actual_keys == set(FAMILY_SENSITIVITY_PINNED_ROW_KEYS), (
        "results/family_sensitivity.csv: the golden file's pinned rows do not match "
        "FAMILY_SENSITIVITY_PINNED_ROW_KEYS"
    )
    _assert_csv_matches_claim_bearing_subset_golden(
        df, golden, FAMILY_SENSITIVITY_IDENTITY, FAMILY_SENSITIVITY_PINNED_FIELDS,
        "results/family_sensitivity.csv", FAMILY_SENSITIVITY_GOLDEN_PATH.name,
    )
    _assert_family_sensitivity_aggregates_match_golden(df, golden)


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
# 12. Pinned digest of results/effect_sizes.csv, every data row and every
#     column: the file Chapter 1's headline figures are drawn from, and the
#     per-cell effect sizes behind Sections 6.1 to 6.9 (the metric3_mechanism
#     and other_metrics blocks) plus the clean-channel-isolation sanity check
#     (the pnl_sanity_check block). Pinned WHOLE, the same way Phase 16
#     converted summary_stats.csv: 185 rows is small enough that a
#     claim-bearing subset would save little and would risk missing a row
#     nobody thought to name in advance. Read-only, and skips (does not fail)
#     if the analysis output is absent on this machine, exactly as items 5
#     to 8 above.
# ---------------------------------------------------------------------------


def test_effect_sizes_csv_matches_pinned_digest():
    _skip_unless_pinned_csv_available(EFFECT_SIZES_CSV_PATH, EFFECT_SIZES_GOLDEN_PATH)

    import pandas as pd

    golden = _load_golden(EFFECT_SIZES_GOLDEN_PATH)
    df = pd.read_csv(EFFECT_SIZES_CSV_PATH)
    _assert_effect_sizes_matches_golden(df, golden)


def test_effect_sizes_pin_bites_on_plausible_regressions():
    """Prove the pin above is not vacuous, on in-memory copies only.

    Four regressions specific to this file: the sign of a metric-3 percent
    change flipping (a reported improvement becomes a reported worsening at
    the same magnitude, which is exactly the kind of regression the sign
    convention pinned alongside it exists to catch), a mean drifting by 1
    percent (a plausible calibration or rounding regression, too small to
    notice by eye), the pnl_sanity_check block's degenerate_paired_diff flag
    flipping (qa_prep A3 points a reader directly at this column carrying
    this value as the evidence for clean channel isolation), and a row
    disappearing. The CSV's own digest is captured before and re-checked
    after, so the file on disk is provably untouched.
    """
    _skip_unless_pinned_csv_available(EFFECT_SIZES_CSV_PATH, EFFECT_SIZES_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(EFFECT_SIZES_CSV_PATH)
    golden = _load_golden(EFFECT_SIZES_GOLDEN_PATH)
    df = pd.read_csv(EFFECT_SIZES_CSV_PATH)

    _assert_effect_sizes_matches_golden(df, golden)  # unperturbed: passes

    # (a) a metric-3 percent change's sign flips: a reported improvement
    #     becomes a reported worsening at the same magnitude.
    flipped = df.copy(deep=True)
    metric3_rows = flipped.index[
        (flipped["block"] == "metric3_mechanism") & (flipped["paired_mean_pct_change"].abs() > 1e-6)
    ]
    assert len(metric3_rows) > 0, "expected at least one non-degenerate metric3_mechanism row"
    row = metric3_rows[0]
    flipped.loc[row, "paired_mean_pct_change"] = -flipped.loc[row, "paired_mean_pct_change"]
    with pytest.raises(AssertionError):
        _assert_effect_sizes_matches_golden(flipped, golden)

    # (b) a mean drifts by 1 percent, small enough to pass a casual eyeball check.
    drifted = df.copy(deep=True)
    nonzero_mean = drifted.index[drifted["mean_ablation"].abs() > 1e-6]
    assert len(nonzero_mean) > 0
    row = nonzero_mean[0]
    drifted.loc[row, "mean_ablation"] = drifted.loc[row, "mean_ablation"] * 1.01
    with pytest.raises(AssertionError):
        _assert_effect_sizes_matches_golden(drifted, golden)

    # (c) the pnl_sanity_check block's degenerate_paired_diff flips; qa_prep
    #     A3 points a reader directly at this column carrying this value.
    undegenerate = df.copy(deep=True)
    degenerate_rows = undegenerate.index[
        (undegenerate["block"] == "pnl_sanity_check") & (undegenerate["degenerate_paired_diff"].eq(True))
    ]
    assert len(degenerate_rows) > 0, "expected the pnl_sanity_check block to be degenerate"
    undegenerate.loc[degenerate_rows[0], "degenerate_paired_diff"] = False
    with pytest.raises(AssertionError):
        _assert_effect_sizes_matches_golden(undegenerate, golden)

    # (d) a row disappears.
    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_effect_sizes_matches_golden(dropped, golden)

    assert _sha256(EFFECT_SIZES_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_effect_sizes_pin_bites_on_a_p_value_magnitude_shift():
    """A p-value moving far in magnitude while staying far below the old
    absolute floor: under the previous rel=1e-9/abs=1e-9 comparison (either
    tolerance sufficing) this exact perturbation PASSED, because both values
    sit under the absolute floor. The assertion inside reproduces that old
    rule and shows it calling the two values equal, immediately before the
    current rule rejects them.
    """
    _skip_unless_pinned_csv_available(EFFECT_SIZES_CSV_PATH, EFFECT_SIZES_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(EFFECT_SIZES_CSV_PATH)
    golden = _load_golden(EFFECT_SIZES_GOLDEN_PATH)
    df = pd.read_csv(EFFECT_SIZES_CSV_PATH)

    _assert_effect_sizes_matches_golden(df, golden)  # unperturbed: passes

    moved = df.copy(deep=True)
    smallest = moved["paired_p"].idxmin()
    original = float(moved.loc[smallest, "paired_p"])
    assert original < 1e-30, "expected a real p-value far below the old absolute floor"
    perturbed_to = 1e-20
    assert perturbed_to == pytest.approx(original, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), (
        f"the old rule really did call {original!r} and {perturbed_to!r} equal"
    )
    moved.loc[smallest, "paired_p"] = perturbed_to
    with pytest.raises(AssertionError):
        _assert_effect_sizes_matches_golden(moved, golden)

    assert _sha256(EFFECT_SIZES_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


EFFECT_SIZES_COLUMNS = (
    "block",
    "k",
    "broker_count",
    "ablation",
    "metric",
    "metric_label",
    "n_seeds",
    "mean_disabled",
    "mean_ablation",
    "paired_mean_diff",
    "paired_mean_pct_change",
    "naive_pct_change_of_means",
    "paired_dz",
    "paired_p",
    "cohens_d_independent",
    "degenerate_paired_diff",
    "max_abs_paired_diff",
    "pnl_physical_leak",
    "metric3_sign_convention",
    "note",
)


def test_effect_sizes_csv_column_set_matches_pinned():
    _skip_unless_results_artifact_available(EFFECT_SIZES_CSV_PATH)
    import pandas as pd

    header = pd.read_csv(EFFECT_SIZES_CSV_PATH, nrows=0).columns
    _assert_exact_column_set(header, EFFECT_SIZES_COLUMNS, "results/effect_sizes.csv")


def test_effect_sizes_csv_column_set_pin_bites_on_growth_and_shrinkage():
    _skip_unless_results_artifact_available(EFFECT_SIZES_CSV_PATH)
    import pandas as pd

    header = list(pd.read_csv(EFFECT_SIZES_CSV_PATH, nrows=0).columns)
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header + ["a_new_column"], EFFECT_SIZES_COLUMNS, "results/effect_sizes.csv")
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header[:-1], EFFECT_SIZES_COLUMNS, "results/effect_sizes.csv")


# ---------------------------------------------------------------------------
# 13. Pinned digest of results/summary_stats_corrected.csv, every data row and
#     every column: the per-comparison, "one row per comparison" detail
#     Section 6.10 promises for the multiple-comparison correction and
#     bootstrap-confidence-interval story. Pinned WHOLE (188 rows, the same
#     scale as summary_stats.csv's 240). Read-only, skip-if-absent, same as
#     item 12 above.
# ---------------------------------------------------------------------------


def test_summary_stats_corrected_csv_matches_pinned_digest():
    _skip_unless_pinned_csv_available(SUMMARY_STATS_CORRECTED_CSV_PATH, SUMMARY_STATS_CORRECTED_GOLDEN_PATH)

    import pandas as pd

    golden = _load_golden(SUMMARY_STATS_CORRECTED_GOLDEN_PATH)
    df = pd.read_csv(SUMMARY_STATS_CORRECTED_CSV_PATH)
    _assert_summary_stats_corrected_matches_golden(df, golden)


def test_summary_stats_corrected_pin_bites_on_plausible_regressions():
    """Prove the pin above is not vacuous, on in-memory copies only.

    Four regressions specific to this file: a t-based confidence interval
    reported low-for-high (mirrors summary_stats.csv's own such bite), a
    Holm-survival flag flipping on one of the 72 headline comparisons Section
    6.10 says all survive, a confidence-interval disagreement appearing on a
    row currently flagged as agreeing (Section 6.10's own claim is "6 of 188
    rows are flagged", so a seventh appearing unannounced is exactly the
    failure mode this pin exists to catch), and a row disappearing.
    """
    _skip_unless_pinned_csv_available(SUMMARY_STATS_CORRECTED_CSV_PATH, SUMMARY_STATS_CORRECTED_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(SUMMARY_STATS_CORRECTED_CSV_PATH)
    golden = _load_golden(SUMMARY_STATS_CORRECTED_GOLDEN_PATH)
    df = pd.read_csv(SUMMARY_STATS_CORRECTED_CSV_PATH)

    _assert_summary_stats_corrected_matches_golden(df, golden)  # unperturbed: passes

    # (a) a t-based confidence interval reported low-for-high.
    inverted = df.copy(deep=True)
    wide = inverted.index[(inverted["ci95_t_hi"] - inverted["ci95_t_lo"]).abs() > 1e-6]
    assert len(wide) > 0, "expected at least one row with a non-degenerate confidence interval"
    row = wide[0]
    inverted.loc[row, ["ci95_t_lo", "ci95_t_hi"]] = [inverted.loc[row, "ci95_t_hi"], inverted.loc[row, "ci95_t_lo"]]
    with pytest.raises(AssertionError):
        _assert_summary_stats_corrected_matches_golden(inverted, golden)

    # (b) a Holm survival flag flips on one of the 72 headline comparisons
    #     Section 6.10 says all survive.
    flipped = df.copy(deep=True)
    surviving = flipped.index[flipped["survives_holm_alpha05"].eq(True)]
    assert len(surviving) > 0, "expected at least one row surviving Holm correction"
    flipped.loc[surviving[0], "survives_holm_alpha05"] = False
    with pytest.raises(AssertionError):
        _assert_summary_stats_corrected_matches_golden(flipped, golden)

    # (c) a confidence-interval disagreement appears and goes unannounced.
    disagreeing = df.copy(deep=True)
    agreeing = disagreeing.index[disagreeing["ci_disagreement_flag"].eq(False)]
    assert len(agreeing) > 0, "expected most rows to have the t and bootstrap intervals agree"
    disagreeing.loc[agreeing[0], "ci_disagreement_flag"] = True
    with pytest.raises(AssertionError):
        _assert_summary_stats_corrected_matches_golden(disagreeing, golden)

    # (d) a row disappears.
    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_summary_stats_corrected_matches_golden(dropped, golden)

    assert _sha256(SUMMARY_STATS_CORRECTED_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_summary_stats_corrected_pin_bites_on_a_p_value_magnitude_shift():
    """A p-value moving 30-plus orders of magnitude, staying far below the
    old absolute floor, on all three of this file's p-value columns.
    """
    _skip_unless_pinned_csv_available(SUMMARY_STATS_CORRECTED_CSV_PATH, SUMMARY_STATS_CORRECTED_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(SUMMARY_STATS_CORRECTED_CSV_PATH)
    golden = _load_golden(SUMMARY_STATS_CORRECTED_GOLDEN_PATH)
    df = pd.read_csv(SUMMARY_STATS_CORRECTED_CSV_PATH)

    _assert_summary_stats_corrected_matches_golden(df, golden)  # unperturbed: passes

    perturbed_to = 1e-20
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
            _assert_summary_stats_corrected_matches_golden(moved, golden)

    assert _sha256(SUMMARY_STATS_CORRECTED_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


SUMMARY_STATS_CORRECTED_COLUMNS = (
    "scope",
    "k",
    "broker_count",
    "ablation",
    "metric",
    "metric_label",
    "n_seeds",
    "paired_mean_diff",
    "paired_mean_pct_change",
    "paired_dz",
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
    "family_size_primary",
    "family_size_alternative",
    "family_size_monopoly_secondary",
    "bootstrap_seed",
    "n_bootstrap",
    "alpha",
    "metric3_sign_convention",
    "notes",
)


def test_summary_stats_corrected_csv_column_set_matches_pinned():
    _skip_unless_results_artifact_available(SUMMARY_STATS_CORRECTED_CSV_PATH)
    import pandas as pd

    header = pd.read_csv(SUMMARY_STATS_CORRECTED_CSV_PATH, nrows=0).columns
    _assert_exact_column_set(header, SUMMARY_STATS_CORRECTED_COLUMNS, "results/summary_stats_corrected.csv")


def test_summary_stats_corrected_csv_column_set_pin_bites_on_growth_and_shrinkage():
    _skip_unless_results_artifact_available(SUMMARY_STATS_CORRECTED_CSV_PATH)
    import pandas as pd

    header = list(pd.read_csv(SUMMARY_STATS_CORRECTED_CSV_PATH, nrows=0).columns)
    with pytest.raises(AssertionError):
        _assert_exact_column_set(
            header + ["a_new_column"], SUMMARY_STATS_CORRECTED_COLUMNS, "results/summary_stats_corrected.csv"
        )
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header[:-1], SUMMARY_STATS_CORRECTED_COLUMNS, "results/summary_stats_corrected.csv")


# ---------------------------------------------------------------------------
# 14. Pinned digest of results/family_sensitivity.csv: a CLAIM-BEARING SUBSET
#     rather than every row (see FAMILY_SENSITIVITY_IDENTITY's comment above
#     for why), plus the file's total row count as a shape guard covering the
#     rows this subset does not otherwise look at. Read-only, skip-if-absent,
#     same as items 12 and 13 above.
# ---------------------------------------------------------------------------


def test_family_sensitivity_csv_matches_pinned_digest():
    _skip_unless_pinned_csv_available(FAMILY_SENSITIVITY_CSV_PATH, FAMILY_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    golden = _load_golden(FAMILY_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH)
    _assert_family_sensitivity_matches_golden(df, golden)


def test_family_sensitivity_pin_bites_on_plausible_regressions():
    """Prove the pin above is not vacuous, on in-memory copies only.

    Three regressions: the "72 of 72 survive" tally dropping by one on a
    family_summary row (the central claim this file backs), a family size
    off by one (mirrors monopoly_comparison.csv's own seed-count-off-by-one
    bite), and a row disappearing ELSEWHERE in the file, outside the eight
    pinned rows, which no identity lookup in this pin's subset can see by
    itself: only the total-row-count guard catches it, which is the whole
    point of pinning that guard separately from the named subset.
    """
    _skip_unless_pinned_csv_available(FAMILY_SENSITIVITY_CSV_PATH, FAMILY_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(FAMILY_SENSITIVITY_CSV_PATH)
    golden = _load_golden(FAMILY_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH)

    _assert_family_sensitivity_matches_golden(df, golden)  # unperturbed: passes

    # (a) the "72 of 72 survive" tally drops by one on a family_summary row.
    regressed = df.copy(deep=True)
    summary_rows = regressed.index[regressed["scope"] == "family_summary"]
    assert len(summary_rows) > 0, "expected the family_summary scope to be present"
    row = summary_rows[0]
    regressed.loc[row, "n_headline_surviving_holm"] = regressed.loc[row, "n_headline_surviving_holm"] - 1
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(regressed, golden)

    # (b) a family size off by one.
    off_by_one = df.copy(deep=True)
    off_by_one.loc[row, "family_size"] = int(off_by_one.loc[row, "family_size"]) - 1
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(off_by_one, golden)

    # (c) a row disappears from OUTSIDE the eight pinned rows: no identity
    #     lookup in this pin's subset notices, so only the total-row-count
    #     guard can catch it. Matched on the full (family_definition, test_id)
    #     pair, not test_id alone: the pinned per-test row's test_id is pooled
    #     into several OTHER family definitions too (F2, F4, F5, F7 all pool
    #     the main sweep), and those rows are not part of the pinned eight.
    pinned_keys = set(FAMILY_SENSITIVITY_PINNED_ROW_KEYS)
    row_keys = list(zip(df["family_definition"], df["test_id"]))
    unpinned_rows = [i for i, key in zip(df.index, row_keys) if key not in pinned_keys]
    assert len(unpinned_rows) > 0
    dropped = df.drop(index=unpinned_rows[0])
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(dropped, golden)

    assert _sha256(FAMILY_SENSITIVITY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_family_sensitivity_pin_bites_on_a_p_value_magnitude_shift():
    """The one pinned per-test row's p-values moving far in magnitude while
    staying far below the old absolute floor. The seven family_summary rows'
    own worst-p figures never get small enough to demonstrate this rule (see
    FAMILY_SENSITIVITY_IDENTITY's comment above), which is the reason that
    eighth row is pinned at all.
    """
    _skip_unless_pinned_csv_available(FAMILY_SENSITIVITY_CSV_PATH, FAMILY_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(FAMILY_SENSITIVITY_CSV_PATH)
    golden = _load_golden(FAMILY_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH)

    _assert_family_sensitivity_matches_golden(df, golden)  # unperturbed: passes

    perturbed_to = 1e-20
    target = df.index[
        (df["family_definition"] == "F1_main_sweep_primary_120")
        & (df["test_id"] == "main|k=1|bc=3|capacity_both|feeder_coefficient_of_variation")
    ]
    assert len(target) == 1, "expected exactly one row for the pinned per-test key"
    row = target[0]
    for column in ("p_uncorrected", "p_holm", "p_bh"):
        moved = df.copy(deep=True)
        original = float(moved.loc[row, column])
        assert original < 1e-30, f"{column}: expected a real p-value far below the old absolute floor"
        assert perturbed_to == pytest.approx(original, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), (
            f"{column}: the old rule really did call {original!r} and {perturbed_to!r} equal"
        )
        moved.loc[row, column] = perturbed_to
        with pytest.raises(AssertionError):
            _assert_family_sensitivity_matches_golden(moved, golden)

    assert _sha256(FAMILY_SENSITIVITY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_family_sensitivity_aggregate_pin_bites_on_an_unpinned_row_flag_flip():
    """A survives_holm_alpha05 flag flipping on a row OUTSIDE the eight
    individually pinned rows: no identity lookup in the claim-bearing subset
    can see this row at all, so only the whole-file aggregate counts catch
    it, on both the affected family_definition's own count and the overall
    count.
    """
    _skip_unless_pinned_csv_available(FAMILY_SENSITIVITY_CSV_PATH, FAMILY_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(FAMILY_SENSITIVITY_CSV_PATH)
    golden = _load_golden(FAMILY_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH)

    _assert_family_sensitivity_matches_golden(df, golden)  # unperturbed: passes

    pinned_keys = set(FAMILY_SENSITIVITY_PINNED_ROW_KEYS)
    row_keys = list(zip(df["family_definition"], df["test_id"]))
    candidates = [
        i
        for i, key, value in zip(df.index, row_keys, df["survives_holm_alpha05"])
        if key not in pinned_keys and value is True
    ]
    assert len(candidates) > 0, "expected an unpinned row currently surviving Holm correction"
    row = candidates[0]
    assert (df.loc[row, "family_definition"], df.loc[row, "test_id"]) not in pinned_keys

    flipped = df.copy(deep=True)
    flipped.loc[row, "survives_holm_alpha05"] = False
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(flipped, golden)

    assert _sha256(FAMILY_SENSITIVITY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_family_sensitivity_aggregate_pin_bites_on_an_unpinned_p_value_magnitude_shift():
    """The file's global-minimum p_holm, which sits on rows OUTSIDE the eight
    individually pinned rows, moving from its real value to 1e-20: exactly the
    old rel=1e-9/abs=1e-9 rule's blind spot, now caught by the relative-only
    min/max aggregate instead of by an individual row pin.

    capacity_both and capacity_pricing_only are bit-identical to each other on
    every metric (Section 6.1: the P&L channel adds no physics once pricing is
    already on), so the row holding the file's minimum p_holm always has an
    exact ablation twin carrying the identical value in the same
    family_definition. Perturbing the minimum alone would leave that twin
    still holding the true minimum and the aggregate would not move at all,
    which is not a weaker bite, it is no bite; every row TIED at the true
    minimum is perturbed together so the minimum is provably gone afterwards.
    """
    _skip_unless_pinned_csv_available(FAMILY_SENSITIVITY_CSV_PATH, FAMILY_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(FAMILY_SENSITIVITY_CSV_PATH)
    golden = _load_golden(FAMILY_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH)

    _assert_family_sensitivity_matches_golden(df, golden)  # unperturbed: passes

    pinned_keys = set(FAMILY_SENSITIVITY_PINNED_ROW_KEYS)
    original = float(df["p_holm"].min())
    assert original < 1e-30, "expected a real p-value far below the old absolute floor"
    perturbed_to = 1e-20
    assert perturbed_to == pytest.approx(original, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), (
        f"the old rule really did call {original!r} and {perturbed_to!r} equal"
    )

    tied = df.index[df["p_holm"] == original]
    assert len(tied) >= 1, "expected at least one row holding the global minimum p_holm"
    for candidate in tied:
        assert (df.loc[candidate, "family_definition"], df.loc[candidate, "test_id"]) not in pinned_keys, (
            "expected every row tied at the global minimum p_holm to sit outside the eight pinned rows"
        )

    moved = df.copy(deep=True)
    moved.loc[tied, "p_holm"] = perturbed_to
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(moved, golden)

    assert _sha256(FAMILY_SENSITIVITY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_family_sensitivity_log_sum_bites_on_a_mid_distribution_p_value_shift_that_min_max_and_counts_miss():
    """The regression class counts and min/max cannot see: a p-value that
    moves in the MIDDLE of the distribution, far from the smallest and
    largest values, and without crossing the alpha=0.05 significance
    boundary. This perturbs exactly one unpinned p_holm five orders of
    magnitude and proves three things at once: the log-sum rejects it, and
    (explicitly asserted, not just claimed in prose) every count and every
    min/max in the same aggregate still match the untouched golden values.
    That combination is what justifies the log-sum's own existence: without
    it, this exact perturbation would pass every other guard in this file.
    """
    _skip_unless_pinned_csv_available(FAMILY_SENSITIVITY_CSV_PATH, FAMILY_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(FAMILY_SENSITIVITY_CSV_PATH)
    golden = _load_golden(FAMILY_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH)

    _assert_family_sensitivity_matches_golden(df, golden)  # unperturbed: passes

    pinned_keys = set(FAMILY_SENSITIVITY_PINNED_ROW_KEYS)
    family_definition = "F1_main_sweep_primary_120"
    test_id = "main|k=1|bc=2|capacity_both|feeder_peak_to_average_ratio"
    assert (family_definition, test_id) not in pinned_keys

    target = df.index[(df["family_definition"] == family_definition) & (df["test_id"] == test_id)]
    assert len(target) == 1
    row = target[0]

    group = df[df["family_definition"] == family_definition]
    original = float(df.loc[row, "p_holm"])
    group_min = float(group["p_holm"].min())
    group_max = float(group["p_holm"].max())
    # Neither the group's minimum nor its maximum: this is a genuinely
    # mid-distribution value, and the shift below stays clear of both.
    assert original == pytest.approx(5.541809e-10, rel=1e-6, abs=0.0)
    assert original > group_min * 1e6 and original < group_max / 1e6

    perturbed_to = original * 1e5  # five orders of magnitude, same sign, still << 1
    assert perturbed_to < 0.05, "perturbation must stay a plausible p-value and keep the significance verdict unchanged"

    moved = df.copy(deep=True)
    moved.loc[row, "p_holm"] = perturbed_to

    # The claim this test exists to prove: counts, min and max ALL still
    # match the untouched golden values after this exact perturbation.
    moved_group = moved[moved["family_definition"] == family_definition]
    assert bool(moved.loc[row, "survives_holm_alpha05"]) is True, (
        "perturbation must not cross the alpha=0.05 boundary, or the survives_holm count would move too"
    )
    unaffected_fields = (
        "n_rows", "n_survives_holm_true", "n_survives_holm_false",
        "n_survives_bh_true", "n_survives_bh_false", "n_degenerate_true",
        "p_uncorrected_min", "p_uncorrected_max", "p_holm_min", "p_holm_max",
        "p_bh_min", "p_bh_max", "n_rows_contributing",
    )
    moved_aggregate = _family_sensitivity_aggregate_row(moved_group)
    golden_aggregate = golden["aggregates"]["by_family_definition"][family_definition]
    for field in unaffected_fields:
        assert moved_aggregate[field] == golden_aggregate[field], (
            f"{field} changed after the perturbation; this test's premise (counts/min/max are blind "
            "to a mid-distribution shift) no longer holds"
        )

    # The log-sum, and only the log-sum, catches it.
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(moved, golden)

    assert _sha256(FAMILY_SENSITIVITY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


_LABEL_ONLY_UNAFFECTED_FIELDS = tuple(FAMILY_SENSITIVITY_AGGREGATE_FIELDS.keys())


def _assert_only_label_counts_moved(actual_group: dict, golden_group: dict, context: str) -> None:
    """Assert every value-column field (counts, extremes, log-sums) still
    matches the untouched golden aggregate, so a test using this helper
    documents, by construction, that the perturbation it applies is invisible
    to everything except the label multiset.
    """
    for field in _LABEL_ONLY_UNAFFECTED_FIELDS:
        assert actual_group[field] == golden_group[field], (
            f"{context}: {field} changed; this perturbation was supposed to touch only a label column"
        )


def test_family_sensitivity_label_multiset_bites_on_metric_ablation_and_is_headline_72_mislabels():
    """The gap an adversarial review reproduced three separate ways against
    an earlier version of this module: a relabelled metric, a swapped
    ablation, and a flipped is_headline_72, each on a row OUTSIDE the eight
    individually pinned, each passing every value-column aggregate above
    (counts, extremes, log-sums are all unchanged, asserted explicitly below,
    not merely claimed) because none of those aggregates ever reads metric,
    ablation or is_headline_72 at all. Only the label multiset sees any of
    the three. Each of the three mirrors the reviewer's own reproduction
    exactly: mislabelled.loc[target, "metric"] = "SWAPPED_WRONG_METRIC_LABEL",
    swapped.loc[target3, "ablation"] = "capacity_pricing_only", and
    flipped.loc[target2, "is_headline_72"] = not before_val.

    A fourth part, added after a follow-up review found the first version of
    this multiset was itself a hand-picked subset of label columns: a
    mislabelled source on a monopoly_supplement row, proving the now-complete
    "every identifier or label column" rule catches a column that earlier
    version left out on purpose.
    """
    _skip_unless_pinned_csv_available(FAMILY_SENSITIVITY_CSV_PATH, FAMILY_SENSITIVITY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(FAMILY_SENSITIVITY_CSV_PATH)
    golden = _load_golden(FAMILY_SENSITIVITY_GOLDEN_PATH)
    df = pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH)

    _assert_family_sensitivity_matches_golden(df, golden)  # unperturbed: passes

    pinned_keys = set(FAMILY_SENSITIVITY_PINNED_ROW_KEYS)

    # (a) a metric relabelled, mirroring the reviewer's own repro exactly.
    family_definition = "F2_main_sweep_with_degenerates_180"
    test_id = "main|k=0.5|bc=2|capacity_both|avg_cost_per_agent_eur"
    assert (family_definition, test_id) not in pinned_keys
    target = df.index[(df["family_definition"] == family_definition) & (df["test_id"] == test_id)]
    assert len(target) == 1

    mislabelled = df.copy(deep=True)
    mislabelled.loc[target, "metric"] = "SWAPPED_WRONG_METRIC_LABEL"
    group = mislabelled[mislabelled["family_definition"] == family_definition]
    _assert_only_label_counts_moved(
        _family_sensitivity_aggregate_row(group),
        golden["aggregates"]["by_family_definition"][family_definition],
        f"metric mislabel: {family_definition}",
    )
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(mislabelled, golden)

    # (b) an ablation swapped, mirroring the reviewer's own repro exactly.
    family_definition = "F5_everything_pooled"
    test_id = "main|k=1.5|bc=3|capacity_both|load_concentration_hhi"
    assert (family_definition, test_id) not in pinned_keys
    target = df.index[(df["family_definition"] == family_definition) & (df["test_id"] == test_id)]
    assert len(target) == 1
    assert df.loc[target[0], "ablation"] == "capacity_both", "expected a real ablation change, not a no-op swap"

    swapped = df.copy(deep=True)
    swapped.loc[target, "ablation"] = "capacity_pricing_only"
    group = swapped[swapped["family_definition"] == family_definition]
    _assert_only_label_counts_moved(
        _family_sensitivity_aggregate_row(group),
        golden["aggregates"]["by_family_definition"][family_definition],
        f"ablation swap: {family_definition}",
    )
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(swapped, golden)

    # (c) is_headline_72 flipped, mirroring the reviewer's own repro exactly.
    family_definition = "F7_everything_pooled_with_degenerates"
    test_id = "main|k=2|bc=5|capacity_both|prosumer_self_sufficiency"
    assert (family_definition, test_id) not in pinned_keys
    target = df.index[(df["family_definition"] == family_definition) & (df["test_id"] == test_id)]
    assert len(target) == 1
    before_val = bool(df.loc[target[0], "is_headline_72"])

    flipped = df.copy(deep=True)
    flipped.loc[target, "is_headline_72"] = not before_val
    group = flipped[flipped["family_definition"] == family_definition]
    _assert_only_label_counts_moved(
        _family_sensitivity_aggregate_row(group),
        golden["aggregates"]["by_family_definition"][family_definition],
        f"is_headline_72 flip: {family_definition}",
    )
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(flipped, golden)

    # (d) source mislabelled, on a monopoly_supplement row: the column a
    # follow-up review found the first version of this fix left out on
    # purpose, without a principled reason to leave it out.
    family_definition = "F4_pooled_with_monopoly"
    test_id = "monopoly|k=0.5|bc=1|capacity_pricing_only|feeder_coefficient_of_variation"
    assert (family_definition, test_id) not in pinned_keys
    target = df.index[(df["family_definition"] == family_definition) & (df["test_id"] == test_id)]
    assert len(target) == 1
    assert df.loc[target[0], "source"] == "monopoly_supplement", "expected a real source change, not a no-op"

    source_mislabelled = df.copy(deep=True)
    source_mislabelled.loc[target, "source"] = "SWAPPED_WRONG_SOURCE_LABEL"
    group = source_mislabelled[source_mislabelled["family_definition"] == family_definition]
    _assert_only_label_counts_moved(
        _family_sensitivity_aggregate_row(group),
        golden["aggregates"]["by_family_definition"][family_definition],
        f"source mislabel: {family_definition}",
    )
    with pytest.raises(AssertionError):
        _assert_family_sensitivity_matches_golden(source_mislabelled, golden)

    assert _sha256(FAMILY_SENSITIVITY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


FAMILY_SENSITIVITY_COLUMNS = (
    "family_definition",
    "family_key",
    "family_size",
    "n_families_in_definition",
    "scope",
    "source",
    "k",
    "broker_count",
    "ablation",
    "cell_is_stamped_not_native",
    "metric",
    "test_id",
    "is_headline_72",
    "degenerate",
    "p_uncorrected",
    "p_holm",
    "p_bh",
    "survives_holm_alpha05",
    "survives_bh_alpha05",
    "alpha",
    "n_headline_tests",
    "n_headline_surviving_holm",
    "n_headline_surviving_bh",
    "worst_headline_p_uncorrected",
    "worst_headline_p_holm",
    "worst_headline_p_bh",
    "bonferroni_alpha_over_family_size",
    "notes",
)


def test_family_sensitivity_csv_column_set_matches_pinned():
    _skip_unless_results_artifact_available(FAMILY_SENSITIVITY_CSV_PATH)
    import pandas as pd

    header = pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH, nrows=0).columns
    _assert_exact_column_set(header, FAMILY_SENSITIVITY_COLUMNS, "results/family_sensitivity.csv")


def test_family_sensitivity_csv_column_set_pin_bites_on_growth_and_shrinkage():
    _skip_unless_results_artifact_available(FAMILY_SENSITIVITY_CSV_PATH)
    import pandas as pd

    header = list(pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH, nrows=0).columns)
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header + ["a_new_column"], FAMILY_SENSITIVITY_COLUMNS, "results/family_sensitivity.csv")
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header[:-1], FAMILY_SENSITIVITY_COLUMNS, "results/family_sensitivity.csv")


# ---------------------------------------------------------------------------
# 15. Pinned digest of results/dose_matched_monopoly.csv, D11's dose-matched-
#     monopoly arm (docs/DECISIONS.md's "## D11." and "## D11 result:"
#     sections, Phase 18.1): every data row and every column, keyed by
#     (scope, arm, k, metric). Pinned WHOLE, the same way effect_sizes.csv and
#     summary_stats_corrected.csv are (see DOSE_MATCHED_MONOPOLY_IDENTITY's
#     comment above): 46 rows is small enough that a claim-bearing subset
#     would save nothing. Read-only, skip-if-absent, same as items 12 and 13.
# ---------------------------------------------------------------------------


def test_dose_matched_monopoly_csv_matches_pinned_digest():
    _skip_unless_pinned_csv_available(DOSE_MATCHED_MONOPOLY_CSV_PATH, DOSE_MATCHED_MONOPOLY_GOLDEN_PATH)

    import pandas as pd

    golden = _load_golden(DOSE_MATCHED_MONOPOLY_GOLDEN_PATH)
    df = pd.read_csv(DOSE_MATCHED_MONOPOLY_CSV_PATH)
    _assert_dose_matched_monopoly_matches_golden(df, golden)


def test_dose_matched_monopoly_pin_bites_on_plausible_regressions():
    """Prove the pin above is not vacuous, on in-memory copies only.

    Six regressions specific to this file: a percent change's sign flipping
    (a reported improvement becomes a reported worsening at the same
    magnitude), the realized-deferred-volume mean drifting by 1 percent (the
    figure D11's whole "dose matched" reading rests on: the arm's verdict
    depends on the volume lining up, not only the effect), a paired-seed
    favorable-count tally off by one, a Holm survival flag flipping, a row's
    own identity silently colliding with another row's (the same arm and
    metric, relabelled from one shipped k level onto the other, which
    _rows_by_identity's own duplicate-key guard must catch), and a row
    disappearing. The CSV's own digest is captured before and re-checked
    after, so the file on disk is provably untouched.
    """
    _skip_unless_pinned_csv_available(DOSE_MATCHED_MONOPOLY_CSV_PATH, DOSE_MATCHED_MONOPOLY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(DOSE_MATCHED_MONOPOLY_CSV_PATH)
    golden = _load_golden(DOSE_MATCHED_MONOPOLY_GOLDEN_PATH)
    df = pd.read_csv(DOSE_MATCHED_MONOPOLY_CSV_PATH)

    _assert_dose_matched_monopoly_matches_golden(df, golden)  # unperturbed: passes

    # (a) a percent change's sign flips: a reported improvement becomes a
    #     reported worsening at the same magnitude.
    flipped = df.copy(deep=True)
    nonzero_pct = flipped.index[flipped["paired_mean_pct_change"].abs() > 1e-6]
    assert len(nonzero_pct) > 0, "expected at least one non-degenerate paired_mean_pct_change"
    row = nonzero_pct[0]
    flipped.loc[row, "paired_mean_pct_change"] = -flipped.loc[row, "paired_mean_pct_change"]
    with pytest.raises(AssertionError):
        _assert_dose_matched_monopoly_matches_golden(flipped, golden)

    # (b) the realized deferred-volume mean drifts by 1 percent: small enough
    #     to pass a casual eyeball check, and this is the figure D11's whole
    #     "dose matched" reading rests on.
    drifted = df.copy(deep=True)
    nonzero_deferred = drifted.index[drifted["total_deferred_kwh_capacity_both_mean"].abs() > 1e-6]
    assert len(nonzero_deferred) > 0
    row = nonzero_deferred[0]
    drifted.loc[row, "total_deferred_kwh_capacity_both_mean"] *= 1.01
    with pytest.raises(AssertionError):
        _assert_dose_matched_monopoly_matches_golden(drifted, golden)

    # (c) a paired-seed favorable-count tally off by one.
    off_by_one = df.copy(deep=True)
    column = "n_favorable_of_n_seeds"
    off_by_one.loc[off_by_one.index[0], column] = int(off_by_one.loc[off_by_one.index[0], column]) - 1
    with pytest.raises(AssertionError):
        _assert_dose_matched_monopoly_matches_golden(off_by_one, golden)

    # (d) a Holm survival flag flips.
    flag_flipped = df.copy(deep=True)
    surviving = flag_flipped.index[flag_flipped["survives_holm_alpha05"].eq(True)]
    assert len(surviving) > 0, "expected at least one row surviving Holm correction"
    flag_flipped.loc[surviving[0], "survives_holm_alpha05"] = False
    with pytest.raises(AssertionError):
        _assert_dose_matched_monopoly_matches_golden(flag_flipped, golden)

    # (e) a row's identity silently collides with another row's: the same
    #     arm and metric, relabelled from one shipped k level to the other,
    #     which _rows_by_identity's own duplicate-key guard must catch.
    collided = df.copy(deep=True)
    mono1_rows = collided.index[
        (collided["arm"] == "mono_divisor_1")
        & (collided["scope"] == "arm_vs_own_baseline")
        & (collided["metric"] == "feeder_peak_to_average_ratio")
    ]
    assert len(mono1_rows) == 2, "expected exactly one k=0.5 and one k=1.0 row for this (scope, arm, metric)"
    collided.loc[mono1_rows[0], "k"] = collided.loc[mono1_rows[1], "k"]
    with pytest.raises(AssertionError):
        _assert_dose_matched_monopoly_matches_golden(collided, golden)

    # (f) a row disappears.
    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_dose_matched_monopoly_matches_golden(dropped, golden)

    assert _sha256(DOSE_MATCHED_MONOPOLY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_dose_matched_monopoly_pin_bites_on_a_p_value_magnitude_shift():
    """A p-value moving far in magnitude while staying far below the old
    absolute floor: under the previous rel=1e-9/abs=1e-9 comparison (either
    tolerance sufficing) this exact perturbation PASSED, because both values
    sit under the absolute floor. The assertion inside reproduces that old
    rule and shows it calling the two values equal, immediately before the
    current rule rejects them. Done for all three correction columns, each
    time picking the smallest NONZERO p in that column: two rows of this file
    carry an exact 0.0, which the "pvalue" kind's own zero branch already
    covers in test_pvalue_comparison_pins_magnitude_and_handles_zero_and_denormals.
    """
    _skip_unless_pinned_csv_available(DOSE_MATCHED_MONOPOLY_CSV_PATH, DOSE_MATCHED_MONOPOLY_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(DOSE_MATCHED_MONOPOLY_CSV_PATH)
    golden = _load_golden(DOSE_MATCHED_MONOPOLY_GOLDEN_PATH)
    df = pd.read_csv(DOSE_MATCHED_MONOPOLY_CSV_PATH)

    _assert_dose_matched_monopoly_matches_golden(df, golden)  # unperturbed: passes

    perturbed_to = 1e-20  # still hopelessly significant, still far below 1e-9
    for column in ("p_uncorrected", "p_holm", "p_bh"):
        moved = df.copy(deep=True)
        nonzero = moved.index[moved[column] > 0.0]
        row = moved.loc[nonzero, column].idxmin()
        original = float(moved.loc[row, column])
        assert original < 1e-30, f"{column}: expected a real p-value far below the old absolute floor"
        assert perturbed_to == pytest.approx(original, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), (
            f"{column}: the old rule really did call {original!r} and {perturbed_to!r} equal"
        )
        moved.loc[row, column] = perturbed_to
        with pytest.raises(AssertionError):
            _assert_dose_matched_monopoly_matches_golden(moved, golden)

    assert _sha256(DOSE_MATCHED_MONOPOLY_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


DOSE_MATCHED_MONOPOLY_COLUMNS = (
    "scope",
    "arm",
    "arm_type",
    "capacity_surcharge_divisor",
    "broker_count",
    "comparator_arm",
    "data_source",
    "k",
    "metric",
    "n_seeds",
    "paired_mean_pct_change",
    "pct_point_diff_mono_minus_comparator",
    "paired_mean_diff_raw",
    "paired_dz",
    "p_uncorrected",
    "p_holm",
    "p_bh",
    "survives_holm_alpha05",
    "survives_bh_alpha05",
    "ci95_t_lo",
    "ci95_t_hi",
    "ci95_boot_lo",
    "ci95_boot_hi",
    "n_favorable_of_n_seeds",
    "total_deferred_kwh_capacity_both_mean",
    "total_deferred_kwh_capacity_disabled_mean",
    "comparator_total_deferred_kwh_capacity_both_mean",
    "deferred_volume_ratio_mono_over_comparator",
    "correction_family",
    "correction_family_size",
    "metric3_sign_convention",
    "notes",
)


def test_dose_matched_monopoly_csv_column_set_matches_pinned():
    _skip_unless_results_artifact_available(DOSE_MATCHED_MONOPOLY_CSV_PATH)
    import pandas as pd

    header = pd.read_csv(DOSE_MATCHED_MONOPOLY_CSV_PATH, nrows=0).columns
    _assert_exact_column_set(header, DOSE_MATCHED_MONOPOLY_COLUMNS, "results/dose_matched_monopoly.csv")


def test_dose_matched_monopoly_csv_column_set_pin_bites_on_growth_and_shrinkage():
    _skip_unless_results_artifact_available(DOSE_MATCHED_MONOPOLY_CSV_PATH)
    import pandas as pd

    header = list(pd.read_csv(DOSE_MATCHED_MONOPOLY_CSV_PATH, nrows=0).columns)
    with pytest.raises(AssertionError):
        _assert_exact_column_set(
            header + ["a_new_column"], DOSE_MATCHED_MONOPOLY_COLUMNS, "results/dose_matched_monopoly.csv"
        )
    with pytest.raises(AssertionError):
        _assert_exact_column_set(header[:-1], DOSE_MATCHED_MONOPOLY_COLUMNS, "results/dose_matched_monopoly.csv")


# ---------------------------------------------------------------------------
# 16. results/sweep_dose_matched_monopoly.parquet (D11's dose-matched-
#     monopoly arm, Phase 18.1): a shape pin (row count and exact column set,
#     matching items 10/11's style for the other four raw sweep parquets)
#     plus a per-(k, capacity_surcharge_divisor, ablation) cell aggregate pin
#     that no CSV happens to cover directly (see
#     DOSE_MATCHED_MONOPOLY_PARQUET_CELL_IDENTITY's comment above): the row
#     count (30 in every cell) and the mean of the six metrics
#     docs/DECISIONS.md's "D11 result" section quotes straight from this
#     parquet.
# ---------------------------------------------------------------------------

# sweep_dose_matched_monopoly.parquet is written by the same per-run metrics
# writer as the other four raw sweep parquets, PLUS the two D11-specific
# columns capacity_surcharge_mode and capacity_surcharge_divisor.
SWEEP_DOSE_MATCHED_MONOPOLY_COLUMNS = SWEEP_RAW_COLUMNS + ("capacity_surcharge_mode", "capacity_surcharge_divisor")
SWEEP_DOSE_MATCHED_MONOPOLY_ROW_COUNT = 360  # 2 k levels x 3 divisors x 2 ablations x 30 seeds


def test_sweep_dose_matched_monopoly_parquet_shape_matches_pinned():
    _skip_unless_results_artifact_available(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    import pandas as pd

    df = pd.read_parquet(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    _assert_parquet_shape(
        df, SWEEP_DOSE_MATCHED_MONOPOLY_ROW_COUNT, SWEEP_DOSE_MATCHED_MONOPOLY_COLUMNS,
        "results/sweep_dose_matched_monopoly.parquet",
    )


def test_sweep_dose_matched_monopoly_parquet_shape_pin_bites_on_growth_shrinkage_and_row_count_drift():
    _skip_unless_results_artifact_available(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    import pandas as pd

    digest_before = _sha256(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    df = pd.read_parquet(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)

    grown = df.copy(deep=True)
    grown["a_new_column"] = 0.0
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            grown, SWEEP_DOSE_MATCHED_MONOPOLY_ROW_COUNT, SWEEP_DOSE_MATCHED_MONOPOLY_COLUMNS,
            "results/sweep_dose_matched_monopoly.parquet",
        )

    shrunk = df.drop(columns=["peak_rss_mb"])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            shrunk, SWEEP_DOSE_MATCHED_MONOPOLY_ROW_COUNT, SWEEP_DOSE_MATCHED_MONOPOLY_COLUMNS,
            "results/sweep_dose_matched_monopoly.parquet",
        )

    appended = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            appended, SWEEP_DOSE_MATCHED_MONOPOLY_ROW_COUNT, SWEEP_DOSE_MATCHED_MONOPOLY_COLUMNS,
            "results/sweep_dose_matched_monopoly.parquet",
        )

    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_parquet_shape(
            dropped, SWEEP_DOSE_MATCHED_MONOPOLY_ROW_COUNT, SWEEP_DOSE_MATCHED_MONOPOLY_COLUMNS,
            "results/sweep_dose_matched_monopoly.parquet",
        )

    assert _sha256(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


def test_sweep_dose_matched_monopoly_parquet_cell_aggregates_match_pinned_digest():
    _skip_unless_pinned_csv_available(
        SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH, SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_CELL_GOLDEN_PATH
    )

    import pandas as pd

    golden = _load_golden(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_CELL_GOLDEN_PATH)
    df = pd.read_parquet(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    _assert_dose_matched_monopoly_parquet_cells_match_golden(df, golden)


def test_sweep_dose_matched_monopoly_parquet_cell_aggregate_pin_bites_on_plausible_regressions():
    """Prove the per-cell aggregate pin above is not vacuous, on in-memory
    copies of the raw (per-seed) parquet rows only.

    Three regressions: a cell's mean drifting by 1 percent (small enough to
    pass a casual eyeball check, and this is exactly the kind of aggregate
    D11 result's peak-to-average table quotes), a single row's own cell
    identity mislabelled so one cell loses a row and its neighbour gains one
    (both cells' n_rows should read exactly 30, the row-count-per-cell
    guarantee this arm's design calls out), and a whole cell vanishing, as if
    a resweep silently skipped one (k, divisor, ablation) combination. The
    parquet's own digest is captured before and re-checked after, so the file
    on disk is provably untouched.
    """
    _skip_unless_pinned_csv_available(
        SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH, SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_CELL_GOLDEN_PATH
    )

    import pandas as pd

    digest_before = _sha256(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    golden = _load_golden(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_CELL_GOLDEN_PATH)
    df = pd.read_parquet(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)

    _assert_dose_matched_monopoly_parquet_cells_match_golden(df, golden)  # unperturbed: passes

    # (a) a cell's mean drifts by 1 percent.
    drifted = df.copy(deep=True)
    target = drifted.index[
        (drifted["k"] == 1.0)
        & (drifted["capacity_surcharge_divisor"] == 3)
        & (drifted["ablation"] == "capacity_both")
    ]
    assert len(target) == 30
    drifted.loc[target, "total_deferred_kwh"] = drifted.loc[target, "total_deferred_kwh"] * 1.01
    with pytest.raises(AssertionError):
        _assert_dose_matched_monopoly_parquet_cells_match_golden(drifted, golden)

    # (b) one row's cell identity mislabelled: a divisor=3 row relabelled as
    #     divisor=2, so the divisor=3 cell loses a row (29, not 30) and the
    #     divisor=2 cell gains one (31, not 30), at the same k and ablation.
    mislabelled = df.copy(deep=True)
    source = mislabelled.index[
        (mislabelled["k"] == 0.5)
        & (mislabelled["capacity_surcharge_divisor"] == 3)
        & (mislabelled["ablation"] == "capacity_both")
    ]
    assert len(source) == 30
    mislabelled.loc[source[0], "capacity_surcharge_divisor"] = 2
    with pytest.raises(AssertionError):
        _assert_dose_matched_monopoly_parquet_cells_match_golden(mislabelled, golden)

    # (c) a whole cell vanishes, as if a resweep silently skipped it.
    vanished_mask = (
        (df["k"] == 1.0) & (df["capacity_surcharge_divisor"] == 5) & (df["ablation"] == "capacity_disabled")
    )
    assert vanished_mask.sum() == 30
    vanished = df[~vanished_mask]
    with pytest.raises(AssertionError):
        _assert_dose_matched_monopoly_parquet_cells_match_golden(vanished, golden)

    assert _sha256(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH) == digest_before, "bite test must not touch the parquet on disk"


# ---------------------------------------------------------------------------
# 17. The two D11 result guardrail invariants (docs/DECISIONS.md's "## D11
#     result:" section): MEASURED there, not merely asserted by construction,
#     and pinned here as their own explicit tests because a later change that
#     silently broke either one would be the single most consequential
#     regression this arm could suffer without D11's own verdict noticing.
# ---------------------------------------------------------------------------

# The seven columns D11 result names explicitly for the first invariant:
# peak-to-average, coefficient of variation, cost, deferred energy, charge,
# fire rate and self-sufficiency.
DOSE_MATCHED_MONOPOLY_GUARDRAIL_COLUMNS = (
    "feeder_peak_to_average_ratio",
    "feeder_coefficient_of_variation",
    "avg_cost_per_agent_eur",
    "total_deferred_kwh",
    "total_capacity_charge_eur",
    "capacity_fire_rate",
    "prosumer_self_sufficiency",
)


def _rows_by_seed(df, k: float, ablation: str, filter_column: str, filter_value):
    subset = df[(df["k"] == k) & (df["ablation"] == ablation) & (df[filter_column] == filter_value)]
    return subset.sort_values("seed").reset_index(drop=True)


def _assert_capacity_disabled_rows_bit_identical(dose_matched_df, monopoly_df, k: float) -> None:
    """The guardrail D11 result measures: with the pricing channel off, the
    divisor cannot do anything, so every capacity_disabled row must be
    EXACTLY the shipped baseline, not approximately. Exact equality (==), not
    a tolerance: D11 result reports this at precisely 0.0 maximum absolute
    divergence, and a tolerance here would hide the thing this test exists to
    catch (a stray dependency of the disabled code path on the divisor).
    """
    divisor2 = _rows_by_seed(dose_matched_df, k, "capacity_disabled", "capacity_surcharge_divisor", 2)
    divisor3 = _rows_by_seed(dose_matched_df, k, "capacity_disabled", "capacity_surcharge_divisor", 3)
    divisor5 = _rows_by_seed(dose_matched_df, k, "capacity_disabled", "capacity_surcharge_divisor", 5)
    shipped = _rows_by_seed(monopoly_df, k, "capacity_disabled", "broker_count", 1)

    assert len(divisor2) == 30 and len(divisor3) == 30 and len(divisor5) == 30 and len(shipped) == 30
    seeds = list(divisor2["seed"])
    for other in (divisor3, divisor5, shipped):
        assert list(other["seed"]) == seeds, f"k={k}: seed ordering must match before a positional == is meaningful"

    for column in DOSE_MATCHED_MONOPOLY_GUARDRAIL_COLUMNS:
        for label, other in (("divisor 3", divisor3), ("divisor 5", divisor5), ("shipped monopoly arm", shipped)):
            assert (divisor2[column].to_numpy() == other[column].to_numpy()).all(), (
                f"k={k}, {column}: divisor 2 is not bit-identical to {label}"
            )


def test_dose_matched_monopoly_capacity_disabled_rows_bit_identical_across_divisors_and_vs_shipped_monopoly():
    _skip_unless_results_artifact_available(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    _skip_unless_results_artifact_available(SWEEP_MONOPOLY_PARQUET_PATH)

    import pandas as pd

    dose_matched_df = pd.read_parquet(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    monopoly_df = pd.read_parquet(SWEEP_MONOPOLY_PARQUET_PATH)

    for k in (0.5, 1.0):
        _assert_capacity_disabled_rows_bit_identical(dose_matched_df, monopoly_df, k)


def test_dose_matched_monopoly_capacity_disabled_guardrail_bites_on_a_nudged_divisor_row():
    _skip_unless_results_artifact_available(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    _skip_unless_results_artifact_available(SWEEP_MONOPOLY_PARQUET_PATH)

    import pandas as pd

    dm_digest_before = _sha256(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    mono_digest_before = _sha256(SWEEP_MONOPOLY_PARQUET_PATH)
    dose_matched_df = pd.read_parquet(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    monopoly_df = pd.read_parquet(SWEEP_MONOPOLY_PARQUET_PATH)

    _assert_capacity_disabled_rows_bit_identical(dose_matched_df, monopoly_df, 0.5)  # unperturbed: passes

    nudged = dose_matched_df.copy(deep=True)
    target = nudged.index[
        (nudged["k"] == 0.5)
        & (nudged["ablation"] == "capacity_disabled")
        & (nudged["capacity_surcharge_divisor"] == 3)
    ]
    assert len(target) == 30
    nudged.loc[target[0], "feeder_peak_to_average_ratio"] += 1e-9
    with pytest.raises(AssertionError):
        _assert_capacity_disabled_rows_bit_identical(nudged, monopoly_df, 0.5)

    assert _sha256(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH) == dm_digest_before, (
        "bite test must not touch the parquet on disk"
    )
    assert _sha256(SWEEP_MONOPOLY_PARQUET_PATH) == mono_digest_before, "bite test must not touch the parquet on disk"


def _assert_divisor2_equals_divisor1_but_cost_differs(dose_matched_df, monopoly_df, k: float) -> None:
    """The other D11 result invariant, and its own vacuity trap made explicit.

    Divisor 2's firing-hour surcharge (capacity_passthrough / 2 = 0.05) and
    divisor 1's (capacity_passthrough / 1 = 0.10) both clip to the same
    deferral factor of 1.0 (see docs/DECISIONS.md's "Why divisor 2 changes
    nothing physical" subsection), so the PHYSICAL columns must be exactly
    identical. Asserting only that would pass against a divisor that is
    silently inert (never applied at all), because an inert divisor would
    leave EVERY column, cost included, identical to divisor 1 too. The second
    half rules that out: cost must actually differ, on every paired seed, not
    merely on average, because cost is not saturated the way the physical
    response is. That second half is the only part of this invariant that
    proves the divisor was actually applied rather than doing nothing; a test
    that asserted only the first half would pass against a broken divisor
    that did nothing at all.
    """
    divisor2 = _rows_by_seed(dose_matched_df, k, "capacity_both", "capacity_surcharge_divisor", 2)
    divisor1 = _rows_by_seed(monopoly_df, k, "capacity_both", "broker_count", 1)

    assert len(divisor2) == 30 and len(divisor1) == 30
    assert list(divisor2["seed"]) == list(divisor1["seed"]), (
        f"k={k}: seed ordering must match before a positional comparison is meaningful"
    )

    physical_columns = ("total_deferred_kwh", "feeder_peak_to_average_ratio", "feeder_coefficient_of_variation")
    for column in physical_columns:
        assert (divisor2[column].to_numpy() == divisor1[column].to_numpy()).all(), (
            f"k={k}, {column}: divisor 2 is not bit-identical to divisor 1"
        )

    # The half that rules out a divisor that does nothing at all: every one
    # of the 30 paired seeds must show a strictly different cost, not merely
    # a different mean (a mean can differ while individual rows coincide).
    cost2 = divisor2["avg_cost_per_agent_eur"].to_numpy()
    cost1 = divisor1["avg_cost_per_agent_eur"].to_numpy()
    assert (cost2 != cost1).all(), f"k={k}: divisor 2's cost must differ from divisor 1's, or the divisor is inert"


def test_dose_matched_monopoly_divisor_2_matches_divisor_1_on_physical_metrics_but_differs_on_cost():
    _skip_unless_results_artifact_available(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    _skip_unless_results_artifact_available(SWEEP_MONOPOLY_PARQUET_PATH)

    import pandas as pd

    dose_matched_df = pd.read_parquet(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    monopoly_df = pd.read_parquet(SWEEP_MONOPOLY_PARQUET_PATH)

    for k in (0.5, 1.0):
        _assert_divisor2_equals_divisor1_but_cost_differs(dose_matched_df, monopoly_df, k)


def test_dose_matched_monopoly_divisor_2_vs_divisor_1_bites_on_a_nudged_row_and_on_a_simulated_inert_divisor():
    """Two bites on the same invariant, each closing a different half.

    First, the identity half: a divisor-2 row nudged so it no longer equals
    divisor 1 on a physical column. Second, and this is the vacuity-trap
    check R2/D11 both call for: a copy where divisor 2's cost is overwritten,
    seed for seed, to equal divisor 1's exactly, simulating a divisor that
    was silently never applied at all. The identity half alone would NOT
    catch this second case (the physical columns are untouched and still
    match); only the "cost must differ" half does, which is exactly why both
    halves are asserted rather than only the first.
    """
    _skip_unless_results_artifact_available(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    _skip_unless_results_artifact_available(SWEEP_MONOPOLY_PARQUET_PATH)

    import pandas as pd

    dm_digest_before = _sha256(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    mono_digest_before = _sha256(SWEEP_MONOPOLY_PARQUET_PATH)
    dose_matched_df = pd.read_parquet(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    monopoly_df = pd.read_parquet(SWEEP_MONOPOLY_PARQUET_PATH)

    _assert_divisor2_equals_divisor1_but_cost_differs(dose_matched_df, monopoly_df, 0.5)  # unperturbed: passes

    # (a) the identity half: a divisor-2 row nudged off divisor 1.
    nudged = dose_matched_df.copy(deep=True)
    target = nudged.index[
        (nudged["k"] == 0.5) & (nudged["ablation"] == "capacity_both") & (nudged["capacity_surcharge_divisor"] == 2)
    ]
    assert len(target) == 30
    nudged.loc[target[0], "total_deferred_kwh"] += 1.0
    with pytest.raises(AssertionError):
        _assert_divisor2_equals_divisor1_but_cost_differs(nudged, monopoly_df, 0.5)

    # (b) the vacuity trap: simulate a divisor that does nothing at all, by
    #     overwriting divisor 2's cost column with divisor 1's, seed for seed
    #     (mapped by seed, not position, so this is correct regardless of
    #     either dataframe's row order). The physical columns are untouched,
    #     so only the "cost must differ" half of the invariant catches this.
    inert = dose_matched_df.copy(deep=True)
    target = inert.index[
        (inert["k"] == 0.5) & (inert["ablation"] == "capacity_both") & (inert["capacity_surcharge_divisor"] == 2)
    ]
    assert len(target) == 30
    reference = monopoly_df[
        (monopoly_df["k"] == 0.5) & (monopoly_df["ablation"] == "capacity_both") & (monopoly_df["broker_count"] == 1)
    ]
    cost_by_seed = reference.set_index("seed")["avg_cost_per_agent_eur"]
    inert.loc[target, "avg_cost_per_agent_eur"] = inert.loc[target, "seed"].map(cost_by_seed).to_numpy()
    with pytest.raises(AssertionError):
        _assert_divisor2_equals_divisor1_but_cost_differs(inert, monopoly_df, 0.5)

    assert _sha256(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH) == dm_digest_before, (
        "bite test must not touch the parquet on disk"
    )
    assert _sha256(SWEEP_MONOPOLY_PARQUET_PATH) == mono_digest_before, "bite test must not touch the parquet on disk"


# ---------------------------------------------------------------------------
# 18. Pinned digest of results/surcharge_mode_comparison.csv (D10's
#     surcharge-distribution control arm, scripts/analyze_surcharge_mode.py):
#     every data row and every column, keyed by (scope,
#     capacity_surcharge_mode, k, broker_count, metric). Pinned WHOLE (24
#     rows). See SURCHARGE_MODE_COMPARISON_IDENTITY/_PINNED_FIELDS's own
#     comments above and the module docstring's item 18 paragraph for the
#     mutation-testing gap this closes. Read-only, skip-if-absent, loaded from
#     tests/golden/surcharge_mode_comparison_pins.json, same as every other
#     item above.
#
#     It was not always. This snapshot shipped as an inline json.loads literal
#     of about a thousand lines right here, because the fix that added
#     the pin was allowed to edit this module but not to add a file under
#     tests/golden/. The pin itself was fine; its STORAGE was not. It was the
#     one pin in this module `python scripts/regenerate_golden_master.py` could
#     not produce, so any legitimate change to that CSV meant hand-editing JSON
#     inside a test module, by hand, at a thousand lines. That script computes
#     it now (see _compute_surcharge_mode_comparison_pins there); nothing about
#     WHAT is pinned changed in the move.
# ---------------------------------------------------------------------------


def test_surcharge_mode_comparison_csv_matches_pinned_digest():
    _skip_unless_pinned_csv_available(SURCHARGE_MODE_COMPARISON_CSV_PATH, SURCHARGE_MODE_COMPARISON_GOLDEN_PATH)

    import pandas as pd

    golden = _load_golden(SURCHARGE_MODE_COMPARISON_GOLDEN_PATH)
    df = pd.read_csv(SURCHARGE_MODE_COMPARISON_CSV_PATH)
    _assert_surcharge_mode_comparison_matches_golden(df, golden)


def test_surcharge_mode_comparison_pin_bites_on_plausible_regressions():
    """Prove the pin above is not vacuous, on in-memory copies only.

    Five regressions specific to this file: a mode_cell percent change's sign
    flipping (a reported improvement becomes a reported worsening at the same
    magnitude, exactly GAP 2's net-export-floor shape at the analysis-output
    level rather than the mechanism level), a mode_cell survives_holm_alpha05
    flag flipping, a gradient_bc2_to_bc5 row's own gradient_pct_points_bc2_to_bc5
    drifting by 1 percent (the number docs/DECISIONS.md's D10 result and
    06_results.md Section 6.9 quote directly), a row's identity silently
    colliding with another row's (two mode_cell rows sharing a broker_count and
    mode, relabelled onto the same k, which _rows_by_identity's own duplicate-key
    guard must catch), and a row disappearing. The CSV's own digest is captured
    before and re-checked after, so the file on disk is provably untouched.
    """
    _skip_unless_pinned_csv_available(SURCHARGE_MODE_COMPARISON_CSV_PATH, SURCHARGE_MODE_COMPARISON_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(SURCHARGE_MODE_COMPARISON_CSV_PATH)
    golden = _load_golden(SURCHARGE_MODE_COMPARISON_GOLDEN_PATH)
    df = pd.read_csv(SURCHARGE_MODE_COMPARISON_CSV_PATH)

    _assert_surcharge_mode_comparison_matches_golden(df, golden)  # unperturbed: passes

    # (a) a mode_cell percent change's sign flips.
    flipped = df.copy(deep=True)
    mode_cell_nonzero = flipped.index[
        (flipped["scope"] == "mode_cell")
        & (flipped["feeder_peak_to_average_ratio_paired_mean_pct_change"].abs() > 1e-6)
    ]
    assert len(mode_cell_nonzero) > 0, "expected at least one non-degenerate mode_cell row"
    row = mode_cell_nonzero[0]
    flipped.loc[row, "feeder_peak_to_average_ratio_paired_mean_pct_change"] = -flipped.loc[
        row, "feeder_peak_to_average_ratio_paired_mean_pct_change"
    ]
    with pytest.raises(AssertionError):
        _assert_surcharge_mode_comparison_matches_golden(flipped, golden)

    # (b) a mode_cell Holm survival flag flips.
    flag_flipped = df.copy(deep=True)
    surviving = flag_flipped.index[
        (flag_flipped["scope"] == "mode_cell")
        & flag_flipped["feeder_peak_to_average_ratio_survives_holm_alpha05"].eq(True)
    ]
    assert len(surviving) > 0, "expected at least one mode_cell row surviving Holm correction"
    flag_flipped.loc[surviving[0], "feeder_peak_to_average_ratio_survives_holm_alpha05"] = False
    with pytest.raises(AssertionError):
        _assert_surcharge_mode_comparison_matches_golden(flag_flipped, golden)

    # (c) a gradient_bc2_to_bc5 row's own headline gradient number drifts by 1
    #     percent: small enough to pass a casual eyeball check, and this is
    #     the exact figure docs/DECISIONS.md's D10 result and
    #     06_results.md Section 6.9 quote (the synchronized -7.93/-8.37 pp
    #     numbers GAP 3's brief names directly).
    drifted = df.copy(deep=True)
    gradient_rows = drifted.index[drifted["scope"] == "gradient_bc2_to_bc5"]
    assert len(gradient_rows) == 6
    row = gradient_rows[0]
    drifted.loc[row, "gradient_pct_points_bc2_to_bc5"] *= 1.01
    with pytest.raises(AssertionError):
        _assert_surcharge_mode_comparison_matches_golden(drifted, golden)

    # (d) a row's identity silently collides with another row's: two mode_cell
    #     rows sharing (capacity_surcharge_mode, broker_count), relabelled from
    #     one shipped k level onto the other, which _rows_by_identity's own
    #     duplicate-key guard must catch.
    collided = df.copy(deep=True)
    prop_bc2_rows = collided.index[
        (collided["scope"] == "mode_cell")
        & (collided["capacity_surcharge_mode"] == "proportional")
        & (collided["broker_count"] == 2)
    ]
    assert len(prop_bc2_rows) == 2, "expected exactly one k=0.5 and one k=1.0 row for this (scope, mode, broker_count)"
    collided.loc[prop_bc2_rows[0], "k"] = collided.loc[prop_bc2_rows[1], "k"]
    with pytest.raises(AssertionError):
        _assert_surcharge_mode_comparison_matches_golden(collided, golden)

    # (e) a row disappears.
    dropped = df.drop(index=df.index[-1])
    with pytest.raises(AssertionError):
        _assert_surcharge_mode_comparison_matches_golden(dropped, golden)

    assert _sha256(SURCHARGE_MODE_COMPARISON_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


def test_surcharge_mode_comparison_pin_bites_on_a_p_value_magnitude_shift():
    """A p-value moving far in magnitude while staying far below the old
    absolute floor: under the previous rel=1e-9/abs=1e-9 comparison (either
    tolerance sufficing) this exact perturbation PASSED, because both values
    sit under the absolute floor. The assertion inside reproduces that old
    rule and shows it calling the two values equal, immediately before the
    current rule rejects them.
    """
    _skip_unless_pinned_csv_available(SURCHARGE_MODE_COMPARISON_CSV_PATH, SURCHARGE_MODE_COMPARISON_GOLDEN_PATH)

    import pandas as pd

    digest_before = _sha256(SURCHARGE_MODE_COMPARISON_CSV_PATH)
    golden = _load_golden(SURCHARGE_MODE_COMPARISON_GOLDEN_PATH)
    df = pd.read_csv(SURCHARGE_MODE_COMPARISON_CSV_PATH)

    _assert_surcharge_mode_comparison_matches_golden(df, golden)  # unperturbed: passes

    moved = df.copy(deep=True)
    column = "feeder_coefficient_of_variation_paired_p"
    nonzero = moved.index[moved[column] > 0.0]
    row = moved.loc[nonzero, column].idxmin()
    original = float(moved.loc[row, column])
    assert original < 1e-30, "expected a real p-value far below the old absolute floor"
    perturbed_to = 1e-20  # still hopelessly significant, still far below 1e-9
    assert perturbed_to == pytest.approx(original, rel=CSV_PIN_REL_TOL, abs=CSV_PIN_ABS_TOL), (
        f"the old rule really did call {original!r} and {perturbed_to!r} equal"
    )
    moved.loc[row, column] = perturbed_to
    with pytest.raises(AssertionError):
        _assert_surcharge_mode_comparison_matches_golden(moved, golden)

    assert _sha256(SURCHARGE_MODE_COMPARISON_CSV_PATH) == digest_before, "bite test must not touch the CSV on disk"


SURCHARGE_MODE_COMPARISON_COLUMNS = (
    "scope",
    "capacity_surcharge_mode",
    "k",
    "broker_count",
    "n_seeds",
    "feeder_peak_to_average_ratio_paired_mean_pct_change",
    "feeder_peak_to_average_ratio_paired_dz",
    "feeder_peak_to_average_ratio_paired_p",
    "feeder_peak_to_average_ratio_p_holm",
    "feeder_peak_to_average_ratio_p_bh",
    "feeder_peak_to_average_ratio_survives_holm_alpha05",
    "feeder_coefficient_of_variation_paired_mean_pct_change",
    "feeder_coefficient_of_variation_paired_dz",
    "feeder_coefficient_of_variation_paired_p",
    "feeder_coefficient_of_variation_p_holm",
    "feeder_coefficient_of_variation_p_bh",
    "feeder_coefficient_of_variation_survives_holm_alpha05",
    "total_deferred_kwh_capacity_both_mean",
    "total_deferred_kwh_capacity_disabled_mean",
    "correction_family",
    "metric3_sign_convention",
    "notes",
    "metric",
    "gradient_pct_at_bc2",
    "gradient_pct_at_bc5",
    "gradient_pct_points_bc2_to_bc5",
    "gradient_dz",
    "gradient_p_uncorrected",
    "gradient_p_holm",
    "gradient_p_bh",
    "gradient_survives_holm_alpha05",
    "gradient_survives_bh_alpha05",
    "gradient_degenerate",
)


def test_surcharge_mode_comparison_csv_column_set_matches_pinned():
    _skip_unless_results_artifact_available(SURCHARGE_MODE_COMPARISON_CSV_PATH)
    import pandas as pd

    header = pd.read_csv(SURCHARGE_MODE_COMPARISON_CSV_PATH, nrows=0).columns
    _assert_exact_column_set(header, SURCHARGE_MODE_COMPARISON_COLUMNS, "results/surcharge_mode_comparison.csv")


def test_surcharge_mode_comparison_csv_column_set_pin_bites_on_growth_and_shrinkage():
    _skip_unless_results_artifact_available(SURCHARGE_MODE_COMPARISON_CSV_PATH)
    import pandas as pd

    header = list(pd.read_csv(SURCHARGE_MODE_COMPARISON_CSV_PATH, nrows=0).columns)
    with pytest.raises(AssertionError):
        _assert_exact_column_set(
            header + ["a_new_column"], SURCHARGE_MODE_COMPARISON_COLUMNS, "results/surcharge_mode_comparison.csv"
        )
    with pytest.raises(AssertionError):
        _assert_exact_column_set(
            header[:-1], SURCHARGE_MODE_COMPARISON_COLUMNS, "results/surcharge_mode_comparison.csv"
        )


# ---------------------------------------------------------------------------
# Regeneration guard: a plain pytest run must never rewrite the golden files.
# scripts/regenerate_golden_master.py sits outside pyproject.toml's testpaths
# ("tests"), so pytest's collector never looks at it and this module never
# imports it either; the check below goes one step further and proves that
# even DIRECTLY importing that script's code (as this test does, to reach its
# module-level constants and functions) does not write anything, because the
# write logic is gated behind `if __name__ == "__main__":`, which is false
# for any import.
#
# The two tests after it use the same import for a different purpose: this
# module and that script carry TWIN definitions of the same things (a
# _metrics_dict, and one IDENTITY/PINNED_FIELDS pair per pinned CSV), and
# every one of those twins is only held together by a comment saying "must
# match". Phase 19 found out what that is worth: the three capacity audit
# fields were added to this module's _metrics_dict and not to the script's, so
# the documented regeneration command stripped six pinned values out of
# tests/golden/short_deterministic_run.json and the suite still passed. The
# comments are still there, and they are still not enforcement; these two
# tests are.
# ---------------------------------------------------------------------------


def _import_regeneration_script():
    """scripts/regenerate_golden_master.py, imported as a module.

    That script's top level inserts into sys.path and installs a warnings
    filter, both process-global and both otherwise outliving the caller for
    the rest of the pytest session. Both are undone here: a test that reaches
    into another module should not leave side effects of its own. Importing it
    writes nothing, which is not assumed but proved by
    test_regeneration_script_import_does_not_write_golden_files below.
    """
    spec = importlib.util.spec_from_file_location("_golden_regen_import_probe", REGENERATE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys_path_before = list(sys.path)
    try:
        with warnings.catch_warnings():
            spec.loader.exec_module(module)  # executes top-level code; module.__name__ != "__main__" here
    finally:
        sys.path[:] = sys_path_before
    return module


def test_the_two_metrics_dict_twins_emit_the_same_keys():
    """_metrics_dict exists twice, here and in the regeneration script, and
    the golden file is only as strong as their AGREEMENT: this module decides
    what is compared, that script decides what is stored, and a key present in
    one but not the other is an unpinned metric either way.

    Fed one MetricsResult with a distinct value in every field, so this
    compares the produced dicts whole rather than only their key sets: a twin
    that read the right key off the wrong attribute would keep the key sets
    equal and still be wrong.
    """
    module = _import_regeneration_script()

    # Distinct, recognisable values, none of them equal to another, so any
    # crossed attribute shows up as a value mismatch rather than cancelling out.
    result = MetricsResult(
        avg_cost_per_agent_eur=1.0,
        avg_cost_per_kwh_eur=2.0,
        broker_load_share={"a": 0.25, "b": 0.75},
        load_concentration_hhi=3.0,
        feeder_coefficient_of_variation=4.0,
        feeder_peak_to_average_ratio=5.0,
        feeder_mean_hourly_ramp_kwh=6.0,
        prosumer_self_sufficiency=7.0,
        total_capacity_charge_eur=8.0,
        capacity_fire_rate=9.0,
        mean_broker_surcharge_eur_per_kwh={"a": 0.1},
        total_deferred_kwh=10.0,
    )

    here = _metrics_dict(result)
    there = module._metrics_dict(result)
    assert set(here) == set(there), (
        "the two _metrics_dict twins no longer emit the same keys "
        f"(only in tests/test_golden_master.py: {sorted(set(here) - set(there))}; "
        f"only in scripts/regenerate_golden_master.py: {sorted(set(there) - set(here))}). "
        "A key this module compares but that script does not write vanishes from "
        "tests/golden/short_deterministic_run.json on the next regeneration; a key that "
        "script writes but this module does not compare is stored and never checked. "
        "Add it to BOTH."
    )
    assert here == there, "the two _metrics_dict twins emit the same keys but disagree on a value"


# Every constant this module and the regeneration script both define under one
# of these suffixes describes a pinned snapshot's SHAPE (which columns identify
# a row, which columns are pinned and how each is compared). Matched by suffix
# rather than by a hand-written list of names, so a pin added later is covered
# without anyone remembering to extend this test.
_TWINNED_CONSTANT_SUFFIXES = (
    "_IDENTITY",
    "_PINNED_FIELDS",
    "_CELL_FIELDS",
    "_CONSTANT_FIELDS",
    "_PINNED_ROW_KEYS",
    "_COLUMNS",
)


def test_the_twinned_pin_shape_constants_agree_between_this_module_and_the_script():
    """Same failure mode as the _metrics_dict twins, one level up: if this
    module pins a column the script does not write (or the reverse), the
    golden file and the check over it stop describing the same thing.

    Only names DEFINED IN BOTH are compared, so a constant that legitimately
    lives in one place alone (this module's whole-header *_COLUMNS tuples, for
    instance, which are checked against the CSV header directly and have no
    counterpart in the script) is not dragged in.
    """
    module = _import_regeneration_script()
    here = globals()

    shared = sorted(
        name
        for name in here
        if name.endswith(_TWINNED_CONSTANT_SUFFIXES) and hasattr(module, name)
    )
    assert shared, "expected at least one twinned pin-shape constant"
    # The pins this module actually compares against a golden file the script
    # writes. Named explicitly so that a twin QUIETLY DISAPPEARING from one
    # side (which would just shrink `shared` above) fails here instead of
    # silently reducing what this test covers.
    for expected_name in (
        "SUMMARY_STATS_IDENTITY",
        "SUMMARY_STATS_PINNED_FIELDS",
        "STRUCTURAL_SENSITIVITY_IDENTITY",
        "STRUCTURAL_SENSITIVITY_PINNED_FIELDS",
        "DEMAND_SOURCE_IDENTITY",
        "DEMAND_SOURCE_PINNED_FIELDS",
        "MONOPOLY_IDENTITY",
        "MONOPOLY_PINNED_FIELDS",
        "EFFECT_SIZES_IDENTITY",
        "EFFECT_SIZES_PINNED_FIELDS",
        "SUMMARY_STATS_CORRECTED_IDENTITY",
        "SUMMARY_STATS_CORRECTED_PINNED_FIELDS",
        "FAMILY_SENSITIVITY_IDENTITY",
        "FAMILY_SENSITIVITY_PINNED_FIELDS",
        "DOSE_MATCHED_MONOPOLY_IDENTITY",
        "DOSE_MATCHED_MONOPOLY_PINNED_FIELDS",
        "DOSE_MATCHED_MONOPOLY_PARQUET_CELL_IDENTITY",
        "DOSE_MATCHED_MONOPOLY_PARQUET_CELL_FIELDS",
        "SURCHARGE_MODE_COMPARISON_IDENTITY",
        "SURCHARGE_MODE_COMPARISON_PINNED_FIELDS",
    ):
        assert expected_name in shared, f"twinned pin-shape constant no longer defined in both: {expected_name}"

    for name in shared:
        assert here[name] == getattr(module, name), (
            f"{name} differs between tests/test_golden_master.py and "
            f"scripts/regenerate_golden_master.py: this module has {here[name]!r}, "
            f"that script has {getattr(module, name)!r}. The snapshot that script "
            "writes and the check this module runs must describe the same columns."
        )


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
        EFFECT_SIZES_GOLDEN_PATH,
        SUMMARY_STATS_CORRECTED_GOLDEN_PATH,
        FAMILY_SENSITIVITY_GOLDEN_PATH,
        DOSE_MATCHED_MONOPOLY_GOLDEN_PATH,
        SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_CELL_GOLDEN_PATH,
        SURCHARGE_MODE_COMPARISON_GOLDEN_PATH,
    ):
        if known_path.is_file():
            assert known_path in golden_paths, f"golden file missing from the no-rewrite snapshot: {known_path}"
    mtimes_before = {path: path.stat().st_mtime_ns for path in golden_paths}

    module = _import_regeneration_script()

    assert module.__name__ != "__main__"
    assert hasattr(module, "main"), "regeneration entry point must be a function, not top-level code"

    mtimes_after = {path: path.stat().st_mtime_ns for path in golden_paths}
    assert mtimes_after == mtimes_before, "importing the regeneration script rewrote a golden file"
