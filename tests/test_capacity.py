"""Tests for the Phase 3 capacity mechanism (D6): the CapacityMechanism unit,
its integration into MicrogridModel behind the master flag, and the four
ablations' observable effects."""

import copy
import statistics

import pytest

from microgrid_sim.agents.consumer import Consumer
from microgrid_sim.brokers.base import Broker
from microgrid_sim.config.loader import ConfigError, load_config, validate_config
from microgrid_sim.environment.capacity import CapacityMechanism
from microgrid_sim.environment.model import MicrogridModel

SHORT_HORIZON = 168  # one week, matching the other fast unit-test conventions


# ---------------------------------------------------------------------------
# config/loader.py validation of the optional capacity_mechanism block
# ---------------------------------------------------------------------------


def _valid_capacity_block():
    return {
        "enabled": True,
        "feedback_pnl": True,
        "feedback_pricing": True,
        "window": 168,
        "k": 1.0,
        "charge_rate_eur_per_kwh": 0.15,
        "capacity_passthrough": 0.10,
        "response_reference_eur_per_kwh": 0.05,
        "deferrable_fraction": 0.2,
        "payback_cap_fraction": 0.5,
    }


def test_capacity_mechanism_block_absent_is_valid():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    del config["capacity_mechanism"]
    validate_config(config)  # must not raise


def test_capacity_mechanism_block_valid_when_present():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"] = _valid_capacity_block()
    validate_config(config)  # must not raise


def test_capacity_mechanism_rejects_window_below_one():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"] = _valid_capacity_block()
    config["capacity_mechanism"]["window"] = 0
    with pytest.raises(ConfigError):
        validate_config(config)


def test_capacity_mechanism_rejects_negative_k():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"] = _valid_capacity_block()
    config["capacity_mechanism"]["k"] = -1.0
    with pytest.raises(ConfigError):
        validate_config(config)


def test_capacity_mechanism_rejects_missing_key():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"] = _valid_capacity_block()
    del config["capacity_mechanism"]["capacity_passthrough"]
    with pytest.raises(ConfigError):
        validate_config(config)


def test_capacity_mechanism_rejects_non_boolean_enabled():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"] = _valid_capacity_block()
    config["capacity_mechanism"]["enabled"] = "yes"
    with pytest.raises(ConfigError):
        validate_config(config)


def test_capacity_mechanism_rejects_deferrable_fraction_out_of_range():
    """D7: deferrable_fraction is a fraction of base demand, must be in [0, 1]."""
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"] = _valid_capacity_block()
    config["capacity_mechanism"]["deferrable_fraction"] = 1.5
    with pytest.raises(ConfigError):
        validate_config(config)


def test_capacity_mechanism_rejects_negative_deferrable_fraction():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"] = _valid_capacity_block()
    config["capacity_mechanism"]["deferrable_fraction"] = -0.1
    with pytest.raises(ConfigError):
        validate_config(config)


def test_capacity_mechanism_rejects_payback_cap_fraction_out_of_range():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"] = _valid_capacity_block()
    config["capacity_mechanism"]["payback_cap_fraction"] = 1.1
    with pytest.raises(ConfigError):
        validate_config(config)


def test_capacity_mechanism_rejects_non_positive_response_reference():
    """D7: response_reference_eur_per_kwh is a divisor in the deferral clip
    formula, so it must be strictly positive (tightened from the previous
    'non-negative' bound now that Consumer/Prosumer both divide by it)."""
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"] = _valid_capacity_block()
    config["capacity_mechanism"]["response_reference_eur_per_kwh"] = 0.0
    with pytest.raises(ConfigError):
        validate_config(config)


def test_all_four_ablation_scenario_files_load_and_validate():
    expected_flags = {
        "capacity_disabled": (False, False, False),
        "capacity_pnl_only": (True, True, False),
        "capacity_pricing_only": (True, False, True),
        "capacity_both": (True, True, True),
    }
    for name, (enabled, feedback_pnl, feedback_pricing) in expected_flags.items():
        config = load_config(f"config/scenarios/{name}.yaml")
        capacity = config["capacity_mechanism"]
        assert capacity["enabled"] is enabled
        assert capacity["feedback_pnl"] is feedback_pnl
        assert capacity["feedback_pricing"] is feedback_pricing
        # Structural coefficients stay fixed across ablations (only the flags vary).
        assert capacity["window"] == 168
        assert capacity["k"] == 1.0
        # D7 guardrail: the deferral coefficients are representative structural
        # values too, held fixed across the ablation/k sweep.
        assert capacity["deferrable_fraction"] == pytest.approx(0.2)
        assert capacity["payback_cap_fraction"] == pytest.approx(0.5)
        assert capacity["response_reference_eur_per_kwh"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# CapacityMechanism unit tests
# ---------------------------------------------------------------------------


def test_window_not_filled_total_charge_is_zero():
    mechanism = CapacityMechanism(window=4, k=1.0, charge_rate_eur_per_kwh=0.5, capacity_passthrough=0.1)
    for value in (10.0, 100.0, 1000.0):  # only 3 samples for a window of 4
        result = mechanism.step(value, {"a": value})
        assert result.threshold_kwh is None
        assert result.excess_kwh == 0.0
        assert result.total_charge_eur == 0.0
        assert result.allocations_eur == {"a": 0.0}
        assert result.surcharges_eur_per_kwh == {"a": 0.0}


def test_feeder_net_import_at_or_below_threshold_gives_zero_excess_and_charge():
    mechanism = CapacityMechanism(window=3, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=1.0)
    # Fill the window with identical values: std = 0, threshold = mean = value,
    # so every subsequent identical value sits exactly AT (not above) threshold.
    for _ in range(5):
        result = mechanism.step(20.0, {"a": 20.0})
        assert result.excess_kwh == 0.0
        assert result.total_charge_eur == 0.0


def test_feeder_net_import_above_threshold_gives_exact_excess_and_charge():
    """Constructed feeder series; expected threshold/excess/charge computed
    independently (via the stdlib statistics module, not by re-deriving the
    mechanism's own code path) and asserted to tight tolerance."""
    window = 3
    k = 1.0
    charge_rate = 2.0
    mechanism = CapacityMechanism(window=window, k=k, charge_rate_eur_per_kwh=charge_rate, capacity_passthrough=0.1)

    # Fill the window with three identical values (excess must be 0 while doing so).
    for _ in range(window):
        result = mechanism.step(20.0, {"a": 20.0})
        assert result.total_charge_eur == 0.0

    # Now push a spike that evicts the oldest 20.0, leaving the trailing
    # window [20.0, 20.0, 80.0].
    series = [20.0, 20.0, 80.0]
    result = mechanism.step(80.0, {"a": 80.0})

    expected_mean = statistics.fmean(series)
    expected_std = statistics.pstdev(series)
    expected_threshold = expected_mean + k * expected_std
    expected_excess = max(0.0, series[-1] - expected_threshold)
    expected_charge = charge_rate * expected_excess

    assert expected_excess > 0.0  # sanity: this construction must actually exceed threshold
    assert result.threshold_kwh == pytest.approx(expected_threshold, abs=1e-9)
    assert result.excess_kwh == pytest.approx(expected_excess, abs=1e-9)
    assert result.total_charge_eur == pytest.approx(expected_charge, abs=1e-9)


def test_zero_excess_hour_yields_zero_total_charge_exactly_at_boundary():
    """A hand-computable, fully exact boundary case: window=2, values [0, V]
    with k chosen so mean + k*std lands exactly on V (population std of a
    2-point window is always half the absolute difference), giving excess
    exactly 0.0, not merely close to it."""
    mechanism = CapacityMechanism(window=2, k=1.0, charge_rate_eur_per_kwh=5.0, capacity_passthrough=0.1)
    mechanism.step(0.0, {"a": 0.0})  # window not filled yet
    result = mechanism.step(10.0, {"a": 10.0})  # window = [0, 10]; mean=5, std=5, threshold=10
    assert result.threshold_kwh == pytest.approx(10.0, abs=1e-12)
    assert result.excess_kwh == 0.0
    assert result.total_charge_eur == 0.0


def test_allocations_sum_to_total_charge_when_positive_contribution_exists():
    window = 3
    mechanism = CapacityMechanism(window=window, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1)
    for _ in range(window):
        mechanism.step(40.0, {"a": 20.0, "b": 20.0, "c": 0.0})

    # Push a spike; contributions sum to the feeder total (80.0), with "c"
    # net-exporting this step (negative contribution, contributes 0 to the pool).
    result = mechanism.step(80.0, {"a": 50.0, "b": 30.0, "c": -10.0})

    assert result.total_charge_eur > 0.0
    assert sum(result.allocations_eur.values()) == pytest.approx(result.total_charge_eur, abs=1e-9)
    assert result.allocations_eur["c"] == 0.0  # non-positive contribution gets no allocation
    # The net-export zero floor applies to the PRICING channel too, not just
    # the P&L channel above: this mechanism defaults to surcharge_mode
    # "proportional", whose formula (capacity.py's else branch) is built from
    # positive_contrib, the same contribution dict clipped to >= 0 that
    # allocations_eur is built from, not the raw signed contributions dict. A
    # mutant that swapped positive_contrib for the raw signed dict in that
    # formula would give net-exporting "c" a NEGATIVE surcharge (passthrough
    # times a negative share) instead of leaving it excluded like its
    # allocation, and this assertion is what catches that.
    #
    # An earlier version of this comment claimed nothing else in this suite
    # read surcharges_eur_per_kwh for a net-exporting broker. That was wrong:
    # tests/test_surcharge_mode.py's
    # test_synchronized_gives_every_broker_an_equal_surcharge_including_zero_and_negative
    # already feeds a -15.0 contributor and pins its surcharge at 0.3/3. What
    # that test does NOT cover, and what this assertion adds, is the
    # PROPORTIONAL branch: synchronized never reads a contribution value at
    # all, so it cannot tell a clipped dict from a signed one. This line is
    # the only place in the suite where the proportional branch is asked what
    # a net exporter pays. (The divisor variant of the same branch is covered
    # in tests/test_surcharge_divisor.py, where the divisor lives.)
    assert result.surcharges_eur_per_kwh["c"] == 0.0


def test_synchronized_mode_does_not_exclude_a_net_exporting_broker():
    """Contrast with the proportional-mode exclusion pinned just above, at the
    identical net-export input: synchronized's formula (capacity.py's
    surcharge_mode == "synchronized" branch) assigns the SAME constant,
    capacity_passthrough / num_brokers, to every broker id and never reads a
    contribution value at all, so there is no zero clip anywhere in it and
    nothing for a negative contribution to be excluded by. A net exporter gets
    the same equal share as every other broker, as long as the step levied a
    charge at all.

    Labelled honestly: this is a LOCALITY DUPLICATE of
    tests/test_surcharge_mode.py's
    test_synchronized_gives_every_broker_an_equal_surcharge_including_zero_and_negative,
    kept here so the three modes' net-export behaviour can be read in one
    place next to each other, not because it covers a mutant that test does
    not. It does not: an adversarial review built the natural "synchronized
    also excludes non-positive contributors" mutant and watched the
    surcharge_mode test kill it on its own.

    An earlier version of this docstring said synchronized "iterates every
    broker_id, not positive_contrib". That is a distinction without a
    difference: capacity.py builds positive_contrib over every key of
    broker_contributions_kwh, so the two dicts always have the same key set.
    Synchronized differs by not consulting the VALUES, which is what the
    wording above now says.
    """
    window = 3
    mechanism = CapacityMechanism(
        window=window, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.3, surcharge_mode="synchronized"
    )
    for _ in range(window):
        mechanism.step(40.0, {"a": 20.0, "b": 20.0, "c": 0.0})
    result = mechanism.step(80.0, {"a": 50.0, "b": 30.0, "c": -10.0})  # "c" net-exports this step

    assert result.total_charge_eur > 0.0
    expected_share = 0.3 / 3  # capacity_passthrough / num_brokers, same for every broker id
    assert result.surcharges_eur_per_kwh["c"] == pytest.approx(expected_share, abs=1e-12)
    assert result.surcharges_eur_per_kwh["c"] == result.surcharges_eur_per_kwh["a"]


def test_renormalized_mode_excludes_a_net_exporting_broker_like_proportional():
    """Contrast with synchronized just above, at the same net-export input:
    renormalized's formula (capacity.py's surcharge_mode == "renormalized"
    branch) multiplies by a contribution taken from positive_contrib, whose
    values have been floored at zero by max(0.0, value), so the net-exporting
    broker "c" is multiplied by 0.0 and gets exactly 0.0 here too, not a share
    of the renormalized total.

    The exclusion is that CLIP ON THE VALUE, not the choice of dict to iterate.
    An earlier version of this docstring said renormalized "iterates
    positive_contrib" as though that were the operative difference; it is not,
    since positive_contrib is built over every key of broker_contributions_kwh
    and the two dicts always have the same key set. The thesis's own wording,
    "floor a broker's contribution at zero before taking shares", was right
    where that docstring was wrong.
    """
    window = 3
    mechanism = CapacityMechanism(
        window=window, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.3, surcharge_mode="renormalized"
    )
    for _ in range(window):
        mechanism.step(40.0, {"a": 20.0, "b": 20.0, "c": 0.0})
    result = mechanism.step(80.0, {"a": 50.0, "b": 30.0, "c": -10.0})  # "c" net-exports this step

    assert result.total_charge_eur > 0.0
    assert result.surcharges_eur_per_kwh["c"] == 0.0


def test_allocation_guard_zero_when_all_contributions_non_positive():
    window = 2
    mechanism = CapacityMechanism(window=window, k=0.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1)
    mechanism.step(10.0, {"a": -5.0, "b": -5.0})
    result = mechanism.step(50.0, {"a": -20.0, "b": -20.0})  # feeder positive overall is impossible here,
    # but exercise the guard directly: all per-broker contributions <= 0.
    assert result.total_charge_eur >= 0.0
    assert result.allocations_eur == {"a": 0.0, "b": 0.0}
    assert result.surcharges_eur_per_kwh == {"a": 0.0, "b": 0.0}


def test_surcharge_strictly_monotone_in_contribution_share():
    """A broker with a strictly larger positive contribution gets a strictly
    larger surcharge (D6's "cross-broker heterogeneity is preserved")."""
    window = 2
    mechanism = CapacityMechanism(window=window, k=0.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.2)
    mechanism.step(10.0, {"big": 5.0, "small": 5.0})
    result = mechanism.step(100.0, {"big": 70.0, "small": 30.0})

    assert result.total_charge_eur > 0.0
    assert result.surcharges_eur_per_kwh["big"] > result.surcharges_eur_per_kwh["small"] > 0.0
    # Exact formula check: surcharge_b = capacity_passthrough * contribution share.
    assert result.surcharges_eur_per_kwh["big"] == pytest.approx(0.2 * (70.0 / 100.0), abs=1e-12)
    assert result.surcharges_eur_per_kwh["small"] == pytest.approx(0.2 * (30.0 / 100.0), abs=1e-12)


def test_allocation_is_strictly_proportional_to_contribution_share_not_merely_ordered():
    """Phase 20 close-out (weakness 1): pins the STATED MECHANISM, not a reported
    number. docs/DECISIONS.md:1351 and docs/thesis/defense_slides.md:76 both assert
    the P&L allocation (capacity.py:173-176) is STRICTLY PROPORTIONAL to each
    broker's contribution share. No test in this suite asserted the RATIO before
    this one: test_allocations_sum_to_total_when_positive_contribution_exists checks
    only sum-to-total and the zero floor on a non-positive contributor, and
    test_surcharge_strictly_monotone_in_contribution_share pins the exact formula
    for the PRICING channel (surcharges_eur_per_kwh) on two brokers, not the P&L
    channel (allocations_eur) this test covers, and not on three DISTINCT shares
    where an equal-share coincidence could hide a wrong rule. An adversarial audit
    found the gap directly: replacing capacity.py:174's
    `contribution / sum_positive_contrib` with a squared-share rule
    (`contribution**2 / sum(contribution**2 for ...)`) preserves the sum-to-total
    property, the zero floor, and "brokers get different charges", and the whole
    469-test suite still passed. Verified standalone (outside the repo, against a
    copy of capacity.py, not the shipped file) that this test's own ratio assertion
    fails under exactly that substitution, and also under an equal-split rule
    (1/n): see the Phase 20 close-out report for the harness and its output.

    Three brokers, three DISTINCT contributions (10.0, 20.0, 30.0; sum 60.0),
    constructed so total_charge_eur is also 60.0 (charge_rate 1.0, excess 60.0 by
    the window/threshold setup below), so the shipped proportional rule returns
    the contributions themselves: (10.0, 20.0, 30.0). For contrast (not asserted,
    stated so a reader can see the rules disagree): the squared-share rule would
    give (60*100/1400, 60*400/1400, 60*900/1400) -- concretely (4.286, 17.143, 38.571); an
    equal split would give (20.0, 20.0, 20.0). All three preserve ordering
    (a < b < c, or tie for equal-split) and sum to 60.0; only the per-broker
    ratio -- allocation / total_charge compared against contribution / total
    contribution -- distinguishes them, which is exactly what this test checks,
    per broker, not in aggregate.

    Note honestly: no REPORTED number moves under the squared-share mutation.
    Per-broker cumulative_capacity_charge_eur is never written to a results CSV
    (only the run-level total_charge_eur and fire_rate are, via
    compute_capacity_audit), and debit_capacity_charge (the P&L channel this
    line feeds) touches neither broker revenue nor any quoted price. This test
    pins a STATED MECHANISM, not a figure that appears in the thesis text.
    """
    window = 2
    mechanism = CapacityMechanism(window=window, k=0.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1)
    mechanism.step(0.0, {"a": 0.0, "b": 0.0, "c": 0.0})  # fills the window; window-not-filled zeroes everything
    result = mechanism.step(120.0, {"a": 10.0, "b": 20.0, "c": 30.0})

    assert result.total_charge_eur == pytest.approx(60.0, abs=1e-9)  # construction sanity, not the claim itself
    total_contribution = 10.0 + 20.0 + 30.0

    for broker_id, contribution in (("a", 10.0), ("b", 20.0), ("c", 30.0)):
        expected_ratio = contribution / total_contribution
        actual_ratio = result.allocations_eur[broker_id] / result.total_charge_eur
        assert actual_ratio == pytest.approx(expected_ratio, abs=1e-12), (
            f"broker {broker_id}: allocation/total_charge ratio does not match contribution/total_contribution"
        )


# ---------------------------------------------------------------------------
# CapacityMechanism.fire_rate() directly. Everywhere else in this suite the
# fire rate is only ever read off a finished model run, and only ever asserted
# to be > 0.0 or == 0.0, which leaves both the zero-observation guard and the
# denominator itself untested: an external mutation-testing pass turned the
# guard into `return 1.0`, deleted it outright, and widened it to
# `hours_observed <= self.window`, and all three mutants passed the whole
# suite. The three tests below are the direct unit coverage that was missing.
# ---------------------------------------------------------------------------


def test_fire_rate_is_zero_on_a_fresh_mechanism_with_no_observed_hours():
    """Called before any step at all, which is the only state in which
    hours_observed is 0. Deleting the guard makes this raise ZeroDivisionError
    rather than return anything, and returning 1.0 instead of 0.0 makes a
    mechanism that has never run report as firing every hour."""
    mechanism = CapacityMechanism(window=168, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1)
    assert mechanism.hours_observed == 0
    assert mechanism.fire_rate() == 0.0


def test_fire_rate_is_zero_while_the_window_is_still_filling():
    """Hours ARE observed here, so the guard above is not what returns 0.0;
    the numerator is, because no charge can be levied until the rolling window
    is full (D6). Distinguishes "no hours yet" from "hours, but no firing"."""
    window = 4
    mechanism = CapacityMechanism(window=window, k=0.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1)
    for hour in range(window - 1):
        result = mechanism.step(10.0 * (hour + 1), {"a": 10.0 * (hour + 1)})
        assert result.threshold_kwh is None  # window not filled: nothing can fire
    assert mechanism.hours_observed == window - 1
    assert mechanism.hours_with_excess == 0
    assert mechanism.fire_rate() == 0.0


def test_fire_rate_denominator_includes_the_window_fill_hours():
    """The exact hour the window first fills: hours_observed == window, and
    that step DOES levy a charge, so the answer is 1/window and not 0.0. This
    is the case that separates the real guard from a widened
    `hours_observed <= self.window` version of it, which would still be inside
    its own early return here and would report 0.0 for a mechanism that has
    demonstrably fired. The docstring's promise that the fill hours count as
    non-firing rather than being excluded is the same claim: they stay in the
    denominator."""
    mechanism = CapacityMechanism(window=2, k=0.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1)
    mechanism.step(0.0, {"a": 0.0})  # window not filled yet
    result = mechanism.step(10.0, {"a": 10.0})  # window = [0, 10]; mean=5, k=0, so threshold=5
    assert result.total_charge_eur > 0.0
    assert mechanism.hours_observed == 2
    assert mechanism.hours_with_excess == 1
    assert mechanism.fire_rate() == 0.5


def test_capacity_mechanism_rejects_invalid_window_and_k():
    with pytest.raises(ValueError):
        CapacityMechanism(window=0, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1)
    with pytest.raises(ValueError):
        CapacityMechanism(window=1, k=-0.5, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1)


# ---------------------------------------------------------------------------
# Model-level integration tests
# ---------------------------------------------------------------------------


def _competitive_config(horizon=SHORT_HORIZON, num_agents=60, seed=3):
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = horizon
    config["simulation"]["seed"] = seed
    config["population"]["num_agents"] = num_agents
    return config


def _run(config, hours):
    model = MicrogridModel(copy.deepcopy(config))
    for _ in range(hours):
        model.step()
    return model


def test_master_flag_off_reproduces_baseline_byte_for_byte():
    """Running with the capacity_mechanism block entirely ABSENT must be
    identical to running with it explicitly present-and-disabled, and both
    must reproduce the plain (pre-Phase-3) baseline: identical four metrics
    AND identical feeder_net_import_history, byte-for-byte (== not approx)."""
    config_with_block = _competitive_config()
    assert config_with_block["capacity_mechanism"]["enabled"] is False  # default off (master flag)

    config_no_block = copy.deepcopy(config_with_block)
    del config_no_block["capacity_mechanism"]

    model_with_block = _run(config_with_block, SHORT_HORIZON)
    model_no_block = _run(config_no_block, SHORT_HORIZON)

    assert model_with_block.feeder_net_import_history == model_no_block.feeder_net_import_history

    metrics_with_block = model_with_block.compute_metrics()
    metrics_no_block = model_no_block.compute_metrics()
    assert metrics_with_block == metrics_no_block


def test_capacity_disabled_scenario_equals_no_capacity_run():
    config_no_block = _competitive_config()
    del config_no_block["capacity_mechanism"]
    model_no_block = _run(config_no_block, SHORT_HORIZON)

    disabled_config = load_config("config/scenarios/capacity_disabled.yaml")
    disabled_config = copy.deepcopy(disabled_config)
    disabled_config["simulation"]["horizon_hours"] = SHORT_HORIZON
    disabled_config["simulation"]["seed"] = disabled_config["simulation"]["seed"]
    disabled_config["population"]["num_agents"] = 60
    # match the seed used by _competitive_config's default (3)
    disabled_config["simulation"]["seed"] = 3
    model_disabled = _run(disabled_config, SHORT_HORIZON)

    assert model_disabled.feeder_net_import_history == model_no_block.feeder_net_import_history
    assert model_disabled.compute_metrics() == model_no_block.compute_metrics()


def _load_ablation(name, horizon=SHORT_HORIZON, num_agents=60, seed=11):
    config = load_config(f"config/scenarios/{name}.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = horizon
    config["simulation"]["seed"] = seed
    config["population"]["num_agents"] = num_agents
    return config


def test_four_ablations_produce_different_trajectories_on_the_same_seed():
    horizon = 500  # long enough to fill the default 168h window and see excess hours
    names = ["capacity_disabled", "capacity_pnl_only", "capacity_pricing_only", "capacity_both"]
    models = {name: _run(_load_ablation(name, horizon=horizon), horizon) for name in names}

    ledger_signatures = {
        name: tuple(
            (broker_id, round(broker.cumulative_revenue_eur, 9), round(broker.cumulative_capacity_charge_eur, 9))
            for broker_id, broker in model.brokers.items()
        )
        for name, model in models.items()
    }
    price_signatures = {name: tuple(model.feeder_net_import_history) for name, model in models.items()}

    combined_signatures = {name: (ledger_signatures[name], price_signatures[name]) for name in names}
    distinct_signatures = set(combined_signatures.values())
    assert len(distinct_signatures) > 1, "all four ablations produced identical trajectories"


def test_capacity_both_allocates_charge_unevenly_across_brokers_by_contribution():
    """Model-level check that the per-broker capacity allocation genuinely
    varies by broker (not spread evenly regardless of contribution) whenever
    the mechanism actually fires under a real, heterogeneous population; the
    CapacityMechanism-level test above already pins the exact
    surcharge-is-strictly-monotone-in-contribution-share formula on two
    constructed brokers, so this is the "real simulation" complement."""
    horizon = 500
    config = _load_ablation("capacity_both", horizon=horizon, num_agents=90, seed=5)
    model = _run(config, horizon)
    result = model.compute_metrics()

    assert result.capacity_fire_rate > 0.0, "construction should produce at least some excess hours"
    total_allocated = sum(broker.cumulative_capacity_charge_eur for broker in model.brokers.values())
    assert total_allocated > 0.0
    charges = [round(broker.cumulative_capacity_charge_eur, 6) for broker in model.brokers.values()]
    assert len(set(charges)) > 1, "capacity charge was spread identically across brokers"


def test_master_flag_off_means_demand_deferral_is_inert():
    """Phase 3b (D7): with the mechanism disabled, broker_surcharges must stay
    all-zero for the whole run, so the price-elastic demand-deferral channel
    (the current physical channel to metric 3, replacing D6's prosumer
    storage response) never activates for ANY agent, consumer or prosumer,
    and every agent's deferred_kwh bucket and total_deferred_kwh audit
    accumulator stay at zero throughout."""
    config = _competitive_config(horizon=48)
    model = MicrogridModel(config)
    for _ in range(48):
        model.step()
        assert all(value == 0.0 for value in model.broker_surcharges.values())
    for agent in model.agents:
        assert agent.deferred_kwh == 0.0
        assert agent.total_deferred_kwh == 0.0


def test_capacity_pnl_only_reproduces_plain_baseline_byte_for_byte():
    """D7 guardrail: capacity_pnl_only debits cumulative_capacity_charge_eur,
    an accumulator that is not part of any of the four MetricsResult metrics,
    and writes no surcharge into any quote (broker_surcharges stays all-zero),
    so it must reproduce the plain baseline's feeder_net_import_history and
    compute_metrics() output byte-for-byte, exactly like capacity_disabled."""
    config_no_block = _competitive_config()
    del config_no_block["capacity_mechanism"]
    model_no_block = _run(config_no_block, SHORT_HORIZON)

    pnl_only_config = _load_ablation("capacity_pnl_only", horizon=SHORT_HORIZON, num_agents=60, seed=3)
    model_pnl_only = _run(pnl_only_config, SHORT_HORIZON)

    assert model_pnl_only.feeder_net_import_history == model_no_block.feeder_net_import_history
    assert model_pnl_only.compute_metrics() == model_no_block.compute_metrics()
    for agent in model_pnl_only.agents:
        assert agent.deferred_kwh == 0.0
        assert agent.total_deferred_kwh == 0.0


def test_ablation_isolation_pnl_only_matches_disabled_pricing_diverges():
    """D7 ablation isolation: capacity_pnl_only's physical feeder series must
    be identical to capacity_disabled's (no surcharge => no deferral => same
    physical dispatch), while capacity_pricing_only (which DOES write a
    surcharge) is free to diverge."""
    horizon = 500
    disabled = _run(_load_ablation("capacity_disabled", horizon=horizon), horizon)
    pnl_only = _run(_load_ablation("capacity_pnl_only", horizon=horizon), horizon)
    pricing_only = _run(_load_ablation("capacity_pricing_only", horizon=horizon), horizon)

    assert pnl_only.feeder_net_import_history == disabled.feeder_net_import_history
    assert pricing_only.feeder_net_import_history != disabled.feeder_net_import_history


def test_total_deferred_kwh_audit_is_positive_under_pricing_and_zero_otherwise():
    """D7: the total_deferred_kwh audit quantity (sum of deferred energy over
    the run, across all agents) must be exactly 0.0 whenever the pricing
    channel never writes a positive surcharge (capacity_disabled,
    capacity_pnl_only), and strictly positive once it does and the deferral
    formula actually fires (capacity_pricing_only, capacity_both)."""
    horizon = 500

    def _metrics(name):
        return _run(_load_ablation(name, horizon=horizon, num_agents=90, seed=5), horizon).compute_metrics()

    disabled = _metrics("capacity_disabled")
    pnl_only = _metrics("capacity_pnl_only")
    pricing_only = _metrics("capacity_pricing_only")
    both = _metrics("capacity_both")

    assert pricing_only.capacity_fire_rate > 0.0  # sanity: construction actually fires
    assert disabled.total_deferred_kwh == 0.0
    assert pnl_only.total_deferred_kwh == 0.0
    assert pricing_only.total_deferred_kwh > 0.0
    assert both.total_deferred_kwh > 0.0


def test_capacity_pricing_only_never_debits_any_broker_ledger():
    """Phase 20 (20.1, mutant B): model.py step 5's `if self.capacity_feedback_pnl:`
    gate must keep capacity_pricing_only's P&L channel silent. Chapter 4
    Section 4.6 states plainly that under capacity_pricing_only "broker
    ledgers are never debited by the scarcity charge itself"; nothing else in
    this suite pins that sentence; test_capacity_pnl_only_reproduces_plain_
    baseline_byte_for_byte above only ever exercises the OTHER ablation
    (feedback_pnl True, feedback_pricing False). A test that never confirms
    the charge actually fires would pass vacuously (a mechanism that never
    levies anything also never debits anything), so capacity_fire_rate > 0.0
    is asserted first, then every broker's ledger is pinned at exactly 0.0."""
    horizon = 500
    config = _load_ablation("capacity_pricing_only", horizon=horizon, num_agents=90, seed=5)
    model = _run(config, horizon)
    metrics = model.compute_metrics()

    assert metrics.capacity_fire_rate > 0.0, "construction should produce at least some excess hours"
    for broker in model.brokers.values():
        assert broker.cumulative_capacity_charge_eur == 0.0


# ---------------------------------------------------------------------------
# Phase 20 (20.1, mutant A): model.py's capacity block, inside
# `if self.capacity_enabled:`, groups a step's broker_contributions_kwh by
# agent.last_broker_id, not by agent.broker.id. The two differ only for an
# agent that switches broker during its OWN step() call: last_broker_id is
# captured in Consumer.step() before the tail-of-step switching logic can
# move self.broker (see agents/consumer.py), so it records the broker that
# actually served (and billed) the agent this hour, while self.broker.id may
# already point at next hour's broker by the time MicrogridModel.step reads
# it. A test that only checks contributions sum to the feeder total cannot
# tell the two apart (the sum is identical either way); the fixtures below
# instead construct a single-step switch and read the resulting PER-BROKER
# capacity-charge split off the real broker ledgers.
#
# broker_count 2 (regulated_utility + volatile_low_cost, no premium_green) is
# used deliberately, mirroring BROKER_COUNTS's bc=2 cell in
# scripts/run_sensitivity_sweep.py: Phase 19's mutation-testing pass reported
# this exact mutant LIVE at broker_count 2, where switching is frequent (mean
# 21.37% of agents over the D8 sweep, docs/thesis/defense_slides.md Slide 13),
# and INERT at the broker_count 3 headline cell.
# ---------------------------------------------------------------------------


class _FixedPriceBroker(Broker):
    """Deterministic-price test double: quote() always returns the same fixed
    price, so a switching decision here depends only on the constructed
    scenario below, never on the real regulated_utility/volatile_low_cost
    brokers' seasonal or stochastic price dynamics."""

    def __init__(self, broker_id, price, greenness=0.5):
        super().__init__(broker_id, broker_id, "stub", greenness)
        self._price = price

    def quote(self, hour, context=None):
        self._record_price(self._price)
        return self._price


def _bc2_stub_model(horizon, capacity_overrides):
    """A broker_count-2 model (regulated_utility + volatile_low_cost ids,
    matching the real bc=2 sweep cell) with num_agents forced to 0 so agents
    can be attached by hand, and the two real brokers swapped for
    _FixedPriceBroker doubles reusing their real ids -- broker_surcharges and
    _surcharge_accum are both built keyed off self.brokers during __init__,
    before this swap, so reusing the same ids keeps them consistent."""
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = horizon
    config["simulation"]["seed"] = 1
    config["population"]["num_agents"] = 0
    config["brokers"] = [b for b in config["brokers"] if b["type"] in ("regulated_utility", "volatile_low_cost")]
    config["capacity_mechanism"] = {
        "enabled": True,
        "feedback_pnl": False,
        "feedback_pricing": False,
        "window": 2,
        "k": 0.0,
        "charge_rate_eur_per_kwh": 1.0,
        "capacity_passthrough": 0.1,
        "response_reference_eur_per_kwh": 0.05,
        # Deliberately 0.0: the demand-deferral channel (D7) is a separate
        # concern from this mutant, and deferrable_fraction 0.0 makes
        # _apply_demand_deferral inert regardless of the surcharge value, so
        # feeder_net_import here stays exactly the hand-set demand_profile,
        # not perturbed by whatever surcharge this scenario also produces.
        "deferrable_fraction": 0.0,
        "payback_cap_fraction": 0.0,
    }
    config["capacity_mechanism"].update(capacity_overrides)
    model = MicrogridModel(config)
    regulated = _FixedPriceBroker("regulated_utility", price=0.30)
    volatile = _FixedPriceBroker("volatile_low_cost", price=0.05)
    model.brokers["regulated_utility"] = regulated
    model.brokers["volatile_low_cost"] = volatile
    return model, regulated, volatile


def test_capacity_attributes_a_same_step_switcher_to_the_broker_that_served_it():
    """See the module-level note above this fixture for the mutant this
    kills. Two agents, three hours, window=2, k=0.0:

    Hours 0-1: demand_profile = 1.0 for both agents (feeder 2.0 each hour),
    filling the rolling window with no excess (mean == the just-observed
    value both times), so no charge is levied yet.

    Hour 2: demand_profile spikes to 5.0 for both agents (feeder 10.0);
    window = [2.0, 10.0], mean = threshold = 6.0 (k=0), excess = 4.0, total
    charge = 4.0 EUR (charge_rate 1.0). "switcher" has breached its own 0.25
    tolerance against "regulated"'s fixed 0.30 price for 3 consecutive hours
    (sustained_breach_hours=3), so it reconsiders and switches to "volatile"
    (0.05) inside this SAME step() call, after last_broker_id has already
    been captured as "regulated_utility" for this hour. "anchor" stays on
    "volatile" throughout (its tolerance never breaches), giving "volatile" a
    genuine contribution of its own to split against.

    Shipped code (last_broker_id): regulated served 5.0 kWh, volatile served
    5.0 kWh this hour -> allocations split 50/50, regulated gets 2.0 EUR,
    volatile gets 2.0 EUR.
    Mutant (agent.broker.id): switcher's kWh is attributed to volatile (its
    NEW broker) instead -> regulated gets 0.0 EUR, volatile gets 4.0 EUR.
    """
    model, regulated, volatile = _bc2_stub_model(horizon=3, capacity_overrides={"feedback_pnl": True})
    model.demand_profile[:3] = [1.0, 1.0, 5.0]

    switcher = Consumer(
        model,
        profile="price_sensitive",
        demand_scale=1.0,
        price_tolerance_eur_per_kwh=0.25,
        volatility_tolerance_eur_per_kwh=0.05,
        greenness_threshold=0.5,
        switching_penalty_eur_per_kwh=0.0,
        sustained_breach_hours=3,
        initial_broker=regulated,
    )
    anchor = Consumer(
        model,
        profile="price_sensitive",
        demand_scale=1.0,
        price_tolerance_eur_per_kwh=1.0,  # 0.05 (volatile's price) never breaches this
        volatility_tolerance_eur_per_kwh=0.05,
        greenness_threshold=0.5,
        switching_penalty_eur_per_kwh=0.0,
        sustained_breach_hours=3,
        initial_broker=volatile,
    )

    for _ in range(3):
        model.step()

    # Sanity: the constructed scenario actually exercises what the docstring
    # claims, before trusting the ledger assertions below.
    assert switcher.switch_count == 1
    assert switcher.broker is volatile
    assert switcher.last_broker_id == "regulated_utility"
    assert anchor.broker is volatile
    assert anchor.last_broker_id == "volatile_low_cost"

    assert regulated.cumulative_capacity_charge_eur == pytest.approx(2.0, abs=1e-9)
    assert volatile.cumulative_capacity_charge_eur == pytest.approx(2.0, abs=1e-9)
