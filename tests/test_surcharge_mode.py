"""Tests for the D10 surcharge-distribution control arm (capacity_surcharge_mode).

Covers the three CapacityMechanism.step() surcharge formulas directly
(proportional, synchronized, renormalized), the loader's validation of the
new optional config key, and two model-level guarantees D10 requires: the
mode is inert wherever the mechanism already is (capacity_disabled,
capacity_pnl_only), and none of the three modes touches the model's seeded
RNG stream.
"""

import copy

import pytest

from microgrid_sim.agents.prosumer import Prosumer
from microgrid_sim.config.loader import ConfigError, load_config, validate_config
from microgrid_sim.environment.capacity import SURCHARGE_MODES, CapacityMechanism
from microgrid_sim.environment.model import MicrogridModel

SHORT_HORIZON = 168

DISABLED_FLAGS = {"enabled": False, "feedback_pnl": False, "feedback_pricing": False}
PNL_ONLY_FLAGS = {"enabled": True, "feedback_pnl": True, "feedback_pricing": False}
BOTH_FLAGS = {"enabled": True, "feedback_pnl": True, "feedback_pricing": True}


def test_default_yaml_ships_proportional_mode():
    config = load_config("config/default.yaml")
    assert config["capacity_mechanism"]["capacity_surcharge_mode"] == "proportional"


def test_capacity_surcharge_mode_absent_is_valid():
    # optional, same as every other capacity_mechanism key that predates it:
    # a block written before D10 existed must still validate.
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    del config["capacity_mechanism"]["capacity_surcharge_mode"]
    validate_config(config)  # must not raise


@pytest.mark.parametrize("mode", SURCHARGE_MODES)
def test_capacity_surcharge_mode_accepts_all_known_values(mode):
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"]["capacity_surcharge_mode"] = mode
    validate_config(config)  # must not raise


def test_capacity_surcharge_mode_rejects_unknown_value_at_loader_level():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"]["capacity_surcharge_mode"] = "bogus_mode"
    with pytest.raises(ConfigError):
        validate_config(config)


def test_capacity_mechanism_rejects_unknown_surcharge_mode_at_construction():
    with pytest.raises(ValueError):
        CapacityMechanism(
            window=1, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1, surcharge_mode="bogus_mode"
        )


def _competitive_config(mode, horizon=SHORT_HORIZON, num_agents=60, seed=7, ablation_flags=BOTH_FLAGS):
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = horizon
    config["simulation"]["seed"] = seed
    config["population"]["num_agents"] = num_agents
    config["capacity_mechanism"]["capacity_surcharge_mode"] = mode
    config["capacity_mechanism"].update(ablation_flags)
    return config


def test_model_construction_raises_for_unknown_mode_when_mechanism_enabled():
    config = _competitive_config("bogus_mode", ablation_flags=BOTH_FLAGS)
    with pytest.raises(ValueError):
        MicrogridModel(config)


def _fired_mechanism(mode, contributions, window=2, k=0.0, charge_rate=1.0, passthrough=0.3):
    """Fill the window with a flat, non-firing baseline (equal to
    contributions' own total, so k=0 alone would not already guarantee
    firing), then push `contributions` as the step that fires."""
    mechanism = CapacityMechanism(
        window=window, k=k, charge_rate_eur_per_kwh=charge_rate, capacity_passthrough=passthrough, surcharge_mode=mode
    )
    baseline = {broker_id: 1.0 for broker_id in contributions}
    for _ in range(window):
        mechanism.step(sum(baseline.values()), baseline)
    return mechanism.step(sum(contributions.values()), contributions)


def test_proportional_is_bit_identical_to_pre_change_behaviour():
    """Pins the exact pre-D10 formula and total: surcharge_b = passthrough *
    share_b, summing to passthrough."""
    contributions = {"a": 60.0, "b": 30.0, "c": 10.0}
    total = sum(contributions.values())
    result = _fired_mechanism("proportional", contributions, passthrough=0.2)
    assert result.total_charge_eur > 0.0
    # Exact equality, not approx: the per-broker value is the identical
    # float expression the pre-D10 code evaluated, so any tolerance here
    # would let a changed order of operations pass as bit identical.
    assert result.surcharges_eur_per_kwh["a"] == 0.2 * (60.0 / total)
    assert result.surcharges_eur_per_kwh["b"] == 0.2 * (30.0 / total)
    assert result.surcharges_eur_per_kwh["c"] == 0.2 * (10.0 / total)
    # The SUM keeps a tolerance: summing three rounded shares need not
    # return exactly passthrough in IEEE754, and that is expected.
    assert sum(result.surcharges_eur_per_kwh.values()) == pytest.approx(0.2, abs=1e-12)


def test_synchronized_gives_every_broker_an_equal_surcharge_including_zero_and_negative():
    contributions = {"positive": 40.0, "zero": 0.0, "negative": -15.0}
    result = _fired_mechanism("synchronized", contributions, passthrough=0.3)
    assert result.total_charge_eur > 0.0
    expected = 0.3 / len(contributions)
    for broker_id in contributions:
        assert result.surcharges_eur_per_kwh[broker_id] == pytest.approx(expected, abs=1e-12)
    assert sum(result.surcharges_eur_per_kwh.values()) == pytest.approx(0.3, abs=1e-12)


def test_synchronized_total_matches_proportional_total():
    contributions = {"a": 25.0, "b": 75.0}
    prop = _fired_mechanism("proportional", contributions, passthrough=0.4)
    sync = _fired_mechanism("synchronized", contributions, passthrough=0.4)
    assert sum(sync.surcharges_eur_per_kwh.values()) == pytest.approx(
        sum(prop.surcharges_eur_per_kwh.values()), abs=1e-12
    )


def test_renormalized_total_is_passthrough_times_broker_count_by_design():
    """renormalized rescales each broker's proportional share by N so the
    MEAN surcharge stays at capacity_passthrough regardless of broker count
    (see the mean-flatness test below); summed across N brokers that mean is
    therefore N * capacity_passthrough, not capacity_passthrough. This is a
    deliberate difference from proportional and synchronized, which are both
    constructed to preserve the total signal strength instead."""
    contributions = {"a": 20.0, "b": 30.0, "c": 50.0}
    total = sum(contributions.values())
    passthrough = 0.15
    n = len(contributions)
    result = _fired_mechanism("renormalized", contributions, passthrough=passthrough)
    assert sum(result.surcharges_eur_per_kwh.values()) == pytest.approx(passthrough * n, abs=1e-12)
    for broker_id, contribution in contributions.items():
        expected = passthrough * (contribution / total) * n
        assert result.surcharges_eur_per_kwh[broker_id] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("n", (2, 3, 5))
def test_renormalized_mean_surcharge_stays_flat_as_broker_count_grows(n):
    passthrough = 0.2
    # Equal contributions isolate the N-dependence: the mean-flatness claim
    # is about broker count, not about how a fixed total is split.
    contributions = {f"broker_{i}": 10.0 for i in range(n)}
    result = _fired_mechanism("renormalized", contributions, passthrough=passthrough)
    mean_surcharge = sum(result.surcharges_eur_per_kwh.values()) / n
    assert mean_surcharge == pytest.approx(passthrough, abs=1e-12)


def test_all_modes_return_zero_surcharges_when_window_not_filled():
    for mode in SURCHARGE_MODES:
        mechanism = CapacityMechanism(
            window=4, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.2, surcharge_mode=mode
        )
        result = mechanism.step(50.0, {"a": 30.0, "b": 20.0})
        assert result.surcharges_eur_per_kwh == {"a": 0.0, "b": 0.0}, mode


def test_all_modes_return_zero_surcharges_when_total_charge_is_zero():
    for mode in SURCHARGE_MODES:
        mechanism = CapacityMechanism(
            window=2, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.2, surcharge_mode=mode
        )
        result = None
        for _ in range(3):
            # identical feeder value every step: std = 0, so nothing ever
            # exceeds mean + k*std and total_charge stays exactly 0.
            result = mechanism.step(10.0, {"a": 5.0, "b": 5.0})
        assert result.total_charge_eur == 0.0, mode
        assert result.surcharges_eur_per_kwh == {"a": 0.0, "b": 0.0}, mode


def test_all_modes_return_zero_surcharges_when_all_contributions_non_positive():
    for mode in SURCHARGE_MODES:
        mechanism = CapacityMechanism(
            window=2, k=0.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.2, surcharge_mode=mode
        )
        mechanism.step(10.0, {"a": -5.0, "b": -5.0})
        result = mechanism.step(50.0, {"a": -20.0, "b": -20.0})
        assert result.surcharges_eur_per_kwh == {"a": 0.0, "b": 0.0}, mode


def test_allocations_are_identical_across_modes():
    """The P&L channel (allocations_eur) does not depend on surcharge_mode:
    D10 only varies how the pricing channel splits the same total signal
    strength, never how the ledger charge itself is billed."""
    contributions = {"a": 70.0, "b": 20.0, "c": 10.0}
    results = {mode: _fired_mechanism(mode, contributions, passthrough=0.25) for mode in SURCHARGE_MODES}
    baseline = results["proportional"].allocations_eur
    for mode, result in results.items():
        assert result.allocations_eur == baseline, mode


def _run(config, hours):
    model = MicrogridModel(copy.deepcopy(config))
    for _ in range(hours):
        model.step()
    return model


@pytest.mark.parametrize(
    "ablation_flags,expect_nonzero_charge",
    ((DISABLED_FLAGS, False), (PNL_ONLY_FLAGS, True)),
    ids=("capacity_disabled", "capacity_pnl_only"),
)
def test_disabled_and_pnl_only_are_byte_for_byte_identical_across_all_modes(ablation_flags, expect_nonzero_charge):
    """Runs the model end to end, at a fixed seed, under each of the three
    modes with the pricing channel off, and checks full equality of the
    recorded output (feeder history, all four thesis metrics and every audit
    field compute_metrics() carries, plus the per-broker ledgers), not just
    one metric."""
    horizon = 500
    models = {
        mode: _run(_competitive_config(mode, horizon=horizon, num_agents=90, seed=5, ablation_flags=ablation_flags), horizon)
        for mode in SURCHARGE_MODES
    }

    reference_mode = SURCHARGE_MODES[0]
    reference = models[reference_mode]
    reference_metrics = reference.compute_metrics()
    reference_total_charge = sum(broker.cumulative_capacity_charge_eur for broker in reference.brokers.values())
    if expect_nonzero_charge:
        assert reference_total_charge > 0.0, "construction should exercise the mechanism at least once"
    else:
        assert reference_total_charge == 0.0

    for mode in SURCHARGE_MODES[1:]:
        model = models[mode]
        assert model.feeder_net_import_history == reference.feeder_net_import_history, mode
        assert model.compute_metrics() == reference_metrics, mode
        total_charge = sum(broker.cumulative_capacity_charge_eur for broker in model.brokers.values())
        assert total_charge == pytest.approx(reference_total_charge, abs=1e-9), mode
        for broker_id, broker in model.brokers.items():
            reference_broker = reference.brokers[broker_id]
            assert broker.cumulative_revenue_eur == pytest.approx(reference_broker.cumulative_revenue_eur, abs=1e-9)


def _population_signature(model):
    signature = []
    for agent in model.agents:
        base = (
            type(agent).__name__,
            agent.profile,
            agent.demand_scale,
            agent.price_tolerance_eur_per_kwh,
            agent.volatility_tolerance_eur_per_kwh,
            agent.greenness_threshold,
            agent.switching_penalty_eur_per_kwh,
            agent.sustained_breach_hours,
            agent.broker.id,
        )
        if isinstance(agent, Prosumer):
            base = base + (agent.pv_capacity_kwp, agent.battery_capacity_kwh)
        signature.append(base)
    return signature


def test_modes_draw_no_rng_population_and_broker_price_draws_are_identical():
    """D10's guardrail: neither new mode touches the model's seeded RNG
    stream, so the same seed must produce an identical sampled population and
    an identical first broker price draw regardless of which mode is
    selected. The volatile broker is the only price-generating broker that
    draws randomly at all (D5); its first quote() call right after
    construction depends only on the model seed, never on surcharge_mode."""
    seed = 4242
    signatures = {}
    first_quotes = {}
    for mode in SURCHARGE_MODES:
        config = _competitive_config(mode, horizon=SHORT_HORIZON, seed=seed, ablation_flags=BOTH_FLAGS)
        model = MicrogridModel(config)
        signatures[mode] = _population_signature(model)
        first_quotes[mode] = model.brokers["volatile_low_cost"].quote(0, {})

    reference_mode = SURCHARGE_MODES[0]
    for mode in SURCHARGE_MODES[1:]:
        assert signatures[mode] == signatures[reference_mode], mode
        assert first_quotes[mode] == first_quotes[reference_mode], mode
