"""Tests for the D11 dose-matched-monopoly control (capacity_surcharge_divisor).

Covers the CapacityMechanism.step() substitution directly (proportional,
synchronized, renormalized), the no-op guarantee when the divisor is absent
or None, the loader's validation of the new optional config key, and three
model-level guarantees D11 requires: the divisor is inert wherever the
mechanism already is (capacity_disabled, capacity_pnl_only), the P&L channel
(allocations_eur) never depends on it, and it touches no seeded RNG draw. See
docs/DECISIONS.md D11 for the full substitution rule this file pins.
"""

import copy

import pytest

from microgrid_sim.config.loader import ConfigError, load_config, validate_config
from microgrid_sim.environment.capacity import CapacityMechanism
from microgrid_sim.environment.model import MicrogridModel

SHORT_HORIZON = 168

DISABLED_FLAGS = {"enabled": False, "feedback_pnl": False, "feedback_pricing": False}
PNL_ONLY_FLAGS = {"enabled": True, "feedback_pnl": True, "feedback_pricing": False}
BOTH_FLAGS = {"enabled": True, "feedback_pnl": True, "feedback_pricing": True}

# A battery of contribution vectors spanning different broker counts and
# value patterns, reused across several "bit-identical" checks below.
CONTRIBUTION_VECTORS = (
    {"a": 60.0, "b": 30.0, "c": 10.0},
    {"a": 1.0},
    {"a": 25.0, "b": 75.0},
    {"a": 20.0, "b": 30.0, "c": 50.0},
    {"a": 40.0, "zero": 0.0, "negative": -15.0},
    {f"broker_{i}": 10.0 * (i + 1) for i in range(5)},
)


def _fired_mechanism(mode, contributions, divisor=None, window=2, k=0.0, charge_rate=1.0, passthrough=0.3):
    """Fill the window with a flat, non-firing baseline (equal to
    contributions' own total, so k=0 alone would not already guarantee
    firing), then push `contributions` as the step that fires. Mirrors
    tests/test_surcharge_mode.py's helper of the same name, with the divisor
    threaded through. Always passes surcharge_divisor explicitly (even when
    None), matching how MicrogridModel._build_capacity_mechanism always calls
    it via capacity_cfg.get("capacity_surcharge_divisor", None): the model
    never omits the argument, so this helper does not either."""
    mechanism = CapacityMechanism(
        window=window,
        k=k,
        charge_rate_eur_per_kwh=charge_rate,
        capacity_passthrough=passthrough,
        surcharge_mode=mode,
        surcharge_divisor=divisor,
    )
    baseline = {broker_id: 1.0 for broker_id in contributions}
    for _ in range(window):
        mechanism.step(sum(baseline.values()), baseline)
    return mechanism.step(sum(contributions.values()), contributions)


def _fired_mechanism_omitting_divisor_kwarg(mode, contributions, window=2, k=0.0, charge_rate=1.0, passthrough=0.3):
    """Same as _fired_mechanism, except surcharge_divisor is never mentioned
    at all, relying purely on CapacityMechanism's own default parameter value
    (used only by the (b) no-op test below, to distinguish "the default is
    None" from "None was passed explicitly")."""
    mechanism = CapacityMechanism(
        window=window, k=k, charge_rate_eur_per_kwh=charge_rate, capacity_passthrough=passthrough, surcharge_mode=mode
    )
    baseline = {broker_id: 1.0 for broker_id in contributions}
    for _ in range(window):
        mechanism.step(sum(baseline.values()), baseline)
    return mechanism.step(sum(contributions.values()), contributions)


# ---------------------------------------------------------------------------
# (a) At N=1, share=1, proportional: surcharge == passthrough / M, to within
# floating-point noise rather than bit-exactly. See the second test below for
# why the distinction is pinned instead of glossed.
# ---------------------------------------------------------------------------

# The shipped capacity_passthrough (config/default.yaml) and the three
# divisors the D11 arm actually runs. Named rather than inlined because the
# precision test below depends on these exact values, not on arbitrary ones.
SHIPPED_PASSTHROUGH = 0.10
ARM_DIVISORS = (2, 3, 5)


@pytest.mark.parametrize("divisor", ARM_DIVISORS)
@pytest.mark.parametrize("passthrough", (0.3, SHIPPED_PASSTHROUGH, 0.234567))
def test_monopolist_proportional_surcharge_is_passthrough_over_divisor(divisor, passthrough):
    result = _fired_mechanism("proportional", {"solo": 42.0}, divisor=divisor, passthrough=passthrough)
    assert result.total_charge_eur > 0.0
    # Relative-only tolerance, abs=0: pytest.approx passes if EITHER tolerance
    # holds, so leaving the default abs in place would let any two values
    # under 1e-12 compare equal, which is the vacuous-pin failure mode this
    # project's own golden-master review found and fixed in Phase 16.
    assert result.surcharges_eur_per_kwh["solo"] == pytest.approx(passthrough / divisor, rel=1e-15, abs=0)


def test_monopolist_dose_matching_is_exact_at_two_of_three_shipped_divisors():
    """Pins the measured float behaviour of the dose match at the SHIPPED
    passthrough, so nothing in this package can claim the monopolist receives
    "exactly" passthrough/M when at one divisor it does not.

    The proportional branch computes passthrough * share * (N / N_eff), which
    at broker_count 1 is passthrough * 1.0 * (1/M) rather than passthrough/M.
    Those two expressions agree bit for bit at M=2 and M=3 and differ by one
    unit in the last place at M=5. The gap is about 1.7e-16 relative, roughly
    fifteen orders of magnitude below the surcharge itself, and the surcharge
    only ever enters the model through clip(surcharge / response_reference,
    0, 1), so it cannot move a deferral decision. It is pinned here rather
    than rounded away because this project has been wrong before by writing
    "exactly" where "to within floating-point noise" was the true statement
    (see the D10 entry's own note on the synchronized total)."""
    surcharges = {
        divisor: _fired_mechanism(
            "proportional", {"solo": 42.0}, divisor=divisor, passthrough=SHIPPED_PASSTHROUGH
        ).surcharges_eur_per_kwh["solo"]
        for divisor in ARM_DIVISORS
    }
    assert surcharges[2] == SHIPPED_PASSTHROUGH / 2
    assert surcharges[3] == SHIPPED_PASSTHROUGH / 3
    assert surcharges[5] != SHIPPED_PASSTHROUGH / 5
    relative_gap = abs(surcharges[5] - SHIPPED_PASSTHROUGH / 5) / (SHIPPED_PASSTHROUGH / 5)
    assert relative_gap < 1e-15


# ---------------------------------------------------------------------------
# (b) divisor=None takes the shipped path: bit-identical to a mechanism built
# with no surcharge_divisor argument at all, across a battery of vectors.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("proportional", "synchronized", "renormalized"))
@pytest.mark.parametrize("contributions", CONTRIBUTION_VECTORS)
def test_divisor_none_is_bit_identical_to_omitting_the_argument(mode, contributions):
    passthrough = 0.234567  # an ugly float, so any accidental rounding would show up
    omitted = _fired_mechanism_omitting_divisor_kwarg(mode, contributions, passthrough=passthrough)
    explicit_none = _fired_mechanism(mode, contributions, divisor=None, passthrough=passthrough)
    assert omitted.surcharges_eur_per_kwh == explicit_none.surcharges_eur_per_kwh
    assert omitted.allocations_eur == explicit_none.allocations_eur
    assert omitted.total_charge_eur == explicit_none.total_charge_eur


# ---------------------------------------------------------------------------
# (c) divisor == N reproduces the no-divisor result bit-identically, for
# proportional and synchronized, at N = 2, 3, 5.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("proportional", "synchronized"))
@pytest.mark.parametrize("n", (2, 3, 5))
def test_divisor_equal_to_broker_count_is_bit_identical_to_no_divisor(mode, n):
    contributions = {f"broker_{i}": 10.0 * (i + 1) for i in range(n)}
    passthrough = 0.17
    no_divisor = _fired_mechanism(mode, contributions, divisor=None, passthrough=passthrough)
    matched_divisor = _fired_mechanism(mode, contributions, divisor=n, passthrough=passthrough)
    assert matched_divisor.surcharges_eur_per_kwh == no_divisor.surcharges_eur_per_kwh


# ---------------------------------------------------------------------------
# (d) renormalized is bit-identical under every divisor (N-invariant).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("contributions", CONTRIBUTION_VECTORS)
@pytest.mark.parametrize("divisor", (None, 2, 3, 5))
def test_renormalized_is_bit_identical_under_every_divisor(contributions, divisor):
    passthrough = 0.1975
    baseline = _fired_mechanism("renormalized", contributions, divisor=None, passthrough=passthrough)
    under_divisor = _fired_mechanism("renormalized", contributions, divisor=divisor, passthrough=passthrough)
    assert under_divisor.surcharges_eur_per_kwh == baseline.surcharges_eur_per_kwh


# ---------------------------------------------------------------------------
# (e) synchronized under divisor M gives passthrough / M for every broker,
# regardless of the actual broker count N.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", (2, 3, 5))
def test_synchronized_under_divisor_ignores_actual_broker_count(n):
    passthrough = 0.4
    divisor = 4
    contributions = {f"broker_{i}": 5.0 * (i + 1) for i in range(n)}
    result = _fired_mechanism("synchronized", contributions, divisor=divisor, passthrough=passthrough)
    expected = passthrough / divisor
    for broker_id in contributions:
        assert result.surcharges_eur_per_kwh[broker_id] == expected


# ---------------------------------------------------------------------------
# (f) allocations_eur (the P&L channel) are bit-identical to the no-divisor
# case under every divisor and every mode.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("proportional", "synchronized", "renormalized"))
@pytest.mark.parametrize("divisor", (None, 2, 3, 5))
def test_allocations_are_bit_identical_regardless_of_divisor(mode, divisor):
    contributions = {"a": 70.0, "b": 20.0, "c": 10.0}
    passthrough = 0.25
    baseline = _fired_mechanism(mode, contributions, divisor=None, passthrough=passthrough)
    under_divisor = _fired_mechanism(mode, contributions, divisor=divisor, passthrough=passthrough)
    assert under_divisor.allocations_eur == baseline.allocations_eur


# ---------------------------------------------------------------------------
# (g) validation: CapacityMechanism construction and the config loader.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("divisor", (None, 1, 2, 5))
def test_capacity_mechanism_accepts_valid_divisor_values(divisor):
    CapacityMechanism(
        window=1, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1, surcharge_divisor=divisor
    )  # must not raise


@pytest.mark.parametrize("divisor", (True, False, 0, -1, 2.0, "3"))
def test_capacity_mechanism_rejects_invalid_divisor_values(divisor):
    with pytest.raises(ValueError):
        CapacityMechanism(
            window=1, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.1, surcharge_divisor=divisor
        )


def test_capacity_surcharge_divisor_absent_is_valid():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    del config["capacity_mechanism"]["capacity_surcharge_divisor"]
    validate_config(config)  # must not raise


def test_default_yaml_ships_null_divisor():
    config = load_config("config/default.yaml")
    assert config["capacity_mechanism"]["capacity_surcharge_divisor"] is None


@pytest.mark.parametrize("divisor", (None, 1, 2, 5))
def test_loader_accepts_valid_divisor_values(divisor):
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"]["capacity_surcharge_divisor"] = divisor
    validate_config(config)  # must not raise


@pytest.mark.parametrize("divisor", (True, False, 0, -1, 2.0, "3"))
def test_loader_rejects_invalid_divisor_values(divisor):
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["capacity_mechanism"]["capacity_surcharge_divisor"] = divisor
    with pytest.raises(ConfigError):
        validate_config(config)


# ---------------------------------------------------------------------------
# (h) the divisor is inert when total_charge is 0 and when the window is not
# yet full: all surcharges 0.0, regardless of divisor or mode.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("proportional", "synchronized", "renormalized"))
@pytest.mark.parametrize("divisor", (2, 3, 5))
def test_divisor_inert_when_window_not_filled(mode, divisor):
    mechanism = CapacityMechanism(
        window=4, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.2, surcharge_mode=mode,
        surcharge_divisor=divisor,
    )
    result = mechanism.step(50.0, {"a": 30.0, "b": 20.0})
    assert result.surcharges_eur_per_kwh == {"a": 0.0, "b": 0.0}


@pytest.mark.parametrize("mode", ("proportional", "synchronized", "renormalized"))
@pytest.mark.parametrize("divisor", (2, 3, 5))
def test_divisor_inert_when_total_charge_is_zero(mode, divisor):
    mechanism = CapacityMechanism(
        window=2, k=1.0, charge_rate_eur_per_kwh=1.0, capacity_passthrough=0.2, surcharge_mode=mode,
        surcharge_divisor=divisor,
    )
    result = None
    for _ in range(3):
        # identical feeder value every step: std = 0, so total_charge stays 0.
        result = mechanism.step(10.0, {"a": 5.0, "b": 5.0})
    assert result.total_charge_eur == 0.0
    assert result.surcharges_eur_per_kwh == {"a": 0.0, "b": 0.0}


# ---------------------------------------------------------------------------
# Model-level integration
# ---------------------------------------------------------------------------


def _competitive_config(divisor, horizon=SHORT_HORIZON, num_agents=60, seed=5, ablation_flags=BOTH_FLAGS):
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = horizon
    config["simulation"]["seed"] = seed
    config["population"]["num_agents"] = num_agents
    config["capacity_mechanism"]["capacity_surcharge_mode"] = "proportional"
    config["capacity_mechanism"]["capacity_surcharge_divisor"] = divisor
    config["capacity_mechanism"].update(ablation_flags)
    return config


def _monopoly_config(divisor, k=1.0, horizon=SHORT_HORIZON, num_agents=60, seed=5, ablation_flags=BOTH_FLAGS):
    config = _competitive_config(divisor, horizon=horizon, num_agents=num_agents, seed=seed, ablation_flags=ablation_flags)
    config["capacity_mechanism"]["k"] = k
    # D11's grid: broker_count fixed at 1, the regulated utility as incumbent
    # monopoly (matches scripts/run_monopoly_arm.py and
    # scripts/run_dose_matched_monopoly.py).
    config["brokers"] = [broker for broker in config["brokers"] if broker["type"] == "regulated_utility"]
    return config


def _run(config, hours):
    model = MicrogridModel(copy.deepcopy(config))
    for _ in range(hours):
        model.step()
    return model


# ---------------------------------------------------------------------------
# (i) capacity_disabled and capacity_pnl_only reproduce the baseline
# byte-for-byte under every divisor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ablation_flags,expect_nonzero_charge",
    ((DISABLED_FLAGS, False), (PNL_ONLY_FLAGS, True)),
    ids=("capacity_disabled", "capacity_pnl_only"),
)
def test_disabled_and_pnl_only_are_byte_for_byte_identical_across_all_divisors(ablation_flags, expect_nonzero_charge):
    """Runs the model end to end, at a fixed seed, under each divisor with
    the pricing channel off (or the mechanism off entirely), and checks full
    equality of the feeder history and all four thesis metrics, not just one."""
    horizon = 500
    divisors = (None, 2, 3, 5)
    models = {
        divisor: _run(
            _competitive_config(divisor, horizon=horizon, num_agents=90, seed=5, ablation_flags=ablation_flags),
            horizon,
        )
        for divisor in divisors
    }

    reference = models[None]
    reference_metrics = reference.compute_metrics()
    reference_total_charge = sum(broker.cumulative_capacity_charge_eur for broker in reference.brokers.values())
    if expect_nonzero_charge:
        assert reference_total_charge > 0.0, "construction should exercise the mechanism at least once"
    else:
        assert reference_total_charge == 0.0

    for divisor in (2, 3, 5):
        model = models[divisor]
        assert model.feeder_net_import_history == reference.feeder_net_import_history, divisor
        assert model.compute_metrics() == reference_metrics, divisor
        total_charge = sum(broker.cumulative_capacity_charge_eur for broker in model.brokers.values())
        assert total_charge == pytest.approx(reference_total_charge, abs=1e-9), divisor
        for broker_id, broker in model.brokers.items():
            reference_broker = reference.brokers[broker_id]
            assert broker.cumulative_revenue_eur == pytest.approx(reference_broker.cumulative_revenue_eur, abs=1e-9)


# ---------------------------------------------------------------------------
# (j) under capacity_both at broker_count 1, a larger divisor produces a
# strictly smaller total_deferred_kwh (ordering only, never a magnitude).
# ---------------------------------------------------------------------------


def test_larger_divisor_strictly_reduces_deferred_volume_at_monopoly():
    horizon = 500
    divisors = (2, 3, 5)
    totals = {}
    for divisor in divisors:
        model = _run(
            _monopoly_config(divisor, horizon=horizon, num_agents=90, seed=5, ablation_flags=BOTH_FLAGS), horizon
        )
        metrics = model.compute_metrics()
        assert metrics.capacity_fire_rate > 0.0, "construction should produce at least some excess hours"
        totals[divisor] = metrics.total_deferred_kwh

    assert totals[2] > totals[3] > totals[5], totals


# ---------------------------------------------------------------------------
# (k) RNG neutrality: the divisor draws no random numbers and touches no
# seeded stream, so population sampling and the first broker price draw are
# identical across divisors at a fixed seed.
# ---------------------------------------------------------------------------


def _population_signature(model):
    from microgrid_sim.agents.prosumer import Prosumer

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


def test_divisor_draws_no_rng_population_and_broker_price_draws_are_identical():
    seed = 4242
    divisors = (None, 2, 3, 5)
    signatures = {}
    first_quotes = {}
    for divisor in divisors:
        config = _competitive_config(divisor, horizon=SHORT_HORIZON, seed=seed, ablation_flags=BOTH_FLAGS)
        model = MicrogridModel(config)
        signatures[divisor] = _population_signature(model)
        first_quotes[divisor] = model.brokers["volatile_low_cost"].quote(0, {})

    reference_divisor = divisors[0]
    for divisor in divisors[1:]:
        assert signatures[divisor] == signatures[reference_divisor], divisor
        assert first_quotes[divisor] == first_quotes[reference_divisor], divisor
