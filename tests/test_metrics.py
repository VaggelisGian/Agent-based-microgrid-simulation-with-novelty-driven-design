import copy
import math

import pytest

from microgrid_sim.agents.consumer import Consumer
from microgrid_sim.brokers.base import Broker
from microgrid_sim.config.loader import load_config
from microgrid_sim.environment import metrics
from microgrid_sim.environment.model import MicrogridModel


class LedgerBroker(Broker):
    def quote(self, hour, context=None):
        return 0.0


def test_compute_average_cost_matches_manual_calculation():
    # broker_a served 500 kWh of imports but also credited 20 kWh of prosumer
    # exports (not counted in cumulative_energy_served_kwh, which is import-only),
    # so its net energy is 480 kWh; broker_b has no exports so both figures match.
    # avg_cost_per_kwh must divide by NET energy (480 + 300), not served energy
    # (500 + 300), so numerator and denominator share the same signed scope (F1).
    broker_a = LedgerBroker("a", "A", "generic", 0.5)
    broker_a.cumulative_revenue_eur = 100.0
    broker_a.cumulative_energy_served_kwh = 500.0
    broker_a.cumulative_net_energy_kwh = 480.0
    broker_b = LedgerBroker("b", "B", "generic", 0.5)
    broker_b.cumulative_revenue_eur = 50.0
    broker_b.cumulative_energy_served_kwh = 300.0
    broker_b.cumulative_net_energy_kwh = 300.0
    brokers = {"a": broker_a, "b": broker_b}

    avg_per_agent, avg_per_kwh = metrics.compute_average_cost(brokers, num_agents=10)

    assert avg_per_agent == pytest.approx((100.0 + 50.0) / 10)
    assert avg_per_kwh == pytest.approx((100.0 + 50.0) / (480.0 + 300.0))


def test_compute_average_cost_handles_zero_agents_and_zero_energy():
    avg_per_agent, avg_per_kwh = metrics.compute_average_cost({}, num_agents=0)
    assert avg_per_agent == 0.0
    assert avg_per_kwh == 0.0


def test_compute_load_distribution_shares_sum_to_one():
    broker_a = LedgerBroker("a", "A", "generic", 0.5)
    broker_a.cumulative_energy_served_kwh = 700.0
    broker_b = LedgerBroker("b", "B", "generic", 0.5)
    broker_b.cumulative_energy_served_kwh = 300.0
    shares, hhi = metrics.compute_load_distribution({"a": broker_a, "b": broker_b})

    assert shares["a"] == pytest.approx(0.7)
    assert shares["b"] == pytest.approx(0.3)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert hhi == pytest.approx(0.7**2 + 0.3**2)


def test_compute_load_distribution_monopoly_hhi_is_one():
    broker_a = LedgerBroker("a", "A", "generic", 0.5)
    broker_a.cumulative_energy_served_kwh = 1000.0
    shares, hhi = metrics.compute_load_distribution({"a": broker_a})
    assert shares["a"] == pytest.approx(1.0)
    assert hhi == pytest.approx(1.0)


def test_compute_load_distribution_returns_zero_shares_and_hhi_when_no_energy_served():
    """total_energy_kwh == 0.0 (no broker has served anything yet: metrics
    queried before any step, or every broker in the config ended up with zero
    customers) is not guaranteed unreachable by any caller. Without the
    early-return guard the shares dict comprehension divides by
    total_energy_kwh directly, which raises ZeroDivisionError on exactly this
    input, so this is genuinely untested logic that could turn metric 2
    (load_concentration_hhi, broker_load_share) into a crash instead of the
    documented 0.0/empty-share result."""
    broker_a = LedgerBroker("a", "A", "generic", 0.5)
    broker_b = LedgerBroker("b", "B", "generic", 0.5)
    shares, hhi = metrics.compute_load_distribution({"a": broker_a, "b": broker_b})
    assert shares == {"a": 0.0, "b": 0.0}
    assert hhi == pytest.approx(0.0)


def test_compute_grid_stability_returns_nan_triple_for_empty_history():
    """feeder_net_import_history == [] (metrics computed before any step) hits
    the size == 0 guard. Genuinely untested: nothing here proves the guard is
    unreachable, and it directly decides what metric 3 (CoV, peak-to-average,
    ramp) reports for a degenerate run rather than a wrong non-nan number."""
    cv, peak_to_average, ramp = metrics.compute_grid_stability([])
    assert math.isnan(cv)
    assert math.isnan(peak_to_average)
    assert math.isnan(ramp)


def test_compute_grid_stability_constant_series_has_zero_variability():
    cv, peak_to_average, ramp = metrics.compute_grid_stability([10.0, 10.0, 10.0, 10.0])
    assert cv == pytest.approx(0.0)
    assert peak_to_average == pytest.approx(1.0)
    assert ramp == pytest.approx(0.0)


def test_compute_grid_stability_variable_series_matches_manual_calculation():
    series = [10.0, 20.0, 10.0, 20.0]
    cv, peak_to_average, ramp = metrics.compute_grid_stability(series)
    assert cv == pytest.approx(5.0 / 15.0)
    assert peak_to_average == pytest.approx(20.0 / 15.0)
    assert ramp == pytest.approx(10.0)


class StubProsumer:
    def __init__(self, ratio):
        self._ratio = ratio

    def self_sufficiency_ratio(self):
        return self._ratio


def test_compute_prosumer_self_sufficiency_averages_ratios():
    prosumers = [StubProsumer(0.2), StubProsumer(0.4), StubProsumer(0.6)]
    assert metrics.compute_prosumer_self_sufficiency(prosumers) == pytest.approx(0.4)


def test_compute_prosumer_self_sufficiency_nan_when_no_prosumers():
    import math

    assert math.isnan(metrics.compute_prosumer_self_sufficiency([]))


def test_compute_metrics_end_to_end_on_short_competitive_run():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = 336
    config["population"]["num_agents"] = 60

    model = MicrogridModel(config)
    for _ in range(336):
        model.step()
    result = model.compute_metrics()

    # Genuine sanity bounds (not tautologies: each could fail under a real bug).
    assert result.avg_cost_per_agent_eur > 0.0
    assert result.avg_cost_per_kwh_eur > 0.0
    assert sum(result.broker_load_share.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.feeder_coefficient_of_variation >= 0.0
    assert result.feeder_mean_hourly_ramp_kwh >= 0.0
    assert 0.0 <= result.prosumer_self_sufficiency <= 1.0

    # F8 fix: exact-value regression pins, replacing what were mathematical
    # tautologies for any 3-way share vector / any non-negative series
    # (hhi >= 1/3 and peak_to_average >= 1.0 hold no matter what the broker-share
    # or feeder computation does, correct or buggy). Pinned to this fixed-seed
    # (config/default.yaml seed 20260704), fixed-horizon (336h), fixed-population
    # (60 agents) run's actual computed output.
    assert result.load_concentration_hhi == pytest.approx(0.4030448039797871, abs=1e-9)
    assert result.feeder_peak_to_average_ratio == pytest.approx(2.5331102715032934, abs=1e-9)
    assert result.avg_cost_per_agent_eur == pytest.approx(31.67159055254317, abs=1e-6)
    assert result.avg_cost_per_kwh_eur == pytest.approx(0.17435024676820962, abs=1e-9)


def test_avg_cost_per_kwh_uses_net_energy_scope_consistently_with_net_revenue():
    """F1 regression: the numerator (net revenue, including export credits) and
    the denominator (net energy, imports minus exports) must share the same
    signed scope, so this metric no longer understates the true unit price by
    mixing a net-of-credit numerator with an import-only denominator (measured
    ~8.5% gap on the pre-fix default config; see docs/DECISIONS.md observation
    note)."""
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = 336
    config["population"]["num_agents"] = 60

    model = MicrogridModel(config)
    for _ in range(336):
        model.step()

    total_revenue_eur = sum(broker.cumulative_revenue_eur for broker in model.brokers.values())
    total_served_kwh = sum(broker.cumulative_energy_served_kwh for broker in model.brokers.values())  # import-only
    total_net_kwh = sum(broker.cumulative_net_energy_kwh for broker in model.brokers.values())  # imports minus exports

    result = model.compute_metrics()

    # This run has prosumers exporting, so the two scopes genuinely differ.
    assert total_net_kwh < total_served_kwh
    # The metric must equal net revenue / net energy, not net revenue / import-only energy.
    assert result.avg_cost_per_kwh_eur == pytest.approx(total_revenue_eur / total_net_kwh, rel=1e-12)
    old_style_value = total_revenue_eur / total_served_kwh
    assert result.avg_cost_per_kwh_eur != pytest.approx(old_style_value, rel=1e-9)
    # Exact-value pin (F8), fixed seed/horizon/population.
    assert result.avg_cost_per_kwh_eur == pytest.approx(0.17435024676820962, abs=1e-9)


def _bc2_config_for_audit(horizon, capacity_overrides):
    """A broker_count-2 config (regulated_utility + volatile_low_cost) with
    num_agents forced to 0, so a single hand-built Consumer can be attached
    for full control over feeder_net_import_kwh via model.demand_profile."""
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
        "deferrable_fraction": 0.0,  # inert deferral: feeder stays exactly the hand-set demand
        "payback_cap_fraction": 0.0,
    }
    config["capacity_mechanism"].update(capacity_overrides)
    return config


def test_compute_capacity_audit_divides_mean_surcharge_by_surcharge_steps():
    """Phase 20 (20.1, mutant C): environment/metrics.py's compute_capacity_audit
    divides model._surcharge_accum by model._surcharge_steps (the number of
    model.step() calls made, every one of which counts whether or not the
    mechanism fired that hour), not steps - 1. At the project's real 8760h
    horizon the two divisors differ by only about 0.011%, too small for most
    tolerances to bite (docs/thesis/defense_slides.md's 30-seed mean-surcharge
    figures were independently recomputed from results/sweep_raw.parquet and
    results/sweep_monopoly.parquet under the shipped `steps` divisor and match
    the printed values to the reported precision, e.g. bc=3 k=1.0:
    premium_green 0.0043, regulated_utility 0.0079, volatile_low_cost 0.0081,
    and passthrough * HHI * fire_rate = 0.0187/0.0101/0.0073/0.0072 at
    bc=1/2/3/5 -- Phase 20 evidence, not re-run here). This test instead uses
    a 5-step run where the gap is 25%: one agent, window=2, k=0.0, demand
    profile [1, 1, 5, 1, 1]. Hour 2 fires once (feeder jumps 1.0 -> 5.0
    against a trailing mean of 3.0, excess 2.0, charge 2.0 EUR, entirely
    attributed to "regulated_utility", the only broker with any contribution)
    and capacity_passthrough=0.5 makes that surcharge exactly 0.5, applied
    once (at hour 3; hour 4's feeder drops back to 1.0, so the window goes
    quiet again). Total over 5 steps is exactly 0.5, so the shipped divisor
    gives 0.5 / 5 = 0.1 and a steps - 1 divisor would give 0.5 / 4 = 0.125.

    The independent total below is read from model.broker_surcharges (public)
    fresh before each step() call -- the exact dict model.py's step() itself
    reads as applied_surcharges -- summed by hand, never touching
    compute_capacity_audit or _surcharge_accum/_surcharge_steps at all, so it
    cannot be affected by the mutation under test.
    """
    horizon = 5
    config = _bc2_config_for_audit(horizon, {"feedback_pricing": True, "capacity_passthrough": 0.5})
    model = MicrogridModel(config)
    Consumer(
        model,
        profile="price_sensitive",
        demand_scale=1.0,
        price_tolerance_eur_per_kwh=1.0,  # never breaches; switching is irrelevant to this mutant
        volatility_tolerance_eur_per_kwh=0.05,
        greenness_threshold=0.5,
        switching_penalty_eur_per_kwh=0.0,
        sustained_breach_hours=999,
        initial_broker=model.brokers["regulated_utility"],
    )
    model.demand_profile[:5] = [1.0, 1.0, 5.0, 1.0, 1.0]

    independent_totals = {"regulated_utility": 0.0, "volatile_low_cost": 0.0}
    for _ in range(horizon):
        for broker_id, surcharge in model.broker_surcharges.items():
            independent_totals[broker_id] += surcharge
        model.step()

    assert independent_totals["regulated_utility"] == pytest.approx(0.5, abs=1e-12)
    assert independent_totals["volatile_low_cost"] == pytest.approx(0.0, abs=1e-12)

    result = model.compute_metrics()
    expected = {broker_id: total / horizon for broker_id, total in independent_totals.items()}
    assert result.mean_broker_surcharge_eur_per_kwh["regulated_utility"] == pytest.approx(
        expected["regulated_utility"], abs=1e-12
    )
    assert result.mean_broker_surcharge_eur_per_kwh["volatile_low_cost"] == pytest.approx(
        expected["volatile_low_cost"], abs=1e-12
    )
    # Tight, hand-derived pin: a steps - 1 divisor gives 0.125 here, a 25%
    # relative gap from the correct 0.1, so no loose tolerance is needed.
    assert result.mean_broker_surcharge_eur_per_kwh["regulated_utility"] == pytest.approx(0.1, abs=1e-9)
