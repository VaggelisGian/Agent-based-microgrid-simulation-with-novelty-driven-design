import copy

import pytest

from microgrid_sim.brokers.base import Broker
from microgrid_sim.config.loader import load_config
from microgrid_sim.environment import metrics
from microgrid_sim.environment.model import MicrogridModel


class LedgerBroker(Broker):
    def quote(self, hour, context=None):
        return 0.0


def test_compute_average_cost_matches_manual_calculation():
    broker_a = LedgerBroker("a", "A", "generic", 0.5)
    broker_a.cumulative_revenue_eur = 100.0
    broker_a.cumulative_energy_served_kwh = 500.0
    broker_b = LedgerBroker("b", "B", "generic", 0.5)
    broker_b.cumulative_revenue_eur = 50.0
    broker_b.cumulative_energy_served_kwh = 300.0
    brokers = {"a": broker_a, "b": broker_b}

    avg_per_agent, avg_per_kwh = metrics.compute_average_cost(brokers, num_agents=10)

    assert avg_per_agent == pytest.approx((100.0 + 50.0) / 10)
    assert avg_per_kwh == pytest.approx((100.0 + 50.0) / (500.0 + 300.0))


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

    assert result.avg_cost_per_agent_eur > 0.0
    assert result.avg_cost_per_kwh_eur > 0.0
    assert sum(result.broker_load_share.values()) == pytest.approx(1.0, abs=1e-6)
    assert 1.0 / 3.0 - 1e-9 <= result.load_concentration_hhi <= 1.0
    assert result.feeder_coefficient_of_variation >= 0.0
    assert result.feeder_peak_to_average_ratio >= 1.0
    assert result.feeder_mean_hourly_ramp_kwh >= 0.0
    assert 0.0 <= result.prosumer_self_sufficiency <= 1.0
