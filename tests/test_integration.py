import copy

from microgrid_sim.config.loader import load_config
from microgrid_sim.environment.model import MicrogridModel

SHORT_HORIZON = 336  # two weeks; short horizon per brief guidance, full year covered by scripts/run_baseline.py


def test_competitive_scenario_runs_end_to_end_and_yields_sane_metrics():
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = SHORT_HORIZON

    model = MicrogridModel(config)
    model.run()

    assert model.current_hour == SHORT_HORIZON
    result = model.compute_metrics()

    assert result.avg_cost_per_agent_eur > 0.0
    assert result.avg_cost_per_kwh_eur > 0.0
    assert set(result.broker_load_share.keys()) == set(model.brokers.keys())
    assert abs(sum(result.broker_load_share.values()) - 1.0) < 1e-6
    assert 1.0 / len(model.brokers) - 1e-9 <= result.load_concentration_hhi <= 1.0
    assert result.feeder_coefficient_of_variation >= 0.0
    assert result.feeder_peak_to_average_ratio >= 1.0
    assert result.feeder_mean_hourly_ramp_kwh >= 0.0
    assert 0.0 <= result.prosumer_self_sufficiency <= 1.0
    assert model.solar_source in ("pvgis_fetch", "generated_sample")
    assert model.demand_source == "generated_sample"


def test_monopoly_baseline_scenario_runs_end_to_end():
    config = load_config("config/scenarios/monopoly_baseline.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = SHORT_HORIZON

    model = MicrogridModel(config)
    model.run()

    result = model.compute_metrics()

    assert result.avg_cost_per_agent_eur > 0.0
    assert result.load_concentration_hhi == 1.0  # a single broker serves all load
    assert all(agent.switch_count == 0 for agent in model.agents)
