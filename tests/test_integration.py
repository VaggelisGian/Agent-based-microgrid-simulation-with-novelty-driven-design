import copy

import pytest

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

    # Genuine sanity bounds (not tautologies).
    assert result.avg_cost_per_agent_eur > 0.0
    assert result.avg_cost_per_kwh_eur > 0.0
    assert set(result.broker_load_share.keys()) == set(model.brokers.keys())
    assert abs(sum(result.broker_load_share.values()) - 1.0) < 1e-6
    assert result.feeder_coefficient_of_variation >= 0.0
    assert result.feeder_mean_hourly_ramp_kwh >= 0.0
    assert 0.0 <= result.prosumer_self_sufficiency <= 1.0

    # F8 fix: exact-value regression pins (hhi >= 1/n and peak_to_average >= 1.0
    # are mathematical identities for any share vector / non-negative series,
    # so they would pass even under a wrong-but-plausible bug). Pinned to
    # config/default.yaml's fixed seed (20260704) at SHORT_HORIZON, default
    # population (200 agents).
    assert result.load_concentration_hhi == pytest.approx(0.3519230863623769, abs=1e-9)
    assert result.feeder_peak_to_average_ratio == pytest.approx(2.4663308301406155, abs=1e-9)

    # allow_network_fetch=False (hardcoded in MicrogridModel) plus the shipped
    # data/samples/solar_thessaloniki_hourly.csv cache (marker "pvgis_fetch")
    # guarantees this is loaded, not a fallback (F8: was a loose membership
    # check that would also have passed on a silent synthetic fallback).
    assert model.solar_source == "pvgis_fetch"
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
