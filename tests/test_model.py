import copy

import pytest

from microgrid_sim.agents.consumer import Consumer
from microgrid_sim.agents.prosumer import Prosumer
from microgrid_sim.config.loader import load_config
from microgrid_sim.environment.model import MicrogridModel

SHORT_HORIZON = 168  # one week, per brief guidance to keep unit tests fast


def _small_competitive_config(horizon=SHORT_HORIZON, num_agents=40, seed=1):
    config = load_config("config/default.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = horizon
    config["simulation"]["seed"] = seed
    config["population"]["num_agents"] = num_agents
    return config


def _small_monopoly_config(horizon=SHORT_HORIZON, num_agents=40, seed=1):
    config = load_config("config/scenarios/monopoly_baseline.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = horizon
    config["simulation"]["seed"] = seed
    config["population"]["num_agents"] = num_agents
    return config


def test_model_step_advances_hour_and_collects_feeder_history():
    model = MicrogridModel(_small_competitive_config(horizon=48))
    for _ in range(48):
        model.step()
    assert model.current_hour == 48
    assert len(model.feeder_net_import_history) == 48


def test_model_same_seed_reproduces_identical_run():
    config = _small_competitive_config(seed=7)
    model_a = MicrogridModel(copy.deepcopy(config))
    model_b = MicrogridModel(copy.deepcopy(config))
    for _ in range(SHORT_HORIZON):
        model_a.step()
    for _ in range(SHORT_HORIZON):
        model_b.step()

    assert model_a.feeder_net_import_history == pytest.approx(model_b.feeder_net_import_history)
    for broker_id in model_a.brokers:
        assert model_a.brokers[broker_id].cumulative_revenue_eur == pytest.approx(
            model_b.brokers[broker_id].cumulative_revenue_eur
        )


def test_model_different_seed_gives_different_run():
    model_a = MicrogridModel(_small_competitive_config(seed=1))
    model_b = MicrogridModel(_small_competitive_config(seed=2))
    for _ in range(SHORT_HORIZON):
        model_a.step()
    for _ in range(SHORT_HORIZON):
        model_b.step()

    assert model_a.feeder_net_import_history != pytest.approx(model_b.feeder_net_import_history)


def test_model_population_split_matches_prosumer_fraction():
    config = _small_competitive_config(num_agents=100)
    config["population"]["prosumer_fraction"] = 0.2
    model = MicrogridModel(config)
    num_consumers = len(model.agents_by_type.get(Consumer, []))
    num_prosumers = len(model.agents_by_type.get(Prosumer, []))
    assert num_prosumers == 20
    assert num_consumers == 80
    assert num_consumers + num_prosumers == 100


def test_monopoly_scenario_all_agents_use_the_single_broker_and_never_switch():
    config = _small_monopoly_config()
    assert len(config["brokers"]) == 1
    model = MicrogridModel(config)
    only_broker_id = config["brokers"][0]["id"]

    for _ in range(SHORT_HORIZON):
        model.step()

    for agent in model.agents:
        assert agent.broker.id == only_broker_id
        assert agent.switch_count == 0


def test_competitive_scenario_has_three_brokers_and_agents_may_be_on_any_of_them():
    config = _small_competitive_config(num_agents=150, horizon=336)
    model = MicrogridModel(config)
    for _ in range(336):
        model.step()

    broker_ids_in_use = {agent.broker.id for agent in model.agents}
    assert broker_ids_in_use.issubset(set(model.brokers.keys()))
    assert len(model.brokers) == 3
