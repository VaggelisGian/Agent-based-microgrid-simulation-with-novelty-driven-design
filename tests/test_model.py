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


def test_initial_broker_assignment_is_profile_compatible_where_possible():
    """F3 fix: initial broker assignment picks a categorically-acceptable
    broker (falling back to overall cheapest only if none is acceptable),
    instead of a uniformly random one, so measured switching afterwards is not
    dominated by a one-time "born on the wrong broker" settling transient.
    Checked immediately after construction, before any step() call, so
    agent.broker is exactly the initial assignment."""
    config = _small_competitive_config(num_agents=150, horizon=48)
    model = MicrogridModel(config)

    volatile = model.brokers["volatile_low_cost"]
    reference_volatility = volatile.reference_volatility_eur_per_kwh()
    saw_green_preferring = False
    saw_stability_oriented = False

    for agent in model.agents:
        if agent.profile == "green_preferring":
            saw_green_preferring = True
            assert agent.broker.greenness >= agent.greenness_threshold
        elif agent.profile == "stability_oriented":
            saw_stability_oriented = True
            if agent.broker.id == volatile.id:
                # Only correct if this agent's own sampled tolerance actually
                # covers the volatile broker's analytically-derived reference
                # volatility (see VolatileLowCostBroker.reference_volatility_
                # eur_per_kwh and docs/DECISIONS.md's F2 observation note: at
                # this calibration this covers only the top end of the sampled
                # volatility_tolerance_eur_per_kwh range, so it is rare but not
                # impossible).
                assert agent.volatility_tolerance_eur_per_kwh >= reference_volatility
            else:
                assert agent.broker.id != volatile.id

    # sanity: the config's profile_mix guarantees both profiles are present
    # in a 150-agent sample, so the assertions above were actually exercised.
    assert saw_green_preferring
    assert saw_stability_oriented


def test_green_preferring_agents_split_into_more_than_one_acceptable_broker_group():
    """F2 fix: greenness_threshold now straddles the regulated broker's
    greenness (0.30), so green-preferring agents split into a group that also
    accepts the regulated broker (threshold <= 0.30) and a strict
    green-only group (threshold > 0.30), instead of every green-preferring
    agent's categorical partition being identical regardless of its sampled
    threshold (see docs/DECISIONS.md observation note for the measured split)."""
    config = _small_competitive_config(num_agents=200, horizon=48)
    model = MicrogridModel(config)
    regulated_greenness = model.brokers["regulated_utility"].greenness

    green_preferring = [agent for agent in model.agents if agent.profile == "green_preferring"]
    assert len(green_preferring) > 0

    accepts_regulated_too = [a for a in green_preferring if a.greenness_threshold <= regulated_greenness]
    strict_green_only = [a for a in green_preferring if a.greenness_threshold > regulated_greenness]

    assert len(accepts_regulated_too) > 0
    assert len(strict_green_only) > 0
    assert len(accepts_regulated_too) + len(strict_green_only) == len(green_preferring)


# NOTE: whether the recalibrated switching_penalty_eur_per_kwh range ([0.02, 0.06])
# actually blocks a reconsideration, and what the annual switch rate is, was measured
# with a standalone full 8760h/200-agent run of the unmodified config/default.yaml (a
# full-year run takes on the order of minutes, too slow for the automated suite; see
# docs/DECISIONS.md's F3 observation note and scratchpad/phase2-fix-report.md for the
# measured values: 1 reconsideration blocked, 0.0% annual switch rate). The fast,
# deterministic regression guard for "the penalty mechanism can block a switch" is
# test_switching_penalty_blocks_marginal_price_improvement in tests/test_agents.py.
