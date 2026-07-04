"""Tests for the Phase 3 capacity mechanism (D6): the CapacityMechanism unit,
its integration into MicrogridModel behind the master flag, and the four
ablations' observable effects."""

import copy
import statistics

import pytest

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


# ---------------------------------------------------------------------------
# CapacityMechanism unit tests
# ---------------------------------------------------------------------------


def test_window_not_filled_total_charge_is_zero():
    mechanism = CapacityMechanism(window=4, k=1.0, charge_rate_eur_per_kwh=0.5, capacity_passthrough=0.1)
    for value in (10.0, 100.0, 1000.0):  # only 3 samples for a window of 4
        result = mechanism.step(value, {"a": value})
        assert result.threshold_eur is None
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
    assert result.threshold_eur == pytest.approx(expected_threshold, abs=1e-9)
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
    assert result.threshold_eur == pytest.approx(10.0, abs=1e-12)
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


def test_master_flag_off_means_prosumer_storage_response_is_inert():
    """With the mechanism disabled, broker_surcharges must stay all-zero for
    the whole run, so the prosumer storage response never activates and the
    unmodified D3 static-reserve dispatch is used throughout."""
    config = _competitive_config(horizon=48)
    model = MicrogridModel(config)
    for _ in range(48):
        model.step()
        assert all(value == 0.0 for value in model.broker_surcharges.values())
