import copy

import pytest

from microgrid_sim.config.loader import ConfigError, load_config, validate_config

DEFAULT_CONFIG_PATH = "config/default.yaml"
MONOPOLY_CONFIG_PATH = "config/scenarios/monopoly_baseline.yaml"


def _minimal_valid_config():
    return {
        "scenario_name": "test",
        "simulation": {"horizon_hours": 168, "seed": 1},
        "population": {
            "num_agents": 10,
            "prosumer_fraction": 0.2,
            "profile_mix": {
                "price_sensitive": 0.4,
                "stability_oriented": 0.3,
                "green_preferring": 0.3,
            },
        },
        "switching": {"enabled": True},
        "agent_parameters": {
            "demand_scale": {"min": 0.6, "max": 1.6},
            "price_tolerance_eur_per_kwh": {"min": 0.18, "max": 0.32},
            "volatility_tolerance_eur_per_kwh": {"min": 0.01, "max": 0.05},
            "greenness_threshold": {"min": 0.35, "max": 0.75},
            "switching_penalty_eur_per_kwh": {"min": 0.003, "max": 0.02},
            "sustained_breach_hours": {"min": 12, "max": 96},
            "prosumer_pv_capacity_kwp": {"min": 1.5, "max": 4.5},
            "prosumer_battery_capacity_kwh": {"min": 4.0, "max": 12.0},
        },
        "prosumer_dispatch": {"reserve_fraction": 0.2, "evening_reserve_hour": 18},
        "data": {
            "solar_profile_path": "data/samples/solar_thessaloniki_hourly.csv",
            "demand_profile_path": "data/samples/residential_demand_thessaloniki_hourly.csv",
        },
        "brokers": [
            {
                "id": "regulated_utility",
                "type": "regulated_utility",
                "name": "Regulated Utility",
                "greenness": 0.3,
                "base_eur_per_kwh": 0.19,
                "seasonal_amplitude_eur_per_kwh": 0.02,
                "seasonal_peak_day": 15,
            }
        ],
    }


def test_load_default_config_has_required_top_level_keys():
    config = load_config(DEFAULT_CONFIG_PATH)
    for key in ("simulation", "population", "switching", "brokers", "data"):
        assert key in config


def test_default_config_has_three_brokers_and_is_valid():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert len(config["brokers"]) == 3
    validate_config(config)


def test_monopoly_config_extends_default_and_overrides_only_brokers_and_switching():
    default_config = load_config(DEFAULT_CONFIG_PATH)
    monopoly_config = load_config(MONOPOLY_CONFIG_PATH)

    assert monopoly_config["switching"]["enabled"] is False
    assert len(monopoly_config["brokers"]) == 1
    assert monopoly_config["brokers"][0]["type"] == "regulated_utility"

    # everything else is reused from default.yaml
    assert monopoly_config["population"] == default_config["population"]
    assert monopoly_config["agent_parameters"] == default_config["agent_parameters"]
    assert monopoly_config["simulation"] == default_config["simulation"]


def test_validate_config_accepts_minimal_valid_config():
    validate_config(_minimal_valid_config())


def test_validate_config_rejects_profile_mix_not_summing_to_one():
    config = _minimal_valid_config()
    config["population"]["profile_mix"]["price_sensitive"] = 0.9
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_prosumer_fraction_out_of_range():
    config = _minimal_valid_config()
    config["population"]["prosumer_fraction"] = 1.5
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_empty_broker_list():
    config = _minimal_valid_config()
    config["brokers"] = []
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_unknown_broker_type():
    config = _minimal_valid_config()
    config["brokers"][0]["type"] = "not_a_real_broker_type"
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_missing_required_broker_field():
    config = _minimal_valid_config()
    del config["brokers"][0]["base_eur_per_kwh"]
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_zero_horizon():
    config = _minimal_valid_config()
    config["simulation"]["horizon_hours"] = 0
    with pytest.raises(ConfigError):
        validate_config(config)


def test_load_config_does_not_mutate_input_between_calls():
    first = load_config(DEFAULT_CONFIG_PATH)
    first["population"]["num_agents"] = 999999
    second = load_config(DEFAULT_CONFIG_PATH)
    assert second["population"]["num_agents"] != 999999


def test_load_config_missing_file_raises_config_error():
    with pytest.raises(ConfigError):
        load_config("config/scenarios/does_not_exist.yaml")


def test_validate_config_rejects_missing_agent_parameter_key():
    config = _minimal_valid_config()
    del config["agent_parameters"]["demand_scale"]
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_inverted_agent_parameter_bounds():
    config = _minimal_valid_config()
    config["agent_parameters"]["demand_scale"] = {"min": 1.6, "max": 0.6}
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_missing_prosumer_dispatch_key():
    config = _minimal_valid_config()
    del config["prosumer_dispatch"]["evening_reserve_hour"]
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_reserve_fraction_out_of_range():
    config = _minimal_valid_config()
    config["prosumer_dispatch"]["reserve_fraction"] = 1.5
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_negative_reserve_fraction():
    config = _minimal_valid_config()
    config["prosumer_dispatch"]["reserve_fraction"] = -0.1
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_rejects_evening_reserve_hour_out_of_range():
    config = _minimal_valid_config()
    config["prosumer_dispatch"]["evening_reserve_hour"] = 24
    with pytest.raises(ConfigError):
        validate_config(config)


def test_validate_config_accepts_boundary_reserve_values():
    config = _minimal_valid_config()
    config["prosumer_dispatch"]["reserve_fraction"] = 0.0
    config["prosumer_dispatch"]["evening_reserve_hour"] = 0
    validate_config(config)  # must not raise
    config["prosumer_dispatch"]["reserve_fraction"] = 1.0
    config["prosumer_dispatch"]["evening_reserve_hour"] = 23
    validate_config(config)  # must not raise
