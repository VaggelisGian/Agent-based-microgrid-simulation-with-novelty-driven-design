"""YAML scenario configuration loading and validation.

Scenario files may set an ``extends: <relative-path>`` key to reuse another
file's values and override only a subset (see config/scenarios/monopoly_baseline.yaml).
The override rule is: dicts merge recursively key by key; any other value
(including lists, such as the broker set) fully replaces the base value.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_KNOWN_BROKER_TYPES = {"regulated_utility", "volatile_low_cost", "premium_green"}

_REQUIRED_BROKER_FIELDS = {
    "regulated_utility": {
        "id",
        "name",
        "greenness",
        "base_eur_per_kwh",
        "seasonal_amplitude_eur_per_kwh",
        "seasonal_peak_day",
    },
    "volatile_low_cost": {
        "id",
        "name",
        "greenness",
        "baseline_eur_per_kwh",
        "mean_reversion_phi",
        "shock_scale_eur_per_kwh",
        "shock_degrees_of_freedom",
        "price_floor_eur_per_kwh",
        "price_cap_multiplier",
    },
    "premium_green": {"id", "name", "greenness", "markup_eur_per_kwh"},
}

_REQUIRED_TOP_LEVEL_KEYS = (
    "simulation",
    "population",
    "switching",
    "agent_parameters",
    "prosumer_dispatch",
    "data",
    "brokers",
)

# F6 fix: the known agent_parameters sub-keys (each a {min, max} sampled range),
# matching what MicrogridModel._build_population expects (environment/model.py).
_AGENT_PARAMETER_KEYS = (
    "demand_scale",
    "price_tolerance_eur_per_kwh",
    "volatility_tolerance_eur_per_kwh",
    "greenness_threshold",
    "switching_penalty_eur_per_kwh",
    "sustained_breach_hours",
    "prosumer_pv_capacity_kwp",
    "prosumer_battery_capacity_kwh",
)

_PROSUMER_DISPATCH_KEYS = ("reserve_fraction", "evening_reserve_hour")

# Phase 3 (D6): the capacity_mechanism block is OPTIONAL at the top level (its
# absence means the mechanism is disabled, matching the master flag's own
# documented default), so it is deliberately NOT in _REQUIRED_TOP_LEVEL_KEYS;
# this keeps every pre-Phase-3 config (including the minimal test fixture in
# tests/test_config_loader.py) valid without edits. When the block IS present,
# every key below is required and validated.
_CAPACITY_MECHANISM_KEYS = (
    "enabled",
    "feedback_pnl",
    "feedback_pricing",
    "window",
    "k",
    "charge_rate_eur_per_kwh",
    "capacity_passthrough",
    "response_reference_eur_per_kwh",
)
_CAPACITY_MECHANISM_BOOL_KEYS = ("enabled", "feedback_pnl", "feedback_pricing")
_CAPACITY_MECHANISM_NONNEGATIVE_NUMERIC_KEYS = (
    "k",
    "charge_rate_eur_per_kwh",
    "capacity_passthrough",
    "response_reference_eur_per_kwh",
)


class ConfigError(ValueError):
    """Raised when a scenario configuration file is missing or invalid."""


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge(base_value, override_value)
        else:
            merged[key] = copy.deepcopy(override_value)
    return merged


def _read_yaml_file(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="ascii") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ConfigError(f"config file did not parse to a mapping: {path}")
    return content


def _load_raw(path: Path) -> dict:
    content = _read_yaml_file(path)
    extends = content.pop("extends", None)
    if extends is None:
        return content
    base_path = (path.parent / extends).resolve()
    base_content = _load_raw(base_path)
    return _deep_merge(base_content, content)


def load_config(path: str | Path) -> dict:
    """Load a scenario YAML file, resolving any ``extends`` chain, and validate it."""
    resolved_path = Path(path)
    try:
        config = _load_raw(resolved_path)
    except ConfigError:
        raise
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read config file {resolved_path}: {exc}") from exc
    validate_config(config)
    return config


def _require_keys(mapping: dict, keys, context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{context} missing required key(s): {sorted(missing)}")


def validate_config(config: dict) -> None:
    """Raise ConfigError if the config dict is structurally or semantically invalid."""
    _require_keys(config, _REQUIRED_TOP_LEVEL_KEYS, "config")

    horizon_hours = config["simulation"].get("horizon_hours")
    if not isinstance(horizon_hours, int) or horizon_hours <= 0:
        raise ConfigError("simulation.horizon_hours must be a positive integer")

    _require_keys(config["switching"], ("enabled",), "switching")
    if not isinstance(config["switching"]["enabled"], bool):
        raise ConfigError("switching.enabled must be a boolean")

    population = config["population"]
    _require_keys(population, ("num_agents", "prosumer_fraction", "profile_mix"), "population")

    if not isinstance(population["num_agents"], int) or population["num_agents"] <= 0:
        raise ConfigError("population.num_agents must be a positive integer")

    prosumer_fraction = population["prosumer_fraction"]
    if not (0.0 <= prosumer_fraction <= 1.0):
        raise ConfigError("population.prosumer_fraction must be within [0, 1]")

    profile_mix = population["profile_mix"]
    if not profile_mix:
        raise ConfigError("population.profile_mix must not be empty")
    mix_total = sum(profile_mix.values())
    if abs(mix_total - 1.0) > 1e-6:
        raise ConfigError(f"population.profile_mix must sum to 1.0, got {mix_total}")

    brokers = config["brokers"]
    if not brokers:
        raise ConfigError("brokers list must not be empty")

    seen_ids = set()
    for broker in brokers:
        _require_keys(broker, ("id", "type"), "broker entry")
        broker_type = broker["type"]
        if broker_type not in _KNOWN_BROKER_TYPES:
            raise ConfigError(f"unknown broker type: {broker_type}")
        _require_keys(
            broker, _REQUIRED_BROKER_FIELDS[broker_type], f"broker '{broker['id']}' ({broker_type})"
        )
        if broker["id"] in seen_ids:
            raise ConfigError(f"duplicate broker id: {broker['id']}")
        seen_ids.add(broker["id"])
        greenness = broker["greenness"]
        if not (0.0 <= greenness <= 1.0):
            raise ConfigError(f"broker '{broker['id']}' greenness must be within [0, 1]")

    # F6 fix: agent_parameters structure and min <= max for every sampled range.
    # A missing key previously surfaced as a raw KeyError deep inside
    # MicrogridModel._build_population, well after config loading appeared to
    # succeed; an inverted bound previously silently reversed the sampled
    # distribution (random.uniform/randint do not validate min <= max).
    agent_parameters = config["agent_parameters"]
    _require_keys(agent_parameters, _AGENT_PARAMETER_KEYS, "agent_parameters")
    for key in _AGENT_PARAMETER_KEYS:
        bounds = agent_parameters[key]
        if not isinstance(bounds, dict):
            raise ConfigError(f"agent_parameters.{key} must be a mapping with 'min' and 'max' keys")
        _require_keys(bounds, ("min", "max"), f"agent_parameters.{key}")
        if bounds["min"] > bounds["max"]:
            raise ConfigError(
                f"agent_parameters.{key}: min ({bounds['min']}) must be <= max ({bounds['max']})"
            )

    # F6 fix: prosumer_dispatch structure and value ranges.
    prosumer_dispatch = config["prosumer_dispatch"]
    _require_keys(prosumer_dispatch, _PROSUMER_DISPATCH_KEYS, "prosumer_dispatch")
    reserve_fraction = prosumer_dispatch["reserve_fraction"]
    if not (0.0 <= reserve_fraction <= 1.0):
        raise ConfigError("prosumer_dispatch.reserve_fraction must be within [0, 1]")
    evening_reserve_hour = prosumer_dispatch["evening_reserve_hour"]
    if not isinstance(evening_reserve_hour, int) or not (0 <= evening_reserve_hour <= 23):
        raise ConfigError("prosumer_dispatch.evening_reserve_hour must be an integer within [0, 23]")

    _validate_capacity_mechanism(config)


def _validate_capacity_mechanism(config: dict) -> None:
    """Phase 3 (D6): validate the optional capacity_mechanism block, if present.

    capacity_passthrough is a signal-strength coefficient (a dimensionless-to-
    EUR/kWh scaling factor for a contribution SHARE), not a EUR/kWh tariff
    rate; charge_rate_eur_per_kwh is the separate CHARGE-magnitude coefficient.
    Both are validated only for basic type/sign sanity here, not against any
    tuned outcome.
    """
    if "capacity_mechanism" not in config:
        return
    capacity = config["capacity_mechanism"]
    if not isinstance(capacity, dict):
        raise ConfigError("capacity_mechanism must be a mapping")
    _require_keys(capacity, _CAPACITY_MECHANISM_KEYS, "capacity_mechanism")

    for key in _CAPACITY_MECHANISM_BOOL_KEYS:
        if not isinstance(capacity[key], bool):
            raise ConfigError(f"capacity_mechanism.{key} must be a boolean")

    window = capacity["window"]
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ConfigError("capacity_mechanism.window must be an integer >= 1")

    for key in _CAPACITY_MECHANISM_NONNEGATIVE_NUMERIC_KEYS:
        value = capacity[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"capacity_mechanism.{key} must be a non-negative number")
