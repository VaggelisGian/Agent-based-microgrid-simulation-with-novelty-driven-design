"""MicrogridModel: the Mesa model tying together brokers, the heterogeneous
consumer/prosumer population, and the hourly step loop."""

from __future__ import annotations

import mesa
import numpy as np

from microgrid_sim.agents.consumer import Consumer
from microgrid_sim.agents.prosumer import Prosumer
from microgrid_sim.brokers.premium_green import PremiumGreenBroker
from microgrid_sim.brokers.regulated_utility import RegulatedUtilityBroker
from microgrid_sim.brokers.volatile_low_cost import VolatileLowCostBroker
from microgrid_sim.data.loaders import load_profiles
from microgrid_sim.environment import metrics as metrics_module

_BROKER_BUILDERS = {}  # populated below, keyed by config "type"


def _build_regulated_utility(cfg, model, regulated_base_eur_per_kwh):
    return RegulatedUtilityBroker(
        cfg["id"],
        cfg["name"],
        cfg["greenness"],
        cfg["base_eur_per_kwh"],
        cfg["seasonal_amplitude_eur_per_kwh"],
        cfg.get("seasonal_peak_day", 15),
    )


def _build_volatile_low_cost(cfg, model, regulated_base_eur_per_kwh):
    # A dedicated numpy Generator seeded from the model's own RNG (D5), so the
    # heavy-tailed shock process is reproducible given the model seed without
    # using an unseeded global source.
    broker_seed = model.random.randrange(2**32)
    rng = np.random.default_rng(broker_seed)
    return VolatileLowCostBroker(
        cfg["id"],
        cfg["name"],
        cfg["greenness"],
        cfg["baseline_eur_per_kwh"],
        cfg["mean_reversion_phi"],
        cfg["shock_scale_eur_per_kwh"],
        cfg["shock_degrees_of_freedom"],
        cfg["price_floor_eur_per_kwh"],
        cfg["price_cap_multiplier"],
        rng=rng,
    )


def _build_premium_green(cfg, model, regulated_base_eur_per_kwh):
    if regulated_base_eur_per_kwh is None:
        raise ValueError(
            "premium_green broker requires a regulated_utility broker in the same "
            "scenario to anchor its markup"
        )
    return PremiumGreenBroker(
        cfg["id"], cfg["name"], cfg["greenness"], regulated_base_eur_per_kwh, cfg["markup_eur_per_kwh"]
    )


_BROKER_BUILDERS.update(
    {
        "regulated_utility": _build_regulated_utility,
        "volatile_low_cost": _build_volatile_low_cost,
        "premium_green": _build_premium_green,
    }
)


class MicrogridModel(mesa.Model):
    def __init__(self, config: dict, seed=None):
        resolved_seed = seed if seed is not None else config.get("simulation", {}).get("seed")
        super().__init__(seed=resolved_seed)

        self.config = config
        self.horizon_hours = config["simulation"]["horizon_hours"]
        self.switching_enabled = config["switching"]["enabled"]
        self.num_agents = config["population"]["num_agents"]
        self.current_hour = 0
        self.current_prices: dict[str, float] = {}
        self.feeder_net_import_history: list[float] = []

        # simulation runs never touch the network: data/samples/ is pre-populated
        # (see docs/DECISIONS.md, "Data provenance"); a run should be hermetic and
        # fast even under a large parallel sweep.
        profiles = load_profiles(config, self.horizon_hours, allow_network_fetch=False)
        self.solar_profile = profiles.solar_kw_per_kwp
        self.demand_profile = profiles.demand_kwh_reference
        self.solar_source = profiles.solar_source
        self.demand_source = profiles.demand_source

        self.brokers = self._build_brokers(config["brokers"])
        self._build_population(config)

    def _build_brokers(self, broker_configs: list[dict]) -> dict:
        regulated_base_eur_per_kwh = None
        for cfg in broker_configs:
            if cfg["type"] == "regulated_utility":
                regulated_base_eur_per_kwh = cfg["base_eur_per_kwh"]
                break

        brokers = {}
        for cfg in broker_configs:
            builder = _BROKER_BUILDERS.get(cfg["type"])
            if builder is None:
                raise ValueError(f"unknown broker type: {cfg['type']}")
            broker = builder(cfg, self, regulated_base_eur_per_kwh)
            brokers[broker.id] = broker
        return brokers

    def _sample_uniform(self, bounds: dict) -> float:
        return self.random.uniform(bounds["min"], bounds["max"])

    def _sample_int(self, bounds: dict) -> int:
        return self.random.randint(int(bounds["min"]), int(bounds["max"]))

    def _build_population(self, config: dict) -> None:
        population_cfg = config["population"]
        params_cfg = config["agent_parameters"]
        dispatch_cfg = config["prosumer_dispatch"]

        num_agents = population_cfg["num_agents"]
        num_prosumers = round(num_agents * population_cfg["prosumer_fraction"])
        prosumer_indices = set(self.random.sample(range(num_agents), num_prosumers))

        profile_names = list(population_cfg["profile_mix"].keys())
        profile_weights = list(population_cfg["profile_mix"].values())
        broker_list = list(self.brokers.values())

        for index in range(num_agents):
            profile = self.random.choices(profile_names, weights=profile_weights, k=1)[0]
            demand_scale = self._sample_uniform(params_cfg["demand_scale"])
            price_tolerance = self._sample_uniform(params_cfg["price_tolerance_eur_per_kwh"])
            volatility_tolerance = self._sample_uniform(params_cfg["volatility_tolerance_eur_per_kwh"])
            greenness_threshold = self._sample_uniform(params_cfg["greenness_threshold"])
            switching_penalty = self._sample_uniform(params_cfg["switching_penalty_eur_per_kwh"])
            sustained_breach_hours = self._sample_int(params_cfg["sustained_breach_hours"])
            initial_broker = self.random.choice(broker_list)

            if index in prosumer_indices:
                Prosumer(
                    self,
                    profile,
                    demand_scale,
                    price_tolerance,
                    volatility_tolerance,
                    greenness_threshold,
                    switching_penalty,
                    sustained_breach_hours,
                    initial_broker,
                    self._sample_uniform(params_cfg["prosumer_pv_capacity_kwp"]),
                    self._sample_uniform(params_cfg["prosumer_battery_capacity_kwh"]),
                    dispatch_cfg["reserve_fraction"],
                    dispatch_cfg["evening_reserve_hour"],
                )
            else:
                Consumer(
                    self,
                    profile,
                    demand_scale,
                    price_tolerance,
                    volatility_tolerance,
                    greenness_threshold,
                    switching_penalty,
                    sustained_breach_hours,
                    initial_broker,
                )

    def step(self) -> None:
        hour = self.current_hour
        context: dict = {}
        self.current_prices = {broker_id: broker.quote(hour, context) for broker_id, broker in self.brokers.items()}

        self.agents.shuffle_do("step")

        feeder_net_import_kwh = sum(agent.last_net_import_kwh for agent in self.agents)
        self.feeder_net_import_history.append(feeder_net_import_kwh)

        self.current_hour += 1

    def run(self, hours: int | None = None) -> None:
        for _ in range(hours if hours is not None else self.horizon_hours):
            self.step()

    def compute_metrics(self) -> metrics_module.MetricsResult:
        return metrics_module.compute_metrics(self)
