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
from microgrid_sim.environment.capacity import CapacityMechanism

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


def _initial_acceptable(profile: str, greenness_threshold: float, volatility_tolerance: float, broker) -> bool:
    """Mirrors Consumer._profile_acceptable's categorical filter (D2), but uses
    each broker's reference_volatility_eur_per_kwh() instead of the live
    price_volatility() rolling-window statistic, since no price history exists
    yet at population-construction time (F3 fix)."""
    if profile == "stability_oriented":
        return broker.reference_volatility_eur_per_kwh() <= volatility_tolerance
    if profile == "green_preferring":
        return broker.greenness >= greenness_threshold
    return True  # price_sensitive has no categorical filter (D2)


def _select_initial_broker(profile: str, greenness_threshold: float, volatility_tolerance: float, broker_list: list):
    """Initial broker assignment (F3 fix): pick the cheapest broker (by
    reference_price_eur_per_kwh()) among those that pass the agent's profile
    categorical filter, falling back to the overall cheapest if none are
    acceptable. Replaces the previous uniform-random initial assignment, which
    caused a one-time "born on the wrong broker, immediately flee" settling
    transient that dominated measured switching (see docs/DECISIONS.md
    observation note)."""
    acceptable = [broker for broker in broker_list if _initial_acceptable(profile, greenness_threshold, volatility_tolerance, broker)]
    candidates = acceptable if acceptable else broker_list
    return min(candidates, key=lambda broker: broker.reference_price_eur_per_kwh())


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
        self._build_capacity_mechanism(config)
        self._build_population(config)

    def _build_capacity_mechanism(self, config: dict) -> None:
        """Phase 3 (D6): set up the optional, additive, OFF-by-default capacity
        mechanism. capacity_mechanism absent from config is equivalent to it
        being present with enabled: false (see config/loader.py's
        _validate_capacity_mechanism); either way the model must reproduce the
        plain baseline byte-for-byte, so nothing here is allowed to alter any
        RNG draw or any arithmetic when disabled.
        """
        capacity_cfg = config.get("capacity_mechanism", {})
        self.capacity_enabled = capacity_cfg.get("enabled", False)
        self.capacity_feedback_pnl = capacity_cfg.get("feedback_pnl", False)
        self.capacity_feedback_pricing = capacity_cfg.get("feedback_pricing", False)
        self.capacity_response_reference_eur_per_kwh = capacity_cfg.get("response_reference_eur_per_kwh", 0.0)
        # Phase 3b (D7): price-elastic demand-deferral coefficients, the
        # current physical channel to metric 3 (replaces D6's prosumer
        # storage response). Both default to 0.0 when the block/keys are
        # absent, which -- combined with broker_surcharges staying all-zero
        # whenever the mechanism is off -- keeps deferral inert by
        # construction, not merely by convention.
        self.capacity_deferrable_fraction = capacity_cfg.get("deferrable_fraction", 0.0)
        self.capacity_payback_cap_fraction = capacity_cfg.get("payback_cap_fraction", 0.0)

        # Carried-over, per-broker NEXT-step surcharge (pricing channel only).
        # Always present (even when disabled) and always all-zero unless the
        # pricing feedback channel is both enabled and has actually computed a
        # positive surcharge, so Consumer._apply_demand_deferral (shared by
        # Prosumer) reading this dict is unconditionally safe and
        # unconditionally inert when the mechanism is off.
        self.broker_surcharges: dict[str, float] = {broker_id: 0.0 for broker_id in self.brokers}

        # Audit accumulators for the "mean surcharge per broker" metric: the
        # surcharge ACTUALLY applied to each step's quote (post-ablation-gate),
        # not the mechanism's raw internal computation.
        self._surcharge_accum: dict[str, float] = {broker_id: 0.0 for broker_id in self.brokers}
        self._surcharge_steps = 0

        self._capacity: CapacityMechanism | None = None
        if self.capacity_enabled:
            self._capacity = CapacityMechanism(
                window=capacity_cfg.get("window", 168),
                k=capacity_cfg.get("k", 1.0),
                charge_rate_eur_per_kwh=capacity_cfg.get("charge_rate_eur_per_kwh", 0.15),
                capacity_passthrough=capacity_cfg.get("capacity_passthrough", 0.10),
                surcharge_mode=capacity_cfg.get("capacity_surcharge_mode", "proportional"),
            )

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
            # F3 fix: deterministic, profile-compatible initial assignment
            # instead of uniform-random (see _select_initial_broker docstring).
            initial_broker = _select_initial_broker(profile, greenness_threshold, volatility_tolerance, broker_list)

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
        # Step 1 (D6 order of operations): brokers quote exactly as before (the
        # volatile broker draws exactly one shock regardless, so the RNG
        # stream is identical across all four ablations); each broker's
        # surcharge CARRIED OVER from the previous step (pricing channel only;
        # all-zero otherwise) is added on top afterwards, never inside quote().
        base_prices = {broker_id: broker.quote(hour, context) for broker_id, broker in self.brokers.items()}
        applied_surcharges = dict(self.broker_surcharges)
        self.current_prices = {
            broker_id: base_prices[broker_id] + applied_surcharges.get(broker_id, 0.0)
            for broker_id in self.brokers
        }
        for broker_id, surcharge in applied_surcharges.items():
            self._surcharge_accum[broker_id] += surcharge
        self._surcharge_steps += 1

        # Step 2: agents act. Every agent (consumer AND prosumer, via the
        # shared Consumer._apply_demand_deferral) reads its own broker's
        # current surcharge (self.broker_surcharges, just applied above) to
        # run the D7 price-elastic demand-deferral channel BEFORE a
        # prosumer's unmodified D3 PV/battery dispatch. Each agent's
        # last_broker_id records which broker actually served it THIS step,
        # even if it switches broker at the tail end of its own step() call.
        self.agents.shuffle_do("step")

        # Step 3: this step's ACTUAL (post-deferral) feeder net import;
        # metric 3 is computed from this same history, unchanged.
        feeder_net_import_kwh = sum(agent.last_net_import_kwh for agent in self.agents)
        self.feeder_net_import_history.append(feeder_net_import_kwh)

        if self.capacity_enabled:
            broker_contributions_kwh = {broker_id: 0.0 for broker_id in self.brokers}
            for agent in self.agents:
                broker_contributions_kwh[agent.last_broker_id] += agent.last_net_import_kwh

            # Step 4: threshold, excess, total charge, allocations.
            result = self._capacity.step(feeder_net_import_kwh, broker_contributions_kwh)

            # Step 5: P&L channel, independently ablatable.
            if self.capacity_feedback_pnl:
                for broker_id, allocation_eur in result.allocations_eur.items():
                    self.brokers[broker_id].debit_capacity_charge(allocation_eur)

            # Step 6: pricing channel, independently ablatable; else the
            # next-step surcharges are explicitly zero.
            if self.capacity_feedback_pricing:
                self.broker_surcharges = dict(result.surcharges_eur_per_kwh)
            else:
                self.broker_surcharges = {broker_id: 0.0 for broker_id in self.brokers}

        self.current_hour += 1

    def run(self, hours: int | None = None) -> None:
        for _ in range(hours if hours is not None else self.horizon_hours):
            self.step()

    def compute_metrics(self) -> metrics_module.MetricsResult:
        return metrics_module.compute_metrics(self)
