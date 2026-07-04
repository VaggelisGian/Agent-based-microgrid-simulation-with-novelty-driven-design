"""Consumer agent: hourly billed demand with lexicographic broker choice under
switching inertia (D1, D2, D4)."""

from __future__ import annotations

import mesa

_VALID_PROFILES = ("price_sensitive", "stability_oriented", "green_preferring")


class Consumer(mesa.Agent):
    def __init__(
        self,
        model,
        profile: str,
        demand_scale: float,
        price_tolerance_eur_per_kwh: float,
        volatility_tolerance_eur_per_kwh: float,
        greenness_threshold: float,
        switching_penalty_eur_per_kwh: float,
        sustained_breach_hours: int,
        initial_broker,
    ):
        super().__init__(model)
        if profile not in _VALID_PROFILES:
            raise ValueError(f"unknown profile: {profile}")
        self.profile = profile
        self.demand_scale = demand_scale
        self.price_tolerance_eur_per_kwh = price_tolerance_eur_per_kwh
        self.volatility_tolerance_eur_per_kwh = volatility_tolerance_eur_per_kwh
        self.greenness_threshold = greenness_threshold
        self.switching_penalty_eur_per_kwh = switching_penalty_eur_per_kwh
        self.sustained_breach_hours = sustained_breach_hours
        self.broker = initial_broker

        self._breach_streak = 0
        self.switch_count = 0
        # Counts reconsiderations where a cheaper acceptable broker existed but
        # the price gap did not clear switching_penalty_eur_per_kwh (F3
        # observability; see docs/DECISIONS.md observation note).
        self.blocked_by_penalty_count = 0
        self.last_demand_kwh = 0.0
        self.last_net_import_kwh = 0.0
        self.last_billed_eur = 0.0
        # Phase 3 (D6): the broker id this agent was actually billed under THIS
        # step, captured before _update_inertia_and_maybe_switch below can
        # mutate self.broker for the NEXT step. MicrogridModel.step groups
        # per-broker customer contributions by this field (not by self.broker,
        # which may already reflect a same-step switch by the time the model
        # reads it), so a broker's contribution to a step's feeder peak is
        # attributed to the broker that actually served that step's demand.
        self.last_broker_id = initial_broker.id

    def step(self) -> None:
        hour = self.model.current_hour
        demand_kwh = self._compute_demand_kwh(hour)
        net_import_kwh = self._compute_net_import_kwh(hour, demand_kwh)

        price = self.model.current_prices[self.broker.id]
        billed_eur = net_import_kwh * price
        self.broker.record_sale(price, net_import_kwh)

        self.last_demand_kwh = demand_kwh
        self.last_net_import_kwh = net_import_kwh
        self.last_billed_eur = billed_eur
        self.last_broker_id = self.broker.id

        if self.model.switching_enabled:
            self._update_inertia_and_maybe_switch(price)

    def _compute_demand_kwh(self, hour: int) -> float:
        return self.model.demand_profile[hour] * self.demand_scale

    def _compute_net_import_kwh(self, hour: int, demand_kwh: float) -> float:
        return demand_kwh

    def _breach_condition(self, current_price: float) -> bool:
        if self.profile == "price_sensitive":
            return current_price > self.price_tolerance_eur_per_kwh
        if self.profile == "stability_oriented":
            return self.broker.price_volatility() > self.volatility_tolerance_eur_per_kwh
        return self.broker.greenness < self.greenness_threshold  # green_preferring

    def _profile_acceptable(self, broker) -> bool:
        if self.profile == "stability_oriented":
            return broker.price_volatility() <= self.volatility_tolerance_eur_per_kwh
        if self.profile == "green_preferring":
            return broker.greenness >= self.greenness_threshold
        return True  # price_sensitive has no categorical filter (D2)

    def _selection_key(self, broker):
        price = self.model.current_prices[broker.id]
        if self.profile == "price_sensitive":
            return (price, broker.price_volatility())  # tie-break on stability (D2)
        return (price,)

    def _update_inertia_and_maybe_switch(self, current_price: float) -> None:
        if self._breach_condition(current_price):
            self._breach_streak += 1
        else:
            self._breach_streak = 0
            return

        if self._breach_streak < self.sustained_breach_hours:
            return

        self._reconsider()
        self._breach_streak = 0

    def _reconsider(self) -> None:
        brokers = list(self.model.brokers.values())
        current_acceptable = self._profile_acceptable(self.broker)
        acceptable = [broker for broker in brokers if self._profile_acceptable(broker)]
        candidates = acceptable if acceptable else brokers
        best = min(candidates, key=self._selection_key)

        if best.id == self.broker.id:
            return

        if not current_acceptable:
            # current broker no longer meets the profile's categorical requirement
            # (D2 filter): flee it even if the alternative is not much cheaper.
            self.broker = best
            self.switch_count += 1
            return

        current_price = self.model.current_prices[self.broker.id]
        best_price = self.model.current_prices[best.id]
        if (current_price - best_price) > self.switching_penalty_eur_per_kwh:
            self.broker = best
            self.switch_count += 1
        else:
            self.blocked_by_penalty_count += 1
