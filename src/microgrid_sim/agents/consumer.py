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
        # Phase 3b (D7): intra-day deferred-energy bucket for the price-elastic
        # demand-deferral channel (see _apply_demand_deferral). Reset to 0.0 at
        # hour_of_day == 0; always exactly 0.0 whenever this agent's own
        # broker's surcharge stays 0 all day, which is the
        # capacity_disabled / capacity_pnl_only case.
        self.deferred_kwh = 0.0
        # Run-level audit accumulator: total energy actually DEFERRED (moved
        # out of a scarcity hour), summed across the whole run; never
        # decremented by a later payback (that is the same energy being
        # served later, not new deferral). See environment/metrics.py's
        # compute_deferral_audit.
        self.total_deferred_kwh = 0.0

    def step(self) -> None:
        hour = self.model.current_hour
        base_demand_kwh = self._compute_demand_kwh(hour)
        demand_kwh = self._apply_demand_deferral(hour, base_demand_kwh)
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

    def _current_capacity_surcharge(self) -> float:
        """Phase 3b (D7): this agent's own broker's CURRENT capacity surcharge
        (carried over from the previous step by MicrogridModel.step; zero
        under capacity_disabled, capacity_pnl_only, a below-threshold hour, or
        whenever the host model carries no capacity-mechanism state at all).
        Reads model state defensively (getattr with an empty-dict / None
        default) so this is inert, not an AttributeError, against any host
        object (e.g. a bare test harness) that does not carry
        capacity-mechanism state -- exactly the "surcharge 0 => served ==
        base" baseline-preserving behaviour D7 requires by construction."""
        broker_surcharges = getattr(self.model, "broker_surcharges", None)
        if not broker_surcharges:
            return 0.0
        return broker_surcharges.get(self.broker.id, 0.0)

    def _apply_demand_deferral(self, hour: int, base_kwh: float) -> float:
        """Phase 3b (D7): price-elastic intra-day demand deferral, the
        physical channel to metric 3 (replaces D6's prosumer storage-response,
        removed). Shared by Consumer and Prosumer (Prosumer does not override
        this); for a Prosumer this runs BEFORE the unchanged D3 PV/battery
        dispatch (see Prosumer._compute_net_import_kwh), so the prosumer's
        battery/PV logic sees the deferred/served demand, not raw base demand.
        Forecast-free (reacts only to the CURRENT surcharge), deterministic,
        draws no RNG. Inert (served == base every hour) whenever the
        surcharge stays identically zero all day -- exactly the
        capacity_disabled / capacity_pnl_only case -- which preserves the
        baseline byte-for-byte.
        """
        hour_of_day = hour % 24
        if hour_of_day == 0:
            self.deferred_kwh = 0.0

        surcharge = self._current_capacity_surcharge()

        if surcharge > 0.0:
            response_reference = getattr(self.model, "capacity_response_reference_eur_per_kwh", 0.0)
            deferrable_fraction = getattr(self.model, "capacity_deferrable_fraction", 0.0)
            factor = min(max(surcharge / response_reference, 0.0), 1.0) if response_reference > 0.0 else 0.0
            defer_kwh = deferrable_fraction * base_kwh * factor
            served_kwh = base_kwh - defer_kwh
            self.deferred_kwh += defer_kwh
            self.total_deferred_kwh += defer_kwh
        elif surcharge == 0.0 and self.deferred_kwh > 0.0:
            payback_cap_fraction = getattr(self.model, "capacity_payback_cap_fraction", 0.0)
            payback_kwh = min(self.deferred_kwh, payback_cap_fraction * base_kwh)
            served_kwh = base_kwh + payback_kwh
            self.deferred_kwh -= payback_kwh
        else:
            served_kwh = base_kwh

        # Daily energy conservation: any residual left in the bucket at the
        # last hour of the day is repaid in full, so total served energy over
        # each day equals total base demand over that day (deferral SHIFTS
        # load, it never deletes it).
        if hour_of_day == 23 and self.deferred_kwh > 0.0:
            served_kwh += self.deferred_kwh
            self.deferred_kwh = 0.0

        return served_kwh

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
