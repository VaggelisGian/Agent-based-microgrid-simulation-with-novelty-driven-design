"""Prosumer agent: consumer plus a small PV and battery with reactive,
forecast-free dispatch (D3)."""

from __future__ import annotations

from microgrid_sim.agents.consumer import Consumer


class Prosumer(Consumer):
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
        pv_capacity_kwp: float,
        battery_capacity_kwh: float,
        reserve_fraction: float,
        evening_reserve_hour: int,
    ):
        super().__init__(
            model,
            profile,
            demand_scale,
            price_tolerance_eur_per_kwh,
            volatility_tolerance_eur_per_kwh,
            greenness_threshold,
            switching_penalty_eur_per_kwh,
            sustained_breach_hours,
            initial_broker,
        )
        self.pv_capacity_kwp = pv_capacity_kwp
        self.battery_capacity_kwh = battery_capacity_kwh
        self.reserve_fraction = reserve_fraction
        self.evening_reserve_hour = evening_reserve_hour
        # Starting at the reserve level is an arbitrary but deterministic and
        # neutral choice (neither empty nor full), so early-horizon behaviour
        # is not biased by an unstated initial-condition assumption.
        self.battery_soc_kwh = battery_capacity_kwh * reserve_fraction

        self.last_pv_kwh = 0.0
        self.total_demand_kwh = 0.0
        self.total_grid_import_kwh = 0.0

    def _capacity_release_fraction(self) -> float:
        """Phase 3 (D6): fraction of the otherwise-withheld D3 reserve this
        prosumer releases this step, driven purely by its own broker's
        CURRENT surcharge (carried over from the previous step; zero under
        capacity_disabled, capacity_pnl_only, or a below-threshold hour, and
        always zero while the model has no capacity mechanism at all). Reads
        model state defensively (getattr with an empty-dict / None default) so
        this is inert, not an AttributeError, against any host object (e.g. a
        test harness) that does not carry capacity-mechanism state -- which is
        exactly the "surcharge 0 -> unchanged D3 dispatch" baseline-preserving
        behaviour this mechanism must have by construction. Forecast-free,
        deterministic, draws no RNG.
        """
        broker_surcharges = getattr(self.model, "broker_surcharges", None)
        if not broker_surcharges:
            return 0.0
        surcharge = broker_surcharges.get(self.broker.id, 0.0)
        if surcharge <= 0.0:
            return 0.0
        response_reference = getattr(self.model, "capacity_response_reference_eur_per_kwh", 0.0)
        if not response_reference:
            return 0.0
        release = surcharge / response_reference
        return min(max(release, 0.0), 1.0)

    def _compute_net_import_kwh(self, hour: int, demand_kwh: float) -> float:
        pv_kwh = self.model.solar_profile[hour] * self.pv_capacity_kwp
        hour_of_day = hour % 24

        if pv_kwh >= demand_kwh:
            surplus_kwh = pv_kwh - demand_kwh
            headroom_kwh = self.battery_capacity_kwh - self.battery_soc_kwh
            charge_kwh = min(surplus_kwh, headroom_kwh)
            self.battery_soc_kwh += charge_kwh
            battery_net_discharge_kwh = -charge_kwh
        else:
            deficit_kwh = demand_kwh - pv_kwh
            if hour_of_day < self.evening_reserve_hour:
                reserve_kwh = self.reserve_fraction * self.battery_capacity_kwh
                normal_dischargeable_kwh = max(0.0, self.battery_soc_kwh - reserve_kwh)
                # Phase 3 (D6) storage response: a positive surcharge releases
                # a proportional fraction of the reserve energy actually held
                # in the battery (bounded by what is really there), on top of
                # the normal (unchanged) dischargeable amount. When surcharge
                # is 0 this is exactly 0.0, so normal_dischargeable_kwh alone
                # is used and the D3 baseline dispatch is preserved exactly.
                withheld_kwh = max(0.0, min(self.battery_soc_kwh, reserve_kwh))
                released_kwh = self._capacity_release_fraction() * withheld_kwh
                dischargeable_kwh = normal_dischargeable_kwh + released_kwh
            else:
                dischargeable_kwh = self.battery_soc_kwh
            # Bounded by the prosumer's own deficit: this is self-consumption
            # toward zero net import, never export-for-credit arbitrage.
            discharge_kwh = min(deficit_kwh, dischargeable_kwh)
            self.battery_soc_kwh -= discharge_kwh
            battery_net_discharge_kwh = discharge_kwh

        net_import_kwh = demand_kwh - pv_kwh - battery_net_discharge_kwh

        self.last_pv_kwh = pv_kwh
        self.total_demand_kwh += demand_kwh
        self.total_grid_import_kwh += max(net_import_kwh, 0.0)
        return net_import_kwh

    def self_sufficiency_ratio(self) -> float:
        """Fraction of this prosumer's demand met by own PV plus battery (metric 4)."""
        if self.total_demand_kwh <= 0.0:
            return 1.0
        return 1.0 - (self.total_grid_import_kwh / self.total_demand_kwh)
