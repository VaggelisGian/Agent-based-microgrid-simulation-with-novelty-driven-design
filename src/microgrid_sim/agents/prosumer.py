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

    def _compute_net_import_kwh(self, hour: int, demand_kwh: float) -> float:
        """Pure D3 reactive dispatch: charge from surplus PV, discharge to
        cover local demand, holding back a fixed evening reserve until
        evening_reserve_hour. No price signal, no forecast, no optimization.
        (Phase 3's D6 scarcity-triggered early reserve-release has been
        removed per D7: the physical channel to metric 3 is now the
        price-elastic demand deferral applied to `demand_kwh` upstream, in
        Consumer._apply_demand_deferral, BEFORE this dispatch runs -- so
        `demand_kwh` here is already the post-deferral served demand, not raw
        base demand, but the dispatch logic itself is unmodified D3.)
        """
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
                dischargeable_kwh = max(0.0, self.battery_soc_kwh - reserve_kwh)
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
