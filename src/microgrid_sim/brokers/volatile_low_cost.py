"""Volatile low-cost broker: cheap baseline plus a mean-reverting, heavy-tailed shock (D5)."""

from __future__ import annotations

import numpy as np

from microgrid_sim.brokers.base import Broker


class VolatileLowCostBroker(Broker):
    def __init__(
        self,
        broker_id: str,
        name: str,
        greenness: float,
        baseline_eur_per_kwh: float,
        mean_reversion_phi: float,
        shock_scale_eur_per_kwh: float,
        shock_degrees_of_freedom: float,
        price_floor_eur_per_kwh: float,
        price_cap_multiplier: float,
        rng: np.random.Generator,
    ):
        super().__init__(broker_id, name, "volatile_low_cost", greenness)
        self.baseline_eur_per_kwh = baseline_eur_per_kwh
        self.mean_reversion_phi = mean_reversion_phi
        self.shock_scale_eur_per_kwh = shock_scale_eur_per_kwh
        self.shock_degrees_of_freedom = shock_degrees_of_freedom
        self.price_floor_eur_per_kwh = price_floor_eur_per_kwh
        self.price_cap_eur_per_kwh = baseline_eur_per_kwh * price_cap_multiplier
        self._rng = rng
        self._shock = 0.0

    def quote(self, hour: int, context: dict | None = None) -> float:
        innovation = self._rng.standard_t(self.shock_degrees_of_freedom) * self.shock_scale_eur_per_kwh
        self._shock = self.mean_reversion_phi * self._shock + innovation
        price = self.baseline_eur_per_kwh + self._shock
        price = min(max(price, self.price_floor_eur_per_kwh), self.price_cap_eur_per_kwh)
        self._record_price(price)
        return price
