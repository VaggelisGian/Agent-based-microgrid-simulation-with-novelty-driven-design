"""Volatile low-cost broker: cheap baseline plus a mean-reverting, heavy-tailed shock (D5)."""

from __future__ import annotations

import math

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
        # Anti-windup (F4 fix): reset the latent AR(1) state to whatever the
        # clipped price implies, so a deep excursion beyond [floor, cap] does not
        # leave the state carrying "memory" of an overshoot that only decays
        # geometrically (phi per hour) while the visible price stays pinned flat
        # at the boundary for many hours. Without this, clipping only the output
        # produced multi-hour flat plateaus at the floor that are an artifact of
        # the clip, not the intended heavy-tailed spiky dynamic (see
        # docs/DECISIONS.md observation note for the measured before/after hit
        # rates).
        self._shock = price - self.baseline_eur_per_kwh
        self._record_price(price)
        return price

    def reference_price_eur_per_kwh(self) -> float:
        """The baseline level (F3 fix): the shock state starts at 0.0 and the
        innovation has zero mean, so this is the process's expected value 'at
        the start', with no dependence on the stochastic path."""
        return self.baseline_eur_per_kwh

    def reference_volatility_eur_per_kwh(self) -> float:
        """Analytically derived stationary standard deviation of the AR(1)
        shock process (F3 fix), used only to evaluate the stability-oriented
        profile's categorical filter at initial assignment time (before any
        real price history exists to compute the live rolling-window
        price_volatility()). For an AR(1) x_t = phi*x_{t-1} + e_t with iid
        innovations e_t, Var(x) = Var(e) / (1 - phi^2). Here e_t = shock_scale *
        T_df, and Var(T_df) = df / (df - 2) for df > 2 (a Student-t distribution
        with df <= 2 has undefined/infinite variance, treated here as an
        unbounded, certainly-not-stable process)."""
        df = self.shock_degrees_of_freedom
        if df <= 2:
            return float("inf")
        innovation_variance = (self.shock_scale_eur_per_kwh**2) * (df / (df - 2))
        stationary_variance = innovation_variance / (1 - self.mean_reversion_phi**2)
        return math.sqrt(stationary_variance)
