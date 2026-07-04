"""Shared broker interface and pricing helpers."""

from __future__ import annotations

import math
import statistics
from collections import deque

_DEFAULT_PRICE_HISTORY_HOURS = 24 * 30  # trailing window used for the volatility signal


class Broker:
    """Base class for the three fixed broker archetypes.

    Subclasses implement quote(hour, context); the contract is that quote is
    called at most once per hour, in increasing hour order, and that each call
    both returns the price and records it into the rolling price history (via
    _record_price), since price_volatility() reads that history.
    """

    def __init__(self, broker_id: str, name: str, archetype: str, greenness: float):
        if not (0.0 <= greenness <= 1.0):
            raise ValueError(f"greenness must be within [0, 1], got {greenness}")
        self.id = broker_id
        self.name = name
        self.archetype = archetype
        self.greenness = greenness
        self._price_history: deque[float] = deque(maxlen=_DEFAULT_PRICE_HISTORY_HOURS)
        self.cumulative_revenue_eur = 0.0
        self.cumulative_energy_served_kwh = 0.0

    def quote(self, hour: int, context: dict | None = None) -> float:
        raise NotImplementedError

    def _record_price(self, price: float) -> None:
        self._price_history.append(price)

    @property
    def price_history(self) -> tuple[float, ...]:
        return tuple(self._price_history)

    def price_volatility(self) -> float:
        """Population stdev of recent quoted prices; 0.0 until at least two exist."""
        if len(self._price_history) < 2:
            return 0.0
        return statistics.pstdev(self._price_history)

    def record_sale(self, price_eur_per_kwh: float, energy_kwh: float) -> None:
        """Record a billing event. energy_kwh may be negative (simple net-metering
        credit for a net-exporting prosumer hour); only the positive part counts
        as grid load served, but revenue reflects the actual signed cash flow."""
        self.cumulative_energy_served_kwh += max(energy_kwh, 0.0)
        self.cumulative_revenue_eur += price_eur_per_kwh * energy_kwh


def seasonal_flat_price(base: float, amplitude: float, day_of_year: int, peak_day: int) -> float:
    """A flat-within-day tariff with a slow cosine seasonal swing peaking at peak_day."""
    return base + amplitude * math.cos(2.0 * math.pi * (day_of_year - peak_day) / 365.0)
