"""Premium green broker: stable premium over the regulated level, high greenness."""

from __future__ import annotations

from microgrid_sim.brokers.base import Broker


class PremiumGreenBroker(Broker):
    def __init__(
        self,
        broker_id: str,
        name: str,
        greenness: float,
        regulated_base_eur_per_kwh: float,
        markup_eur_per_kwh: float,
    ):
        super().__init__(broker_id, name, "premium_green", greenness)
        # Priced as a flat markup over the regulated broker's base level (not its
        # seasonal swing), reflecting a fixed-rate green tariff product that is
        # deliberately more stable than the regulated broker it is benchmarked against.
        self.price_eur_per_kwh = regulated_base_eur_per_kwh + markup_eur_per_kwh

    def quote(self, hour: int, context: dict | None = None) -> float:
        self._record_price(self.price_eur_per_kwh)
        return self.price_eur_per_kwh
