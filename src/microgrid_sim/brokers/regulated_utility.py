"""Regulated utility broker: flat seasonal tariff, stable and relatively expensive."""

from __future__ import annotations

from microgrid_sim.brokers.base import Broker, seasonal_flat_price


class RegulatedUtilityBroker(Broker):
    def __init__(
        self,
        broker_id: str,
        name: str,
        greenness: float,
        base_eur_per_kwh: float,
        seasonal_amplitude_eur_per_kwh: float,
        seasonal_peak_day: int = 15,
    ):
        super().__init__(broker_id, name, "regulated_utility", greenness)
        self.base_eur_per_kwh = base_eur_per_kwh
        self.seasonal_amplitude_eur_per_kwh = seasonal_amplitude_eur_per_kwh
        self.seasonal_peak_day = seasonal_peak_day

    def quote(self, hour: int, context: dict | None = None) -> float:
        day_of_year = (hour // 24) % 365
        price = seasonal_flat_price(
            self.base_eur_per_kwh,
            self.seasonal_amplitude_eur_per_kwh,
            day_of_year,
            self.seasonal_peak_day,
        )
        self._record_price(price)
        return price
