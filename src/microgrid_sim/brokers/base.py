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
        # Signed running total (imports minus exports); used together with
        # cumulative_revenue_eur (also signed) so the per-kWh price metric's
        # numerator and denominator share the same energy scope (F1 fix, see
        # docs/DECISIONS.md observation note).
        self.cumulative_net_energy_kwh = 0.0
        # Phase 3 (D6): total capacity-charge allocation debited to this
        # broker's ledger over the run, via the capacity_feedback_pnl channel
        # only. Deliberately a SEPARATE accumulator from cumulative_revenue_eur
        # (which is sales revenue collected from customers, possibly already
        # inflated by the OTHER, independent pricing-surcharge channel), so
        # the two channels' effects can be attributed separately and the
        # existing cost metric's scope is not corrupted by this addition.
        self.cumulative_capacity_charge_eur = 0.0

    def quote(self, hour: int, context: dict | None = None) -> float:
        raise NotImplementedError

    def reference_price_eur_per_kwh(self) -> float:
        """Deterministic 'at the start' price used only to rank brokers for
        initial broker assignment (F3 fix), before quote() has ever been called.
        Must not depend on stochastic internal state or price history, since at
        population-construction time neither exists yet. Subclasses override."""
        raise NotImplementedError

    def reference_volatility_eur_per_kwh(self) -> float:
        """Deterministic 'typical' price volatility used only to evaluate a
        stability-oriented profile's categorical filter at initial broker
        assignment time (F3 fix), before any real price history exists (so the
        live price_volatility() rolling-window statistic, which needs at least
        two recorded quotes, cannot yet be used). Default: archetypally stable
        (0.0), which is exactly correct for brokers with a deterministic or flat
        price process; the volatile broker overrides this with the analytically
        derived stationary standard deviation of its shock process. This does
        not change price_volatility() itself, which is still what agents use for
        every categorical check after the simulation starts."""
        return 0.0

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
        as grid load served (cumulative_energy_served_kwh, used for the load-share
        metric), but revenue and cumulative_net_energy_kwh both reflect the actual
        signed value (imports minus exports), so a per-kWh price computed from
        them is scope-consistent (F1 fix)."""
        self.cumulative_energy_served_kwh += max(energy_kwh, 0.0)
        self.cumulative_net_energy_kwh += energy_kwh
        self.cumulative_revenue_eur += price_eur_per_kwh * energy_kwh

    def debit_capacity_charge(self, amount_eur: float) -> None:
        """Phase 3 (D6), capacity_feedback_pnl channel: reduce this broker's
        P&L by its allocated share of a step's scarcity charge. Pure
        accounting -- does not touch cumulative_revenue_eur (sales revenue)
        or write anything into prices; see cumulative_capacity_charge_eur."""
        self.cumulative_capacity_charge_eur += amount_eur


def seasonal_flat_price(base: float, amplitude: float, day_of_year: int, peak_day: int) -> float:
    """A flat-within-day tariff with a slow cosine seasonal swing peaking at peak_day."""
    return base + amplitude * math.cos(2.0 * math.pi * (day_of_year - peak_day) / 365.0)
