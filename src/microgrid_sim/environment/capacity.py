"""Capacity mechanism (D6): shared-feeder scarcity charge, billing signal only.

No power flow, no voltage, no line limits: this module only ever consumes a
scalar "feeder net import" figure (already computed elsewhere from demand
minus PV minus battery discharge) and a per-broker contribution breakdown of
that same figure, and produces EUR charge/allocation/surcharge numbers.

Deterministic given feeder state: this module draws ZERO random numbers, so
enabling it, or flipping either feedback channel, never perturbs the RNG
stream consumed elsewhere in the model (broker shocks, population sampling).
All four ablations therefore share an identical RNG draw sequence at a given
seed; any measured difference between them is the mechanism, not divergent
randomness.

capacity_passthrough is a signal-strength coefficient that scales a
dimensionless contribution share into a next-step price adder. It is NOT a
EUR/kWh tariff rate and must never be described as one; charge_rate_eur_per_kwh
(a EUR per kWh-of-excess coefficient) is the separate quantity that sets the
CHARGE magnitude.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CapacityStepResult:
    """Everything the mechanism computed for one hourly step.

    threshold_eur is None while the rolling window is not yet full (no charge
    can be levied yet). allocations_eur and surcharges_eur_per_kwh are keyed
    by the same broker ids as the contributions dict passed into step(), and
    are exactly 0.0 for every broker whenever total_charge_eur is 0.0 or no
    broker has a positive contribution.
    """

    threshold_eur: float | None
    excess_kwh: float
    total_charge_eur: float
    allocations_eur: dict = field(default_factory=dict)
    surcharges_eur_per_kwh: dict = field(default_factory=dict)


class CapacityMechanism:
    """Stateful rolling-window scarcity-charge calculator (D6).

    Owns only: the trailing window of feeder net import samples, and
    run-level audit accumulators (total charge levied, hours observed, hours
    with positive excess). It does NOT own broker ledgers or next-step price
    state; MicrogridModel applies allocations_eur / surcharges_eur_per_kwh
    selectively depending on which feedback channel(s) are ablated in, which
    keeps the two feedback channels cleanly, independently switchable.
    """

    def __init__(
        self,
        window: int,
        k: float,
        charge_rate_eur_per_kwh: float,
        capacity_passthrough: float,
    ):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.window = window
        self.k = k
        self.charge_rate_eur_per_kwh = charge_rate_eur_per_kwh
        self.capacity_passthrough = capacity_passthrough

        self._history: deque[float] = deque(maxlen=window)

        # Run-level audit accumulators (Phase 5/6 use); see
        # environment/metrics.py's compute_capacity_audit.
        self.total_charge_eur = 0.0
        self.hours_observed = 0
        self.hours_with_excess = 0

    def step(self, feeder_net_import_kwh: float, broker_contributions_kwh: dict) -> CapacityStepResult:
        """Advance the mechanism by one hourly step.

        feeder_net_import_kwh: the ACTUAL (post-storage-response) feeder net
        import for this step (sum over all agents of demand minus PV minus
        battery discharge).
        broker_contributions_kwh: mapping broker_id -> the SIGNED sum of net
        import of that broker's current customers this step (may be negative
        for a broker whose customers net-exported this step).

        Per the order of operations this is called AFTER the sample has
        conceptually occurred, and this call itself both appends the sample to
        the rolling window and evaluates the threshold/excess/charge against
        that same, just-appended window (i.e. the trailing W hours ending at
        and including this step).
        """
        self._history.append(feeder_net_import_kwh)
        self.hours_observed += 1
        broker_ids = list(broker_contributions_kwh.keys())

        if len(self._history) < self.window:
            # Window not filled: no charge is levied yet (D6).
            zero_allocations = {broker_id: 0.0 for broker_id in broker_ids}
            zero_surcharges = {broker_id: 0.0 for broker_id in broker_ids}
            return CapacityStepResult(None, 0.0, 0.0, zero_allocations, zero_surcharges)

        mean = statistics.fmean(self._history)
        std = statistics.pstdev(self._history)
        threshold = mean + self.k * std
        excess = max(0.0, feeder_net_import_kwh - threshold)
        total_charge = self.charge_rate_eur_per_kwh * excess

        self.total_charge_eur += total_charge
        if total_charge > 0.0:
            self.hours_with_excess += 1

        positive_contrib = {broker_id: max(0.0, value) for broker_id, value in broker_contributions_kwh.items()}
        sum_positive_contrib = sum(positive_contrib.values())

        if total_charge <= 0.0 or sum_positive_contrib <= 0.0:
            allocations = {broker_id: 0.0 for broker_id in broker_ids}
            surcharges = {broker_id: 0.0 for broker_id in broker_ids}
        else:
            allocations = {
                broker_id: total_charge * (contribution / sum_positive_contrib)
                for broker_id, contribution in positive_contrib.items()
            }
            surcharges = {
                broker_id: self.capacity_passthrough * (contribution / sum_positive_contrib)
                for broker_id, contribution in positive_contrib.items()
            }

        return CapacityStepResult(threshold, excess, total_charge, allocations, surcharges)

    def fire_rate(self) -> float:
        """Fraction of observed hours with a positive excess (a positive
        charge levied), including the window-not-filled hours as non-firing."""
        if self.hours_observed == 0:
            return 0.0
        return self.hours_with_excess / self.hours_observed
