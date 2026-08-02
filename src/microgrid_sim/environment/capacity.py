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

surcharge_mode (D10) controls only how the SAME total signal strength is
spread across brokers' next-step surcharges: "proportional" (default, the
original behaviour) splits it by contribution share; "synchronized" and
"renormalized" are analysis-only controls that separate desynchronized
deferral timing from deferred volume (see docs/DECISIONS.md D10). The
allocations (P&L channel) are always strictly proportional, in every mode.

surcharge_divisor (D11) decouples the pricing channel's effective broker
count from the actual one, so a single broker can be evaluated as though it
faced the per-broker signal strength an M-broker market would deliver. It is
an ARTIFICIAL ANALYSIS-ONLY CONTROL WITH NO MARKET INTERPRETATION: under the
shipped allocation rule a broker cannot choose to receive passthrough/M, it
receives the full passthrough its contribution share entitles it to, by
construction. Its only purpose is the D11 dose-matched monopoly arm, which
needs to hold broker_count at 1 (one supplier) while separately dialing the
per-supplier dose to whatever an M-supplier market would deliver, something
the shipped design cannot do because broker_count moves both the supplier
count and the per-broker signal strength at once. Default None means "use the
actual active broker count", which is the shipped behaviour, bit for bit; the
substitution never runs at all in that case (see step() below), and every
existing result and golden-master pin is therefore untouched by this control's
existence. The allocations (P&L channel) are never affected by the divisor,
exactly as they are never affected by surcharge_mode.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CapacityStepResult:
    """Everything the mechanism computed for one hourly step.

    threshold_kwh is None while the rolling window is not yet full (no charge
    can be levied yet). It is a threshold on the feeder's own net import
    (mean + k * std over the trailing window), so its unit is kWh, not EUR;
    an earlier name of threshold_eur said otherwise and was wrong.
    allocations_eur and surcharges_eur_per_kwh are keyed
    by the same broker ids as the contributions dict passed into step(), and
    are exactly 0.0 for every broker whenever total_charge_eur is 0.0 or no
    broker has a positive contribution.
    """

    threshold_kwh: float | None
    excess_kwh: float
    total_charge_eur: float
    allocations_eur: dict = field(default_factory=dict)
    surcharges_eur_per_kwh: dict = field(default_factory=dict)


SURCHARGE_MODES = ("proportional", "synchronized", "renormalized")


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
        surcharge_mode: str = "proportional",
        surcharge_divisor: int | None = None,
    ):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if surcharge_mode not in SURCHARGE_MODES:
            raise ValueError(f"surcharge_mode must be one of {SURCHARGE_MODES}, got {surcharge_mode!r}")
        # bool is an int subclass in Python, so isinstance(x, bool) has to be
        # checked before isinstance(x, int) or True/False would silently pass
        # as 1/0 (the same trap _validate_capacity_mechanism in
        # config/loader.py already guards against for every other numeric key).
        if surcharge_divisor is not None and (
            isinstance(surcharge_divisor, bool) or not isinstance(surcharge_divisor, int) or surcharge_divisor < 1
        ):
            raise ValueError(f"surcharge_divisor must be None or an int >= 1, got {surcharge_divisor!r}")
        self.window = window
        self.k = k
        self.charge_rate_eur_per_kwh = charge_rate_eur_per_kwh
        self.capacity_passthrough = capacity_passthrough
        self.surcharge_mode = surcharge_mode
        self.surcharge_divisor = surcharge_divisor

        self._history: deque[float] = deque(maxlen=window)

        # Run-level audit accumulators (Phase 5/6 use); see
        # environment/metrics.py's compute_capacity_audit.
        self.total_charge_eur = 0.0
        self.hours_observed = 0
        self.hours_with_excess = 0

    def step(self, feeder_net_import_kwh: float, broker_contributions_kwh: dict) -> CapacityStepResult:
        """Advance the mechanism by one hourly step.

        feeder_net_import_kwh: the ACTUAL (post-demand-deferral) feeder net
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
            # Allocations (the P&L channel) stay strictly proportional to
            # contribution share in every mode: D10 only varies how the
            # pricing channel spreads the same total signal strength across
            # brokers, never how the ledger charge itself is billed.
            allocations = {
                broker_id: total_charge * (contribution / sum_positive_contrib)
                for broker_id, contribution in positive_contrib.items()
            }
            num_brokers = len(broker_ids)
            # D11: the divisor substitutes N_eff (self.surcharge_divisor) for
            # the actual broker count N wherever the pricing-channel formula
            # depends on N, so a broker_count-1 monopolist can be evaluated as
            # though it faced an M-supplier market's per-broker signal
            # strength. When the divisor is None (the default), N_eff is N and
            # the branch below takes the ORIGINAL, pre-D11 expression with no
            # extra arithmetic at all (not a multiply by a computed 1.0), so
            # the divisor being absent from a config is provably a no-op down
            # to the bit, not merely "close" (guardrail 1).
            if self.surcharge_mode == "synchronized":
                # Every broker gets the identical surcharge, including
                # brokers with zero or negative contribution, so long as the
                # step levied a charge at all (the early exit above still
                # zeroes every mode). The sum across brokers is
                # capacity_passthrough up to float rounding, matching
                # proportional's total, so total signal strength is preserved
                # while cross-broker variation is removed entirely (D10).
                # D11: surcharge_b = passthrough / N_eff; N_eff is N when the
                # divisor is unset, which is exactly the pre-D11 expression.
                if self.surcharge_divisor is None:
                    surcharges = {broker_id: self.capacity_passthrough / num_brokers for broker_id in broker_ids}
                else:
                    surcharges = {
                        broker_id: self.capacity_passthrough / self.surcharge_divisor for broker_id in broker_ids
                    }
            elif self.surcharge_mode == "renormalized":
                # Rescales proportional's share by broker count so the MEAN
                # surcharge across brokers stays capacity_passthrough
                # regardless of N, holding deferred volume roughly constant
                # across broker counts while cross-broker variation survives
                # (D10); its own total is therefore capacity_passthrough * N,
                # not capacity_passthrough.
                #
                # D11's substitution (share_b * N/N_eff for share_b, N_eff for
                # N) gives passthrough * share_b * (N/N_eff) * N_eff, which is
                # algebraically identical to passthrough * share_b * N for any
                # N_eff: the N_eff introduced by the substitution and the one
                # the formula multiplies back by cancel exactly. Renormalized
                # is therefore N-invariant BY CONSTRUCTION, precisely because
                # its whole point is a mean surcharge of capacity_passthrough
                # regardless of broker count; pretending the count is M cannot
                # change that. Computing the substituted form literally
                # (divide by N_eff, then multiply by N_eff) would reintroduce
                # a division/multiplication round-trip that need not return
                # the original float in IEEE754, so this stays written in the
                # shipped form (no self.surcharge_divisor reference at all)
                # to keep it exactly, not just approximately, unchanged.
                surcharges = {
                    broker_id: self.capacity_passthrough * (contribution / sum_positive_contrib) * num_brokers
                    for broker_id, contribution in positive_contrib.items()
                }
            else:
                # proportional. D11: surcharge_b = passthrough * share_b *
                # (N/N_eff); N_eff is N when the divisor is unset, and
                # N/N_eff would then be exactly 1.0, but that branch is
                # written as the literal pre-D11 expression below rather than
                # as this formula times a computed 1.0, so no extra
                # arithmetic happens when the divisor is absent.
                if self.surcharge_divisor is None:
                    surcharges = {
                        broker_id: self.capacity_passthrough * (contribution / sum_positive_contrib)
                        for broker_id, contribution in positive_contrib.items()
                    }
                else:
                    surcharges = {
                        broker_id: self.capacity_passthrough
                        * (contribution / sum_positive_contrib)
                        * (num_brokers / self.surcharge_divisor)
                        for broker_id, contribution in positive_contrib.items()
                    }

        return CapacityStepResult(threshold, excess, total_charge, allocations, surcharges)

    def fire_rate(self) -> float:
        """Fraction of observed hours with a positive excess (a positive
        charge levied), including the window-not-filled hours as non-firing."""
        if self.hours_observed == 0:
            return 0.0
        return self.hours_with_excess / self.hours_observed
