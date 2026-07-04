"""The four thesis metrics, computed from accumulated model state.

Metric 3 (grid stability) is expected to be roughly insensitive to broker
competition under this plain baseline; that is the known gap the Phase 3
capacity mechanism will address, not something to fix here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from microgrid_sim.agents.prosumer import Prosumer


@dataclass(frozen=True)
class MetricsResult:
    avg_cost_per_agent_eur: float
    avg_cost_per_kwh_eur: float
    broker_load_share: dict = field(default_factory=dict)
    load_concentration_hhi: float = 0.0
    feeder_coefficient_of_variation: float = 0.0
    feeder_peak_to_average_ratio: float = 0.0
    feeder_mean_hourly_ramp_kwh: float = 0.0
    prosumer_self_sufficiency: float = float("nan")
    # Phase 3 (D6) audit quantities, for the Phase 5/6 ablation sweep. All
    # three are exactly 0.0 / empty-mean when the capacity mechanism is
    # disabled (or absent), so they do not corrupt any existing metric's scope.
    total_capacity_charge_eur: float = 0.0
    capacity_fire_rate: float = 0.0
    mean_broker_surcharge_eur_per_kwh: dict = field(default_factory=dict)
    # Phase 3b (D7) audit quantity: total energy actually deferred (shifted
    # out of a scarcity hour) over the run, summed across every agent. Exactly
    # 0.0 whenever the pricing channel never writes a positive surcharge
    # (capacity_disabled, capacity_pnl_only, or the mechanism absent).
    total_deferred_kwh: float = 0.0


def compute_average_cost(brokers: dict, num_agents: int) -> tuple[float, float]:
    """avg_cost_per_agent_eur: total net revenue (import charges minus export
    credits) divided by the number of agents; netting is correct here since this
    is a per-agent net-bill figure.

    avg_cost_per_kwh_eur (F1 fix): effective net price = the SAME total net
    revenue divided by total NET energy (imports minus exports), i.e. numerator
    and denominator share one signed energy scope. Previously the denominator
    counted only positive (import) energy while the numerator already netted out
    export credits, which understated this metric by about 8.5% on the default
    config (see docs/DECISIONS.md observation note for the measured before/after
    values). Broker.cumulative_energy_served_kwh (import-only, unsigned) is
    intentionally left out of this computation and is used only for the load
    distribution metric (compute_load_distribution), where "load served" is the
    correct, unsigned notion.
    """
    total_revenue_eur = sum(broker.cumulative_revenue_eur for broker in brokers.values())
    total_net_energy_kwh = sum(broker.cumulative_net_energy_kwh for broker in brokers.values())
    avg_cost_per_agent = total_revenue_eur / num_agents if num_agents > 0 else 0.0
    avg_cost_per_kwh = total_revenue_eur / total_net_energy_kwh if total_net_energy_kwh > 0 else 0.0
    return avg_cost_per_agent, avg_cost_per_kwh


def compute_load_distribution(brokers: dict) -> tuple[dict, float]:
    total_energy_kwh = sum(broker.cumulative_energy_served_kwh for broker in brokers.values())
    if total_energy_kwh <= 0:
        return {broker_id: 0.0 for broker_id in brokers}, 0.0
    shares = {
        broker_id: broker.cumulative_energy_served_kwh / total_energy_kwh
        for broker_id, broker in brokers.items()
    }
    hhi = sum(share**2 for share in shares.values())
    return shares, hhi


def compute_grid_stability(feeder_net_import_history: list) -> tuple[float, float, float]:
    series = np.asarray(feeder_net_import_history, dtype=float)
    if series.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = series.mean()
    coefficient_of_variation = series.std() / mean if mean != 0 else float("nan")
    peak_to_average_ratio = series.max() / mean if mean != 0 else float("nan")
    mean_hourly_ramp_kwh = float(np.mean(np.abs(np.diff(series)))) if series.size > 1 else 0.0
    return float(coefficient_of_variation), float(peak_to_average_ratio), mean_hourly_ramp_kwh


def compute_prosumer_self_sufficiency(prosumers: list) -> float:
    if not prosumers:
        return float("nan")
    ratios = [prosumer.self_sufficiency_ratio() for prosumer in prosumers]
    return float(np.mean(ratios))


def compute_capacity_audit(model) -> tuple[float, float, dict]:
    """Phase 3 (D6) audit quantities: total capacity charge levied over the
    run, fire rate (fraction of hours with a positive excess), and the mean,
    over all steps, of the surcharge ACTUALLY applied to each broker's quote
    (post-ablation-gate: exactly 0.0 whenever capacity_feedback_pricing is
    off, or the mechanism is disabled/absent entirely).
    """
    capacity = getattr(model, "_capacity", None)
    broker_ids = list(model.brokers.keys())
    if capacity is None:
        return 0.0, 0.0, {broker_id: 0.0 for broker_id in broker_ids}

    steps = model._surcharge_steps
    mean_surcharge = {
        broker_id: (model._surcharge_accum[broker_id] / steps if steps > 0 else 0.0) for broker_id in broker_ids
    }
    return capacity.total_charge_eur, capacity.fire_rate(), mean_surcharge


def compute_deferral_audit(model) -> float:
    """Phase 3b (D7) audit quantity: total energy actually deferred (moved out
    of a scarcity hour, not counting later paybacks of that same energy) over
    the run, summed across every agent (consumer and prosumer alike). Reads
    each agent's total_deferred_kwh accumulator (see agents/consumer.py's
    Consumer._apply_demand_deferral) defensively, so a host population of
    agents that predate this accumulator would contribute 0.0 rather than
    raising."""
    return float(sum(getattr(agent, "total_deferred_kwh", 0.0) for agent in model.agents))


def compute_metrics(model) -> MetricsResult:
    avg_cost_per_agent, avg_cost_per_kwh = compute_average_cost(model.brokers, model.num_agents)
    broker_load_share, load_concentration_hhi = compute_load_distribution(model.brokers)
    coefficient_of_variation, peak_to_average_ratio, mean_hourly_ramp_kwh = compute_grid_stability(
        model.feeder_net_import_history
    )
    prosumers = list(model.agents_by_type.get(Prosumer, []))
    prosumer_self_sufficiency = compute_prosumer_self_sufficiency(prosumers)
    total_capacity_charge_eur, capacity_fire_rate, mean_broker_surcharge_eur_per_kwh = compute_capacity_audit(model)
    total_deferred_kwh = compute_deferral_audit(model)

    return MetricsResult(
        avg_cost_per_agent_eur=avg_cost_per_agent,
        avg_cost_per_kwh_eur=avg_cost_per_kwh,
        broker_load_share=broker_load_share,
        load_concentration_hhi=load_concentration_hhi,
        feeder_coefficient_of_variation=coefficient_of_variation,
        feeder_peak_to_average_ratio=peak_to_average_ratio,
        feeder_mean_hourly_ramp_kwh=mean_hourly_ramp_kwh,
        prosumer_self_sufficiency=prosumer_self_sufficiency,
        total_capacity_charge_eur=total_capacity_charge_eur,
        capacity_fire_rate=capacity_fire_rate,
        mean_broker_surcharge_eur_per_kwh=mean_broker_surcharge_eur_per_kwh,
        total_deferred_kwh=total_deferred_kwh,
    )
