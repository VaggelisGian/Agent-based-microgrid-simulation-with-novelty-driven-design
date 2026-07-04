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


def compute_metrics(model) -> MetricsResult:
    avg_cost_per_agent, avg_cost_per_kwh = compute_average_cost(model.brokers, model.num_agents)
    broker_load_share, load_concentration_hhi = compute_load_distribution(model.brokers)
    coefficient_of_variation, peak_to_average_ratio, mean_hourly_ramp_kwh = compute_grid_stability(
        model.feeder_net_import_history
    )
    prosumers = list(model.agents_by_type.get(Prosumer, []))
    prosumer_self_sufficiency = compute_prosumer_self_sufficiency(prosumers)

    return MetricsResult(
        avg_cost_per_agent_eur=avg_cost_per_agent,
        avg_cost_per_kwh_eur=avg_cost_per_kwh,
        broker_load_share=broker_load_share,
        load_concentration_hhi=load_concentration_hhi,
        feeder_coefficient_of_variation=coefficient_of_variation,
        feeder_peak_to_average_ratio=peak_to_average_ratio,
        feeder_mean_hourly_ramp_kwh=mean_hourly_ramp_kwh,
        prosumer_self_sufficiency=prosumer_self_sufficiency,
    )
