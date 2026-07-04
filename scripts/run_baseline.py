"""Run the single-broker monopoly baseline for a full year and print the four
thesis metrics. Phase 1 placeholder runner; later phases may add scenario
selection and result persistence under results/.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from microgrid_sim.config.loader import load_config
from microgrid_sim.environment.model import MicrogridModel

# Mesa 3.5.1 still supports seed= (the brief's mandated API) but nudges toward
# rng=; functionally identical per Mesa's own warning message.
warnings.filterwarnings(
    "ignore", message="The use of the `seed` keyword argument is deprecated", category=FutureWarning
)


def main() -> None:
    config = load_config("config/scenarios/monopoly_baseline.yaml")
    model = MicrogridModel(config)
    model.run()

    result = model.compute_metrics()

    print(f"scenario: {config['scenario_name']}")
    print(f"horizon_hours: {model.horizon_hours}")
    print(f"num_agents: {model.num_agents}")
    print(f"solar_source: {model.solar_source}")
    print(f"demand_source: {model.demand_source}")
    print()
    print("Metric 1 - average cost per agent:")
    print(f"  avg_cost_per_agent_eur = {result.avg_cost_per_agent_eur:.4f}")
    print(f"  avg_cost_per_kwh_eur   = {result.avg_cost_per_kwh_eur:.4f}")
    print()
    print("Metric 2 - load distribution across brokers:")
    for broker_id, share in result.broker_load_share.items():
        print(f"  {broker_id}: {share:.4f}")
    print(f"  load_concentration_hhi = {result.load_concentration_hhi:.4f}")
    print()
    print("Metric 3 - grid stability (expected roughly insensitive under this baseline):")
    print(f"  feeder_coefficient_of_variation = {result.feeder_coefficient_of_variation:.4f}")
    print(f"  feeder_peak_to_average_ratio    = {result.feeder_peak_to_average_ratio:.4f}")
    print(f"  feeder_mean_hourly_ramp_kwh     = {result.feeder_mean_hourly_ramp_kwh:.4f}")
    print()
    print("Metric 4 - prosumer self-sufficiency:")
    print(f"  prosumer_self_sufficiency = {result.prosumer_self_sufficiency:.4f}")


if __name__ == "__main__":
    main()
