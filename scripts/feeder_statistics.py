"""Phase 20 item 20.7, task 4: the appendix's feeder-statistics generator.

docs/thesis/appendix_validation.md (A.2 "Methods") describes exactly how its
own feeder-shape statistics were computed, then states plainly: "The script
itself (compute_feeder_stats.py) is a throwaway analysis tool kept in the
scratch working area used for this check, not part of the repository."
docs/PROGRESS.md's closing note names that absence directly ("the appendix's
absent feeder-statistics script") as one half of the structural gap this
phase item closes: results are pinned and reproduce, the prose describing
them is not. This script is that generator, committed for real, with its
outputs pinned in tests/golden/feeder_statistics_pins.json instead of
recomputed by hand each time a reviewer asks whether the appendix still
matches the shipped model.

WHAT THIS DRIVES

The real MicrogridModel (src/microgrid_sim/environment/model.py), constructed
exactly as appendix_validation.md's A.2 describes: "MicrogridModel was
constructed exactly as scripts/run_baseline.py does, then stepped one hour at
a time for the full horizon ... so that per-agent state could be read after
every hour." Two configs, both unmodified from the shipped repository,
exactly as A.2 lists them:

  - config/default.yaml (capacity_mechanism.enabled: false), the "baseline".
  - config/scenarios/capacity_both.yaml (extends default.yaml; both feedback
    channels on, k=1.0, broker_count=3, the Chapter 5-6 headline arm), the
    "capacity_both" run.

Nothing here computes broker prices, agent decisions, or capacity-mechanism
arithmetic: every statistic below is a plain aggregate (mean, max, argmax,
...) of model.feeder_net_import_history (the same series metric 3 is
computed from) or of each agent's own public last_demand_kwh /
last_net_import_kwh, read after real model.step() calls.

FORMULAS (verbatim from appendix_validation.md A.2, restated here only so
the code that implements them sits next to a citation of where they came
from, not as a second, competing definition):

  - load_factor = mean(feeder) / max(feeder).
  - coincidence_factor = feeder peak / sum of each agent's own annual peak
    last_net_import_kwh. diversity_factor is its reciprocal.
  - peak_to_average_ratio = feeder peak / feeder mean (same quantity as
    MetricsResult.feeder_peak_to_average_ratio; not cross-checked against it
    here to avoid a second model construction, but it is the identical
    formula over the identical series).
  - mean_demand_per_household_kwh / mean_net_import_per_household_kwh = the
    per-agent sum of last_demand_kwh / last_net_import_kwh over the run,
    averaged over all agents. Appendix A.2 calls these "annual" because its
    own runs are always the full 8760-hour year; this script's `horizon_hours`
    can be shorter (see "SMOKE RUN" below), so the field names here drop
    "annual" and horizon_hours is reported alongside them instead.
  - Daily shape: reshape the feeder series into whole days of 24 hours and
    average over days; hour_of_daily_peak / hour_of_daily_trough are that
    profile's argmax / argmin. night_valley_depth = 1 - (that profile's mean
    over hours 01:00-04:00, i.e. hour-of-day in {1,2,3,4}, divided by the
    profile's own peak). The four-hour reading (not three) was checked
    against the appendix's own printed baseline value (0.709) and is the one
    that reproduces it; that comparison is the whole check, not repeated
    elsewhere in the repo.
  - Calendar month of each hour: hour_of_year // 24 is the (0-indexed) day of
    a non-leap 365-day year starting at hour 0 = January 1st 00:00 (matching
    data/loaders.py's own day_of_year = (hour_index // 24) % 365 convention,
    though that module never needs a calendar MONTH, only a day-of-year
    angle, so there is no existing helper to reuse here); this script maps
    that day to a calendar month via Python's own datetime, anchored at an
    arbitrary non-leap reference year (2001).
  - month_of_peak_hour = the calendar month containing the single hour of
    the feeder series' yearly maximum. winter_mean_kwh / summer_mean_kwh =
    the feeder series' mean restricted to December-January-February /
    June-July-August hours; winter_summer_ratio is their ratio. Over a run
    shorter than a full year these can be undefined (no hours in a season at
    all); undefined values are reported as JSON null, never invented, and
    never silently coerced to 0.0 (see "SMOKE RUN" below).
  - min_max_ratio = min(feeder) / max(feeder); fraction_negative_hours =
    the fraction of hours with a negative feeder value.
  - feeder_coefficient_of_variation = std(feeder) / mean(feeder) (the
    "two figures reported for completeness" in appendix A.3).
  - total_energy_served_kwh = sum, over every agent and every hour, of
    last_demand_kwh (i.e. mean_demand_per_household_kwh * num_agents; the
    same sum, not divided). This was checked, not assumed: an initial version
    of this script used sum(broker.cumulative_energy_served_kwh), the
    grid-import-only quantity environment/metrics.py's compute_load_distribution
    sums for metric 2, and measured 740,119 kWh against the appendix's stated
    858,996.5 kWh, a 14 percent gap, not a rounding difference. Multiplying
    this script's own mean_demand_per_household_kwh by num_agents instead
    gives 858,996.53 and 858,971.85 for the two configs, matching the
    appendix's 858,996.5 and 858,971.9 to the precision it prints, and
    matching the appendix's own stated reason the two configs nearly agree
    ("demand deferral shifts energy in time and does not delete it" -- true
    of total demand, which deferral only reschedules, not of grid-imported
    energy, which a deferral-driven PV/battery interaction could in principle
    shift by more). So "energy served to the population" in appendix A.3 is
    total demand met (by any source: grid, PV, or battery), not grid draw
    alone, and this script now computes it that way.

SMOKE RUN

A full run is two 8760-hour simulations (baseline, capacity_both); measured
532.65 seconds combined for the two full-year runs (docs/PROGRESS.md, Phase
20.7), so the pytest test that checks the full run is marked slow and
skipped by default. A short "smoke" horizon covers everything except the
calendar-season fields (month_of_peak_hour, winter_mean_kwh, summer_mean_kwh,
winter_summer_ratio), which are legitimately undefined (JSON null) over a
horizon of only a few hundred hours starting at hour 0 (all within January,
so there are no June-July-August hours to average, and no ratio to take).
That is a real property of a short run, not a bug in this script.

CLI

    python scripts/feeder_statistics.py [--horizon-hours N] [--out PATH] [--regenerate [PATH]]

Two output modes: a single run's JSON, printed to stdout and, if --out is
given, also written to that path; or, via --regenerate, build_pins() run for
both the full and smoke horizons and both configs, with the combined pins
file written out, independent of --horizon-hours.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from microgrid_sim.config.loader import load_config  # noqa: E402
from microgrid_sim.environment.model import MicrogridModel  # noqa: E402

DEFAULT_HORIZON_HOURS = 8760  # the mandated full year (D8), matching appendix_validation.md A.2
DEFAULT_SMOKE_HORIZON_HOURS = 240  # 10 days from hour 0 (Jan 1): see module docstring, "SMOKE RUN"
DEFAULT_PINS_PATH = _REPO_ROOT / "tests" / "golden" / "feeder_statistics_pins.json"

CONFIG_PATHS = {
    "baseline": _REPO_ROOT / "config" / "default.yaml",
    "capacity_both": _REPO_ROOT / "config" / "scenarios" / "capacity_both.yaml",
}

# Single source of truth for the per-run field list: tests/test_golden_master.py's
# _metrics_dict once duplicated this list by hand and drifted out of sync
# with it (docs/PROGRESS.md "19.3"); iterating FIELD_NAMES instead of
# retyping it avoids that twin-defect risk here.
FIELD_NAMES = (
    "horizon_hours",
    "num_agents",
    "seed",
    "load_factor",
    "coincidence_factor",
    "diversity_factor",
    "peak_to_average_ratio",
    "mean_demand_per_household_kwh",
    "mean_net_import_per_household_kwh",
    "hour_of_daily_peak",
    "hour_of_daily_trough",
    "month_of_peak_hour",
    "winter_mean_kwh",
    "summer_mean_kwh",
    "winter_summer_ratio",
    "min_max_ratio",
    "fraction_negative_hours",
    "night_valley_depth",
    "feeder_coefficient_of_variation",
    "total_energy_served_kwh",
)

# Metadata and quantities that are exact integers/counts by construction;
# everything else is a computed ratio/mean and is compared by relative
# tolerance only in the test module (never rel AND abs together: pytest.approx
# treats rel and abs as an OR, not an AND, so pairing both on one field
# weakens the check by silently widening the effective tolerance).
EXACT_FIELDS = frozenset({"horizon_hours", "num_agents", "seed", "hour_of_daily_peak", "hour_of_daily_trough", "month_of_peak_hour"})
REL_TOLERANCE_FIELDS = frozenset(FIELD_NAMES) - EXACT_FIELDS

NIGHT_VALLEY_HOURS_OF_DAY = (1, 2, 3, 4)  # "01:00-04:00" read as 4 hourly samples; see module docstring
WINTER_MONTHS = (12, 1, 2)
SUMMER_MONTHS = (6, 7, 8)
_CALENDAR_REFERENCE_YEAR = 2001  # an arbitrary non-leap year, used only for date arithmetic


def _month_of_hour(hour_index: int) -> int:
    """Calendar month (1-12) of an hour in an 8760-hour, non-leap, hour
    0 = January 1st 00:00 calendar (appendix_validation.md A.2's own stated
    convention, matching data/loaders.py's day_of_year = hour_index // 24)."""
    day_of_year = hour_index // 24
    return (date(_CALENDAR_REFERENCE_YEAR, 1, 1) + timedelta(days=day_of_year)).month


def compute_feeder_statistics(config_path: str | Path, horizon_hours: int) -> dict:
    """Run one config for `horizon_hours` and measure every field in
    FIELD_NAMES, following appendix_validation.md A.2's methods (see module
    docstring)."""
    config = copy.deepcopy(load_config(str(config_path)))
    config["simulation"]["horizon_hours"] = horizon_hours
    model = MicrogridModel(config)

    # "captured once, immediately after construction" (A.2), so a later
    # broker switch cannot change which Python objects "the population" means.
    agents = list(model.agents)
    num_agents = len(agents)

    total_demand_kwh = [0.0] * num_agents
    total_net_import_kwh = [0.0] * num_agents
    peak_net_import_kwh = [float("-inf")] * num_agents

    for _ in range(horizon_hours):
        model.step()
        for index, agent in enumerate(agents):
            total_demand_kwh[index] += agent.last_demand_kwh
            net = agent.last_net_import_kwh
            total_net_import_kwh[index] += net
            if net > peak_net_import_kwh[index]:
                peak_net_import_kwh[index] = net

    feeder = np.asarray(model.feeder_net_import_history, dtype=float)
    feeder_mean = float(feeder.mean())
    feeder_max = float(feeder.max())
    feeder_min = float(feeder.min())
    feeder_std = float(feeder.std())
    sum_of_peaks = float(sum(peak_net_import_kwh))

    load_factor = feeder_mean / feeder_max
    peak_to_average_ratio = feeder_max / feeder_mean
    coincidence_factor = feeder_max / sum_of_peaks
    diversity_factor = sum_of_peaks / feeder_max
    min_max_ratio = feeder_min / feeder_max
    fraction_negative_hours = float(np.mean(feeder < 0.0))
    feeder_coefficient_of_variation = feeder_std / feeder_mean

    sum_total_demand_kwh = sum(total_demand_kwh)
    mean_demand_per_household_kwh = sum_total_demand_kwh / num_agents
    mean_net_import_per_household_kwh = sum(total_net_import_kwh) / num_agents

    # See module docstring's total_energy_served_kwh entry for why this is
    # total demand met (any source), not sum(broker.cumulative_energy_served_kwh)
    # (grid-import-only): the latter measured 740,119 kWh against the
    # appendix's stated 858,996.5 kWh, a 14 percent gap, while this matches to
    # the precision the appendix prints.
    total_energy_served_kwh = sum_total_demand_kwh

    # Daily shape (A.2): reshape into whole days, average over days. Any
    # trailing partial day (horizon_hours not a multiple of 24) is dropped
    # from this reshape only, exactly as a fixed-width reshape requires;
    # every other statistic above still uses the full, un-reshaped series.
    num_days = horizon_hours // 24
    usable_hours = num_days * 24
    if num_days > 0:
        daily_profile = feeder[:usable_hours].reshape(num_days, 24).mean(axis=0)
        hour_of_daily_peak = int(daily_profile.argmax())
        hour_of_daily_trough = int(daily_profile.argmin())
        night_valley_mean = float(daily_profile[list(NIGHT_VALLEY_HOURS_OF_DAY)].mean())
        night_valley_depth = 1.0 - (night_valley_mean / float(daily_profile.max()))
    else:
        hour_of_daily_peak = -1
        hour_of_daily_trough = -1
        night_valley_depth = float("nan")

    months = np.array([_month_of_hour(hour) for hour in range(horizon_hours)])
    peak_hour_index = int(feeder.argmax())
    month_of_peak_hour = int(months[peak_hour_index])

    winter_mask = np.isin(months, WINTER_MONTHS)
    summer_mask = np.isin(months, SUMMER_MONTHS)
    # None (JSON null), not NaN or 0.0: a run short enough to contain no
    # hours of a season has genuinely no value for that season's mean, and
    # this must be visible as "undefined", not silently averaged over zero
    # hours or coerced into a number that looks measured.
    winter_mean_kwh = float(feeder[winter_mask].mean()) if winter_mask.any() else None
    summer_mean_kwh = float(feeder[summer_mask].mean()) if summer_mask.any() else None
    if winter_mean_kwh is not None and summer_mean_kwh not in (None, 0.0):
        winter_summer_ratio = winter_mean_kwh / summer_mean_kwh
    else:
        winter_summer_ratio = None

    result = {
        "horizon_hours": horizon_hours,
        "num_agents": num_agents,
        "seed": config["simulation"]["seed"],
        "load_factor": load_factor,
        "coincidence_factor": coincidence_factor,
        "diversity_factor": diversity_factor,
        "peak_to_average_ratio": peak_to_average_ratio,
        "mean_demand_per_household_kwh": mean_demand_per_household_kwh,
        "mean_net_import_per_household_kwh": mean_net_import_per_household_kwh,
        "hour_of_daily_peak": hour_of_daily_peak,
        "hour_of_daily_trough": hour_of_daily_trough,
        "month_of_peak_hour": month_of_peak_hour,
        "winter_mean_kwh": winter_mean_kwh,
        "summer_mean_kwh": summer_mean_kwh,
        "winter_summer_ratio": winter_summer_ratio,
        "min_max_ratio": min_max_ratio,
        "fraction_negative_hours": fraction_negative_hours,
        "night_valley_depth": night_valley_depth,
        "feeder_coefficient_of_variation": feeder_coefficient_of_variation,
        "total_energy_served_kwh": total_energy_served_kwh,
    }
    assert set(result) == set(FIELD_NAMES), "compute_feeder_statistics's output drifted from FIELD_NAMES"
    return result


def run_feeder_statistics(horizon_hours: int) -> dict:
    """Both configs (appendix_validation.md A.2), keyed by name."""
    return {name: compute_feeder_statistics(path, horizon_hours) for name, path in CONFIG_PATHS.items()}


def build_pins(full_horizon_hours: int = DEFAULT_HORIZON_HOURS, smoke_horizon_hours: int = DEFAULT_SMOKE_HORIZON_HOURS) -> dict:
    """The two runs tests/golden/feeder_statistics_pins.json pins: "full"
    (8760h x 2 configs, the appendix's own methodology, marked slow) and
    "smoke" (a short default-speed run of the same code path for the default
    test suite; see module docstring, "SMOKE RUN")."""
    return {
        "full": run_feeder_statistics(full_horizon_hours),
        "smoke": run_feeder_statistics(smoke_horizon_hours),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS, help=f"hours to simulate per config (default {DEFAULT_HORIZON_HOURS})"
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON to this path")
    parser.add_argument(
        "--regenerate",
        nargs="?",
        const=str(DEFAULT_PINS_PATH),
        default=None,
        metavar="PATH",
        help=(
            "run BOTH the full (8760h) and smoke horizons for both configs via "
            f"build_pins() and write the combined pins file (default {DEFAULT_PINS_PATH}), "
            "independent of --horizon-hours"
        ),
    )
    args = parser.parse_args()

    start = time.perf_counter()
    result = run_feeder_statistics(horizon_hours=args.horizon_hours)
    elapsed = time.perf_counter() - start

    ordered = {config_name: {field: result[config_name][field] for field in FIELD_NAMES} for config_name in CONFIG_PATHS}
    text = json.dumps(ordered, indent=2) + "\n"
    sys.stdout.write(text)

    print(
        f"[feeder_statistics] {args.horizon_hours} hours x {len(CONFIG_PATHS)} configs in {elapsed:.2f}s wall",
        file=sys.stderr,
    )

    if args.out:
        args.out.write_text(text, encoding="utf-8")
    if args.regenerate:
        pins = build_pins(full_horizon_hours=DEFAULT_HORIZON_HOURS, smoke_horizon_hours=DEFAULT_SMOKE_HORIZON_HOURS)
        regen_path = Path(args.regenerate)
        regen_path.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
        print(f"[feeder_statistics] wrote pins ('full' + 'smoke') to {regen_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
