"""Phase 20 item 20.7: commit the live-demo instrumentation as a real,
testable script instead of the throwaway drive that measured
docs/thesis/demo_instructions.md's numbers.

docs/PROGRESS.md's closing note ("The pattern across both audits") records
that results/*.csv are pinned and reproduce, but the prose describing the
live dashboard is not pinned anywhere, and that gap is where every audit
finding across five passes actually lived. One re-audit disputed six of
demo_instructions.md's measured numbers and was wrong on all six; settling
that cost three separate manual remeasurements (docs/PROGRESS.md, "19.4 A
finding that did not reproduce"). This script exists so the next drift, real
or imagined, is a failing test instead of a fourth manual remeasurement.

WHAT THIS DRIVES

The real dashboard.DemoSession (dashboard/model_state.py), not a
reimplementation of it: this script imports DemoSession and reads its own
public state after real DemoSession.step_batch(1) calls, the same calls the
dashboard's Step button makes. Every quantity below is either read directly
off DemoSession's public attributes, or, where a quantity is computed inside
DemoSession._step_once but only its CLIPPED half is kept (dashboard/
model_state.py lines 211-214: the per-broker signed contribution sum, before
the max(0.0, ...) floor that turns it into the area-chart share), this script
reproduces that exact expression itself, over the same public per-agent
attributes (agent.last_broker_id, agent.last_net_import_kwh) at the same
point in the step cycle, rather than inventing a second definition of it.
The comment at that computation below cites the file:line it mirrors.

THE CELL

k = 1.0, ablation = "capacity_both" ("Both channels (full mechanism)"),
broker_count = 3: DemoSession's own DEFAULT_K / DEFAULT_ABLATION /
DEFAULT_BROKER_COUNT (dashboard/model_state.py), which DemoSession.__init__
already resets to, so simply constructing DemoSession() lands on the thesis's
headline cell with no cell-selection logic duplicated here.

THE SEED

DemoSession never takes a seed argument: DemoSession.reset() always builds
its config through dashboard/model_state.py's build_demo_config(), which
calls run_sensitivity_sweep.build_run_config(..., DEMO_SEED, ...) with
DEMO_SEED = 20260704 hardcoded in model_state.py, itself a comment-documented
copy of config/default.yaml's own simulation.seed (also 20260704). So the
seed is fixed and deterministic by construction, not chosen by this script;
it is recorded here (SEED, below) only for the report, not passed anywhere.

THE HORIZON MEASURED

docs/thesis/demo_instructions.md's per-hour claims (lines 43-71) are all
stated "over a 2000-hour run at this cell", a prefix of DemoSession's own
8760-hour horizon (D8), not the full horizon itself. run_instrumentation's
`hours` argument defaults to 2000 to match; a shorter value is used by the
default (non-marked) pytest test as a fast smoke check of the same code path,
documented in tests/test_demo_instrumentation.py and in
docs/verification/demo_instrumentation.md.

DETERMINISM

The emitted JSON contains only quantities computed from the model's own
deterministic state; no wall-clock timing is included in it (timing is
printed separately, to stderr), specifically so that two invocations of
`python scripts/demo_instrumentation.py` produce byte-identical stdout. This
was verified manually (see docs/verification/demo_instrumentation.md) and is
not re-verified by every pytest run, since the model itself provides no
mechanism by which two runs at a fixed seed could differ.

CLI

    python scripts/demo_instrumentation.py [--hours N] [--out PATH] [--regenerate [PATH]]

--hours: how many DemoSession steps to take (default 2000, matching
  demo_instructions.md).
--out: also write the JSON to this path (in addition to stdout).
--regenerate: write the JSON to tests/golden/demo_instrumentation_pins.json
  (or an explicit PATH), i.e. treat this run's output as the new pin. This is
  the ONLY place tests/golden/demo_instrumentation_pins.json is written from;
  tests/test_demo_instrumentation.py only ever reads it. Regenerating is a
  deliberate act: running the plain test suite never rewrites this file.
  FIELD_NAMES below is the single source of truth for the field list both the
  regeneration path and the test's comparison use, so the two cannot grow a
  twin the way tests/test_golden_master.py's _metrics_dict once did
  (docs/PROGRESS.md, "19.3 What the reviews found that the audits did not").
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD_DIR = _REPO_ROOT / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

# model_state.py itself inserts src/ and scripts/ onto sys.path and builds the
# real MicrogridModel through run_sensitivity_sweep.build_run_config, so
# importing it here reuses that setup rather than re-deriving it.
from model_state import DEMO_SEED, WINDOW_HOURS, DemoSession  # noqa: E402

DEFAULT_HOURS = 2000
# A short run past the 168-hour window fill, used only for a fast default
# (non-marked) pytest smoke test: it exercises every code path above at low
# cost, but does NOT reproduce demo_instructions.md's own numbers, which are
# stated over a 2000-hour run specifically. See docs/verification/
# demo_instrumentation.md for why 2000 hours itself needed a marker: measured
# 40-48s on this machine (busy with a concurrent mutation run at the time),
# close enough to the "about 60 seconds" guideline that a slower CI box could
# cross it.
DEFAULT_SMOKE_HOURS = 250
DEFAULT_PINS_PATH = _REPO_ROOT / "tests" / "golden" / "demo_instrumentation_pins.json"

# Single source of truth for the field list (see module docstring's CLI
# section). Order here is the order written to JSON via sort_keys=False is
# NOT relied upon for correctness anywhere (comparisons are by name, never by
# position, per the brief), only for stable, readable file diffs.
FIELD_NAMES = (
    "seed",
    "k",
    "ablation",
    "broker_count",
    "hours_measured",
    "window_hours",
    "capacity_passthrough",
    "firing_hours",
    "same_hour_max_gap",
    "prev_hour_max_gap_documented_variant",
    "prev_hour_max_gap_previous_firing_hour_variant",
    "prev_hour_max_gap_all_hours_variant",
    "post_window_hours",
    "all_zero_bar_hours",
    "all_zero_bar_fraction",
    "net_export_hours",
    "net_export_and_firing_hours",
    "area_share_final_hour_regulated_utility",
    "area_share_final_hour_volatile_low_cost",
    "area_share_final_hour_premium_green",
    "card_load_share_regulated_utility",
    "card_load_share_volatile_low_cost",
    "card_load_share_premium_green",
)

# Fields compared by EXACT equality (tests/test_demo_instrumentation.py):
# integer/string metadata, hour counts, and same_hour_max_gap, which is
# exactly 0.0 by construction (proportional mode's bar and the same-hour area
# share are both `capacity_passthrough * (contribution / sum_positive_contrib)`
# over the identical `contributions` dict; see environment/capacity.py's
# proportional branch and model_state.py's _step_once). Every other field is
# a ratio or a cross-hour gap and is compared by RELATIVE tolerance only, with
# no absolute term: pytest.approx's default of rel AND abs together would
# make an assertion on a value near/at 0.0 (same_hour_max_gap) vacuous, which
# is exactly the hazard this project was warned to avoid.
EXACT_FIELDS = frozenset(
    {
        "seed",
        "k",
        "ablation",
        "broker_count",
        "hours_measured",
        "window_hours",
        "capacity_passthrough",
        "firing_hours",
        "same_hour_max_gap",
        "post_window_hours",
        "all_zero_bar_hours",
        "net_export_hours",
        "net_export_and_firing_hours",
    }
)
REL_TOLERANCE_FIELDS = frozenset(FIELD_NAMES) - EXACT_FIELDS


def run_instrumentation(hours: int = DEFAULT_HOURS) -> dict:
    """Drive DemoSession for `hours` steps at the headline cell and measure
    every field in FIELD_NAMES. Returns a dict keyed exactly by FIELD_NAMES.
    """
    session = DemoSession()  # __init__ already resets to the headline cell

    # dashboard/model_state.py:103 reads model._capacity the same way to
    # install its instrumentation wrapper; capacity_passthrough is the
    # mechanism's own signal-strength coefficient (environment/capacity.py),
    # read here rather than hardcoded so a config change cannot silently
    # desync this script from what the model actually used.
    passthrough = session.model._capacity.capacity_passthrough

    scarcity_this_hour: list[bool] = []
    threshold_is_set: list[bool] = []
    # bar_this_hour[t][broker_id]: the surcharge dashboard/figures.py's
    # surcharge_figure plots, read from session.surcharge_current, which
    # model_state.py._step_once sets to dict(model.broker_surcharges) right
    # after model.step() returns (the same dict Step 6 of MicrogridModel.step
    # just overwrote; model.py lines 291-296).
    bar_this_hour: list[dict] = []
    # area_this_hour[t][broker_id]: the CLIPPED, normalized share
    # dashboard/figures.py's broker_share_figure plots, read from
    # session.broker_share_history (model_state.py._step_once, lines 217-221).
    area_this_hour: list[dict] = []
    # signed_this_hour[t][broker_id]: the UNCLIPPED per-broker aggregate net
    # import for that hour. This is the pre-clip half of the exact same
    # expression model_state.py._step_once computes at lines 211-213 (and
    # model.py computes independently, identically, at lines 279-281) to
    # build `contributions`; reproduced here, over the same public per-agent
    # attributes at the same point in the step cycle, only because
    # DemoSession itself never stores the unclipped value anywhere public,
    # and a negative aggregate (a net-exporting broker) is exactly what the
    # clipped share cannot show.
    signed_this_hour: list[dict] = []

    for _ in range(hours):
        session.step_batch(1)

        scarcity_this_hour.append(bool(session.scarcity[-1]))
        threshold_is_set.append(session.threshold[-1] is not None)
        bar_this_hour.append(dict(session.surcharge_current))
        area_this_hour.append({broker_id: series[-1] for broker_id, series in session.broker_share_history.items()})

        signed = {broker_id: 0.0 for broker_id in session.model.brokers}
        for agent in session.model.agents:
            signed[agent.last_broker_id] += agent.last_net_import_kwh
        signed_this_hour.append(signed)

    firing_hours = sum(1 for fired in scarcity_this_hour if fired)

    # same_hour_max_gap: over firing hours, |bar - passthrough * SAME hour's
    # area share|, per broker (demo_instructions.md lines 43-44).
    same_hour_gaps = [
        abs(bar_this_hour[t][broker_id] - passthrough * area_this_hour[t][broker_id])
        for t in range(hours)
        if scarcity_this_hour[t]
        for broker_id in bar_this_hour[t]
    ]
    same_hour_max_gap = max(same_hour_gaps) if same_hour_gaps else 0.0

    # prev_hour_max_gap_documented_variant: over firing hours, |bar -
    # passthrough * the IMMEDIATELY PRECEDING step's area share| (whether or
    # not that preceding step itself fired). This is the variant
    # demo_instructions.md lines 44-45 states (3.815e-03).
    documented_gaps = [
        abs(bar_this_hour[t][broker_id] - passthrough * area_this_hour[t - 1][broker_id])
        for t in range(1, hours)
        if scarcity_this_hour[t]
        for broker_id in bar_this_hour[t]
    ]
    prev_hour_max_gap_documented_variant = max(documented_gaps) if documented_gaps else 0.0

    # prev_hour_max_gap_previous_firing_hour_variant: over firing hours,
    # |bar - passthrough * the area share at the PREVIOUS FIRING hour|
    # (skipping any non-firing hours in between). Natural variant #2
    # (docs/PROGRESS.md 19.4: 3.896e-03).
    firing_indices = [t for t in range(hours) if scarcity_this_hour[t]]
    prev_firing_gaps = [
        abs(bar_this_hour[firing_indices[i]][broker_id] - passthrough * area_this_hour[firing_indices[i - 1]][broker_id])
        for i in range(1, len(firing_indices))
        for broker_id in bar_this_hour[firing_indices[i]]
    ]
    prev_hour_max_gap_previous_firing_hour_variant = max(prev_firing_gaps) if prev_firing_gaps else 0.0

    # prev_hour_max_gap_all_hours_variant: the same "previous step" gap as
    # the documented variant, but over ALL hours, not only firing ones
    # (most bars are 0.0 on a non-firing hour while the previous hour's area
    # share is not, so this variant is naturally much larger). Natural
    # variant #3 (docs/PROGRESS.md 19.4: 6.497e-02).
    all_hours_gaps = [
        abs(bar_this_hour[t][broker_id] - passthrough * area_this_hour[t - 1][broker_id])
        for t in range(1, hours)
        for broker_id in bar_this_hour[t]
    ]
    prev_hour_max_gap_all_hours_variant = max(all_hours_gaps) if all_hours_gaps else 0.0

    # post_window_hours: hours where CapacityMechanism.step had a full
    # 168-hour window and so returned a non-None threshold (demo_
    # instructions.md line 57), read from the model's own output rather than
    # hardcoding WINDOW_HOURS - 1 as the first such index.
    post_window_hours = sum(1 for is_set in threshold_is_set if is_set)

    # all_zero_bar_hours: among the post-window hours, every broker's bar
    # reads exactly 0.0 (demo_instructions.md lines 57-58).
    all_zero_bar_hours = sum(
        1
        for t in range(hours)
        if threshold_is_set[t] and all(value == 0.0 for value in bar_this_hour[t].values())
    )
    all_zero_bar_fraction = (all_zero_bar_hours / post_window_hours) if post_window_hours else float("nan")

    # net_export_hours / net_export_and_firing_hours: an hour has a
    # net-exporting broker when that broker's UNCLIPPED aggregate contribution
    # is negative (demo_instructions.md lines 62-67).
    net_export_hours = sum(1 for t in range(hours) if any(value < 0.0 for value in signed_this_hour[t].values()))
    net_export_and_firing_hours = sum(
        1
        for t in range(hours)
        if scarcity_this_hour[t] and any(value < 0.0 for value in signed_this_hour[t].values())
    )

    final_area = area_this_hour[-1] if area_this_hour else {}
    # "the card": metrics.broker_load_share, the SAME MetricsResult field
    # dashboard/app.py's _metrics_children() reads (app.py line 212), backed
    # by cumulative import-only energy served over the whole run
    # (environment/metrics.py's compute_load_distribution), not the per-hour
    # area share. session.metrics is refreshed by the final step_batch(1)
    # call above (model_state.py's step_batch calls compute_metrics() every
    # batch), so no extra call is needed here.
    card = session.metrics.broker_load_share

    result = {
        "seed": DEMO_SEED,
        "k": session.k,
        "ablation": session.ablation,
        "broker_count": session.broker_count,
        "hours_measured": hours,
        "window_hours": WINDOW_HOURS,
        "capacity_passthrough": passthrough,
        "firing_hours": firing_hours,
        "same_hour_max_gap": same_hour_max_gap,
        "prev_hour_max_gap_documented_variant": prev_hour_max_gap_documented_variant,
        "prev_hour_max_gap_previous_firing_hour_variant": prev_hour_max_gap_previous_firing_hour_variant,
        "prev_hour_max_gap_all_hours_variant": prev_hour_max_gap_all_hours_variant,
        "post_window_hours": post_window_hours,
        "all_zero_bar_hours": all_zero_bar_hours,
        "all_zero_bar_fraction": all_zero_bar_fraction,
        "net_export_hours": net_export_hours,
        "net_export_and_firing_hours": net_export_and_firing_hours,
        "area_share_final_hour_regulated_utility": final_area.get("regulated_utility", float("nan")),
        "area_share_final_hour_volatile_low_cost": final_area.get("volatile_low_cost", float("nan")),
        "area_share_final_hour_premium_green": final_area.get("premium_green", float("nan")),
        "card_load_share_regulated_utility": card.get("regulated_utility", float("nan")),
        "card_load_share_volatile_low_cost": card.get("volatile_low_cost", float("nan")),
        "card_load_share_premium_green": card.get("premium_green", float("nan")),
    }
    assert set(result) == set(FIELD_NAMES), "run_instrumentation's output drifted from FIELD_NAMES"
    return result


def build_pins(full_hours: int = DEFAULT_HOURS, smoke_hours: int = DEFAULT_SMOKE_HOURS) -> dict:
    """The two runs tests/golden/demo_instrumentation_pins.json pins: "full"
    (the 2000-hour run demo_instructions.md's own numbers are stated over,
    marked slow in the test module) and "smoke" (a short default-speed run of
    the same code path, for the pytest suite's default, non-marked test).
    Both are produced by the SAME run_instrumentation function at different
    `hours` values, so there is only ever one definition of any field here to
    diverge.
    """
    return {
        "full": run_instrumentation(hours=full_hours),
        "smoke": run_instrumentation(hours=smoke_hours),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS, help=f"steps to take (default {DEFAULT_HOURS})")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON to this path")
    parser.add_argument(
        "--regenerate",
        nargs="?",
        const=str(DEFAULT_PINS_PATH),
        default=None,
        metavar="PATH",
        help=(
            "run BOTH the full (--hours) and smoke (DEFAULT_SMOKE_HOURS) "
            f"horizons via build_pins() and write the combined {{'full':..., "
            f"'smoke':...}} golden pins file (default {DEFAULT_PINS_PATH}), "
            "independent of any single-run --hours/--out output below"
        ),
    )
    args = parser.parse_args()

    start = time.perf_counter()
    result = run_instrumentation(hours=args.hours)
    elapsed = time.perf_counter() - start

    # Field order is fixed (FIELD_NAMES) rather than alphabetized, and no
    # timing value is included, so stdout is byte-identical across runs.
    ordered = {name: result[name] for name in FIELD_NAMES}
    text = json.dumps(ordered, indent=2) + "\n"
    sys.stdout.write(text)

    print(f"[demo_instrumentation] {args.hours} hours in {elapsed:.2f}s wall ({elapsed / max(args.hours, 1) * 1000:.3f} ms/hour)", file=sys.stderr)

    if args.out:
        args.out.write_text(text, encoding="utf-8")
    if args.regenerate:
        pins = build_pins(full_hours=DEFAULT_HOURS, smoke_hours=DEFAULT_SMOKE_HOURS)
        regen_path = Path(args.regenerate)
        regen_path.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
        print(f"[demo_instrumentation] wrote pins ('full' + 'smoke') to {regen_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
