"""D10 structural instrumentation: does a net-exporting broker's SIZE, not any
self-suppressing property of the mechanism, decide whether it is excluded from
a scarcity-hour surcharge?

Why this exists. docs/DECISIONS.md's D10 result correction section reports three
divergence rates (0.00, 0.20, 7.49 percent) as evidence that a small exporting
broker is excluded from a scarcity hour's surcharge routinely, while a large one
suppresses the very charge it would otherwise be excluded from. Those numbers
were originally produced by an ad hoc instrumentation run and quoted in
docs/thesis/06_results.md, docs/thesis/07_discussion.md and docs/thesis/qa_prep.md,
but the instrumentation itself was never checked into the repository, so the
figures could not be reproduced or re-verified. This script is that
instrumentation, checked in so the three numbers are a shipped, re-runnable
artifact rather than a claim.

What it does. It instantiates CapacityMechanism directly (window=24, k=1.0,
charge_rate_eur_per_kwh=0.15, capacity_passthrough=0.10, surcharge_mode=
"proportional", the shipped default) and drives it with three synthetic
brokers, A, B and C, over 800 steps per seed and 10 seeds (random.seed(0)
through random.seed(9)). Each step draws a base load base = 100 + 40 *
random.random() and splits it 62/38 between A and B; C is a net exporter whose
share of the feeder varies across three runs (0.20, 0.08, 0.04 of base). On
roughly two-thirds of steps (random.random() > 0.35) C contributes its share
like a normal load; on the rest it net-exports 0.75 of its share instead,
which is the condition D10 checks for exclusion. A scarcity hour is any step
where the mechanism actually levies a charge (total_charge_eur > 0.0); a
divergence event is a scarcity hour in which A carries a positive surcharge
while C's surcharge is zero, i.e. C was excluded from a charge A still paid.

Expected output, matching docs/DECISIONS.md's D10 result correction section
and the figures cited from it in docs/thesis/:

    exporting broker share 0.20:   0 divergence events in 1469 scarcity hours   (0.00 percent)
    exporting broker share 0.08:   3 divergence events in 1530 scarcity hours   (0.20 percent)
    exporting broker share 0.04: 110 divergence events in 1468 scarcity hours   (7.49 percent)

Run from the repository root:
    python scripts/check_surcharge_timing.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from microgrid_sim.environment.capacity import CapacityMechanism  # noqa: E402  (path set above)

WINDOW_HOURS = 24
K = 1.0
CHARGE_RATE_EUR_PER_KWH = 0.15
CAPACITY_PASSTHROUGH = 0.10
SURCHARGE_MODE = "proportional"

DEFAULT_SEEDS = tuple(range(10))
DEFAULT_STEPS = 800
DEFAULT_EXPORT_SHARES = (0.20, 0.08, 0.04)
EXPORT_PROBABILITY_THRESHOLD = 0.35


def _run_one_seed(export_share: float, seed: int, steps: int) -> tuple[int, int]:
    """Drive one seeded 800-step run; return (divergence_events, scarcity_hours)."""
    random.seed(seed)
    mechanism = CapacityMechanism(
        window=WINDOW_HOURS,
        k=K,
        charge_rate_eur_per_kwh=CHARGE_RATE_EUR_PER_KWH,
        capacity_passthrough=CAPACITY_PASSTHROUGH,
        surcharge_mode=SURCHARGE_MODE,
    )
    divergence_events = 0
    scarcity_hours = 0
    for _ in range(steps):
        base = 100 + 40 * random.random()
        contributions = {
            "A": base * (1 - export_share) * 0.62,
            "B": base * (1 - export_share) * 0.38,
            "C": base * export_share if random.random() > EXPORT_PROBABILITY_THRESHOLD else -base * export_share * 0.75,
        }
        feeder_net_import_kwh = sum(contributions.values())
        result = mechanism.step(feeder_net_import_kwh, contributions)
        if result.total_charge_eur > 0.0:
            scarcity_hours += 1
            a_pays = result.surcharges_eur_per_kwh["A"] > 0.0
            c_excluded = result.surcharges_eur_per_kwh["C"] <= 0.0
            if a_pays and c_excluded:
                divergence_events += 1
    return divergence_events, scarcity_hours


def run_export_share(export_share: float, seeds: tuple[int, ...], steps: int) -> tuple[int, int]:
    """Sum divergence events and scarcity hours for export_share across all seeds."""
    total_divergence = 0
    total_scarcity = 0
    for seed in seeds:
        divergence, scarcity = _run_one_seed(export_share, seed, steps)
        total_divergence += divergence
        total_scarcity += scarcity
    return total_divergence, total_scarcity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds",
        type=int,
        default=len(DEFAULT_SEEDS),
        help=f"number of seeds, 0..N-1 (default {len(DEFAULT_SEEDS)}, matching the D10 figures)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"steps per seed (default {DEFAULT_STEPS}, matching the D10 figures)",
    )
    parser.add_argument(
        "--export-shares",
        type=float,
        nargs="+",
        default=list(DEFAULT_EXPORT_SHARES),
        help=f"exporting broker C's share(s) of base load (default {list(DEFAULT_EXPORT_SHARES)})",
    )
    args = parser.parse_args(argv)

    seeds = tuple(range(args.seeds))
    for export_share in args.export_shares:
        divergence, scarcity = run_export_share(export_share, seeds, args.steps)
        pct = 100.0 * divergence / scarcity if scarcity else 0.0
        print(
            f"exporting broker share {export_share:.2f}: {divergence:3d} divergence events "
            f"in {scarcity} scarcity hours   ({pct:.2f} percent)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
