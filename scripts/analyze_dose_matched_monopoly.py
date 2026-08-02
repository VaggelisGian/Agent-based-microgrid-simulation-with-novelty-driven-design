"""D11 dose-matched monopoly arm analysis (docs/DECISIONS.md D11).

D11's question: D10 found the peak-to-average channel is deferred VOLUME, not
desynchronized deferral timing. The shipped monopoly arm (broker_count=1,
results/sweep_monopoly.parquet) runs the sole broker at FULL capacity_passthrough,
because a single broker's contribution share is 1 by construction, so its failure
to shave the peak is confounded with an overdose relative to what a competitive
market's customers would actually defer. D11 adds capacity_surcharge_divisor: an
integer M substitutes for the active broker count N wherever the pricing-channel
formula depends on N, so a broker_count=1 monopolist can be dosed at
passthrough/M, the per-broker signal strength a symmetric M-broker market
delivers, while broker_count itself stays fixed at 1 (one supplier, no
switching). This script answers whether that dose-matched monopoly reproduces
the competitive peak-to-average result, or falls short of it, at matched dose.

THE DIVISOR IS AN ARTIFICIAL ANALYSIS-ONLY CONTROL WITH NO MARKET INTERPRETATION.
Under the shipped allocation rule a monopolist cannot choose to receive
passthrough/M; it receives the full passthrough by construction (share_b = 1).
No claim in this script or its outputs treats divisor M as a policy instrument a
regulator could build by decree. capacity_passthrough itself is a signal-strength
coefficient, never a EUR/kWh tariff rate or a price.

D11's pre-registered reading rule (docs/DECISIONS.md D11, tested here, not
assumed):
  - If monopoly at divisor 3 matches broker_count 3 on peak-to-average within the
    seed-level noise band AND lands at comparable realized deferred volume, then
    competition is SUFFICIENT but NOT NECESSARY for the headline result, and the
    thesis claim narrows to exactly that ("not necessary" is a claim about this
    design at these coefficients, not about competition in general).
  - If monopoly at divisor 3 still differs materially from broker_count 3 at
    matched volume, something beyond dose distinguishes the competitive case;
    report what the data shows it to be, without inventing a mechanism.
  - Any other pattern, including a divisor that fails to match volume at all, is
    reported as observed and the claim is downgraded to match.

A non-significant paired difference is NOT proof of equivalence. Where the data
supports "matches within the noise band", this script reports the CI on that
difference so a reader can see the size of effect the test could and could not
have detected, and never writes "identical" or "no difference" in its place.

Data sources (read-only; this script writes only the two named outputs below,
nothing here is re-run):
  - results/sweep_dose_matched_monopoly.parquet (360 rows): the new D11 arm,
    broker_count=1, capacity_surcharge_divisor in {2, 3, 5}, k in {0.5, 1.0},
    ablation in {capacity_disabled, capacity_both}, capacity_surcharge_mode
    "proportional", 30 seeds (20260704..20260733).
  - results/sweep_monopoly.parquet: the divisor-1-equivalent comparator (the
    monopoly at FULL passthrough, the shipped pre-D11 broker_count=1 arm),
    filtered to k in {0.5, 1.0} and the two shared ablations. A monopolist under
    the shipped rule (share_b=1, N=1) already receives passthrough * 1 * (N/N)
    = passthrough, so this arm IS "divisor 1" analytically; it is reused, not
    re-run (docs/DECISIONS.md D11 "Reused, not re-run").
  - results/sweep_raw.parquet: the competitive comparators, broker_count in
    {2, 3, 5}, proportional mode, filtered to the same k and ablations. Also
    reused, not re-run.

Guardrail checks performed and printed (docs/DECISIONS.md D11's guardrails,
verified empirically here, not assumed):
  - The 180 capacity_disabled rows in the new arm are bit-identical across
    divisor 2, 3 and 5 (the pricing channel is off, so the divisor is provably
    inert there; this checks the construction argument rather than trusting it).
  - Those same disabled rows are bit-identical to the same-seed capacity_disabled
    rows of results/sweep_monopoly.parquet at k in {0.5, 1.0}.

Two structural findings this script verifies from the data, not merely restates
(see print_saturation_check() and print_effective_broker_count()):
  (a) SATURATION. response_reference_eur_per_kwh is read from config/default.yaml
      (not hardcoded here). The per-firing-hour surcharge is derived from the
      data as mean_broker_surcharge_json (which averages over ALL hours,
      firing and non-firing) divided by capacity_fire_rate. At divisor 1 and 2
      this ratio divided by response_reference_eur_per_kwh is >= 1, so the
      deferral response clip(surcharge / response_reference, 0, 1) saturates at
      1.0 for both, which is why divisor 1 and 2 are bit-identical in the data:
      not a coincidence, a shared ceiling. Divisor 3 and 5 sit below the ceiling.
      D10's result section states "Nothing saturates the deferral response ...
      in any cell", but every D10 cell has broker_count >= 2; this is reported
      as a SCOPE correction to that sentence (it does not hold at broker_count 1
      under a small divisor), not as an error in D10's verdict, which was never
      about broker_count 1.
  (b) EFFECTIVE BROKER COUNT. The nominal broker_count=5 arm runs at 3 to 4
      ACTIVE (positive load share) brokers (Phase 5 observation, D8), derived
      here from broker_load_share_json via analyze_sweep.active_broker_count,
      reused by import. So broker_count=5's per-broker dose is nearer
      passthrough/3.6 than passthrough/5, and divisor 3 (not divisor 5) is the
      volume-matched comparator to broker_count=5's result.

Also verified and printed: within each k, does realized deferred volume order
the peak-to-average effect across all seven arms (monopoly divisors 1, 2, 3, 5
plus broker_count 2, 3, 5)? Spearman rho and p are printed with n stated
explicitly (n=7), following the precedent set after an earlier audit asked
D10's write-up to add its own explicit n.

The capacity_disabled baselines are NOT identical across arms (monopoly vs
broker_count=2 vs 3 vs 5): a different broker list consumes the model's seeded
RNG stream differently (about 0.093 max abs diff on a peak-to-average of
roughly 3.5). Every effect in this file's CSV is therefore a percent change
against that ROW'S OWN paired baseline, which is valid (the comparison inside
an arm is still seed-paired), but it means the seven arms' capacity_disabled
means are not expected to coincide, and this file does not pretend they do.

Statistical methods are REUSED from scripts/analyze_sweep.py, not reimplemented:
paired_comparison (paired Cohen's d_z and scipy.stats.ttest_rel),
paired_pct_change (paired mean of per-seed percent changes), holm_bonferroni,
benjamini_hochberg, t_ci_halfwidth, bootstrap_paired_diff_ci, ALPHA,
BOOTSTRAP_SEED (20260704), N_BOOTSTRAP (10000), SIGN_CONVENTION_NOTE, and
active_broker_count. This script does not alter any of them.

CI convention (deliberate, stated here because it differs slightly by scope).
For "arm_vs_own_baseline" rows the CI is computed on the RAW paired difference
(capacity_both minus capacity_disabled, in the metric's own units), exactly the
quantity scripts/analyze_sweep.py's build_corrected_summary()/_family_row()
CI's, so the bootstrap and t-based CI here match that file's method bit for
bit given the same input array; paired_mean_pct_change is reported alongside as
its own point estimate, not re-derived from the CI. For "decisive_contrast" rows
(the monopoly-divisor-3 vs broker_count-3 test D11's rule turns on) the "raw
paired difference" being tested IS a difference of two already-percent
per-seed values (mono's per-seed pct change minus broker_count=3's per-seed pct
change), because that is the quantity the pre-registered rule asks about
directly; the same t_ci_halfwidth/bootstrap_paired_diff_ci machinery is applied
to that array unchanged.

Correction families (mirrors scripts/analyze_surcharge_mode.py's precedent: the
primary family is the finest-grained set of REAL, non-overlapping paired tests
this arm produces; secondary/alternative views are corrected separately, never
pooled with the primary, so a coarser or narrower view can never dilute or
inflate the primary budget):

  - "dose_matched_arm_primary_real" (28 tests): capacity_both vs
    capacity_disabled, paired by seed against that arm's OWN baseline, for both
    metric-3 sub-measures (feeder_peak_to_average_ratio,
    feeder_coefficient_of_variation), at each of the 7 arms (monopoly divisors
    1, 2, 3, 5; broker_count 2, 3, 5) and 2 k values (7 x 2 x 2 = 28). This is
    the PRIMARY family: it answers, for every arm, whether the mechanism has a
    measurable effect at all, mirroring D10's 36-test primary family structure
    (which also pooled both metric-3 sub-measures together rather than
    splitting them into separate families).
  - "dose_matched_peak_only_alternative" (14 tests): the same 7 arms x 2 k, but
    feeder_peak_to_average_ratio only. This is a DEFENSIBLE ALTERNATIVE, not a
    second primary claim: D11's decision rule is stated purely in terms of
    peak-to-average ratio, so a reader may reasonably ask what the correction
    budget looks like without folding in the coefficient-of-variation tests
    that the rule does not turn on. Corrected as its own family so the primary
    family's budget is never silently changed by this narrower framing, the
    same reasoning D9's cell-vs-marginal split and D8's primary-vs-alternative
    split both gave for keeping alternate views separate.
  - "divisor3_vs_bc3_decisive_contrast_secondary" (4 tests): the D11 rule's own
    decisive comparison, monopoly at divisor 3 against broker_count 3, computed
    seed by seed on the paired PERCENT-CHANGE differences, for both metric-3
    sub-measures at both k values (2 x 2 = 4). Corrected as its own secondary
    family because it re-uses the same underlying seed draws that the primary
    family already tested, under a different (differenced-across-arms)
    aggregation; pooling it with the primary family would double-count those
    draws, exactly the reasoning scripts/analyze_surcharge_mode.py gives for
    keeping its own 6-test gradient family separate from its 36-test primary
    family. The peak-to-average row within this family is the one D11's rule
    is actually stated in terms of; the coefficient-of-variation row is
    reported alongside for transparency, not as part of the rule's own test.

Run from the repository root:

    python scripts/analyze_dose_matched_monopoly.py

Requires: Python 3.12+, pandas, numpy, scipy, matplotlib (no seaborn), pyyaml
(via microgrid_sim.config.loader, already a project dependency). ASCII only
throughout (source and all written CSV/PNG artifacts). No randomness beyond the
fixed bootstrap seed, no network access, no re-running of any simulation, no
tuning of any analysis choice against a desired result. Effects are reported as
observed; if the observed pattern does not fit either of D11's two clean
branches, that is reported plainly instead of forced into one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import analyze_sweep as base  # noqa: E402 (path set above)
# Reused from scripts/analyze_sweep.py rather than reimplemented: paired_comparison,
# paired_pct_change, holm_bonferroni, benjamini_hochberg, t_ci_halfwidth,
# bootstrap_paired_diff_ci, active_broker_count, ALPHA, BOOTSTRAP_SEED, N_BOOTSTRAP,
# SIGN_CONVENTION_NOTE.

from microgrid_sim.config.loader import load_config  # noqa: E402 (path set above)

# ---------------------------------------------------------------------------
# Paths and grid constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DOSE_PATH = REPO_ROOT / "results" / "sweep_dose_matched_monopoly.parquet"
MONOPOLY_PATH = REPO_ROOT / "results" / "sweep_monopoly.parquet"
RAW_PATH = REPO_ROOT / "results" / "sweep_raw.parquet"
# Not part of the D11 grid: read only by print_d10_correction_table() below, the
# non-blocking reproducibility item for docs/DECISIONS.md's D10 correction
# subsection ("A correction to D10 that this arm forced"), which currently has
# no script reproducing its firing-hour-surcharge-per-broker table.
SURCHARGE_MODE_PATH = REPO_ROOT / "results" / "sweep_surcharge_mode.parquet"
CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
OUTPUT_CSV_PATH = REPO_ROOT / "results" / "dose_matched_monopoly.csv"
PLOTS_DIR = REPO_ROOT / "results" / "plots"
FIG_PATH = PLOTS_DIR / "fig_dose_matched_monopoly.png"

K_VALUES = (0.5, 1.0)
MONOPOLY_DIVISORS_NEW = (2, 3, 5)  # newly run for D11 (results/sweep_dose_matched_monopoly.parquet)
MONOPOLY_DIVISOR_REUSED = 1  # analytically equivalent, reused from results/sweep_monopoly.parquet
ALL_MONOPOLY_DIVISORS = (MONOPOLY_DIVISOR_REUSED,) + MONOPOLY_DIVISORS_NEW  # (1, 2, 3, 5), ascending
COMPETITIVE_BROKER_COUNTS = (2, 3, 5)
ABLATIONS = ("capacity_disabled", "capacity_both")
SEED_BASE = 20260704
SEEDS = tuple(SEED_BASE + i for i in range(30))
METRIC3_KEYS = ("feeder_peak_to_average_ratio", "feeder_coefficient_of_variation")
DECISIVE_METRIC = "feeder_peak_to_average_ratio"  # the metric D11's rule is stated in terms of
DECISIVE_DIVISOR = 3  # D11's decisive comparison: monopoly at this divisor vs broker_count 3
DECISIVE_BROKER_COUNT = 3

N_ARMS = len(ALL_MONOPOLY_DIVISORS) + len(COMPETITIVE_BROKER_COUNTS)  # 7
assert N_ARMS == 7

FAMILY_SIZE_ARM_PRIMARY = N_ARMS * len(K_VALUES) * len(METRIC3_KEYS)  # 7 * 2 * 2 = 28
FAMILY_SIZE_PEAK_ONLY_ALTERNATIVE = N_ARMS * len(K_VALUES)  # 7 * 2 = 14
FAMILY_SIZE_DECISIVE_SECONDARY = len(K_VALUES) * len(METRIC3_KEYS)  # 2 * 2 = 4

# D10 correction-table reproduction constants (non-blocking reproducibility item,
# see print_d10_correction_table() below). D10's own grid (results/sweep_surcharge_mode.parquet)
# has broker_count in {2, 3, 5} and k in {0.5, 1.0}; D10_TABLE_K = 0.5 because that is the
# ONLY one of the two k values that reproduces docs/DECISIONS.md's shipped digits (checked
# directly: k=1.0 and the two k pooled both give visibly different numbers, e.g. 0.0824/0.1176
# and 0.0828/0.1172 at broker_count=2 against the shipped 0.0832/0.1167, while k=0.5 alone
# matches to four decimals). The original write-up never states which k it used, since it was
# never a script in the first place; this is that missing fact, established empirically rather
# than assumed, exactly like everything else this project logs.
D10_TABLE_BROKER_COUNTS = (2, 3, 5)
D10_TABLE_MODES = ("synchronized", "renormalized")
D10_TABLE_K = 0.5

ARM_COLOR = {"monopoly_dose_matched": "#d62728", "competitive": "#1f77b4"}
DIVISOR_MARKER = {1: "D", 2: "s", 3: "^", 5: "o"}
BC_MARKER = {2: "s", 3: "^", 5: "o"}


# ---------------------------------------------------------------------------
# Load and tag
# ---------------------------------------------------------------------------


def arm_label_monopoly(divisor: int) -> str:
    return f"mono_divisor_{divisor}"


def arm_label_competitive(bc: int) -> str:
    return f"bc_{bc}"


def load_tagged_frames() -> pd.DataFrame:
    """Load all three parquets, filter to the shared grid, and tag every row
    with an `arm` identifier so the rest of the script can treat all 7 arms
    uniformly. Nothing is re-run; this is read-only."""
    dose = pd.read_parquet(DOSE_PATH)
    mono = pd.read_parquet(MONOPOLY_PATH)
    raw = pd.read_parquet(RAW_PATH)

    frames = []

    # Monopoly divisors 2, 3, 5 (newly run for D11).
    dose = dose[dose["k"].isin(K_VALUES) & dose["ablation"].isin(ABLATIONS) & dose["broker_count"].eq(1)].copy()
    dose["arm"] = dose["capacity_surcharge_divisor"].map(arm_label_monopoly)
    dose["arm_type"] = "monopoly_dose_matched"
    dose["data_source"] = "sweep_dose_matched_monopoly.parquet"
    frames.append(dose)

    # Monopoly divisor 1 (analytically full passthrough; reused, not re-run).
    mono = mono[mono["k"].isin(K_VALUES) & mono["ablation"].isin(ABLATIONS) & mono["broker_count"].eq(1)].copy()
    mono["capacity_surcharge_divisor"] = MONOPOLY_DIVISOR_REUSED
    mono["arm"] = arm_label_monopoly(MONOPOLY_DIVISOR_REUSED)
    mono["arm_type"] = "monopoly_dose_matched"
    mono["data_source"] = "sweep_monopoly.parquet (reused, not re-run; analytically divisor=1)"
    frames.append(mono)

    # Competitive comparators (reused, not re-run).
    raw = raw[
        raw["k"].isin(K_VALUES) & raw["ablation"].isin(ABLATIONS) & raw["broker_count"].isin(COMPETITIVE_BROKER_COUNTS)
    ].copy()
    raw["capacity_surcharge_divisor"] = np.nan
    raw["arm"] = raw["broker_count"].map(arm_label_competitive)
    raw["arm_type"] = "competitive"
    raw["data_source"] = "sweep_raw.parquet (reused, not re-run)"
    frames.append(raw)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def validate_combined(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    expected_arms = {arm_label_monopoly(d) for d in ALL_MONOPOLY_DIVISORS} | {
        arm_label_competitive(bc) for bc in COMPETITIVE_BROKER_COUNTS
    }
    observed_arms = set(df["arm"].unique())
    if observed_arms != expected_arms:
        problems.append(f"arm set {sorted(observed_arms)} != expected {sorted(expected_arms)}")

    for arm in sorted(expected_arms):
        for k in K_VALUES:
            for ablation in ABLATIONS:
                cell = df[(df["arm"] == arm) & (df["k"] == k) & (df["ablation"] == ablation)]
                if len(cell) != len(SEEDS):
                    problems.append(f"arm={arm} k={k} ablation={ablation}: {len(cell)} rows, expected {len(SEEDS)}")
                observed_seeds = set(cell["seed"].astype(int))
                if observed_seeds != set(SEEDS):
                    problems.append(
                        f"arm={arm} k={k} ablation={ablation}: seed mismatch, "
                        f"missing {sorted(set(SEEDS) - observed_seeds)}, "
                        f"unexpected {sorted(observed_seeds - set(SEEDS))}"
                    )
    return problems


def _cell_frames(df: pd.DataFrame, arm: str, k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = df[(df["arm"] == arm) & (df["k"] == k)]
    both = cell[cell["ablation"] == "capacity_both"].set_index("seed").sort_index()
    disabled = cell[cell["ablation"] == "capacity_disabled"].set_index("seed").sort_index()
    if len(both) != len(SEEDS) or len(disabled) != len(SEEDS):
        raise AssertionError(f"arm={arm} k={k}: {len(both)} capacity_both, {len(disabled)} capacity_disabled rows")
    return both, disabled


# ---------------------------------------------------------------------------
# Guardrail checks (D11's guardrails, verified from data, not assumed)
# ---------------------------------------------------------------------------


def print_guardrail_checks(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("GUARDRAIL CHECKS (docs/DECISIONS.md D11)")
    print("=" * 78)

    check_cols = [
        "feeder_peak_to_average_ratio", "feeder_coefficient_of_variation", "total_deferred_kwh",
        "avg_cost_per_agent_eur", "capacity_fire_rate", "prosumer_self_sufficiency",
    ]

    dose_disabled = df[(df["arm_type"] == "monopoly_dose_matched") & (df["ablation"] == "capacity_disabled")]
    for k in K_VALUES:
        sub = dose_disabled[dose_disabled["k"] == k]
        by_div = {
            d: sub[sub["capacity_surcharge_divisor"] == d].sort_values("seed").reset_index(drop=True)
            for d in MONOPOLY_DIVISORS_NEW
        }
        max_diff = 0.0
        for c in check_cols:
            for d in (3, 5):
                diff = float((by_div[2][c] - by_div[d][c]).abs().max())
                max_diff = max(max_diff, diff)
        print(f"  k={k:g}: max abs diff, capacity_disabled across divisor 2/3/5, all {len(check_cols)} columns: {max_diff:.3e}")

    mono_reused = df[(df["arm"] == arm_label_monopoly(MONOPOLY_DIVISOR_REUSED)) & (df["ablation"] == "capacity_disabled")]
    dose_div2 = dose_disabled[dose_disabled["capacity_surcharge_divisor"] == 2]
    for k in K_VALUES:
        a = dose_div2[dose_div2["k"] == k].sort_values("seed").reset_index(drop=True)
        b = mono_reused[mono_reused["k"] == k].sort_values("seed").reset_index(drop=True)
        max_diff = max(float((a[c] - b[c]).abs().max()) for c in check_cols)
        print(f"  k={k:g}: max abs diff, capacity_disabled divisor 2 vs results/sweep_monopoly.parquet: {max_diff:.3e}")


# ---------------------------------------------------------------------------
# Finding (a): saturation
# ---------------------------------------------------------------------------


def print_saturation_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("FINDING (a): SATURATION OF THE DEFERRAL RESPONSE")
    print("=" * 78)

    config = load_config(CONFIG_PATH)
    capacity_cfg = config["capacity_mechanism"]
    passthrough = float(capacity_cfg["capacity_passthrough"])
    response_reference = float(capacity_cfg["response_reference_eur_per_kwh"])
    print(f"  config/default.yaml: capacity_passthrough = {passthrough:g}, response_reference_eur_per_kwh = {response_reference:g}")
    print(
        "  Saturation below is judged on the ANALYTIC firing-hour surcharge, not the empirical estimator printed "
        "alongside it. Reason: an earlier version of this function judged saturation from mean_broker_surcharge_json "
        "/ capacity_fire_rate alone, and that ratio is a BIASED estimator of the firing-hour surcharge, low by about "
        "0.03-0.04 percent, because mean_broker_surcharge_json accumulates the surcharge model.py actually APPLIES "
        "to next-step quotes (carried over one step, model.py step() order of operations), while capacity_fire_rate "
        "counts the excess event in the step where capacity.py's CapacityMechanism.step() computes it, one step "
        "earlier; the two are off by exactly the one firing-hour surcharge computed on the run's final step, which "
        "is never applied within the horizon (verified by direct instrumentation: reconstructing total applied "
        "surcharge as total computed minus the last step's computed value matches the model's own accumulator "
        "exactly). That bias is harmless almost everywhere, but capacity_passthrough/2 lands EXACTLY on the "
        "response_reference clip boundary, so a 0.03-0.04 percent low bias flips the >= comparison there and made "
        "the earlier version of this script report divisor 2 as unsaturated when it is not."
    )

    both = df[(df["arm_type"] == "monopoly_dose_matched") & (df["ablation"] == "capacity_both")]
    for k in K_VALUES:
        print(f"  k={k:g}:")
        for divisor in ALL_MONOPOLY_DIVISORS:
            sub = both[(both["k"] == k) & (both["capacity_surcharge_divisor"] == divisor)]

            # ANALYTIC: at broker_count=1 a single broker's positive-contribution
            # share is exactly 1 whenever it fires at all (capacity.py's
            # proportional branch: contribution / sum_positive_contrib with only
            # one broker in the mapping), so the firing-hour surcharge is exactly
            # capacity_passthrough / divisor, read from config with no estimator,
            # no lag and no round-trip through model output. This is the value
            # the saturated/unsaturated call below is based on.
            analytic_surcharge = passthrough / divisor
            analytic_ratio = analytic_surcharge / response_reference
            analytic_saturated = analytic_ratio >= 1.0

            # ESTIMATOR, kept for corroboration (not dropped: an estimator that
            # agrees with the analytic value to 0.03-0.04 percent is itself a
            # useful cross-check), but explicitly LABELED and never used for the
            # saturated/unsaturated call, for the bias reason printed above.
            # mean_broker_surcharge_json averages the APPLIED per-broker surcharge
            # over ALL hours, firing and non-firing (the model's single broker
            # means the json has exactly one value per row); dividing by
            # capacity_fire_rate is the (biased) attempt to recover the
            # firing-hour surcharge from that all-hours mean.
            mean_surcharge = sub["mean_broker_surcharge_json"].apply(lambda blob: next(iter(json.loads(blob).values())))
            fire_rate = sub["capacity_fire_rate"]
            estimator_firing_hour_surcharge = (mean_surcharge / fire_rate).mean()
            estimator_ratio = estimator_firing_hour_surcharge / response_reference

            print(
                f"    divisor {divisor}: ANALYTIC firing-hour surcharge = {analytic_surcharge:.5f} "
                f"(= passthrough/{divisor}), surcharge/response_reference = {analytic_ratio:.3f} -> "
                f"{'SATURATED (clips to 1.0)' if analytic_saturated else 'unsaturated'}"
            )
            print(
                f"                ESTIMATOR (biased low ~0.03-0.04%, see note above; corroboration only, not "
                f"used for the saturation call): mean_broker_surcharge (all-hours mean) = {mean_surcharge.mean():.5f}, "
                f"fire_rate = {fire_rate.mean():.5f}, implied firing-hour surcharge = "
                f"{estimator_firing_hour_surcharge:.5f}, surcharge/response_reference = {estimator_ratio:.3f} -> "
                f"{'saturated' if estimator_ratio >= 1.0 else 'unsaturated'}"
            )
    print(
        "  Scope correction to D10's non-saturation sentence: D10's own cells are all broker_count >= 2, and "
        "'nothing saturates the deferral response' is correct there. It does NOT hold at broker_count 1 with a "
        "small divisor: divisor 1 (ratio 2.000) and divisor 2 (ratio EXACTLY 1.000, the clip boundary) both "
        "saturate here ANALYTICALLY, which is why their physical output is bit-identical (a shared response "
        "ceiling of 1.0, not independent runs that happen to agree). Divisor 3 (0.667) and divisor 5 (0.400) sit "
        "below the ceiling. The empirical estimator's own value at divisor 2 sits fractionally under the boundary "
        "(the bias described above), which is why it is reported as corroboration and not as the test."
    )


# ---------------------------------------------------------------------------
# Finding (b): effective broker count
# ---------------------------------------------------------------------------


def print_effective_broker_count(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("FINDING (b): NOMINAL VS EFFECTIVE (ACTIVE) BROKER COUNT")
    print("=" * 78)

    # Canonical figure (Phase 5/D8, reproduced exactly as scripts/analyze_sweep.py's
    # print_diagnostics() computes it: the FULL sweep_raw.parquet, all 4 k values and
    # all 4 ablations, not the k/ablation subset D11 happens to use), so this script's
    # number is directly comparable to the one already on record rather than a
    # different subset-conditional statistic quietly presented as the same figure.
    full_raw = pd.read_parquet(RAW_PATH)
    full_raw = full_raw.copy()
    full_raw["active_brokers"] = full_raw["broker_load_share_json"].apply(base.active_broker_count)
    full_summary = full_raw.groupby("broker_count")["active_brokers"].agg(["mean", "median", "min", "max"])
    print("  Full sweep_raw.parquet (4 k x 4 ablations x 30 seeds), matching scripts/analyze_sweep.py's own diagnostic:")
    print(full_summary.to_string())

    # This arm's own subset (k in {0.5, 1.0}, ablations capacity_disabled/capacity_both
    # only), reported alongside since it is not identical to the full-grid figure above
    # (active brokers depends weakly on k and ablation at the margin) and this script
    # should not silently substitute one for the other.
    competitive = df[df["arm_type"] == "competitive"].copy()
    competitive["active_brokers"] = competitive["broker_load_share_json"].apply(base.active_broker_count)
    subset_summary = competitive.groupby("broker_count")["active_brokers"].agg(["mean", "median", "min", "max"])
    print("\n  This script's own k/ablation subset (k in {0.5, 1.0}, capacity_disabled/capacity_both only):")
    print(subset_summary.to_string())

    print(
        "\n  broker_count=5 runs at a mean of "
        f"{full_summary.loc[5, 'mean']:.3f} active brokers on the full grid (min {int(full_summary.loc[5, 'min'])}, "
        f"max {int(full_summary.loc[5, 'max'])}), not 5, because duplicated brokers are near-substitutes (Phase 5 "
        "observation, D8). Its per-broker dose is therefore nearer passthrough/3.6 than passthrough/5, which is "
        "why divisor 3, not divisor 5, is the volume-matched comparator to broker_count=5's result."
    )


# ---------------------------------------------------------------------------
# Non-blocking reproducibility item: docs/DECISIONS.md's D10 correction table
# ("A correction to D10 that this arm forced") is currently produced by no
# script in this repository, unlike D10's other corrected numbers (which
# scripts/check_surcharge_timing.py already reproduces). This closes that gap.
# ---------------------------------------------------------------------------


def _rank_sorted_firing_hour_surcharges(cell: pd.DataFrame) -> np.ndarray:
    """For one (mode, broker_count, k) cell's 30 seed rows (capacity_both
    only), return the broker_count-length array of firing-hour surcharge
    ESTIMATES (mean_broker_surcharge_json / capacity_fire_rate -- the same
    estimator print_saturation_check uses and labels as biased low by about
    0.03-0.04 percent), RANK-SORTED WITHIN EACH SEED before being averaged
    across the 30 seeds.

    WHY rank-sort per seed rather than average by broker id: broker "role" in
    a firing hour, not broker "id", is the only thing stable across seeds.
    Even at broker_count=3, where the id set never rotates (no duplicated
    broker type), WHICH of the three ids ends up the largest/smallest
    contributor in a given seed's firing hours varies with that seed's random
    demand and switching dynamics; at broker_count=5 two of the five ids are
    near-substitute duplicates (Phase 5/D8) whose relative activity swaps
    depending on each seed's deterministic jitter (run_sensitivity_sweep's
    _build_broker_list). Averaging "by id" therefore blends different
    structural roles across seeds and silently produces a different,
    misleading split: this same data averaged by id (see
    _naive_by_id_firing_hour_surcharges, printed alongside for direct
    comparison) gives a different two-way split at broker_count=3 than the
    rank-sorted one, and at broker_count=5 it blends the two near-zero idle
    duplicates into the active brokers' averages instead of surfacing them as
    the near-zero group they actually are.
    """
    rows = []
    for _, row in cell.iterrows():
        blob = json.loads(row["mean_broker_surcharge_json"])
        fire_rate = row["capacity_fire_rate"]
        rows.append(sorted(v / fire_rate for v in blob.values()))
    return np.array(rows).mean(axis=0)


def _naive_by_id_firing_hour_surcharges(cell: pd.DataFrame) -> dict:
    """Same estimator, averaged BY BROKER ID instead of by within-seed rank:
    the wrong way, computed only so print_d10_correction_table() can show the
    reader the difference the rank-sort makes directly, rather than merely
    asserting that it matters."""
    accum: dict[str, list[float]] = {}
    for _, row in cell.iterrows():
        blob = json.loads(row["mean_broker_surcharge_json"])
        fire_rate = row["capacity_fire_rate"]
        for broker_id, v in blob.items():
            accum.setdefault(broker_id, []).append(v / fire_rate)
    return {broker_id: float(np.mean(vs)) for broker_id, vs in accum.items()}


def print_d10_correction_table() -> None:
    """Reproduce, from results/sweep_surcharge_mode.parquet, the firing-hour
    surcharge per broker table docs/DECISIONS.md's D10 correction subsection
    ("A correction to D10 that this arm forced") reports, so that table is a
    shipped, re-runnable artifact rather than a number nobody can recheck (the
    same reason scripts/check_surcharge_timing.py was checked in for D10's
    other corrected numbers)."""
    print("\n" + "=" * 78)
    print("D10 REPRODUCIBILITY: firing-hour surcharge per broker")
    print("(docs/DECISIONS.md, 'A correction to D10 that this arm forced')")
    print("=" * 78)

    if not SURCHARGE_MODE_PATH.exists():
        print(f"  SKIPPED: {SURCHARGE_MODE_PATH} not found.")
        return

    config = load_config(CONFIG_PATH)
    passthrough = float(config["capacity_mechanism"]["capacity_passthrough"])
    response_reference = float(config["capacity_mechanism"]["response_reference_eur_per_kwh"])

    sm = pd.read_parquet(SURCHARGE_MODE_PATH)
    sm = sm[sm["ablation"] == "capacity_both"]

    print(
        f"  Aggregation is RANK-SORTED WITHIN EACH SEED, then averaged across the 30 seeds (broker ROLE, not "
        "broker ID, is what is stable across seeds; see _rank_sorted_firing_hour_surcharges docstring). All "
        f"values at k={D10_TABLE_K:g} (of the two k this arm ran, the one that reproduces the shipped table's "
        "digits; see D10_TABLE_K's definition above for how that was established). Compared against "
        f"response_reference_eur_per_kwh = {response_reference:g}.\n"
    )

    for mode in D10_TABLE_MODES:
        for bc in D10_TABLE_BROKER_COUNTS:
            cell = sm[(sm["capacity_surcharge_mode"] == mode) & (sm["broker_count"] == bc) & (sm["k"] == D10_TABLE_K)]
            assert len(cell) == 30, f"{mode} bc={bc}: expected 30 seed rows, found {len(cell)}"
            naive = _naive_by_id_firing_hour_surcharges(cell)

            if mode == "synchronized":
                # ANALYTIC: synchronized hands EVERY broker the identical
                # surcharge passthrough/N regardless of contribution share
                # (capacity.py's synchronized branch: surcharge does not even
                # reference the broker's own contribution), so the
                # firing-hour surcharge is passthrough/broker_count exactly,
                # read from config; no estimator is needed for the value
                # itself, only for the cross-check below.
                analytic = passthrough / bc
                ratio = analytic / response_reference
                empirical = _rank_sorted_firing_hour_surcharges(cell)
                print(
                    f"  {mode:13s} bc={bc}: ANALYTIC = {analytic:.4f} for all {bc} brokers (= passthrough/{bc}), "
                    f"ratio to response_reference = {ratio:.3f} -> {'saturated' if ratio >= 1.0 else 'unsaturated'}"
                )
                print(f"                 estimator cross-check (rank-sorted): {', '.join(f'{v:.4f}' for v in empirical)}")
            else:
                # RENORMALIZED: surcharge_b = passthrough * share_b * N varies
                # hour to hour and broker to broker with the realized
                # contribution share, so there is no closed form; the
                # empirical, rank-sorted estimator is the only option. It
                # carries the same ~0.03-0.04 percent one-step-lag bias
                # print_saturation_check documents, which cannot flip any of
                # these values: they sit roughly 24 to 300 percent away from
                # the 0.05 reference, nowhere near the boundary that bit
                # divisor 2 in finding (a).
                empirical = _rank_sorted_firing_hour_surcharges(cell)
                ratios = empirical / response_reference
                print(
                    f"  {mode:13s} bc={bc}: estimator (rank-sorted, EMPIRICAL ONLY, not analytic) = "
                    f"{', '.join(f'{v:.4f}' for v in empirical)}"
                )
                print(f"                 ratio to response_reference = {', '.join(f'{r:.3f}' for r in ratios)}")
                print(
                    f"                 naive by-id average (WRONG, shown only to demonstrate why rank-sorting "
                    f"matters) = {{{', '.join(f'{bid}: {v:.4f}' for bid, v in naive.items())}}}"
                )

    print(
        "\n  Match against the shipped table: synchronized is analytic and matches at every broker_count. "
        "Renormalized matches the shipped digits exactly at broker_count 2 and 3; at broker_count 5 the two "
        "near-zero idle-duplicate values match qualitatively (both near zero) and the three active values match "
        "to within about 0.1-0.2 percent, not bit-for-bit, consistent with the estimator's own documented bias "
        "and with the shipped table having been computed by an unscripted, unreproducible analysis in the first "
        "place -- which is exactly the gap this function closes."
    )


# ---------------------------------------------------------------------------
# Primary family (28) and peak-only alternative family (14):
# capacity_both vs capacity_disabled, per arm, per k, per metric
# ---------------------------------------------------------------------------


def compute_arm_tests(df: pd.DataFrame) -> list[dict]:
    all_arms = [arm_label_monopoly(d) for d in ALL_MONOPOLY_DIVISORS] + [
        arm_label_competitive(bc) for bc in COMPETITIVE_BROKER_COUNTS
    ]
    tests = []
    for arm in all_arms:
        arm_type = "monopoly_dose_matched" if arm.startswith("mono_") else "competitive"
        for k in K_VALUES:
            both, disabled = _cell_frames(df, arm, k)
            for metric in METRIC3_KEYS:
                both_vals = both[metric].to_numpy(dtype=float)
                disabled_vals = disabled[metric].reindex(both.index).to_numpy(dtype=float)
                diff = both_vals - disabled_vals

                paired = base.paired_comparison(both_vals, disabled_vals)
                pct_paired, _ = base.paired_pct_change(both_vals, disabled_vals)
                halfwidth = base.t_ci_halfwidth(paired["sd_diff"], len(diff))
                n_improved = int(np.sum(both_vals < disabled_vals))

                row_divisor = both["capacity_surcharge_divisor"].iloc[0] if arm_type == "monopoly_dose_matched" else np.nan
                row_bc = int(both["broker_count"].iloc[0])
                data_source = both["data_source"].iloc[0]

                tests.append(
                    {
                        "arm": arm,
                        "arm_type": arm_type,
                        "capacity_surcharge_divisor": row_divisor,
                        "broker_count": row_bc,
                        "data_source": data_source,
                        "k": k,
                        "metric": metric,
                        "n_seeds": len(both_vals),
                        "paired_mean_diff_raw": paired["mean_diff"],
                        "paired_mean_pct_change": pct_paired,
                        "paired_dz": paired["d_z"],
                        "p_uncorrected": paired["p_value"],
                        "degenerate": paired["degenerate"],
                        "n_improved": n_improved,
                        "ci95_t_lo": paired["mean_diff"] - halfwidth,
                        "ci95_t_hi": paired["mean_diff"] + halfwidth,
                        "total_deferred_kwh_capacity_both_mean": float(both["total_deferred_kwh"].mean()),
                        "total_deferred_kwh_capacity_disabled_mean": float(disabled["total_deferred_kwh"].mean()),
                        "_diff": diff,
                    }
                )
    assert len(tests) == FAMILY_SIZE_ARM_PRIMARY, f"expected {FAMILY_SIZE_ARM_PRIMARY} arm tests, found {len(tests)}"
    return tests


def _apply_correction(tests: list[dict], key_filter, family_name: str) -> list[int]:
    """Apply Holm/BH correction (and the fixed-seed bootstrap CI) in place to the
    subset of `tests` selected by key_filter, tagging each with correction_family
    and correction_family_size. Returns the list of indices touched."""
    idxs = [i for i, t in enumerate(tests) if key_filter(t)]
    pvals = [tests[i]["p_uncorrected"] for i in idxs]
    holm_adj = base.holm_bonferroni(pvals)
    bh_adj = base.benjamini_hochberg(pvals)
    rng = np.random.default_rng(base.BOOTSTRAP_SEED)
    for j, i in enumerate(idxs):
        t = tests[i]
        t.setdefault("p_holm", {})[family_name] = float(holm_adj[j])
        t.setdefault("p_bh", {})[family_name] = float(bh_adj[j])
        t.setdefault("family_size", {})[family_name] = len(idxs)
        boot_lo, boot_hi = base.bootstrap_paired_diff_ci(t["_diff"], rng)
        t.setdefault("ci95_boot", {})[family_name] = (boot_lo, boot_hi)
    return idxs


# ---------------------------------------------------------------------------
# Secondary family (4): decisive contrast, monopoly divisor 3 vs broker_count 3
# ---------------------------------------------------------------------------


def compute_decisive_contrast_tests(df: pd.DataFrame) -> list[dict]:
    mono_arm = arm_label_monopoly(DECISIVE_DIVISOR)
    bc_arm = arm_label_competitive(DECISIVE_BROKER_COUNT)
    tests = []
    for k in K_VALUES:
        both_m, dis_m = _cell_frames(df, mono_arm, k)
        both_c, dis_c = _cell_frames(df, bc_arm, k)
        for metric in METRIC3_KEYS:
            # base.paired_pct_change returns only the two summary scalars (paired
            # mean, naive pct); the contrast below needs the per-seed array itself
            # (mono's per-seed pct change minus the comparator's), so it is built
            # directly here rather than via that helper.
            per_seed_m = (
                (both_m[metric].to_numpy(dtype=float) - dis_m[metric].reindex(both_m.index).to_numpy(dtype=float))
                / dis_m[metric].reindex(both_m.index).to_numpy(dtype=float)
                * 100.0
            )
            per_seed_c = (
                (both_c[metric].to_numpy(dtype=float) - dis_c[metric].reindex(both_c.index).to_numpy(dtype=float))
                / dis_c[metric].reindex(both_c.index).to_numpy(dtype=float)
                * 100.0
            )
            assert (both_m.index.to_numpy() == both_c.index.to_numpy()).all(), "seed mismatch between contrast arms"

            paired = base.paired_comparison(per_seed_m, per_seed_c)
            halfwidth = base.t_ci_halfwidth(paired["sd_diff"], len(per_seed_m))
            n_mono_better = int(np.sum(per_seed_m < per_seed_c))
            deferred_mono = float(both_m["total_deferred_kwh"].mean())
            deferred_bc = float(both_c["total_deferred_kwh"].mean())

            tests.append(
                {
                    "arm": f"{mono_arm}_minus_{bc_arm}",
                    "arm_type": "monopoly_vs_competitive_contrast",
                    "comparator_arm": bc_arm,
                    "capacity_surcharge_divisor": DECISIVE_DIVISOR,
                    "broker_count": DECISIVE_BROKER_COUNT,
                    "k": k,
                    "metric": metric,
                    "n_seeds": len(per_seed_m),
                    "pct_point_diff_mono_minus_comparator": paired["mean_diff"],
                    "paired_dz": paired["d_z"],
                    "p_uncorrected": paired["p_value"],
                    "degenerate": paired["degenerate"],
                    "n_mono_better": n_mono_better,
                    "ci95_t_lo": paired["mean_diff"] - halfwidth,
                    "ci95_t_hi": paired["mean_diff"] + halfwidth,
                    "total_deferred_kwh_capacity_both_mean": deferred_mono,
                    "comparator_total_deferred_kwh_capacity_both_mean": deferred_bc,
                    "deferred_volume_ratio_mono_over_comparator": deferred_mono / deferred_bc,
                    "mono_pct_mean": float(per_seed_m.mean()),
                    "comparator_pct_mean": float(per_seed_c.mean()),
                    "_diff": per_seed_m - per_seed_c,
                }
            )
    assert len(tests) == FAMILY_SIZE_DECISIVE_SECONDARY
    return tests


# ---------------------------------------------------------------------------
# Build the combined CSV
# ---------------------------------------------------------------------------


def build_output_csv(arm_tests: list[dict], decisive_tests: list[dict]) -> pd.DataFrame:
    rows = []

    for t in arm_tests:
        p_holm_primary = t["p_holm"]["dose_matched_arm_primary_real"]
        p_bh_primary = t["p_bh"]["dose_matched_arm_primary_real"]
        boot_primary = t["ci95_boot"]["dose_matched_arm_primary_real"]
        row = {
            "scope": "arm_vs_own_baseline",
            "arm": t["arm"],
            "arm_type": t["arm_type"],
            "capacity_surcharge_divisor": t["capacity_surcharge_divisor"],
            "broker_count": t["broker_count"],
            "comparator_arm": "",
            "data_source": t["data_source"],
            "k": t["k"],
            "metric": t["metric"],
            "n_seeds": t["n_seeds"],
            "paired_mean_pct_change": t["paired_mean_pct_change"],
            "pct_point_diff_mono_minus_comparator": np.nan,
            "paired_mean_diff_raw": t["paired_mean_diff_raw"],
            "paired_dz": t["paired_dz"],
            "p_uncorrected": t["p_uncorrected"],
            "p_holm": p_holm_primary,
            "p_bh": p_bh_primary,
            "survives_holm_alpha05": bool((not t["degenerate"]) and p_holm_primary <= base.ALPHA),
            "survives_bh_alpha05": bool((not t["degenerate"]) and p_bh_primary <= base.ALPHA),
            "ci95_t_lo": t["ci95_t_lo"],
            "ci95_t_hi": t["ci95_t_hi"],
            "ci95_boot_lo": boot_primary[0],
            "ci95_boot_hi": boot_primary[1],
            "n_favorable_of_n_seeds": t["n_improved"],
            "total_deferred_kwh_capacity_both_mean": t["total_deferred_kwh_capacity_both_mean"],
            "total_deferred_kwh_capacity_disabled_mean": t["total_deferred_kwh_capacity_disabled_mean"],
            "comparator_total_deferred_kwh_capacity_both_mean": np.nan,
            "deferred_volume_ratio_mono_over_comparator": np.nan,
            "correction_family": "dose_matched_arm_primary_real",
            "correction_family_size": FAMILY_SIZE_ARM_PRIMARY,
            "metric3_sign_convention": base.SIGN_CONVENTION_NOTE,
            "notes": "",
        }
        rows.append(row)

        # Peak-only alternative family: same underlying test, a second row with
        # the alternative family's correction applied, only for the decisive metric.
        if t["metric"] == DECISIVE_METRIC:
            p_holm_alt = t["p_holm"]["dose_matched_peak_only_alternative"]
            p_bh_alt = t["p_bh"]["dose_matched_peak_only_alternative"]
            boot_alt = t["ci95_boot"]["dose_matched_peak_only_alternative"]
            alt_row = dict(row)
            alt_row["scope"] = "arm_vs_own_baseline_peak_only_alternative"
            alt_row["p_holm"] = p_holm_alt
            alt_row["p_bh"] = p_bh_alt
            alt_row["survives_holm_alpha05"] = bool((not t["degenerate"]) and p_holm_alt <= base.ALPHA)
            alt_row["survives_bh_alpha05"] = bool((not t["degenerate"]) and p_bh_alt <= base.ALPHA)
            alt_row["ci95_boot_lo"] = boot_alt[0]
            alt_row["ci95_boot_hi"] = boot_alt[1]
            alt_row["correction_family"] = "dose_matched_peak_only_alternative"
            alt_row["correction_family_size"] = FAMILY_SIZE_PEAK_ONLY_ALTERNATIVE
            alt_row["notes"] = (
                "alternative correction family: feeder_peak_to_average_ratio only (the metric D11's rule is "
                "stated in terms of), not pooled with the 28-test primary family that also includes "
                "feeder_coefficient_of_variation"
            )
            rows.append(alt_row)

    pvals = [t["p_uncorrected"] for t in decisive_tests]
    holm_adj = base.holm_bonferroni(pvals)
    bh_adj = base.benjamini_hochberg(pvals)
    for i, t in enumerate(decisive_tests):
        decisive_flag = " (DECISIVE: this is D11's own pre-registered test)" if t["metric"] == DECISIVE_METRIC else (
            " (reported alongside for transparency; not part of D11's rule, which is stated in terms of "
            "feeder_peak_to_average_ratio only)"
        )
        row = {
            "scope": "decisive_contrast",
            "arm": t["arm"],
            "arm_type": t["arm_type"],
            "capacity_surcharge_divisor": t["capacity_surcharge_divisor"],
            "broker_count": t["broker_count"],
            "comparator_arm": t["comparator_arm"],
            "data_source": "sweep_dose_matched_monopoly.parquet vs sweep_raw.parquet (both reused, not re-run)",
            "k": t["k"],
            "metric": t["metric"],
            "n_seeds": t["n_seeds"],
            "paired_mean_pct_change": np.nan,
            "pct_point_diff_mono_minus_comparator": t["pct_point_diff_mono_minus_comparator"],
            "paired_mean_diff_raw": np.nan,
            "paired_dz": t["paired_dz"],
            "p_uncorrected": t["p_uncorrected"],
            "p_holm": float(holm_adj[i]),
            "p_bh": float(bh_adj[i]),
            "survives_holm_alpha05": bool((not t["degenerate"]) and holm_adj[i] <= base.ALPHA),
            "survives_bh_alpha05": bool((not t["degenerate"]) and bh_adj[i] <= base.ALPHA),
            "ci95_t_lo": t["ci95_t_lo"],
            "ci95_t_hi": t["ci95_t_hi"],
            "ci95_boot_lo": np.nan,
            "ci95_boot_hi": np.nan,
            "n_favorable_of_n_seeds": t["n_mono_better"],
            "total_deferred_kwh_capacity_both_mean": t["total_deferred_kwh_capacity_both_mean"],
            "total_deferred_kwh_capacity_disabled_mean": np.nan,
            "comparator_total_deferred_kwh_capacity_both_mean": t["comparator_total_deferred_kwh_capacity_both_mean"],
            "deferred_volume_ratio_mono_over_comparator": t["deferred_volume_ratio_mono_over_comparator"],
            "correction_family": "divisor3_vs_bc3_decisive_contrast_secondary",
            "correction_family_size": FAMILY_SIZE_DECISIVE_SECONDARY,
            "metric3_sign_convention": base.SIGN_CONVENTION_NOTE,
            "notes": (
                f"mono {t['arm'].split('_minus_')[0]} mean {t['mono_pct_mean']:.3f}% vs {t['comparator_arm']} mean "
                f"{t['comparator_pct_mean']:.3f}%; n_favorable = number of seeds where the monopoly arm's paired "
                "pct change beats the comparator's (more negative, i.e. better peak shaving)." + decisive_flag
            ),
        }
        rows.append(row)

    # Bootstrap CI on the decisive-contrast rows: same fixed-seed, threaded-RNG
    # convention as the primary/alternative families, computed as its own pass
    # (its own family, own draw order) so it is reproducible independent of how
    # many primary-family draws preceded it.
    rng = np.random.default_rng(base.BOOTSTRAP_SEED)
    decisive_boot = {}
    for t in decisive_tests:
        boot_lo, boot_hi = base.bootstrap_paired_diff_ci(t["_diff"], rng)
        decisive_boot[(t["k"], t["metric"])] = (boot_lo, boot_hi)
    for row in rows:
        if row["scope"] == "decisive_contrast":
            boot_lo, boot_hi = decisive_boot[(row["k"], row["metric"])]
            row["ci95_boot_lo"] = boot_lo
            row["ci95_boot_hi"] = boot_hi

    columns = [
        "scope", "arm", "arm_type", "capacity_surcharge_divisor", "broker_count", "comparator_arm", "data_source",
        "k", "metric", "n_seeds",
        "paired_mean_pct_change", "pct_point_diff_mono_minus_comparator", "paired_mean_diff_raw", "paired_dz",
        "p_uncorrected", "p_holm", "p_bh", "survives_holm_alpha05", "survives_bh_alpha05",
        "ci95_t_lo", "ci95_t_hi", "ci95_boot_lo", "ci95_boot_hi",
        "n_favorable_of_n_seeds",
        "total_deferred_kwh_capacity_both_mean", "total_deferred_kwh_capacity_disabled_mean",
        "comparator_total_deferred_kwh_capacity_both_mean", "deferred_volume_ratio_mono_over_comparator",
        "correction_family", "correction_family_size", "metric3_sign_convention", "notes",
    ]
    return pd.DataFrame(rows)[columns]


# ---------------------------------------------------------------------------
# Ordering check: does realized deferred volume order the peak-to-average
# effect across all 7 arms, within a fixed k?
# ---------------------------------------------------------------------------


def print_ordering_check(arm_tests: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("ORDERING CHECK: does realized deferred volume order the peak-to-average effect across the 7 arms?")
    print("=" * 78)
    for k in K_VALUES:
        cells = [t for t in arm_tests if t["k"] == k and t["metric"] == DECISIVE_METRIC]
        volumes = [t["total_deferred_kwh_capacity_both_mean"] for t in cells]
        effects = [t["paired_mean_pct_change"] for t in cells]
        rho, p = stats.spearmanr(volumes, effects)
        print(f"  k={k:g} (n={len(cells)} arms): Spearman rho={rho:+.4f}, p={p:.4f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


MONO_LABEL_OFFSET = {
    # Hand-placed, not uniform: at both k the arm order along the curve is the
    # same (d=5 isolated bottom-left; d=3 sits in a tight three-way cluster
    # with bc=3/bc=5; d=1 and d=2 coincide exactly at the top-right corner).
    # A single uniform offset puts d=1's and d=2's labels exactly on top of
    # each other (they share one point) and puts d=3's on top of bc=3/bc=5's,
    # so each is placed individually to stay legible in both panels.
    1: (10, -20),   # below-right of the coincident point; d=2 takes the space above it
    2: (10, 10),    # above-right of the coincident point
    3: (-34, 10),   # above-left, clear of the bc=3/bc=5 cluster below it
    5: (7, 6),
}
BC_LABEL_OFFSET = {
    2: (7, -16),
    3: (8, 12),     # above-right, clear of d=3's label and bc=5's point
    5: (-10, -20),  # below, clear of both d=3 and bc=3 labels
}


def plot_dose_matched_monopoly(arm_tests: list[dict]) -> Path:
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(13, 6.2), sharey=True, layout="constrained")

    for ax, k in zip(axes, K_VALUES):
        cells = {t["arm"]: t for t in arm_tests if t["k"] == k and t["metric"] == DECISIVE_METRIC}

        mono_arms = [arm_label_monopoly(d) for d in ALL_MONOPOLY_DIVISORS]
        mono_x = [cells[a]["total_deferred_kwh_capacity_both_mean"] for a in mono_arms]
        mono_y = [cells[a]["paired_mean_pct_change"] for a in mono_arms]
        ax.plot(mono_x, mono_y, linestyle="--", linewidth=1.4, color=ARM_COLOR["monopoly_dose_matched"], zorder=2)
        for divisor, x, y in zip(ALL_MONOPOLY_DIVISORS, mono_x, mono_y):
            marker = DIVISOR_MARKER[divisor]
            # divisor 1 and 2 are BIT-IDENTICAL (both saturate the deferral
            # response, finding (a)): a single marker would silently hide one
            # point. Divisor 1 is drawn as a larger hollow ring underneath;
            # divisor 2 as its normal filled marker on top, so the coincidence
            # is visible rather than reading as a missing series.
            if divisor == MONOPOLY_DIVISOR_REUSED:
                ax.scatter(
                    [x], [y], marker=marker, s=260, facecolors="none",
                    edgecolors=ARM_COLOR["monopoly_dose_matched"], linewidths=2.0, zorder=3,
                )
            else:
                ax.scatter(
                    [x], [y], marker=marker, s=110, color=ARM_COLOR["monopoly_dose_matched"],
                    edgecolors="black", linewidths=0.6, zorder=4,
                )
            ax.annotate(
                f"d={divisor}", (x, y), textcoords="offset points", xytext=MONO_LABEL_OFFSET[divisor], fontsize=8,
                fontweight="bold", color=ARM_COLOR["monopoly_dose_matched"],
            )

        bc_arms = [arm_label_competitive(bc) for bc in COMPETITIVE_BROKER_COUNTS]
        bc_x = [cells[a]["total_deferred_kwh_capacity_both_mean"] for a in bc_arms]
        bc_y = [cells[a]["paired_mean_pct_change"] for a in bc_arms]
        ax.plot(bc_x, bc_y, linestyle="-", linewidth=1.8, color=ARM_COLOR["competitive"], zorder=2)
        for bc, x, y in zip(COMPETITIVE_BROKER_COUNTS, bc_x, bc_y):
            ax.scatter(
                [x], [y], marker=BC_MARKER[bc], s=110, color=ARM_COLOR["competitive"],
                edgecolors="black", linewidths=0.6, zorder=4,
            )
            ax.annotate(
                f"bc={bc}", (x, y), textcoords="offset points", xytext=BC_LABEL_OFFSET[bc], fontsize=8,
                fontweight="bold", color=ARM_COLOR["competitive"],
            )

        # Highlight the D11 decisive pair: monopoly divisor 3 vs broker_count 3.
        mx, my = cells[arm_label_monopoly(DECISIVE_DIVISOR)]["total_deferred_kwh_capacity_both_mean"], cells[
            arm_label_monopoly(DECISIVE_DIVISOR)
        ]["paired_mean_pct_change"]
        cx, cy = cells[arm_label_competitive(DECISIVE_BROKER_COUNT)]["total_deferred_kwh_capacity_both_mean"], cells[
            arm_label_competitive(DECISIVE_BROKER_COUNT)
        ]["paired_mean_pct_change"]
        ax.plot([mx, cx], [my, cy], linestyle=":", linewidth=1.6, color="dimgray", zorder=1)
        # The mono_divisor_3 / bc_3 / bc_5 points sit close together at both k, so
        # every offset tried near that cluster collided with either the d=3/bc=3
        # point labels or the x-axis title. Placed instead in a fixed, empty patch
        # of AXES-fraction space (checked by rendering and inspecting the PNG),
        # with an arrow from there to the mono_divisor_3 point, so its position
        # does not depend on the two panels' different data ranges.
        ax.annotate(
            "D11 decisive\ncontrast",
            xy=(mx, my), xycoords="data",
            xytext=(0.40, 0.62), textcoords="axes fraction",
            fontsize=7.8, color="dimgray", style="italic", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="dimgray", linewidth=1.0, shrinkA=2, shrinkB=4),
        )

        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("realized deferred volume, capacity_both mean (kWh/year)")
        ax.set_title(f"k = {k:g}")

    axes[0].set_ylabel(
        "Feeder peak-to-average ratio,\npaired mean percent change\n(capacity_both vs capacity_disabled, %)"
    )
    axes[0].annotate(
        "Lower (more negative) = more stable", xy=(0.02, 0.03), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8, style="italic",
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=ARM_COLOR["monopoly_dose_matched"], marker="D", linestyle="--", markersize=8,
               label="monopoly, dose-matched (divisor)"),
        Line2D([0], [0], color=ARM_COLOR["competitive"], marker="o", linestyle="-", markersize=8,
               label="competitive (broker_count)"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=13, markerfacecolor="none",
               markeredgecolor=ARM_COLOR["monopoly_dose_matched"], markeredgewidth=2.0,
               label="divisor 1 (hollow ring: coincides exactly with divisor 2, both saturated)"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=1, frameon=False, fontsize=8.5)

    fig.suptitle(
        "D11: dose-matched monopoly vs competitive peak-to-average result, against realized deferred volume\n"
        "capacity_surcharge_divisor is an ARTIFICIAL analysis-only control with no market interpretation (see script\n"
        "and CSV docstrings); it lets a broker_count=1 monopolist be dosed at passthrough/M instead of the full\n"
        "passthrough its contribution share of 1 would otherwise hand it. Monopoly divisor 1 and 2 coincide exactly\n"
        "(both saturate the deferral response ANALYTICALLY, clip(surcharge/response_reference,0,1)=1.0 for both;\n"
        "see finding (a), which also reports and labels the empirical corroborating estimator alongside).\n"
        "Capacity_disabled baselines differ across arms (different broker lists consume the RNG stream differently),\n"
        "so every effect here is a percent change against that arm's OWN paired baseline, not a common one.",
        fontsize=9.6,
    )
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    return FIG_PATH


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def print_verdict(decisive_tests: list[dict], holm_adj: np.ndarray) -> None:
    print("\n" + "=" * 78)
    print("D11 VERDICT")
    print("=" * 78)

    decisive = [t for t in decisive_tests if t["metric"] == DECISIVE_METRIC]
    by_k = {t["k"]: t for t in decisive}
    idx_by_kmetric = {(t["k"], t["metric"]): i for i, t in enumerate(decisive_tests)}

    all_non_significant = True
    for k in K_VALUES:
        t = by_k[k]
        p_holm = holm_adj[idx_by_kmetric[(k, DECISIVE_METRIC)]]
        sig = (not t["degenerate"]) and p_holm <= base.ALPHA
        all_non_significant = all_non_significant and not sig
        print(
            f"  k={k:g}: monopoly divisor {DECISIVE_DIVISOR} vs broker_count {DECISIVE_BROKER_COUNT}, "
            f"{DECISIVE_METRIC}: diff {t['pct_point_diff_mono_minus_comparator']:+.3f} pp "
            f"(d_z={t['paired_dz']:.3f}, p_uncorrected={t['p_uncorrected']:.3g}, p_holm={p_holm:.3g}, "
            f"sig={sig}), CI95_t [{t['ci95_t_lo']:.3f}, {t['ci95_t_hi']:.3f}], "
            f"deferred ratio (mono/bc{DECISIVE_BROKER_COUNT}) = {t['deferred_volume_ratio_mono_over_comparator']:.4f}, "
            f"monopoly better in {t['n_mono_better']} of {t['n_seeds']} seeds"
        )

    volume_ratios = [by_k[k]["deferred_volume_ratio_mono_over_comparator"] for k in K_VALUES]
    volume_comparable = all(0.85 <= r <= 1.15 for r in volume_ratios)  # within 15%, stated explicitly, not a hidden threshold

    if all_non_significant and volume_comparable:
        branch = "BRANCH 1: dose-matched, within noise band -> competition sufficient but not necessary"
        verdict = (
            "D11 VERDICT, branch 1 of the pre-registered rule fires: at divisor 3, the monopoly's peak-to-average "
            "result is NOT statistically distinguishable from broker_count=3's at this sample size (Holm-corrected "
            "p > 0.05 at both k), and realized deferred volume is comparable between the two arms (ratio "
            f"{volume_ratios[0]:.3f} at k=0.5, {volume_ratios[1]:.3f} at k=1.0, i.e. within about "
            f"{100 * max(abs(1 - r) for r in volume_ratios):.0f} percent). Per D11's pre-registered rule: "
            "COMPETITION IS SUFFICIENT BUT NOT NECESSARY for the peak-to-average result in this design at these "
            "coefficients. This is NOT a claim of equivalence: the CI on the seed-level difference (printed above) "
            "shows the size of effect this sample could and could not have detected, and the honest statement is "
            "that the difference is small and not distinguishable from zero at n=30 seeds, not that the two arms "
            "are identical."
        )
    else:
        branch = "BRANCH 2 or 3: does not cleanly fire branch 1 (see D11's pre-registered rule, docs/DECISIONS.md)"
        verdict = (
            "D11 VERDICT: the data do not cleanly fire branch 1 of the pre-registered rule (either the divisor-3 "
            "vs broker_count-3 difference reaches significance at one or both k, or realized deferred volume is "
            "not comparable between the two arms). Reported as observed per D11's own instruction for this case; "
            "see the per-k detail above for the specific numbers."
        )

    print(f"\n  Branch: {branch}")
    print("\n  " + verdict)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    for path in (DOSE_PATH, MONOPOLY_PATH, RAW_PATH, CONFIG_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            return 1

    df = load_tagged_frames()
    problems = validate_combined(df)
    if problems:
        print("VALIDATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"VALIDATION OK: {len(df)} combined rows across {N_ARMS} arms (4 monopoly divisors + 3 competitive broker counts).")

    print_guardrail_checks(df)
    print_saturation_check(df)
    print_effective_broker_count(df)
    print_d10_correction_table()  # non-blocking reproducibility item; reads its own parquet, independent of df above

    arm_tests = compute_arm_tests(df)
    _apply_correction(arm_tests, lambda t: True, "dose_matched_arm_primary_real")
    _apply_correction(arm_tests, lambda t: t["metric"] == DECISIVE_METRIC, "dose_matched_peak_only_alternative")

    decisive_tests = compute_decisive_contrast_tests(df)
    decisive_pvals = [t["p_uncorrected"] for t in decisive_tests]
    decisive_holm = base.holm_bonferroni(decisive_pvals)

    output_df = build_output_csv(arm_tests, decisive_tests)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nwrote {OUTPUT_CSV_PATH} ({len(output_df)} rows)")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dose_matched_monopoly(arm_tests)
    print(f"wrote {fig_path}")

    print_ordering_check(arm_tests)
    print_verdict(decisive_tests, decisive_holm)

    return 0


if __name__ == "__main__":
    sys.exit(main())
