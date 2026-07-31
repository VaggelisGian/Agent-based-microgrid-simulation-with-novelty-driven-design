"""Phase 6 analysis of the Phase 5 capacity-mechanism sensitivity sweep.

Reads the raw per-run sweep data (results/sweep_raw.parquet: 1440 rows = 4 k values
x 3 broker_count values x 4 ablation configs x 30 seeds) and writes:

  - results/summary_stats.csv    per (k, broker_count, ablation, metric) descriptive
                                  stats (mean, std, 95 percent t-based CI) plus, for
                                  every non-disabled ablation, effect sizes versus
                                  capacity_disabled at the same (k, broker_count):
                                  independent-form Cohen's d, paired Cohen's d_z, and
                                  a paired t-test p-value.
  - results/effect_sizes.csv      a smaller, targeted table: (a) the mechanism effect
                                  on metric 3 (feeder_coefficient_of_variation and
                                  feeder_peak_to_average_ratio) for capacity_pricing_only
                                  and capacity_both versus capacity_disabled, (b) the
                                  same structure for metrics 1, 2, 4, and (c) a sanity
                                  block confirming capacity_pnl_only is statistically
                                  indistinguishable from capacity_disabled on the
                                  physical metrics.
  - results/plots/fig_metric3_vs_k.png
  - results/plots/fig_ablation_bars.png
  - results/plots/fig_fire_rate.png
  - results/plots/fig_broker_heterogeneity.png

If results/sweep_monopoly.parquet is also present (a supplementary broker_count = 1
monopoly arm, same schema, 4 k values x 1 broker_count x 4 ablation configs x 30 seeds
= 480 rows), the script additionally writes:

  - results/monopoly_comparison.csv     one row per (broker_count in {1,2,3,5}, k in
                                         {0.5,1.0,1.5,2.0}): the capacity_pricing_only
                                         vs capacity_disabled paired comparison for
                                         both metric-3 sub-measures (feeder CoV and
                                         feeder peak-to-average), plus mean fire rate,
                                         mean deferred energy, and the consumer-cost
                                         percent change. Purpose: show whether the
                                         metric-3 improvement is driven by the
                                         capacity-charge MECHANISM (present even at
                                         broker_count = 1, a monopoly with nowhere to
                                         switch) or by broker COMPETITION (should
                                         scale with broker_count).
  - results/plots/fig_monopoly_comparison.png   two panels, CoV and peak-to-average
                                         paired percent change vs broker_count, one
                                         line per k, broker_count = 1 marked as the
                                         monopoly reference point.

If results/sweep_monopoly.parquet is absent, this section is skipped with a printed
note; all other outputs are unaffected.

This script additionally writes:

  - results/summary_stats_corrected.csv   multiple-comparison correction (Holm-Bonferroni
                                  primary, Benjamini-Hochberg FDR secondary) and 10000-resample
                                  seed-level paired bootstrap 95 percent CIs, alongside the
                                  existing t-based CIs and uncorrected p-values, for every paired
                                  comparison in the correction family. The family is reported two
                                  ways (see build_corrected_summary()'s docstring for the full
                                  reasoning): FAMILY_SIZE_PRIMARY = 120 real, computed paired
                                  t-tests in the main D8 sweep (capacity_pricing_only and
                                  capacity_both vs capacity_disabled, 5 metrics x 4 k x 3
                                  broker_count), and FAMILY_SIZE_ALTERNATIVE = 180, which folds in
                                  the 60 capacity_pnl_only vs capacity_disabled contrasts that are
                                  deterministic identities (exact zero paired difference, p=1.0 by
                                  convention, not a computed test) rather than real hypothesis
                                  tests. The supplementary broker_count=1 monopoly arm's own 8
                                  real tests (results/monopoly_comparison.csv) are corrected as a
                                  separate SECONDARY family (FAMILY_SIZE_MONOPOLY_SECONDARY = 8),
                                  not pooled into the primary 120, since it is a later addition
                                  analyzed in its own file, not part of the D8 sweep grid itself.
                                  This file is purely additive: new output only, and building it
                                  does not alter anything build_summary_stats() or
                                  build_effect_sizes() compute or write.

Design notes (see docs/DECISIONS.md D6, D7, D8 and the Phase 5 observation section):

  - Seeds are SHARED across ablations at a given (k, broker_count), so ablation
    comparisons are PAIRED by seed. Every effect-size computation in this script
    joins on seed before differencing.
  - capacity_disabled is the reference arm. capacity_pnl_only debits broker ledgers
    but writes nothing into prices, so it is expected to be inert (equal to
    capacity_disabled up to float noise) on the physical metrics (load concentration,
    feeder stability, prosumer self-sufficiency) and on consumer cost. This script
    checks that expectation and flags loudly if it is violated.
  - Sign convention for metric 3 (feeder_coefficient_of_variation and
    feeder_peak_to_average_ratio): a NEGATIVE difference (ablation minus disabled)
    means IMPROVED stability (lower CoV / lower peak-to-average ratio). This is
    stated in the CSV outputs, not just in this docstring.
  - The nominal broker_count for the broker_count == 5 arm runs at 3 to 4 EFFECTIVE
    (positive load share) brokers, not 5 (see docs/DECISIONS.md, Phase 5 observation
    section). This script logs the effective count to stdout as a diagnostic; it does
    not change any per-run metric.

Run from the repository root:

    python scripts/analyze_sweep.py

The script is deterministic: it only reads results/sweep_raw.parquet and writes the
files listed above. No randomness, no network access, no tuning of any analysis
choice against a desired result. Effects are reported as observed.

Requires: Python 3.12+, pandas, numpy, scipy, matplotlib (no seaborn).
ASCII only throughout (source and all written CSV/PNG artifacts).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "results" / "sweep_raw.parquet"
MONOPOLY_PATH = REPO_ROOT / "results" / "sweep_monopoly.parquet"
SUMMARY_PATH = REPO_ROOT / "results" / "summary_stats.csv"
EFFECTS_PATH = REPO_ROOT / "results" / "effect_sizes.csv"
MONOPOLY_COMPARISON_PATH = REPO_ROOT / "results" / "monopoly_comparison.csv"
PLOTS_DIR = REPO_ROOT / "results" / "plots"

# ---------------------------------------------------------------------------
# Sweep grid constants (must match docs/DECISIONS.md D8 / scripts/run_sensitivity_sweep.py)
# ---------------------------------------------------------------------------

K_VALUES = (0.5, 1.0, 1.5, 2.0)
BROKER_COUNTS = (2, 3, 5)
MONOPOLY_BROKER_COUNT = 1
ALL_BROKER_COUNTS = (MONOPOLY_BROKER_COUNT,) + BROKER_COUNTS
DISABLED = "capacity_disabled"
ABLATION_ORDER = ("capacity_disabled", "capacity_pnl_only", "capacity_pricing_only", "capacity_both")
NON_DISABLED = ("capacity_pnl_only", "capacity_pricing_only", "capacity_both")
ENABLED_PRICING = ("capacity_pricing_only", "capacity_both")  # the two arms that activate deferral
SEED_N = 30
CI_CONF = 0.95
DEGENERATE_TOL = 1e-9  # abs tolerance on sd(paired diff) below which d_z/p are set by convention
PNL_LEAK_RELATIVE_TOL = 1e-6  # relative tolerance for the pnl-vs-disabled sanity flag

ABLATION_SHORT = {
    "capacity_disabled": "disabled",
    "capacity_pnl_only": "pnl_only",
    "capacity_pricing_only": "pricing_only",
    "capacity_both": "both",
}

# metric key -> (display label, is this one of the two metric-3 stability sub-metrics)
METRICS = {
    "avg_cost_per_agent_eur": ("Avg cost per agent (EUR/year)", False),
    "load_concentration_hhi": ("Load concentration HHI", False),
    "feeder_peak_to_average_ratio": ("Feeder peak-to-average ratio", True),
    "feeder_coefficient_of_variation": ("Feeder coefficient of variation", True),
    "prosumer_self_sufficiency": ("Prosumer self-sufficiency", False),
}
METRIC3_KEYS = ("feeder_peak_to_average_ratio", "feeder_coefficient_of_variation")
OTHER_METRIC_KEYS = ("avg_cost_per_agent_eur", "load_concentration_hhi", "prosumer_self_sufficiency")

SIGN_CONVENTION_NOTE = (
    "metric3 sign convention: negative (ablation - disabled) diff means IMPROVED "
    "stability (lower CoV / lower peak-to-average ratio); positive means worsened."
)

CORRECTED_SUMMARY_PATH = REPO_ROOT / "results" / "summary_stats_corrected.csv"

# ---------------------------------------------------------------------------
# Multiple-comparison correction and bootstrap CI constants
# ---------------------------------------------------------------------------
#
# Family definition. The PRIMARY correction family is the 120 REAL, computed
# paired t-tests in the main D8 sweep: capacity_pricing_only and capacity_both
# vs capacity_disabled, each of the 5 thesis-relevant metrics, at each of the
# 4 k values and 3 broker counts (2 ablations x 5 metrics x 4 k x 3
# broker_count = 120). It excludes the 60 capacity_pnl_only vs
# capacity_disabled contrasts (1 ablation x 5 metrics x 4 k x 3 broker_count):
# those are DETERMINISTIC IDENTITIES the model guarantees by construction (the
# P&L channel debits a ledger and writes nothing into any price or physical
# quantity, so its paired difference on every metric is exactly 0 for every
# seed), not estimated hypothesis tests -- paired_comparison()'s `degenerate`
# branch sets p=1.0 by convention rather than computing it from a
# zero-variance t-test. Folding a deterministic identity into a correction
# family cannot guard against any real false-positive risk (a p=1.0 entry can
# never survive any correction) while it does dilute the correction budget
# available to the tests that are actual estimates, so these are excluded
# from the primary family and reported separately, in full, for auditability
# (the ALTERNATIVE family size below, which simply adds them back in).
#
# The supplementary broker_count=1 monopoly arm (Phase 8, results/
# monopoly_comparison.csv) is a later addition, analyzed in its own separate
# output file, and is not part of "the sweep" as D8 defines it (broker_count
# in {2, 3, 5}). Its own 8 real paired tests (capacity_pricing_only vs
# capacity_disabled, both metric-3 sub-metrics, 4 k values -- the only
# monopoly comparisons already quoted as evidence in
# docs/thesis/06_results.md Section 6.9 and qa_prep.md question C5) are
# corrected as their own SECONDARY family, not pooled into the primary 120, so
# that a later supplementary arm can never silently shrink or inflate the
# primary family's correction budget.
FAMILY_SIZE_PRIMARY = 120  # main-sweep real tests only (excludes the 60 degenerate pnl_only contrasts)
FAMILY_SIZE_ALTERNATIVE = 180  # main-sweep real (120) + main-sweep degenerate (60); for auditability
FAMILY_SIZE_MONOPOLY_SECONDARY = 8  # bc=1 supplementary arm, corrected separately, see note above

ALPHA = 0.05
BOOTSTRAP_SEED = 20260704  # the project's canonical run seed, reused here (see docs/DECISIONS.md
                            # F1-F4 fix notes for its other appearances) purely to seed the bootstrap
                            # resampling RNG below; it draws no simulation data and touches no model
                            # RNG stream.
N_BOOTSTRAP = 10000
CI_DISAGREEMENT_WIDTH_RATIO = 2.0  # flag a disagreement if one CI is >= 2x the width of the other

# Default matplotlib color cycle, fixed per ablation so colors are consistent across figures.
_DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ABLATION_COLOR = {ablation: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, ablation in enumerate(ABLATION_ORDER)}

# Draw styles for the coincident ablation pairs on the physical metrics. On those
# metrics capacity_pnl_only reproduces capacity_disabled exactly, and capacity_both
# reproduces capacity_pricing_only exactly, because the P&L channel debits broker
# ledgers and writes nothing into any price (clean channel isolation, see
# docs/thesis/06_results.md Section 6.1 and the module docstring above). Drawn with
# one uniform style the overplotted member of each pair is completely invisible, so
# the legend advertises four series while the panel shows two and the figure reads as
# broken rather than as the result it is. Each underlying series is therefore drawn
# thick and solid with a large marker, and the series that lands exactly on top of it
# thin and dashed with a small marker, so both members stay readable and the
# concentric markers show the coincidence. Presentation only: no plotted value is
# offset, rescaled, or otherwise altered.
ABLATION_LINE_STYLE = {
    "capacity_disabled": {"linestyle": "-", "linewidth": 3.6, "markersize": 8.5, "zorder": 2},
    "capacity_pnl_only": {"linestyle": (0, (5, 2.5)), "linewidth": 1.5, "markersize": 3.4, "zorder": 4},
    "capacity_pricing_only": {"linestyle": "-", "linewidth": 3.6, "markersize": 8.5, "zorder": 3},
    "capacity_both": {"linestyle": (0, (5, 2.5)), "linewidth": 1.5, "markersize": 3.4, "zorder": 5},
}


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------


def t_ci_halfwidth(std: float, n: int = SEED_N, conf: float = CI_CONF) -> float:
    """t-based 95 percent CI half-width, df = n - 1."""
    if n < 2 or not np.isfinite(std):
        return float("nan")
    sem = std / np.sqrt(n)
    t_crit = stats.t.ppf(1 - (1 - conf) / 2, df=n - 1)
    return float(t_crit * sem)


def cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    """Standard (unpaired, pooled-sd) Cohen's d for a vs b, treated as two independent samples."""
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    denom = n1 + n2 - 2
    if denom <= 0:
        return 0.0
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / denom
    pooled_sd = np.sqrt(pooled_var)
    if not np.isfinite(pooled_sd) or pooled_sd == 0.0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def paired_comparison(ablation_vals: np.ndarray, disabled_vals: np.ndarray, tol: float = DEGENERATE_TOL) -> dict:
    """Paired diff, paired Cohen's d_z, and a paired t-test, matched by seed order.

    Handles the degenerate case (sd(diff) ~ 0, e.g. capacity_pnl_only vs capacity_disabled
    on a physical metric) without raising: d_z is set to 0.0 and p to 1.0, and the row is
    flagged via `degenerate=True`.
    """
    diff = np.asarray(ablation_vals, dtype=float) - np.asarray(disabled_vals, dtype=float)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    max_abs_diff = float(np.max(np.abs(diff)))
    degenerate = sd_diff < tol
    if degenerate:
        d_z = 0.0
        p_value = 1.0
    else:
        d_z = mean_diff / sd_diff
        _, p_value = stats.ttest_rel(ablation_vals, disabled_vals)
        p_value = float(p_value)
    return {
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "max_abs_diff": max_abs_diff,
        "d_z": float(d_z),
        "p_value": p_value,
        "degenerate": bool(degenerate),
    }


def paired_pct_change(ablation_vals: np.ndarray, disabled_vals: np.ndarray) -> tuple:
    """Return (paired mean of per-seed percent changes, naive percent-change-of-means).

    Per D8/task instructions the paired mean of per-seed percent changes is the preferred
    figure; the naive percent-change-of-means is also returned for comparison.
    """
    ablation_vals = np.asarray(ablation_vals, dtype=float)
    disabled_vals = np.asarray(disabled_vals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_seed_pct = np.where(disabled_vals != 0, (ablation_vals - disabled_vals) / disabled_vals * 100.0, np.nan)
    paired_mean_pct = float(np.nanmean(per_seed_pct))
    disabled_mean = float(np.mean(disabled_vals))
    ablation_mean = float(np.mean(ablation_vals))
    naive_pct = float((ablation_mean - disabled_mean) / disabled_mean * 100.0) if disabled_mean != 0 else float("nan")
    return paired_mean_pct, naive_pct


def active_broker_count(load_share_json: str, tol: float = 1e-9) -> int:
    """Number of brokers with a positive (> tol) load share, parsed from broker_load_share_json."""
    shares = json.loads(load_share_json)
    return sum(1 for v in shares.values() if v > tol)


# ---------------------------------------------------------------------------
# Multiple-comparison correction (Holm-Bonferroni primary, BH FDR secondary)
# ---------------------------------------------------------------------------


def holm_bonferroni(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down family-wise error rate correction.

    Standard algorithm (Holm 1979): sort p ascending, p_(1) <= ... <= p_(m).
    At sorted rank i (1-indexed) the raw-adjusted value is (m - i + 1) * p_(i);
    the adjusted p-value is the running MAXIMUM of that quantity up through
    rank i, which is what enforces the required monotonicity (adjusted p
    cannot decrease as the sorted rank increases), capped at 1.0. Returns
    corrected p-values in the SAME order as the input (not sorted).
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)
    candidate = (m - ranks + 1) * sorted_p
    adjusted_sorted = np.minimum(np.maximum.accumulate(candidate), 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg step-up false discovery rate correction (q-values).

    Standard algorithm (Benjamini and Hochberg 1995): sort p ascending,
    p_(1) <= ... <= p_(m). At sorted rank i the raw-adjusted value is
    p_(i) * m / i; the adjusted q-value is the running MINIMUM of that
    quantity from the largest rank down to rank i, which is what enforces the
    required monotonicity from the top, capped at 1.0. Returns corrected
    q-values in the SAME order as the input (not sorted).
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)
    candidate = sorted_p * m / ranks
    adjusted_sorted = np.minimum(np.minimum.accumulate(candidate[::-1])[::-1], 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def bootstrap_paired_diff_ci(diff, rng: np.random.Generator, n_boot: int = N_BOOTSTRAP, conf: float = CI_CONF) -> tuple:
    """Bootstrap 95 percent CI for the MEAN of paired differences.

    Resamples the SEED-LEVEL paired differences with replacement, not the two
    ablation arms independently: each of the n_boot resamples draws len(diff)
    indices with replacement from the n seed-level differences and takes
    their mean. This is the correct resampling unit for a paired design,
    because the seed indexes a shared population sample and weather draw
    common to both arms being differenced (docs/DECISIONS.md D8: "seeds are
    shared across ablations at a given (k, broker_count), so ablation
    comparisons are PAIRED by seed"); the seed-level difference is therefore
    the exchangeable unit under resampling. Resampling the two arms
    independently would break that deliberate pairing and would misstate the
    true sampling variability of the paired mean difference.
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    resample_means = diff[idx].mean(axis=1)
    lo = float(np.percentile(resample_means, (1 - conf) / 2 * 100))
    hi = float(np.percentile(resample_means, (1 + conf) / 2 * 100))
    return lo, hi


def _ci_disagreement(t_lo: float, t_hi: float, boot_lo: float, boot_hi: float,
                      width_ratio_threshold: float = CI_DISAGREEMENT_WIDTH_RATIO) -> tuple:
    """Flag a material disagreement between the t-based and bootstrap CIs.

    Two independent triggers, either one is enough to flag: (1) the two
    intervals disagree about whether zero lies inside them; (2) the wider
    interval is at least `width_ratio_threshold` times the width of the
    narrower one. Returns (flag: bool, detail: str, width_ratio: float).
    """
    t_excludes_zero = not (t_lo <= 0.0 <= t_hi)
    boot_excludes_zero = not (boot_lo <= 0.0 <= boot_hi)
    zero_flag = t_excludes_zero != boot_excludes_zero

    width_t = t_hi - t_lo
    width_boot = boot_hi - boot_lo
    if width_t <= 0.0 and width_boot <= 0.0:
        width_ratio = 1.0
    elif min(width_t, width_boot) <= 0.0:
        width_ratio = float("inf")
    else:
        width_ratio = max(width_t, width_boot) / min(width_t, width_boot)
    width_flag = width_ratio >= width_ratio_threshold

    flag = bool(zero_flag or width_flag)
    details = []
    if zero_flag:
        details.append(
            f"zero-exclusion disagreement (t excludes zero={t_excludes_zero}, bootstrap excludes zero={boot_excludes_zero})"
        )
    if width_flag:
        details.append(f"width ratio {width_ratio:.2f}x >= {width_ratio_threshold:g}x threshold")
    return flag, "; ".join(details), width_ratio


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(RAW_PATH)
    expected_rows = len(K_VALUES) * len(BROKER_COUNTS) * len(ABLATION_ORDER) * SEED_N
    if len(df) != expected_rows:
        print(
            f"WARNING: expected {expected_rows} rows (4 k x 3 broker_count x 4 ablation x "
            f"{SEED_N} seeds), found {len(df)}."
        )
    return df


def load_monopoly_combined(raw_df: pd.DataFrame) -> pd.DataFrame | None:
    """Load results/sweep_monopoly.parquet (broker_count = 1) and concatenate it with
    the already-loaded main sweep frame on the shared schema, for the monopoly-vs-
    competition comparison.

    Returns None (and prints a note) if results/sweep_monopoly.parquet does not exist,
    so the rest of the script can run unaffected on machines/checkouts that only have
    the main sweep.
    """
    if not MONOPOLY_PATH.exists():
        print(
            f"\nNOTE: {MONOPOLY_PATH} not found; skipping monopoly-vs-competition "
            "comparison (results/monopoly_comparison.csv and "
            "results/plots/fig_monopoly_comparison.png will not be written)."
        )
        return None

    mono_df = pd.read_parquet(MONOPOLY_PATH)
    expected_rows = len(K_VALUES) * 1 * len(ABLATION_ORDER) * SEED_N
    if len(mono_df) != expected_rows:
        print(
            f"WARNING: expected {expected_rows} monopoly rows (4 k x 1 broker_count x "
            f"4 ablation x {SEED_N} seeds), found {len(mono_df)}."
        )
    if set(mono_df["broker_count"].unique()) != {MONOPOLY_BROKER_COUNT}:
        print(
            f"WARNING: {MONOPOLY_PATH} contains broker_count values other than "
            f"{MONOPOLY_BROKER_COUNT}: {sorted(mono_df['broker_count'].unique())}"
        )

    combined = pd.concat([raw_df, mono_df], ignore_index=True)
    return combined


def _cell_by_ablation(df: pd.DataFrame, k: float, broker_count: int) -> dict:
    """Split one (k, broker_count) cell into per-ablation frames indexed by seed."""
    cell = df[(df["k"] == k) & (df["broker_count"] == broker_count)]
    return {ablation: cell[cell["ablation"] == ablation].set_index("seed").sort_index() for ablation in ABLATION_ORDER}


# ---------------------------------------------------------------------------
# Deliverable 2: results/summary_stats.csv
# ---------------------------------------------------------------------------


def build_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        for bc in BROKER_COUNTS:
            by_ablation = _cell_by_ablation(df, k, bc)
            disabled_frame = by_ablation[DISABLED]
            for metric, (label, is_metric3) in METRICS.items():
                sign_note = SIGN_CONVENTION_NOTE if is_metric3 else ""
                for ablation in ABLATION_ORDER:
                    frame = by_ablation[ablation]
                    vals = frame[metric].to_numpy(dtype=float)
                    n = len(vals)
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1))
                    halfwidth = t_ci_halfwidth(std, n)

                    row = {
                        "k": k,
                        "broker_count": bc,
                        "ablation": ablation,
                        "metric": metric,
                        "metric_label": label,
                        "n_seeds": n,
                        "mean": mean,
                        "std": std,
                        "ci95_lo": mean - halfwidth,
                        "ci95_hi": mean + halfwidth,
                        "ci95_halfwidth": halfwidth,
                        "vs_disabled_cohens_d_independent": np.nan,
                        "vs_disabled_paired_dz": np.nan,
                        "vs_disabled_paired_p": np.nan,
                        "vs_disabled_paired_mean_diff": np.nan,
                        "vs_disabled_paired_mean_pct_change": np.nan,
                        "vs_disabled_degenerate_paired_diff": "",
                        "metric3_sign_convention": sign_note,
                        "notes": "",
                    }

                    if ablation == DISABLED:
                        row["notes"] = "reference arm (capacity_disabled); vs_disabled_* not applicable"
                    else:
                        disabled_vals = disabled_frame[metric].reindex(frame.index).to_numpy(dtype=float)
                        d_indep = cohens_d_independent(vals, disabled_vals)
                        paired = paired_comparison(vals, disabled_vals)
                        pct_paired, _pct_naive = paired_pct_change(vals, disabled_vals)

                        row["vs_disabled_cohens_d_independent"] = d_indep
                        row["vs_disabled_paired_dz"] = paired["d_z"]
                        row["vs_disabled_paired_p"] = paired["p_value"]
                        row["vs_disabled_paired_mean_diff"] = paired["mean_diff"]
                        row["vs_disabled_paired_mean_pct_change"] = pct_paired
                        row["vs_disabled_degenerate_paired_diff"] = paired["degenerate"]
                        if paired["degenerate"]:
                            row["notes"] = (
                                "degenerate paired diff (sd(diff) < "
                                f"{DEGENERATE_TOL:g}, max abs diff {paired['max_abs_diff']:.3e}); "
                                "d_z set to 0.0 and p set to 1.0 by convention, not computed from "
                                "a zero-variance t-test"
                            )
                    rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deliverable 3: results/effect_sizes.csv
# ---------------------------------------------------------------------------


def _effect_row(df, k, bc, ablation, metric, label, block, is_metric3, by_ablation=None):
    if by_ablation is None:
        by_ablation = _cell_by_ablation(df, k, bc)
    disabled_frame = by_ablation[DISABLED]
    frame = by_ablation[ablation]
    ablation_vals = frame[metric].to_numpy(dtype=float)
    disabled_vals = disabled_frame[metric].reindex(frame.index).to_numpy(dtype=float)

    paired = paired_comparison(ablation_vals, disabled_vals)
    pct_paired, pct_naive = paired_pct_change(ablation_vals, disabled_vals)
    d_indep = cohens_d_independent(ablation_vals, disabled_vals)

    return {
        "block": block,
        "k": k,
        "broker_count": bc,
        "ablation": ablation,
        "metric": metric,
        "metric_label": label,
        "n_seeds": len(ablation_vals),
        "mean_disabled": float(np.mean(disabled_vals)),
        "mean_ablation": float(np.mean(ablation_vals)),
        "paired_mean_diff": paired["mean_diff"],
        "paired_mean_pct_change": pct_paired,
        "naive_pct_change_of_means": pct_naive,
        "paired_dz": paired["d_z"],
        "paired_p": paired["p_value"],
        "cohens_d_independent": d_indep,
        "degenerate_paired_diff": paired["degenerate"],
        "max_abs_paired_diff": paired["max_abs_diff"],
        "pnl_physical_leak": "",
        "metric3_sign_convention": SIGN_CONVENTION_NOTE if is_metric3 else "",
        "note": "",
    }


def build_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # (a) mechanism effect on metric 3, pricing_only and both vs disabled, per (k, broker_count)
    for k in K_VALUES:
        for bc in BROKER_COUNTS:
            by_ablation = _cell_by_ablation(df, k, bc)
            for metric in METRIC3_KEYS:
                label, _ = METRICS[metric]
                for ablation in ENABLED_PRICING:
                    rows.append(
                        _effect_row(df, k, bc, ablation, metric, label, "metric3_mechanism", True, by_ablation)
                    )

    # (b) same structure for metrics 1, 2, 4
    for k in K_VALUES:
        for bc in BROKER_COUNTS:
            by_ablation = _cell_by_ablation(df, k, bc)
            for metric in OTHER_METRIC_KEYS:
                label, _ = METRICS[metric]
                for ablation in ENABLED_PRICING:
                    rows.append(
                        _effect_row(df, k, bc, ablation, metric, label, "other_metrics", False, by_ablation)
                    )

    # (c) sanity check: capacity_pnl_only vs capacity_disabled on the physical metrics
    # (load concentration, both metric-3 sub-metrics, prosumer self-sufficiency), plus
    # avg_cost_per_agent_eur included as a bonus check (D6 says pnl_only should also be
    # inert on consumer cost, since only the pricing channel writes into quotes).
    sanity_metrics = ("load_concentration_hhi",) + METRIC3_KEYS + ("prosumer_self_sufficiency", "avg_cost_per_agent_eur")
    required_scope = set(("load_concentration_hhi",) + METRIC3_KEYS + ("prosumer_self_sufficiency",))

    # per-(k, broker_count) cell rows
    per_cell_max_diff = {metric: [] for metric in sanity_metrics}
    for k in K_VALUES:
        for bc in BROKER_COUNTS:
            by_ablation = _cell_by_ablation(df, k, bc)
            disabled_frame = by_ablation[DISABLED]
            frame = by_ablation["capacity_pnl_only"]
            for metric in sanity_metrics:
                label, is_metric3 = METRICS[metric]
                ablation_vals = frame[metric].to_numpy(dtype=float)
                disabled_vals = disabled_frame[metric].reindex(frame.index).to_numpy(dtype=float)
                paired = paired_comparison(ablation_vals, disabled_vals)
                per_cell_max_diff[metric].append(paired["max_abs_diff"])

                scale = float(np.mean(np.abs(disabled_vals))) or 1.0
                tolerance_abs = PNL_LEAK_RELATIVE_TOL * scale
                leak = metric in required_scope and paired["max_abs_diff"] > tolerance_abs

                rows.append(
                    {
                        "block": "pnl_sanity_check",
                        "k": k,
                        "broker_count": bc,
                        "ablation": "capacity_pnl_only",
                        "metric": metric,
                        "metric_label": label,
                        "n_seeds": len(ablation_vals),
                        "mean_disabled": float(np.mean(disabled_vals)),
                        "mean_ablation": float(np.mean(ablation_vals)),
                        "paired_mean_diff": paired["mean_diff"],
                        "paired_mean_pct_change": np.nan,
                        "naive_pct_change_of_means": np.nan,
                        "paired_dz": paired["d_z"],
                        "paired_p": paired["p_value"],
                        "cohens_d_independent": cohens_d_independent(ablation_vals, disabled_vals),
                        "degenerate_paired_diff": paired["degenerate"],
                        "max_abs_paired_diff": paired["max_abs_diff"],
                        "pnl_physical_leak": leak if metric in required_scope else "n/a (bonus check, not in required scope 2/3/4)",
                        "metric3_sign_convention": SIGN_CONVENTION_NOTE if is_metric3 else "",
                        "note": (
                            "tolerance_abs=%.3e (relative tol %.0e x mean|disabled|=%.6g)"
                            % (tolerance_abs, PNL_LEAK_RELATIVE_TOL, scale)
                        ),
                    }
                )

    # grid-wide (k=ALL, broker_count=ALL) grand summary row per metric
    for metric in sanity_metrics:
        label, is_metric3 = METRICS[metric]
        grand_max = float(np.max(per_cell_max_diff[metric]))
        # scale from the full disabled arm across the whole grid
        disabled_all = df[df["ablation"] == DISABLED][metric].to_numpy(dtype=float)
        scale = float(np.mean(np.abs(disabled_all))) or 1.0
        tolerance_abs = PNL_LEAK_RELATIVE_TOL * scale
        leak = metric in required_scope and grand_max > tolerance_abs
        rows.append(
            {
                "block": "pnl_sanity_check",
                "k": "ALL",
                "broker_count": "ALL",
                "ablation": "capacity_pnl_only",
                "metric": metric,
                "metric_label": label,
                "n_seeds": len(K_VALUES) * len(BROKER_COUNTS) * SEED_N,
                "mean_disabled": float(np.mean(disabled_all)),
                "mean_ablation": float(np.mean(df[df["ablation"] == "capacity_pnl_only"][metric])),
                "paired_mean_diff": np.nan,
                "paired_mean_pct_change": np.nan,
                "naive_pct_change_of_means": np.nan,
                "paired_dz": np.nan,
                "paired_p": np.nan,
                "cohens_d_independent": np.nan,
                "degenerate_paired_diff": "",
                "max_abs_paired_diff": grand_max,
                "pnl_physical_leak": leak if metric in required_scope else "n/a (bonus check, not in required scope 2/3/4)",
                "metric3_sign_convention": SIGN_CONVENTION_NOTE if is_metric3 else "",
                "note": (
                    "GRAND max abs paired diff over the entire sweep (all k, all broker_count, "
                    f"all {SEED_N} seeds); tolerance_abs=%.3e (relative tol %.0e x mean|disabled|=%.6g)"
                    % (tolerance_abs, PNL_LEAK_RELATIVE_TOL, scale)
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_metric3_vs_k(summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(BROKER_COUNTS), figsize=(13, 7.5), sharex=True, layout="constrained")
    metric_rows = [
        ("feeder_peak_to_average_ratio", "Feeder peak-to-average ratio"),
        ("feeder_coefficient_of_variation", "Feeder coefficient of variation"),
    ]

    for row_idx, (metric, label) in enumerate(metric_rows):
        for col_idx, bc in enumerate(BROKER_COUNTS):
            ax = axes[row_idx, col_idx]
            sub = summary_df[(summary_df["broker_count"] == bc) & (summary_df["metric"] == metric)]
            for ablation in ABLATION_ORDER:
                line = sub[sub["ablation"] == ablation].sort_values("k")
                x = line["k"].to_numpy(dtype=float)
                y = line["mean"].to_numpy(dtype=float)
                halfwidth = line["ci95_halfwidth"].to_numpy(dtype=float)
                color = ABLATION_COLOR[ablation]
                ax.plot(x, y, marker="o", label=ABLATION_SHORT[ablation], color=color, **ABLATION_LINE_STYLE[ablation])
                ax.fill_between(x, y - halfwidth, y + halfwidth, color=color, alpha=0.15)
            ax.set_title(f"broker_count = {bc}")
            if row_idx == len(metric_rows) - 1:
                ax.set_xlabel("k (capacity threshold multiplier)")
            if col_idx == 0:
                ax.set_ylabel(label)
            ax.set_xticks(list(K_VALUES))

    axes[0, 0].annotate(
        "Lower = more stable",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        style="italic",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(ABLATION_ORDER),
        frameon=False,
        title=(
            "Each coincident pair is drawn as a thick solid line with its partner dashed on top, "
            "so both members of the pair stay visible."
        ),
        title_fontsize=9,
    )
    fig.suptitle(
        "Metric 3 (feeder stability) vs k, by broker_count, with 95 percent CI bands\n"
        "Lower values = more stable feeder load. Shaded band = 95 percent CI across 30 seeds.\n"
        "Only two distinct curves appear per panel, by construction: pnl_only lies on disabled to within\n"
        "floating-point noise (about 4.4e-16 on 3 of 360 rows), and both lies exactly on pricing_only (0.0\n"
        "on all rows), because the P&L channel writes nothing into prices (clean channel isolation, Section 6.1).",
        fontsize=11,
    )
    fig.savefig(PLOTS_DIR / "fig_metric3_vs_k.png", dpi=150)
    plt.close(fig)


def plot_ablation_bars(summary_df: pd.DataFrame) -> None:
    k_ref, bc_ref = 1.0, 3
    metrics_for_panel = [
        ("avg_cost_per_agent_eur", "Avg cost per agent (EUR/year)"),
        ("load_concentration_hhi", "Load concentration HHI"),
        ("feeder_peak_to_average_ratio", "Feeder peak-to-average ratio (lower = more stable)"),
        ("prosumer_self_sufficiency", "Prosumer self-sufficiency"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for ax, (metric, label) in zip(axes.flat, metrics_for_panel):
        sub = summary_df[
            (summary_df["k"] == k_ref) & (summary_df["broker_count"] == bc_ref) & (summary_df["metric"] == metric)
        ].set_index("ablation").reindex(ABLATION_ORDER)
        x = np.arange(len(ABLATION_ORDER))
        means = sub["mean"].to_numpy(dtype=float)
        errs = sub["ci95_halfwidth"].to_numpy(dtype=float)
        colors = [ABLATION_COLOR[a] for a in ABLATION_ORDER]
        ax.bar(x, means, yerr=errs, capsize=4, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([ABLATION_SHORT[a] for a in ABLATION_ORDER], rotation=20)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("mean (95 percent CI)")

    fig.suptitle(f"Four thesis metrics by ablation at k={k_ref}, broker_count={bc_ref} (error bars = 95 percent CI, n=30 seeds)")
    fig.savefig(PLOTS_DIR / "fig_ablation_bars.png", dpi=150)
    plt.close(fig)


def plot_fire_rate(df: pd.DataFrame) -> None:
    # capacity_fire_rate is a property of the scarcity charge/threshold, essentially equal
    # across the mechanism-enabled ablations (capacity_pnl_only, capacity_pricing_only,
    # capacity_both). We use capacity_both here (see docs/DECISIONS.md D6/D7): it is
    # byte-for-byte identical to capacity_pricing_only in this data (the P&L channel does
    # not touch the physical dynamics that the threshold/fire-rate calc depends on), and
    # differs only very slightly from capacity_pnl_only, which runs the same threshold
    # mechanic WITHOUT the deferral response feeding back into feeder net import.
    #
    # Presentation note: the broker_count = 3 and broker_count = 5 curves nearly
    # coincide in the measured data, closely enough that with one uniform style the
    # bc = 3 curve is drawn over completely and disappears. This is a NEAR-coincidence
    # in the measured means, not an identity the model guarantees (unlike the
    # capacity_pnl_only / capacity_disabled pair in fig_metric3_vs_k.png), so the two
    # are separated by linestyle, marker shape and linewidth only, and the measured
    # gap is stated in the panel and computed below from the same means that are
    # plotted. No plotted value is offset or altered.
    style_by_bc = {
        2: {"linestyle": "-", "linewidth": 1.8, "marker": "o", "markersize": 6.0, "zorder": 2},
        3: {"linestyle": "-", "linewidth": 3.6, "marker": "o", "markersize": 9.0, "zorder": 3},
        5: {"linestyle": (0, (5, 2.5)), "linewidth": 1.6, "marker": "s", "markersize": 3.8, "zorder": 4},
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    sub_both = df[df["ablation"] == "capacity_both"]
    mean_by_bc = {}
    for bc in BROKER_COUNTS:
        line = sub_both[sub_both["broker_count"] == bc].groupby("k")["capacity_fire_rate"].mean().reset_index()
        line = line.sort_values("k")
        mean_by_bc[bc] = line["capacity_fire_rate"].to_numpy(dtype=float)
        ax.plot(line["k"], line["capacity_fire_rate"], label=f"broker_count = {bc}", **style_by_bc[bc])

    # Closeness of the two near-coincident curves, measured from the plotted means.
    # The relative bound is rounded UP to one decimal so the stated "under" figure is
    # never smaller than the measured maximum.
    gap = np.abs(mean_by_bc[3] - mean_by_bc[5])
    rel_gap_pct = 100.0 * np.max(gap / mean_by_bc[5])
    rel_gap_bound = np.ceil(rel_gap_pct * 10.0) / 10.0
    ax.text(
        0.02,
        0.06,
        "broker_count = 3 and 5 nearly coincide in the measured data (a near-coincidence,\n"
        f"not an identity): largest gap {np.max(gap):.1e} in fire rate, under {rel_gap_bound:.1f} percent of the\n"
        "broker_count = 5 value at every k. Drawn thick solid vs thin dashed so both read.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        style="italic",
    )
    ax.set_xlabel("k (capacity threshold multiplier)")
    ax.set_ylabel("capacity_fire_rate (fraction of hours charged)")
    ax.set_xticks(list(K_VALUES))
    ax.set_title(
        "Capacity charge fire rate vs k (capacity_both shown;\n"
        "~equal across capacity_pnl_only / capacity_pricing_only / capacity_both)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig_fire_rate.png", dpi=150)
    plt.close(fig)


def plot_broker_heterogeneity(df: pd.DataFrame) -> None:
    k_ref, bc_ref, ablation_ref = 1.0, 3, "capacity_both"
    sub = df[(df["k"] == k_ref) & (df["broker_count"] == bc_ref) & (df["ablation"] == ablation_ref)]

    shares = sub["broker_load_share_json"].apply(json.loads)
    surcharges = sub["mean_broker_surcharge_json"].apply(json.loads)
    share_df = pd.DataFrame(list(shares))
    surcharge_df = pd.DataFrame(list(surcharges))

    broker_order = sorted(share_df.columns, key=lambda b: share_df[b].mean())
    share_mean = share_df[broker_order].mean()
    share_std = share_df[broker_order].std(ddof=1)
    surcharge_mean = surcharge_df[broker_order].mean()
    surcharge_std = surcharge_df[broker_order].std(ddof=1)

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")

    x = np.arange(len(broker_order))
    ax_bar.bar(x, surcharge_mean.to_numpy(), yerr=surcharge_std.to_numpy(), capsize=4, color=_DEFAULT_COLORS[: len(broker_order)])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(broker_order, rotation=20)
    ax_bar.set_ylabel("mean broker surcharge (EUR/kWh quote adder)")
    ax_bar.set_title(f"Per-broker mean surcharge\n(k={k_ref}, broker_count={bc_ref}, {ablation_ref}, n={len(sub)} seeds)")

    for i, broker in enumerate(broker_order):
        ax_scatter.errorbar(
            share_mean[broker],
            surcharge_mean[broker],
            xerr=share_std[broker],
            yerr=surcharge_std[broker],
            fmt="o",
            markersize=9,
            color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
            label=broker,
        )
    ax_scatter.set_xlabel("mean load share")
    ax_scatter.set_ylabel("mean surcharge (EUR/kWh)")
    ax_scatter.set_title("Load share vs surcharge (monotone: bigger contributor,\nstrictly bigger surcharge)")
    ax_scatter.legend()

    fig.suptitle(
        "Broker heterogeneity under capacity_both: the surcharge is a EUR/kWh quote adder,\n"
        "equal to capacity_passthrough (a signal-strength coefficient, not a tariff rate) x contribution share",
        fontsize=10,
    )
    fig.savefig(PLOTS_DIR / "fig_broker_heterogeneity.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Deliverable (supplementary): results/monopoly_comparison.csv
# ---------------------------------------------------------------------------

COMPARISON_ABLATION = "capacity_pricing_only"


def build_monopoly_comparison(combined_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (broker_count in {1,2,3,5}, k in {0.5,1.0,1.5,2.0}): the
    capacity_pricing_only vs capacity_disabled paired-by-seed comparison for both
    metric-3 sub-measures, plus supporting capacity_both / cost figures.

    Purpose: separate the capacity-charge MECHANISM (should hold even at
    broker_count = 1, a monopoly with no switching) from broker COMPETITION (should
    scale with broker_count). See docs/DECISIONS.md and the Phase 8 monopoly note.
    """
    rows = []
    for bc in ALL_BROKER_COUNTS:
        for k in K_VALUES:
            by_ablation = _cell_by_ablation(combined_df, k, bc)
            disabled_frame = by_ablation[DISABLED]
            pricing_frame = by_ablation[COMPARISON_ABLATION]
            both_frame = by_ablation["capacity_both"]

            row = {
                "broker_count": bc,
                "is_monopoly": bc == MONOPOLY_BROKER_COUNT,
                "k": k,
                "n_seeds": len(pricing_frame),
            }

            for metric in METRIC3_KEYS:
                ablation_vals = pricing_frame[metric].to_numpy(dtype=float)
                disabled_vals = disabled_frame[metric].reindex(pricing_frame.index).to_numpy(dtype=float)
                paired = paired_comparison(ablation_vals, disabled_vals)
                pct_paired, _pct_naive = paired_pct_change(ablation_vals, disabled_vals)
                n_improved = int(np.sum(ablation_vals < disabled_vals))

                row[f"{metric}_paired_mean_pct_change"] = pct_paired
                row[f"{metric}_paired_dz"] = paired["d_z"]
                row[f"{metric}_paired_p"] = paired["p_value"]
                row[f"{metric}_n_improved_of_{len(ablation_vals)}"] = n_improved

            # supporting figures at capacity_both: mean fire rate and mean deferred energy
            row["capacity_both_mean_fire_rate"] = float(both_frame["capacity_fire_rate"].mean())
            row["capacity_both_mean_total_deferred_kwh"] = float(both_frame["total_deferred_kwh"].mean())

            # consumer cost: pricing_only vs disabled, paired percent change
            cost_ablation_vals = pricing_frame["avg_cost_per_agent_eur"].to_numpy(dtype=float)
            cost_disabled_vals = disabled_frame["avg_cost_per_agent_eur"].reindex(pricing_frame.index).to_numpy(dtype=float)
            cost_pct_paired, _cost_pct_naive = paired_pct_change(cost_ablation_vals, cost_disabled_vals)
            row["avg_cost_per_agent_eur_paired_mean_pct_change"] = cost_pct_paired

            row["metric3_sign_convention"] = SIGN_CONVENTION_NOTE
            row["note"] = (
                f"{COMPARISON_ABLATION} vs {DISABLED}, paired by seed; "
                "cost pct change is also pricing_only vs disabled (positive = costs more)"
            )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_monopoly_comparison(comparison_df: pd.DataFrame) -> None:
    fig, (ax_cov, ax_peak) = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    x = list(ALL_BROKER_COUNTS)

    panels = [
        (ax_cov, "feeder_coefficient_of_variation_paired_mean_pct_change", "Feeder coefficient of variation (CoV)"),
        (ax_peak, "feeder_peak_to_average_ratio_paired_mean_pct_change", "Feeder peak-to-average ratio"),
    ]
    for ax, col, label in panels:
        for k in K_VALUES:
            line = comparison_df[comparison_df["k"] == k].sort_values("broker_count")
            ax.plot(line["broker_count"], line[col], marker="o", label=f"k = {k:g}")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(MONOPOLY_BROKER_COUNT, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xlabel("broker_count (1 = monopoly, no switching)")
        ax.set_ylabel(f"{label}\npaired mean pct change, pricing_only vs disabled (%)")
        ax.set_title(label)
        ax.annotate(
            "monopoly",
            xy=(MONOPOLY_BROKER_COUNT, ax.get_ylim()[1]),
            xytext=(MONOPOLY_BROKER_COUNT + 0.08, ax.get_ylim()[1]),
            fontsize=8,
            ha="left",
            va="top",
            style="italic",
            color="gray",
        )

    handles, labels = ax_cov.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(K_VALUES), frameon=False)
    fig.suptitle(
        "Monopoly (broker_count=1) vs competition: capacity_pricing_only vs capacity_disabled,\n"
        "paired mean percent change by broker_count and k. Negative = improved stability.\n"
        "Left: CoV improvement is nearly flat across broker_count (mechanism-driven).\n"
        "Right: peak-to-average improvement grows with broker_count, absent at broker_count=1 (competition-dependent).",
        fontsize=10,
    )
    fig.savefig(PLOTS_DIR / "fig_monopoly_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostics (stdout only; not written to any CSV)
# ---------------------------------------------------------------------------


def print_diagnostics(df: pd.DataFrame, effects_df: pd.DataFrame) -> None:
    print(f"rows analyzed: {len(df)}")

    df = df.copy()
    df["active_brokers"] = df["broker_load_share_json"].apply(active_broker_count)
    active_summary = df.groupby("broker_count")["active_brokers"].agg(["mean", "median", "min", "max"])
    print("\nEffective (active) broker count by nominal broker_count (see D8 Phase 5 observation):")
    print(active_summary.to_string())

    sanity = effects_df[(effects_df["block"] == "pnl_sanity_check") & (effects_df["k"] == "ALL")]
    print("\ncapacity_pnl_only vs capacity_disabled, grid-wide max abs paired diff (sanity check):")
    print(sanity[["metric", "max_abs_paired_diff", "pnl_physical_leak", "note"]].to_string(index=False))

    any_leak = sanity["pnl_physical_leak"].apply(lambda v: v is True).any()
    print(f"\npnl_physical_leak flagged anywhere: {any_leak}")


# ---------------------------------------------------------------------------
# Deliverable: results/summary_stats_corrected.csv (multiple-comparison
# correction and bootstrap CIs; additive to the existing summary/effect files)
# ---------------------------------------------------------------------------


def _family_row(scope: str, k: float, bc: int, ablation: str, metric: str, by_ablation: dict, correction_family: str) -> dict:
    """One row's worth of paired-comparison statistics, computed the same way
    as build_summary_stats()/build_effect_sizes() (same paired_comparison()
    and paired_pct_change() helpers), plus a t-based CI on the PAIRED
    DIFFERENCE itself (not the per-arm descriptive CI already reported in
    results/summary_stats.csv)."""
    label, is_metric3 = METRICS[metric]
    disabled_frame = by_ablation[DISABLED]
    frame = by_ablation[ablation]
    ablation_vals = frame[metric].to_numpy(dtype=float)
    disabled_vals = disabled_frame[metric].reindex(frame.index).to_numpy(dtype=float)
    diff = ablation_vals - disabled_vals

    paired = paired_comparison(ablation_vals, disabled_vals)
    pct_paired, _pct_naive = paired_pct_change(ablation_vals, disabled_vals)
    halfwidth = t_ci_halfwidth(paired["sd_diff"], len(diff))

    return {
        "scope": scope,
        "k": k,
        "broker_count": bc,
        "ablation": ablation,
        "metric": metric,
        "metric_label": label,
        "n_seeds": len(diff),
        "paired_mean_diff": paired["mean_diff"],
        "paired_mean_pct_change": pct_paired,
        "paired_dz": paired["d_z"],
        "p_uncorrected": paired["p_value"],
        "degenerate": paired["degenerate"],
        "correction_family": correction_family,
        "ci95_t_lo": paired["mean_diff"] - halfwidth,
        "ci95_t_hi": paired["mean_diff"] + halfwidth,
        "metric3_sign_convention": SIGN_CONVENTION_NOTE if is_metric3 else "",
        "_diff": diff,
    }


def build_corrected_summary(df: pd.DataFrame, mono_df: pd.DataFrame | None) -> pd.DataFrame:
    """Build the full correction-family table: one row per paired comparison,
    with Holm-Bonferroni and Benjamini-Hochberg corrected p-values side by
    side with the uncorrected p, plus a t-based and a bootstrap CI on the
    paired mean difference. See the FAMILY_SIZE_* constants' docstring above
    for the full reasoning behind which rows are in which correction family.

    Three blocks of rows, distinguished by the `correction_family` column:
      - "primary_main_sweep_real" (120 rows): capacity_pricing_only and
        capacity_both vs capacity_disabled, all 5 metrics, all (k,
        broker_count) cells of the main D8 sweep. This is the PRIMARY
        correction family (FAMILY_SIZE_PRIMARY).
      - "excluded_degenerate_main_sweep" (60 rows): capacity_pnl_only vs
        capacity_disabled, same metrics/cells. Deterministic identities, not
        hypothesis tests (see module-level constants docstring); p_holm and
        p_bh are set to 1.0 by convention, not computed. Adding these back to
        the primary 120 gives FAMILY_SIZE_ALTERNATIVE = 180.
      - "secondary_monopoly_supplement_real" (8 rows, only if
        results/sweep_monopoly.parquet is available): capacity_pricing_only
        vs capacity_disabled at broker_count=1, both metric-3 sub-metrics,
        all 4 k values. Corrected as its OWN separate family
        (FAMILY_SIZE_MONOPOLY_SECONDARY), not pooled with the primary 120.

    Bootstrap CIs are computed for every row in all three blocks (188 rows x
    10000 resamples is cheap), not only the metric-3/cost headline subset.
    """
    rows: list[dict] = []

    for k in K_VALUES:
        for bc in BROKER_COUNTS:
            by_ablation = _cell_by_ablation(df, k, bc)
            for ablation in ENABLED_PRICING:
                for metric in METRICS:
                    rows.append(_family_row("main_sweep", k, bc, ablation, metric, by_ablation, "primary_main_sweep_real"))
    n_primary = sum(1 for r in rows if r["correction_family"] == "primary_main_sweep_real")
    assert n_primary == FAMILY_SIZE_PRIMARY, f"expected {FAMILY_SIZE_PRIMARY} primary real tests, found {n_primary}"

    degenerate_start = len(rows)
    for k in K_VALUES:
        for bc in BROKER_COUNTS:
            by_ablation = _cell_by_ablation(df, k, bc)
            for metric in METRICS:
                rows.append(
                    _family_row("main_sweep", k, bc, "capacity_pnl_only", metric, by_ablation, "excluded_degenerate_main_sweep")
                )
    n_degenerate = len(rows) - degenerate_start
    expected_degenerate = FAMILY_SIZE_ALTERNATIVE - FAMILY_SIZE_PRIMARY
    assert n_degenerate == expected_degenerate, f"expected {expected_degenerate} degenerate rows, found {n_degenerate}"
    for r in rows[degenerate_start:]:
        assert r["degenerate"], "capacity_pnl_only vs capacity_disabled must be degenerate by construction (D6)"

    if mono_df is not None and len(mono_df) > 0:
        mono_start = len(rows)
        for k in K_VALUES:
            by_ablation = _cell_by_ablation(mono_df, k, MONOPOLY_BROKER_COUNT)
            for metric in METRIC3_KEYS:
                rows.append(
                    _family_row(
                        "monopoly_supplement", k, MONOPOLY_BROKER_COUNT, "capacity_pricing_only",
                        metric, by_ablation, "secondary_monopoly_supplement_real",
                    )
                )
        n_mono = len(rows) - mono_start
        assert n_mono == FAMILY_SIZE_MONOPOLY_SECONDARY, f"expected {FAMILY_SIZE_MONOPOLY_SECONDARY} monopoly tests, found {n_mono}"
    else:
        print(
            "\nNOTE: monopoly data not available; results/summary_stats_corrected.csv will not include "
            f"the {FAMILY_SIZE_MONOPOLY_SECONDARY}-row monopoly_supplement secondary family."
        )

    # --- Holm and BH correction, one family at a time ---
    family_indices: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        family_indices.setdefault(r["correction_family"], []).append(i)

    for family, idxs in family_indices.items():
        if family == "excluded_degenerate_main_sweep":
            for i in idxs:
                rows[i]["p_holm"] = 1.0
                rows[i]["p_bh"] = 1.0
                rows[i]["correction_family_size"] = 0  # excluded by design, not corrected
            continue
        pvals = [rows[i]["p_uncorrected"] for i in idxs]
        holm_adj = holm_bonferroni(pvals)
        bh_adj = benjamini_hochberg(pvals)
        for j, i in enumerate(idxs):
            rows[i]["p_holm"] = float(holm_adj[j])
            rows[i]["p_bh"] = float(bh_adj[j])
            rows[i]["correction_family_size"] = len(idxs)

    # --- bootstrap CI + t-vs-bootstrap disagreement flag, every row ---
    # One Generator is threaded through every row rather than reseeded per row,
    # so the draws for row n depend on how many rows preceded it. That is
    # reproducible, but only while the row-building order above stays fixed:
    # reordering those loops would silently change every bootstrap interval in
    # the output without changing a single input. If the row order ever needs
    # to change, seed per row from a stable key (the grouping tuple) instead of
    # sharing this one, so an interval depends on which comparison it is and
    # not on where it happens to sit in the list.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for r in rows:
        boot_lo, boot_hi = bootstrap_paired_diff_ci(r["_diff"], rng)
        flag, detail, width_ratio = _ci_disagreement(r["ci95_t_lo"], r["ci95_t_hi"], boot_lo, boot_hi)
        r["ci95_boot_lo"] = boot_lo
        r["ci95_boot_hi"] = boot_hi
        r["ci_width_ratio"] = width_ratio
        r["ci_disagreement_flag"] = flag
        r["ci_disagreement_detail"] = detail
        r["survives_holm_alpha05"] = bool((not r["degenerate"]) and r["p_holm"] <= ALPHA)
        r["survives_bh_alpha05"] = bool((not r["degenerate"]) and r["p_bh"] <= ALPHA)
        r["family_size_primary"] = FAMILY_SIZE_PRIMARY
        r["family_size_alternative"] = FAMILY_SIZE_ALTERNATIVE
        r["family_size_monopoly_secondary"] = FAMILY_SIZE_MONOPOLY_SECONDARY
        r["bootstrap_seed"] = BOOTSTRAP_SEED
        r["n_bootstrap"] = N_BOOTSTRAP
        r["alpha"] = ALPHA
        if r["degenerate"]:
            r["notes"] = (
                "deterministic identity: capacity_pnl_only cannot alter this quantity by construction "
                "(D6, the P&L channel writes nothing into any price or physical quantity); excluded "
                "from the primary correction family; p_holm/p_bh set to 1.0 by convention, not "
                "computed from a zero-variance test"
            )
        elif r["correction_family"] == "secondary_monopoly_supplement_real":
            r["notes"] = (
                "supplementary broker_count=1 monopoly arm (Phase 8); corrected within its own "
                f"{FAMILY_SIZE_MONOPOLY_SECONDARY}-test secondary family, not pooled with the "
                f"{FAMILY_SIZE_PRIMARY}-test primary main-sweep family"
            )
        else:
            r["notes"] = ""
        del r["_diff"]

    columns = [
        "scope", "k", "broker_count", "ablation", "metric", "metric_label", "n_seeds",
        "paired_mean_diff", "paired_mean_pct_change", "paired_dz",
        "p_uncorrected", "p_holm", "p_bh", "survives_holm_alpha05", "survives_bh_alpha05",
        "ci95_t_lo", "ci95_t_hi", "ci95_boot_lo", "ci95_boot_hi",
        "ci_disagreement_flag", "ci_disagreement_detail", "ci_width_ratio",
        "degenerate", "correction_family", "correction_family_size",
        "family_size_primary", "family_size_alternative", "family_size_monopoly_secondary",
        "bootstrap_seed", "n_bootstrap", "alpha",
        "metric3_sign_convention", "notes",
    ]
    return pd.DataFrame(rows)[columns]


def print_correction_diagnostics(corrected_df: pd.DataFrame) -> None:
    primary = corrected_df[corrected_df["correction_family"] == "primary_main_sweep_real"]
    print(
        f"\nPrimary correction family size: {FAMILY_SIZE_PRIMARY} real tests "
        f"(alternative, including the degenerate pnl_only identities: {FAMILY_SIZE_ALTERNATIVE})"
    )
    print(
        f"Primary family: {int(primary['survives_holm_alpha05'].sum())} of {len(primary)} survive Holm at "
        f"alpha={ALPHA}; {int(primary['survives_bh_alpha05'].sum())} of {len(primary)} survive BH at alpha={ALPHA}"
    )
    mono = corrected_df[corrected_df["correction_family"] == "secondary_monopoly_supplement_real"]
    if len(mono):
        print(
            f"Monopoly secondary family ({FAMILY_SIZE_MONOPOLY_SECONDARY} tests): "
            f"{int(mono['survives_holm_alpha05'].sum())} survive Holm; {int(mono['survives_bh_alpha05'].sum())} survive BH"
        )
    n_disagree = int(corrected_df["ci_disagreement_flag"].sum())
    print(f"t-vs-bootstrap CI disagreements flagged: {n_disagree} of {len(corrected_df)} rows")
    if n_disagree:
        print(
            corrected_df.loc[
                corrected_df["ci_disagreement_flag"],
                ["scope", "k", "broker_count", "ablation", "metric", "ci_disagreement_detail"],
            ].to_string(index=False)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()

    summary_df = build_summary_stats(df)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"wrote {SUMMARY_PATH} ({len(summary_df)} rows)")

    effects_df = build_effect_sizes(df)
    effects_df.to_csv(EFFECTS_PATH, index=False)
    print(f"wrote {EFFECTS_PATH} ({len(effects_df)} rows)")

    plot_metric3_vs_k(summary_df)
    plot_ablation_bars(summary_df)
    plot_fire_rate(df)
    plot_broker_heterogeneity(df)
    print(f"wrote 4 PNGs to {PLOTS_DIR}")

    print_diagnostics(df, effects_df)

    combined_df = load_monopoly_combined(df)
    if combined_df is not None:
        comparison_df = build_monopoly_comparison(combined_df)
        comparison_df.to_csv(MONOPOLY_COMPARISON_PATH, index=False)
        print(f"wrote {MONOPOLY_COMPARISON_PATH} ({len(comparison_df)} rows)")

        plot_monopoly_comparison(comparison_df)
        print(f"wrote {PLOTS_DIR / 'fig_monopoly_comparison.png'}")

    mono_only_df = None
    if combined_df is not None:
        mono_only_df = combined_df[combined_df["broker_count"] == MONOPOLY_BROKER_COUNT].reset_index(drop=True)

    corrected_df = build_corrected_summary(df, mono_only_df)
    corrected_df.to_csv(CORRECTED_SUMMARY_PATH, index=False)
    print(f"wrote {CORRECTED_SUMMARY_PATH} ({len(corrected_df)} rows)")
    print_correction_diagnostics(corrected_df)


if __name__ == "__main__":
    main()
