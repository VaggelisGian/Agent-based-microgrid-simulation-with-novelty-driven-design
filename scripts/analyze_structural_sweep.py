"""D9 structural-coefficient sensitivity analysis (docs/DECISIONS.md D9).

Reads results/sweep_structural.parquet (2880 rows: deferrable_fraction in
{0.05, 0.10, 0.20, 0.30} x response_reference_eur_per_kwh in {0.0125, 0.025,
0.10, 0.20} x payback_cap_fraction in {0.25, 0.5, 1.0} x ablation in
{capacity_disabled, capacity_both} x 30 seeds, all at the headline cell
k=1.0, broker_count=3) and answers the question the thesis's metric-3 result
has to survive: does the D7 demand-deferral response's headline effect on
feeder stability depend on the three structural coefficients D7/D8 held
fixed and never swept?

This script writes:

  - results/structural_sensitivity.csv   one row per full-factorial coefficient
                                  cell (48 cells x 2 metric-3 sub-measures = 96
                                  rows), one row per marginal per-coefficient-level
                                  comparison ((4+4+3) levels x 2 sub-measures = 22
                                  rows), and one row per per-coefficient verdict
                                  (3 coefficients x 2 sub-measures = 6 rows).
                                  Self-describing: every row carries its own
                                  scope, metric, and correction-family columns.
  - results/plots/fig_structural_sensitivity.png   top row: marginal paired
                                  percent-change vs each coefficient's own
                                  levels, with approximate 95 percent CI, for
                                  both metric-3 sub-measures. Bottom row:
                                  deferrable_fraction x payback_cap_fraction
                                  interaction heatmaps (averaged over
                                  response_reference), marking D9's own named
                                  plausible failure corner and the shipped
                                  calibration point, plus a text panel with
                                  the per-coefficient verdict.

Design notes, reusing scripts/analyze_sweep.py's idioms rather than
reimplementing them (paired_comparison, paired_pct_change,
cohens_d_independent, t_ci_halfwidth, holm_bonferroni, benjamini_hochberg,
bootstrap_paired_diff_ci, the CI-disagreement flag, and the metric-3 sign
convention are all imported from there, not re-derived):

  - Sign convention for both metric-3 sub-measures (unchanged from D6/D7/D8):
    a NEGATIVE paired difference (capacity_both minus capacity_disabled)
    means IMPROVED stability; POSITIVE means worsened.

  - The disabled arm is provably invariant to all three coefficients (D9;
    verified empirically at 8 corners x 5 seeds, max divergence 0.0, see
    results/sweep_structural_invariance_check.json). scripts/run_structural_sweep.py
    exploits this: it runs the disabled arm once per seed and MATERIALIZES
    the remaining 1400 disabled cells by copying that seed's canonical row.
    The parquet still carries 2880 rows so every downstream comparison here
    is a plain paired-by-seed comparison; this script never treats the 48
    disabled copies of one seed as 48 independent baselines and never lets
    that duplication inflate any n or degree of freedom. Every marginal
    (averaged-over-two-axes) comparison below is built by averaging the
    ABLATION arm per seed first and pairing against that seed's single
    disabled value, so n stays at 30 seeds throughout, never 30 x (number
    of cells averaged over).

  - Multiple-comparison correction family (see FAMILY_SIZE_CELL_PRIMARY and
    FAMILY_SIZE_MARGINAL_SECONDARY below for the full reasoning): the PRIMARY
    family is the 96 real, computed paired tests at this sweep's finest
    resolution -- the 48 full-factorial coefficient cells (4 deferrable_fraction
    x 4 response_reference x 3 payback_cap_fraction) times the 2 metric-3
    sub-measures, each capacity_both vs capacity_disabled, paired by seed.
    This mirrors scripts/analyze_sweep.py's own precedent of choosing the
    finest-grained, non-overlapping real tests as its primary correction
    family. The 22 marginal (per-coefficient-level) comparisons are corrected
    as their own SEPARATE secondary family, because each one averages over an
    overlapping subset of the same 48 cells and so is not an independent test
    of the primary family -- pooling them would double-count the same
    underlying seed draws under a different aggregation.

  - The D9 interaction check (docs/DECISIONS.md D9's independent rationale):
    "a large deferrable_fraction combined with a tight payback_cap_fraction is
    precisely the corner where the peak-to-average result could invert" (this
    happened historically in Phase 3c, before the D7 revision). This script
    explicitly scans the full 48-cell box for a peak-to-average sign flip and
    reports one loudly wherever it occurs, not only at the named corner.

Run from the repository root:

    python scripts/analyze_structural_sweep.py

For testing against a fixture instead of the real (not-yet-existing)
results/sweep_structural.parquet, and to avoid writing into results/ before
the sweep this script analyzes has finished:

    python scripts/analyze_structural_sweep.py --parquet <fixture.parquet> --output-dir <scratch dir>

Requires: Python 3.12+, pandas, numpy, scipy (via scripts/analyze_sweep.py),
matplotlib (no seaborn). ASCII only throughout (source and all written
CSV/PNG artifacts). No randomness beyond the seeded bootstrap resampling
already used by scripts/analyze_sweep.py; no network access; no tuning of
any analysis choice against a desired result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import analyze_sweep as base  # noqa: E402  (path set above)
# Reused from scripts/analyze_sweep.py rather than reimplemented: paired_comparison,
# paired_pct_change, cohens_d_independent, t_ci_halfwidth, holm_bonferroni,
# benjamini_hochberg, bootstrap_paired_diff_ci, _ci_disagreement, METRICS,
# SIGN_CONVENTION_NOTE, BOOTSTRAP_SEED, N_BOOTSTRAP, ALPHA, CI_CONF.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET_PATH = REPO_ROOT / "results" / "sweep_structural.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results"
MAIN_SWEEP_RAW_PATH = REPO_ROOT / "results" / "sweep_raw.parquet"

# ---------------------------------------------------------------------------
# D9 grid constants (must match docs/DECISIONS.md D9 / scripts/run_structural_sweep.py)
# ---------------------------------------------------------------------------

DEFERRABLE_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
RESPONSE_REFERENCES_EUR_PER_KWH = (0.0125, 0.025, 0.10, 0.20)
PAYBACK_CAP_FRACTIONS = (0.25, 0.5, 1.0)
ABLATIONS = ("capacity_disabled", "capacity_both")
SEED_BASE = 20260704
SEEDS = tuple(SEED_BASE + offset for offset in range(30))  # must match
# scripts/run_sensitivity_sweep.py's SEEDS, which scripts/run_structural_sweep.py
# imports directly (docs/DECISIONS.md D9: "30 seeds, same seeds as the main sweep").

SHIPPED_K = 1.0
SHIPPED_BROKER_COUNT = 3

EXPECTED_ROWS = (
    len(DEFERRABLE_FRACTIONS) * len(RESPONSE_REFERENCES_EUR_PER_KWH) * len(PAYBACK_CAP_FRACTIONS) * len(ABLATIONS) * len(SEEDS)
)
assert EXPECTED_ROWS == 2880, f"D9 grid constants do not multiply out to 2880: got {EXPECTED_ROWS}"

# The shipped/calibration cell (config/default.yaml). deferrable_fraction and
# payback_cap_fraction ARE swept levels of this grid; response_reference_eur_per_kwh
# is NOT (docs/DECISIONS.md D9: the four swept response_reference levels bracket
# 0.05 symmetrically in log space, but 0.05 itself is the geometric centre, not
# one of the four grid points). The shipped cell lives in the MAIN sweep
# (results/sweep_raw.parquet at k=1.0, broker_count=3); shipped_cell_anchor()
# below recovers its effect from that file rather than re-simulating it here.
SHIPPED_DFRAC = 0.20
SHIPPED_RREF = 0.05
SHIPPED_PCAP = 0.5

# D9's own named plausible failure mode (docs/DECISIONS.md D9's independent
# rationale): large deferrable_fraction combined with a tight payback_cap_fraction
# is where a payback rebound could invert the peak-to-average result (this is what
# happened historically in Phase 3c). Deterministic from the grid constants above,
# not chosen to match any particular result.
FLAGGED_CORNER = (max(DEFERRABLE_FRACTIONS), min(PAYBACK_CAP_FRACTIONS))

VALID_PROVENANCE = {"direct", "verification", "materialized_from_canonical_seed_run"}

METRIC3_KEYS = ("feeder_coefficient_of_variation", "feeder_peak_to_average_ratio")

COEFFICIENT_LEVELS = {
    "deferrable_fraction": DEFERRABLE_FRACTIONS,
    "response_reference_eur_per_kwh": RESPONSE_REFERENCES_EUR_PER_KWH,
    "payback_cap_fraction": PAYBACK_CAP_FRACTIONS,
}
# Maps a coefficient name to the SNAPPED raw-data column built by _snap_axes();
# used only when filtering the raw per-seed frame (build_marginal_table,
# _dfrac_pcap_marginal). The already-aggregated cell_df/marginal_df tables use
# the coefficient name itself as the column name (see _stats_row), so they do
# not need this mapping.
RAW_AXIS_COLUMN = {
    "deferrable_fraction": "_dfrac",
    "response_reference_eur_per_kwh": "_rref",
    "payback_cap_fraction": "_pcap",
}

FAMILY_SIZE_CELL_PRIMARY = (
    len(DEFERRABLE_FRACTIONS) * len(RESPONSE_REFERENCES_EUR_PER_KWH) * len(PAYBACK_CAP_FRACTIONS) * len(METRIC3_KEYS)
)
assert FAMILY_SIZE_CELL_PRIMARY == 96
FAMILY_SIZE_MARGINAL_SECONDARY = sum(len(levels) for levels in COEFFICIENT_LEVELS.values()) * len(METRIC3_KEYS)
assert FAMILY_SIZE_MARGINAL_SECONDARY == 22

# Mirrors scripts/run_structural_sweep.py's ROW_COLUMNS (presence only; order
# does not matter for validation).
RUNNER_ROW_COLUMNS = (
    "deferrable_fraction", "response_reference_eur_per_kwh", "payback_cap_fraction", "ablation", "seed",
    "k", "broker_count", "avg_cost_per_agent_eur", "avg_cost_per_kwh_eur", "load_concentration_hhi",
    "broker_load_share_json", "feeder_coefficient_of_variation", "feeder_peak_to_average_ratio",
    "feeder_mean_hourly_ramp_kwh", "prosumer_self_sufficiency", "total_capacity_charge_eur",
    "capacity_fire_rate", "mean_broker_surcharge_json", "broker_load_share_gini", "total_deferred_kwh",
    "switching_rate", "n_brokers", "run_wallclock_sec", "peak_rss_mb", "row_provenance",
)

# Mirrors scripts/run_structural_sweep.py's NUMERIC_COMPARISON_COLUMNS: every
# metric/audit column that is not a config echo, provenance tag, or timing/
# resource column. Used by disabled_arm_within_seed_spread() below to
# double-check, on the ASSEMBLED parquet's full 1440 disabled rows (not just
# the 40-run verification sample the runner itself checked), that a given
# seed's disabled-arm outcome really is identical across all 48 coefficient
# cells.
DISABLED_ARM_NUMERIC_COLUMNS = (
    "avg_cost_per_agent_eur", "avg_cost_per_kwh_eur", "load_concentration_hhi",
    "feeder_coefficient_of_variation", "feeder_peak_to_average_ratio", "feeder_mean_hourly_ramp_kwh",
    "prosumer_self_sufficiency", "total_capacity_charge_eur", "capacity_fire_rate",
    "broker_load_share_gini", "total_deferred_kwh", "switching_rate", "n_brokers",
)
SPREAD_WARN_TOL = 1e-6  # generous relative to the runner's own 1e-9 abs + 1e-9 rel invariance tolerance

CSV_COLUMNS = [
    "scope",
    "deferrable_fraction", "response_reference_eur_per_kwh", "payback_cap_fraction",
    "coefficient", "coefficient_level",
    "metric", "metric_label", "n_seeds",
    "mean_disabled", "mean_both",
    "paired_mean_diff", "paired_mean_pct_change", "naive_pct_change_of_means",
    "paired_dz", "cohens_d_independent",
    "p_uncorrected", "p_holm", "p_bh", "survives_holm_alpha05", "survives_bh_alpha05",
    "ci95_t_lo", "ci95_t_hi", "ci95_boot_lo", "ci95_boot_hi",
    "ci_disagreement_flag", "ci_disagreement_detail", "ci_width_ratio",
    "degenerate",
    "correction_family", "correction_family_size",
    "is_flagged_d9_corner", "worsened_vs_disabled", "is_shipped_axis_level",
    "min_pct_change_across_levels", "max_pct_change_across_levels",
    "sign_consistent_across_marginal_levels", "significant_at_every_level_holm",
    "cell_level_sign_consistent_within_every_level", "levels_with_within_level_cell_sign_disagreement",
    "verdict",
    "bootstrap_seed", "n_bootstrap", "alpha",
    "metric3_sign_convention", "notes",
]

COV_COLOR = "#1f77b4"
P2A_COLOR = "#d62728"
METRIC_PLOT_STYLE = {
    "feeder_coefficient_of_variation": (COV_COLOR, "Coefficient of variation"),
    "feeder_peak_to_average_ratio": (P2A_COLOR, "Peak-to-average ratio"),
}


# ---------------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------------


def _missing_parquet_message(parquet_path: Path) -> str:
    return (
        f"ERROR: {parquet_path} not found.\n"
        "The D9 structural-coefficient sensitivity sweep "
        "(scripts/run_structural_sweep.py --mode full) has not finished yet. "
        "This analyzer requires the completed 2880-row results/sweep_structural.parquet. "
        "Wait for the sweep to finish, then re-run this script (no arguments needed for "
        "the real run), or pass --parquet to point at a completed parquet elsewhere."
    )


def validate_structural_sweep(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable validation failures (empty = valid).

    Checks: row count, exact axis values, k and broker_count fixed at the
    headline cell everywhere, exactly the 30 project seeds, no duplicate
    (coefficients, ablation, seed) tuples, no missing/extra cells, and that
    row_provenance is well-formed (capacity_both must be entirely "direct").
    """
    problems: list[str] = []

    missing_columns = set(RUNNER_ROW_COLUMNS) - set(df.columns)
    if missing_columns:
        problems.append(f"missing expected columns: {sorted(missing_columns)}")
        return problems  # cannot safely run the remaining checks without these columns

    if len(df) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} rows, found {len(df)}")

    work = df.copy()
    work["_dfrac"] = work["deferrable_fraction"].astype(float).round(6)
    work["_rref"] = work["response_reference_eur_per_kwh"].astype(float).round(6)
    work["_pcap"] = work["payback_cap_fraction"].astype(float).round(6)

    observed_dfrac = set(work["_dfrac"].unique())
    expected_dfrac = {round(v, 6) for v in DEFERRABLE_FRACTIONS}
    if observed_dfrac != expected_dfrac:
        problems.append(f"deferrable_fraction values {sorted(observed_dfrac)} != expected {sorted(expected_dfrac)}")

    observed_rref = set(work["_rref"].unique())
    expected_rref = {round(v, 6) for v in RESPONSE_REFERENCES_EUR_PER_KWH}
    if observed_rref != expected_rref:
        problems.append(f"response_reference_eur_per_kwh values {sorted(observed_rref)} != expected {sorted(expected_rref)}")

    observed_pcap = set(work["_pcap"].unique())
    expected_pcap = {round(v, 6) for v in PAYBACK_CAP_FRACTIONS}
    if observed_pcap != expected_pcap:
        problems.append(f"payback_cap_fraction values {sorted(observed_pcap)} != expected {sorted(expected_pcap)}")

    observed_ablations = set(work["ablation"].unique())
    if observed_ablations != set(ABLATIONS):
        problems.append(f"ablation values {sorted(observed_ablations)} != expected {sorted(ABLATIONS)}")

    bad_k = work.loc[work["k"].astype(float) != SHIPPED_K, "k"].unique()
    if len(bad_k):
        problems.append(f"k must be {SHIPPED_K} on every row; found other values {sorted(bad_k)}")

    bad_bc = work.loc[work["broker_count"].astype(int) != SHIPPED_BROKER_COUNT, "broker_count"].unique()
    if len(bad_bc):
        problems.append(f"broker_count must be {SHIPPED_BROKER_COUNT} on every row; found other values {sorted(bad_bc)}")

    observed_seeds = set(work["seed"].astype(int).unique())
    if len(observed_seeds) != 30:
        problems.append(f"expected 30 distinct seeds, found {len(observed_seeds)}")
    if observed_seeds != set(SEEDS):
        missing_seeds = sorted(set(SEEDS) - observed_seeds)
        extra_seeds = sorted(observed_seeds - set(SEEDS))
        problems.append(f"seed set does not match the project's 30 seeds: missing {missing_seeds}, unexpected {extra_seeds}")

    key_cols = ["_dfrac", "_rref", "_pcap", "ablation", "seed"]
    dup_counts = work.groupby(key_cols).size()
    duplicates = dup_counts[dup_counts > 1]
    if len(duplicates):
        problems.append(
            f"{len(duplicates)} duplicate (coefficients, ablation, seed) tuples found, "
            f"e.g. {duplicates.index[:5].tolist()}"
        )

    expected_keys = {
        (d, r, p, a, s)
        for d in (round(v, 6) for v in DEFERRABLE_FRACTIONS)
        for r in (round(v, 6) for v in RESPONSE_REFERENCES_EUR_PER_KWH)
        for p in (round(v, 6) for v in PAYBACK_CAP_FRACTIONS)
        for a in ABLATIONS
        for s in SEEDS
    }
    actual_keys = set(work[key_cols].itertuples(index=False, name=None))
    missing_cells = expected_keys - actual_keys
    if missing_cells:
        problems.append(f"{len(missing_cells)} missing (coefficients, ablation, seed) cells, e.g. {sorted(missing_cells)[:5]}")
    extra_cells = actual_keys - expected_keys
    if extra_cells:
        problems.append(f"{len(extra_cells)} unexpected (coefficients, ablation, seed) cells outside the D9 grid, e.g. {sorted(extra_cells)[:5]}")

    bad_provenance = set(work["row_provenance"].unique()) - VALID_PROVENANCE
    if bad_provenance:
        problems.append(f"invalid row_provenance values: {sorted(bad_provenance)}")

    both_provenance = set(work.loc[work["ablation"] == "capacity_both", "row_provenance"].unique())
    if both_provenance - {"direct"}:
        problems.append(f"capacity_both rows must all have row_provenance='direct'; found {sorted(both_provenance)}")

    return problems


def _snap_axes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_dfrac"] = df["deferrable_fraction"].astype(float).round(6)
    df["_rref"] = df["response_reference_eur_per_kwh"].astype(float).round(6)
    df["_pcap"] = df["payback_cap_fraction"].astype(float).round(6)
    return df


def provenance_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["ablation", "row_provenance"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["ablation", "row_provenance"])
        .reset_index(drop=True)
    )


def disabled_arm_within_seed_spread(df: pd.DataFrame) -> pd.DataFrame:
    """For every seed, the spread (max - min) of each metric/audit column across
    all 48 coefficient cells of the disabled arm. Should be ~0 everywhere: D9's
    disabled-arm reuse rests on the disabled baseline being the SAME simulated
    outcome for a given seed regardless of which coefficient cell it is attached
    to. results/sweep_structural_invariance_check.json only covers a 40-run
    verification sample; this checks all 1440 assembled disabled rows.
    """
    disabled = df[df["ablation"] == "capacity_disabled"]
    cols = [c for c in DISABLED_ARM_NUMERIC_COLUMNS if c in disabled.columns]
    return disabled.groupby("seed")[cols].agg(lambda s: float(s.max() - s.min()))


# ---------------------------------------------------------------------------
# Cell-level and marginal statistics (reuses scripts/analyze_sweep.py helpers)
# ---------------------------------------------------------------------------


def _stats_row(dfrac, rref, pcap, coefficient: str, coefficient_level, metric: str,
                both_vals: np.ndarray, disabled_vals: np.ndarray) -> dict:
    label = base.METRICS[metric][0]
    diff = both_vals - disabled_vals
    paired = base.paired_comparison(both_vals, disabled_vals)
    pct_paired, pct_naive = base.paired_pct_change(both_vals, disabled_vals)
    d_indep = base.cohens_d_independent(both_vals, disabled_vals)
    halfwidth = base.t_ci_halfwidth(paired["sd_diff"], len(diff))
    return {
        "deferrable_fraction": dfrac,
        "response_reference_eur_per_kwh": rref,
        "payback_cap_fraction": pcap,
        "coefficient": coefficient,
        "coefficient_level": coefficient_level,
        "metric": metric,
        "metric_label": label,
        "n_seeds": len(diff),
        "mean_disabled": float(np.mean(disabled_vals)),
        "mean_both": float(np.mean(both_vals)),
        "paired_mean_diff": paired["mean_diff"],
        "paired_mean_pct_change": pct_paired,
        "naive_pct_change_of_means": pct_naive,
        "paired_dz": paired["d_z"],
        "cohens_d_independent": d_indep,
        "p_uncorrected": paired["p_value"],
        "degenerate": paired["degenerate"],
        "ci95_t_lo": paired["mean_diff"] - halfwidth,
        "ci95_t_hi": paired["mean_diff"] + halfwidth,
        "metric3_sign_convention": base.SIGN_CONVENTION_NOTE,
        "notes": "",
        "_diff": diff,
    }


def _apply_correction_family(rows: list[dict], family_name: str, family_size: int) -> None:
    assert len(rows) == family_size, f"{family_name}: expected {family_size} rows, got {len(rows)}"
    pvals = [r["p_uncorrected"] for r in rows]
    holm_adj = base.holm_bonferroni(pvals)
    bh_adj = base.benjamini_hochberg(pvals)
    for i, r in enumerate(rows):
        r["p_holm"] = float(holm_adj[i])
        r["p_bh"] = float(bh_adj[i])
        r["correction_family"] = family_name
        r["correction_family_size"] = family_size
        r["survives_holm_alpha05"] = bool((not r["degenerate"]) and r["p_holm"] <= base.ALPHA)
        r["survives_bh_alpha05"] = bool((not r["degenerate"]) and r["p_bh"] <= base.ALPHA)


def _apply_bootstrap(rows: list[dict], rng: np.random.Generator) -> None:
    """Thread ONE rng through the given rows in the order given. Reproducible
    only while callers keep a fixed, documented row order (same caveat as
    scripts/analyze_sweep.py's build_corrected_summary): run_analysis() below
    always applies this to cell rows first, then marginal rows.
    """
    for r in rows:
        diff = r.pop("_diff")
        boot_lo, boot_hi = base.bootstrap_paired_diff_ci(diff, rng)
        flag, detail, ratio = base._ci_disagreement(r["ci95_t_lo"], r["ci95_t_hi"], boot_lo, boot_hi)
        r["ci95_boot_lo"] = boot_lo
        r["ci95_boot_hi"] = boot_hi
        r["ci_disagreement_flag"] = flag
        r["ci_disagreement_detail"] = detail
        r["ci_width_ratio"] = ratio
        r["bootstrap_seed"] = base.BOOTSTRAP_SEED
        r["n_bootstrap"] = base.N_BOOTSTRAP
        r["alpha"] = base.ALPHA


def build_cell_table(df: pd.DataFrame) -> list[dict]:
    """Full factorial: one row per (coefficient cell, metric3 sub-measure),
    48 cells x 2 metrics = 96 rows, each a real paired capacity_both vs
    capacity_disabled t-test at n=30 seeds. df must already carry the
    _dfrac/_rref/_pcap snapped columns (see _snap_axes)."""
    rows = []
    for dfrac in DEFERRABLE_FRACTIONS:
        for rref in RESPONSE_REFERENCES_EUR_PER_KWH:
            for pcap in PAYBACK_CAP_FRACTIONS:
                cell = df[np.isclose(df["_dfrac"], dfrac) & np.isclose(df["_rref"], rref) & np.isclose(df["_pcap"], pcap)]
                both = cell[cell["ablation"] == "capacity_both"].set_index("seed").sort_index()
                disabled = cell[cell["ablation"] == "capacity_disabled"].set_index("seed").sort_index()
                if len(both) != len(SEEDS) or len(disabled) != len(SEEDS):
                    raise AssertionError(
                        f"cell (dfrac={dfrac}, rref={rref}, pcap={pcap}) has {len(both)} capacity_both rows "
                        f"and {len(disabled)} capacity_disabled rows; expected {len(SEEDS)} each"
                    )
                for metric in METRIC3_KEYS:
                    both_vals = both[metric].to_numpy(dtype=float)
                    disabled_vals = disabled[metric].reindex(both.index).to_numpy(dtype=float)
                    rows.append(_stats_row(dfrac, rref, pcap, "", "", metric, both_vals, disabled_vals))
    _apply_correction_family(rows, "cell_full_factorial_primary", FAMILY_SIZE_CELL_PRIMARY)
    return rows


def _marginal_arrays(df: pd.DataFrame, coefficient: str, level: float, metric: str) -> tuple:
    """Average per seed over the OTHER two coefficients' full cartesian
    product, holding `coefficient` at `level`, separately for capacity_both
    and capacity_disabled, then return the two length-30 (one per seed)
    arrays paired by seed. Because the disabled arm's value for a given seed
    is identical across every one of the cells being averaged over (see
    module docstring), the disabled-side average is exactly that seed's
    single simulated value; this never manufactures 48 independent
    baselines or inflates n beyond 30."""
    col = RAW_AXIS_COLUMN[coefficient]
    sub = df[np.isclose(df[col], level)]
    both = sub[sub["ablation"] == "capacity_both"].groupby("seed")[metric].mean().sort_index()
    disabled = sub[sub["ablation"] == "capacity_disabled"].groupby("seed")[metric].mean().sort_index()
    expected_seeds = sorted(SEEDS)
    if list(both.index) != expected_seeds or list(disabled.index) != expected_seeds:
        raise AssertionError(
            f"marginal averaging for {coefficient}={level}, metric={metric} did not yield all "
            f"{len(expected_seeds)} seeds for both arms (both={len(both)}, disabled={len(disabled)})"
        )
    return both.to_numpy(dtype=float), disabled.to_numpy(dtype=float)


def build_marginal_table(df: pd.DataFrame) -> list[dict]:
    """One row per (coefficient level, metric3 sub-measure), marginalized
    (averaged) over the other two coefficients: (4+4+3) levels x 2 metrics
    = 22 rows. Answers "does the effect depend on this coefficient" as a
    main-effect view; build_cell_table() above is what catches an
    interaction this view would average away."""
    rows = []
    for coefficient, levels in COEFFICIENT_LEVELS.items():
        for level in levels:
            for metric in METRIC3_KEYS:
                both_vals, disabled_vals = _marginal_arrays(df, coefficient, level, metric)
                dfrac = level if coefficient == "deferrable_fraction" else "ALL"
                rref = level if coefficient == "response_reference_eur_per_kwh" else "ALL"
                pcap = level if coefficient == "payback_cap_fraction" else "ALL"
                rows.append(_stats_row(dfrac, rref, pcap, coefficient, level, metric, both_vals, disabled_vals))
    _apply_correction_family(rows, "marginal_per_coefficient_secondary", FAMILY_SIZE_MARGINAL_SECONDARY)
    return rows


def _dfrac_pcap_marginal(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """deferrable_fraction x payback_cap_fraction grid of the paired mean
    percent change, averaged (marginalized) over response_reference_eur_per_kwh.
    This is the view D9's independent rationale names by name (a large
    deferrable_fraction combined with a tight payback_cap_fraction is where a
    payback rebound could invert the peak-to-average result). df must already
    carry the _dfrac/_rref/_pcap snapped columns."""
    data = np.zeros((len(DEFERRABLE_FRACTIONS), len(PAYBACK_CAP_FRACTIONS)))
    for i, dfrac in enumerate(DEFERRABLE_FRACTIONS):
        for j, pcap in enumerate(PAYBACK_CAP_FRACTIONS):
            sub = df[np.isclose(df["_dfrac"], dfrac) & np.isclose(df["_pcap"], pcap)]
            both = sub[sub["ablation"] == "capacity_both"].groupby("seed")[metric].mean().sort_index()
            disabled = sub[sub["ablation"] == "capacity_disabled"].groupby("seed")[metric].mean().sort_index()
            pct, _ = base.paired_pct_change(both.to_numpy(dtype=float), disabled.to_numpy(dtype=float))
            data[i, j] = pct
    return pd.DataFrame(data, index=list(DEFERRABLE_FRACTIONS), columns=list(PAYBACK_CAP_FRACTIONS))


# ---------------------------------------------------------------------------
# Interaction check (task item 3) and per-coefficient verdict (task item 4)
# ---------------------------------------------------------------------------


def interaction_report(cell_df: pd.DataFrame) -> dict:
    """Scan the full 48-cell coefficient box for a peak-to-average sign flip
    (worsened, i.e. paired_mean_diff > 0 under the fixed sign convention),
    with the D9-flagged corner (large deferrable_fraction, tight
    payback_cap_fraction) reported explicitly regardless of outcome."""
    p2a = cell_df[cell_df["metric"] == "feeder_peak_to_average_ratio"]
    all_flip = p2a[p2a["worsened_vs_disabled"]].sort_values("paired_mean_pct_change", ascending=False)
    corner = p2a[
        np.isclose(p2a["deferrable_fraction"].astype(float), FLAGGED_CORNER[0])
        & np.isclose(p2a["payback_cap_fraction"].astype(float), FLAGGED_CORNER[1])
    ].sort_values("response_reference_eur_per_kwh")
    corner_flip = corner[corner["worsened_vs_disabled"]]
    return {
        "all_flip_cells": all_flip,
        "corner_cells": corner,
        "corner_flip_cells": corner_flip,
        "any_flip_anywhere": bool(len(all_flip)),
        "flip_at_flagged_corner": bool(len(corner_flip)),
    }


def _cell_rows_at_level(cell_df: pd.DataFrame, coefficient: str, level: float, metric: str) -> pd.DataFrame:
    """Rows of cell_df (the 96-row full-factorial table) where `coefficient`
    equals `level` and metric matches, i.e. the subset of cells this
    coefficient's marginal level was averaged over."""
    return cell_df[(cell_df["metric"] == metric) & np.isclose(cell_df[coefficient].astype(float), level)]


def build_verdict_table(cell_df: pd.DataFrame, marginal_df: pd.DataFrame) -> pd.DataFrame:
    """Per (coefficient, metric3 sub-measure): is the effect robust across the
    swept range, or fragile? Three criteria, all data-driven off the fixed
    sign convention (negative = improvement):
      1. sign consistency across the coefficient's own marginal levels
      2. significance (Holm, alpha=0.05) at every one of those levels
      3. sign consistency of the underlying cells WITHIN each level (an
         interaction check: a marginal average can stay negative even while
         individual cells behind it disagree in sign, exactly the failure
         mode a one-at-a-time scan would miss -- see docs/DECISIONS.md D9).
    """
    rows = []
    for coefficient, levels in COEFFICIENT_LEVELS.items():
        for metric in METRIC3_KEYS:
            marg = marginal_df[(marginal_df["coefficient"] == coefficient) & (marginal_df["metric"] == metric)].copy()
            marg["coefficient_level"] = marg["coefficient_level"].astype(float)
            marg = marg.sort_values("coefficient_level")

            pct = marg["paired_mean_pct_change"].to_numpy(dtype=float)
            diffs = marg["paired_mean_diff"].to_numpy(dtype=float)
            nonzero_signs = np.sign(diffs)[np.sign(diffs) != 0]
            sign_consistent = bool(len(np.unique(nonzero_signs)) <= 1)
            holm_ok = bool(marg["survives_holm_alpha05"].all())

            interaction_flip_levels = []
            for level in levels:
                cell_sub = _cell_rows_at_level(cell_df, coefficient, level, metric)
                cell_signs = np.sign(cell_sub["paired_mean_diff"].to_numpy(dtype=float))
                cell_signs_nonzero = cell_signs[cell_signs != 0]
                if len(np.unique(cell_signs_nonzero)) > 1:
                    interaction_flip_levels.append(level)
            interaction_clean = len(interaction_flip_levels) == 0

            if not sign_consistent:
                verdict = "FRAGILE: marginal effect changes sign across this coefficient's levels"
            elif not holm_ok:
                verdict = "FRAGILE: marginal effect loses significance (Holm, alpha=0.05) at some level(s)"
            elif not interaction_clean:
                verdict = "ROBUST ON MARGINAL AVERAGE BUT INTERACTION-FRAGILE: individual cells disagree in sign (see interaction check)"
            else:
                verdict = "ROBUST across the swept range"

            rows.append(
                {
                    "scope": "coefficient_verdict",
                    "coefficient": coefficient,
                    "metric": metric,
                    "metric_label": base.METRICS[metric][0],
                    "min_pct_change_across_levels": float(np.min(pct)),
                    "max_pct_change_across_levels": float(np.max(pct)),
                    "sign_consistent_across_marginal_levels": sign_consistent,
                    "significant_at_every_level_holm": holm_ok,
                    "cell_level_sign_consistent_within_every_level": interaction_clean,
                    "levels_with_within_level_cell_sign_disagreement": str(interaction_flip_levels),
                    "verdict": verdict,
                    "metric3_sign_convention": base.SIGN_CONVENTION_NOTE,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shipped-cell anchor (recovered from the main sweep, not re-simulated)
# ---------------------------------------------------------------------------


def shipped_cell_anchor(raw_path: Path = MAIN_SWEEP_RAW_PATH) -> dict | None:
    """Recover the shipped calibration point's metric-3 effect
    (deferrable_fraction=0.20, response_reference=0.05, payback_cap=0.5) from
    the MAIN sweep (results/sweep_raw.parquet at k=1.0, broker_count=3), since
    response_reference=0.05 is not one of the four D9-swept levels. Returns
    None (and the caller prints a note) if that file is not present."""
    if not raw_path.exists():
        return None
    raw = pd.read_parquet(raw_path)
    cell = raw[(raw["k"].astype(float) == SHIPPED_K) & (raw["broker_count"].astype(int) == SHIPPED_BROKER_COUNT)]
    both = cell[cell["ablation"] == "capacity_both"].set_index("seed").sort_index()
    disabled = cell[cell["ablation"] == "capacity_disabled"].set_index("seed").sort_index()
    result = {"dfrac": SHIPPED_DFRAC, "rref": SHIPPED_RREF, "pcap": SHIPPED_PCAP, "n_seeds": len(both)}
    for metric in METRIC3_KEYS:
        both_vals = both[metric].to_numpy(dtype=float)
        disabled_vals = disabled[metric].reindex(both.index).to_numpy(dtype=float)
        pct_paired, _ = base.paired_pct_change(both_vals, disabled_vals)
        result[f"{metric}_paired_mean_pct_change"] = pct_paired
    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _approx_pct_ci(row: pd.Series) -> tuple:
    """Approximate percent-change CI bounds for the figure only, by scaling
    the exact t-based CI on the ABSOLUTE paired difference (ci95_t_lo/hi in
    results/structural_sensitivity.csv, the authoritative figures) by the
    ratio of the reported paired percent-change to the paired absolute
    difference. Percent change is used on the figure, rather than absolute
    units, because it puts the two very differently-scaled metric-3
    sub-measures (CoV around 0.7, peak-to-average around 3.5) on one
    comparable axis."""
    diff = row["paired_mean_diff"]
    pct = row["paired_mean_pct_change"]
    if diff == 0:
        return pct, pct
    scale = pct / diff
    lo, hi = row["ci95_t_lo"] * scale, row["ci95_t_hi"] * scale
    return min(lo, hi), max(lo, hi)


def plot_structural_sensitivity(cell_df: pd.DataFrame, marginal_df: pd.DataFrame, verdict_df: pd.DataFrame,
                                 heatmaps: dict, anchor: dict | None, plots_dir: Path) -> Path:
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    coeff_panels = [
        ("deferrable_fraction", "Deferrable fraction", False, SHIPPED_DFRAC),
        ("response_reference_eur_per_kwh", "Response reference (EUR/kWh, log scale)", True, SHIPPED_RREF),
        ("payback_cap_fraction", "Payback cap fraction", False, SHIPPED_PCAP),
    ]

    for col_idx, (coefficient, xlabel, log_x, shipped_level) in enumerate(coeff_panels):
        ax = fig.add_subplot(gs[0, col_idx])
        for metric, (color, mlabel) in METRIC_PLOT_STYLE.items():
            sub = marginal_df[(marginal_df["coefficient"] == coefficient) & (marginal_df["metric"] == metric)].copy()
            sub["coefficient_level"] = sub["coefficient_level"].astype(float)
            sub = sub.sort_values("coefficient_level")
            x = sub["coefficient_level"].to_numpy(dtype=float)
            y = sub["paired_mean_pct_change"].to_numpy(dtype=float)
            bounds = [_approx_pct_ci(r) for _, r in sub.iterrows()]
            los = np.array([b[0] for b in bounds])
            his = np.array([b[1] for b in bounds])
            yerr = np.vstack([y - los, his - y])
            ax.errorbar(x, y, yerr=yerr, marker="o", color=color, label=mlabel, capsize=4)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        is_swept_level = any(np.isclose(shipped_level, lvl) for lvl in COEFFICIENT_LEVELS[coefficient])
        if is_swept_level:
            ax.axvline(shipped_level, color="gray", linewidth=1.2, linestyle=":")
        else:
            ax.annotate(
                f"shipped value {shipped_level:g} is off this grid\n(geometric centre of the bracket)",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7, style="italic", color="gray",
            )
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        if col_idx == 0:
            ax.set_ylabel("Paired mean percent change\ncapacity_both vs capacity_disabled (%)")
        ax.set_title(f"Marginal effect vs {coefficient}", fontsize=10)
        ax.set_xticks(list(COEFFICIENT_LEVELS[coefficient]))

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", frameon=False, fontsize=9)

    for panel_idx, metric in enumerate(METRIC3_KEYS):
        ax = fig.add_subplot(gs[1, panel_idx])
        grid = heatmaps[metric]
        vmax = float(np.nanmax(np.abs(grid.to_numpy()))) or 1.0
        im = ax.imshow(grid.to_numpy(), cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(PAYBACK_CAP_FRACTIONS)))
        ax.set_xticklabels([f"{p:g}" for p in PAYBACK_CAP_FRACTIONS])
        ax.set_yticks(range(len(DEFERRABLE_FRACTIONS)))
        ax.set_yticklabels([f"{d:g}" for d in DEFERRABLE_FRACTIONS])
        ax.set_xlabel("payback_cap_fraction")
        ax.set_ylabel("deferrable_fraction")
        ax.set_title(f"{METRIC_PLOT_STYLE[metric][1]}: % change\n(marginal over response_reference)", fontsize=9)
        for i, dfrac in enumerate(DEFERRABLE_FRACTIONS):
            for j, pcap in enumerate(PAYBACK_CAP_FRACTIONS):
                val = grid.iloc[i, j]
                is_corner = np.isclose(dfrac, FLAGGED_CORNER[0]) and np.isclose(pcap, FLAGGED_CORNER[1])
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=8,
                        fontweight="bold" if is_corner else "normal")
                if is_corner:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))
        if any(np.isclose(SHIPPED_DFRAC, DEFERRABLE_FRACTIONS)):
            i_shipped = DEFERRABLE_FRACTIONS.index(SHIPPED_DFRAC)
            j_shipped = PAYBACK_CAP_FRACTIONS.index(SHIPPED_PCAP)
            ax.plot(j_shipped, i_shipped, marker="*", markersize=18, color="blue", markeredgecolor="white")
        fig.colorbar(im, ax=ax, shrink=0.8, label="% change (negative = improved)")

    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")
    lines = ["Per-coefficient verdict (CoV / Peak-to-average):", ""]
    for coefficient in COEFFICIENT_LEVELS:
        sub = verdict_df[verdict_df["coefficient"] == coefficient]
        cov_v = sub[sub["metric"] == "feeder_coefficient_of_variation"]["verdict"].iloc[0]
        p2a_v = sub[sub["metric"] == "feeder_peak_to_average_ratio"]["verdict"].iloc[0]
        lines.append(f"{coefficient}:")
        lines.append(f"  CoV: {cov_v}")
        lines.append(f"  Peak-to-avg: {p2a_v}")
        lines.append("")
    if anchor is not None:
        lines.append("Shipped-cell anchor (main sweep, k=1.0 bc=3,")
        lines.append(f"  dfrac={SHIPPED_DFRAC}, rref={SHIPPED_RREF} off-grid, pcap={SHIPPED_PCAP}):")
        lines.append(f"  CoV {anchor['feeder_coefficient_of_variation_paired_mean_pct_change']:+.2f}%   "
                      f"Peak-to-avg {anchor['feeder_peak_to_average_ratio_paired_mean_pct_change']:+.2f}%")
    ax_text.text(0.0, 1.0, "\n".join(lines), transform=ax_text.transAxes, fontsize=8, va="top", ha="left",
                 family="monospace", wrap=True)

    fig.suptitle(
        "D9: does the metric-3 result depend on the three fixed structural coefficients?\n"
        "Top row: marginal paired percent change vs each coefficient's own levels (other two averaged out), "
        "approximate 95 percent CI; star/dotted line = shipped calibration.\n"
        "Bottom row: deferrable_fraction x payback_cap_fraction interaction (response_reference averaged out); "
        "black box = D9's named plausible failure corner (large deferrable, tight payback cap).",
        fontsize=10,
    )

    fig_path = plots_dir / "fig_structural_sensitivity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


# ---------------------------------------------------------------------------
# Stdout diagnostics
# ---------------------------------------------------------------------------


def _print_interaction_report(report: dict) -> None:
    print("\n" + "=" * 78)
    print("INTERACTION CHECK: peak-to-average sign across the full 48-cell coefficient box")
    print("D9's flagged failure mode: large deferrable_fraction (0.30) x tight payback_cap_fraction (0.25)")
    print("=" * 78)
    corner = report["corner_cells"]
    print(
        f"\nFlagged corner (deferrable_fraction={FLAGGED_CORNER[0]}, payback_cap_fraction={FLAGGED_CORNER[1]}), "
        f"across all {len(corner)} response_reference levels:"
    )
    print(corner[["response_reference_eur_per_kwh", "paired_mean_pct_change", "p_holm", "worsened_vs_disabled"]].to_string(index=False))

    if report["flip_at_flagged_corner"]:
        print("\n*** SIGN FLIP DETECTED AT THE D9-FLAGGED CORNER ***")
        print("The peak-to-average effect WORSENS (positive paired diff) at:")
        print(
            report["corner_flip_cells"][
                ["deferrable_fraction", "response_reference_eur_per_kwh", "payback_cap_fraction", "paired_mean_pct_change", "p_holm"]
            ].to_string(index=False)
        )
    else:
        print("\nNo sign flip at the flagged corner.")

    if report["any_flip_anywhere"]:
        n = len(report["all_flip_cells"])
        print(f"\n*** SIGN FLIP DETECTED SOMEWHERE IN THE 48-CELL COEFFICIENT BOX: {n} of 48 cells worsen ***")
        print(
            report["all_flip_cells"][
                ["deferrable_fraction", "response_reference_eur_per_kwh", "payback_cap_fraction", "paired_mean_pct_change", "p_holm"]
            ].to_string(index=False)
        )
    else:
        print("\nNo peak-to-average sign flip anywhere in the 48-cell coefficient box.")
    print("=" * 78)


def _print_verdict_summary(verdict_df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("PER-COEFFICIENT VERDICT: robust or fragile across the swept range?")
    print("=" * 78)
    for _, row in verdict_df.iterrows():
        print(
            f"\n{row['coefficient']} / {row['metric_label']}: range "
            f"{row['min_pct_change_across_levels']:+.2f}% to {row['max_pct_change_across_levels']:+.2f}% "
            "(paired mean pct change across this coefficient's own levels)"
        )
        print(f"  sign consistent across marginal levels: {row['sign_consistent_across_marginal_levels']}")
        print(f"  significant (Holm, alpha=0.05) at every level: {row['significant_at_every_level_holm']}")
        print(f"  cell-level sign consistent within every level (interaction check): {row['cell_level_sign_consistent_within_every_level']}")
        print(f"  VERDICT: {row['verdict']}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_analysis(parquet_path: Path, output_dir: Path) -> dict:
    if not parquet_path.exists():
        print(_missing_parquet_message(parquet_path), file=sys.stderr)
        return {"ok": False}

    df = pd.read_parquet(parquet_path)

    problems = validate_structural_sweep(df)
    if problems:
        print(f"VALIDATION FAILED for {parquet_path}:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return {"ok": False, "problems": problems}

    print(f"VALIDATION OK: {len(df)} rows, axes/schema match the D9 grid exactly.")
    df = _snap_axes(df)

    prov = provenance_breakdown(df)
    print("\nRow provenance breakdown:")
    print(prov.to_string(index=False))

    spread = disabled_arm_within_seed_spread(df)
    max_spread = float(spread.to_numpy().max()) if len(spread) else 0.0
    print(
        f"\nDisabled-arm same-seed spread across all 48 coefficient cells (max over seeds and columns): "
        f"{max_spread:.3e}"
    )
    print("(should be ~0: a seed's disabled-arm outcome must not depend on which coefficient cell it is attached to)")
    if max_spread > SPREAD_WARN_TOL:
        print(
            f"WARNING: disabled-arm spread exceeds {SPREAD_WARN_TOL:g}; the materialized-baseline assumption "
            "may not hold in this data.",
            file=sys.stderr,
        )

    cell_rows = build_cell_table(df)
    marginal_rows = build_marginal_table(df)
    n_degenerate = sum(1 for r in cell_rows + marginal_rows if r["degenerate"])
    if n_degenerate:
        print(
            f"\nNOTE: {n_degenerate} of {len(cell_rows) + len(marginal_rows)} paired comparisons were degenerate "
            "(near-zero variance in the paired difference); unlike the D8 sweep's capacity_pnl_only channel, "
            "nothing about this sweep's capacity_both vs capacity_disabled comparison is expected to be a "
            "deterministic identity, so this is worth a second look if it occurs."
        )

    rng = np.random.default_rng(base.BOOTSTRAP_SEED)
    _apply_bootstrap(cell_rows, rng)      # fixed order: cell rows first ...
    _apply_bootstrap(marginal_rows, rng)  # ... then marginal rows (see _apply_bootstrap's docstring)

    cell_df = pd.DataFrame(cell_rows)
    marginal_df = pd.DataFrame(marginal_rows)
    cell_df["scope"] = "cell_full_factorial"
    marginal_df["scope"] = "marginal_per_coefficient"

    cell_df["is_flagged_d9_corner"] = (
        np.isclose(cell_df["deferrable_fraction"].astype(float), FLAGGED_CORNER[0])
        & np.isclose(cell_df["payback_cap_fraction"].astype(float), FLAGGED_CORNER[1])
    )
    cell_df["worsened_vs_disabled"] = cell_df["paired_mean_diff"] > 0
    cell_df["is_shipped_axis_level"] = (
        np.isclose(cell_df["deferrable_fraction"].astype(float), SHIPPED_DFRAC)
        & np.isclose(cell_df["payback_cap_fraction"].astype(float), SHIPPED_PCAP)
    )

    marginal_df["worsened_vs_disabled"] = marginal_df["paired_mean_diff"] > 0
    marginal_df["is_shipped_axis_level"] = marginal_df.apply(
        lambda r: bool(
            (r["coefficient"] == "deferrable_fraction" and np.isclose(float(r["coefficient_level"]), SHIPPED_DFRAC))
            or (r["coefficient"] == "payback_cap_fraction" and np.isclose(float(r["coefficient_level"]), SHIPPED_PCAP))
        ),
        axis=1,
    )
    marginal_df.loc[marginal_df["coefficient"] == "response_reference_eur_per_kwh", "notes"] = (
        "shipped response_reference_eur_per_kwh=0.05 is not one of the 4 swept levels here (it is the "
        "geometric centre of the bracket); see the shipped-cell anchor recovered from results/sweep_raw.parquet"
    )

    verdict_df = build_verdict_table(cell_df, marginal_df)

    interaction = interaction_report(cell_df)
    _print_interaction_report(interaction)
    _print_verdict_summary(verdict_df)

    anchor = shipped_cell_anchor()
    if anchor is not None:
        print(
            f"\nShipped calibration cell anchor (results/sweep_raw.parquet, k=1.0, bc=3, "
            f"deferrable_fraction={SHIPPED_DFRAC}, response_reference={SHIPPED_RREF} [off this sweep's grid], "
            f"payback_cap_fraction={SHIPPED_PCAP}), n_seeds={anchor['n_seeds']}:"
        )
        for metric in METRIC3_KEYS:
            print(f"  {metric}: {anchor[f'{metric}_paired_mean_pct_change']:+.2f}% (paired mean pct change, capacity_both vs capacity_disabled)")
    else:
        print(f"\nNOTE: {MAIN_SWEEP_RAW_PATH} not found; skipping the shipped-cell anchor comparison.")

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([cell_df, marginal_df, verdict_df], ignore_index=True, sort=False)
    combined = combined.reindex(columns=CSV_COLUMNS)
    csv_path = output_dir / "structural_sensitivity.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path} ({len(combined)} rows)")

    heatmaps = {metric: _dfrac_pcap_marginal(df, metric) for metric in METRIC3_KEYS}
    fig_path = plot_structural_sensitivity(cell_df, marginal_df, verdict_df, heatmaps, anchor, plots_dir)
    print(f"wrote {fig_path}")

    return {
        "ok": True,
        "cell_df": cell_df,
        "marginal_df": marginal_df,
        "verdict_df": verdict_df,
        "interaction": interaction,
        "csv_path": csv_path,
        "fig_path": fig_path,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D9 structural-coefficient sensitivity analysis (docs/DECISIONS.md D9).")
    parser.add_argument("--parquet", default=None, help="Path to sweep_structural.parquet (default: results/sweep_structural.parquet)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write structural_sensitivity.csv and plots/ into (default: results/). "
             "Point this at a scratch directory for testing; the real run against results/ happens "
             "once results/sweep_structural.parquet exists.",
    )
    args = parser.parse_args(argv)

    parquet_path = Path(args.parquet) if args.parquet else DEFAULT_PARQUET_PATH
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    result = run_analysis(parquet_path, output_dir)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
