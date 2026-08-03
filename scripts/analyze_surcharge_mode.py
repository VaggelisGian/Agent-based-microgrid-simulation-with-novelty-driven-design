"""D10 surcharge-distribution control arm analysis (docs/DECISIONS.md D10).

D10's defect: the thesis claims the peak-to-average improvement that grows with
broker_count is caused by DESYNCHRONIZED deferral timing across brokers. Under the
shipped ("proportional") surcharge rule a larger broker_count also means a smaller
per-broker surcharge and therefore less total deferred energy, so "more brokers,
better peak shaving" has two candidate explanations the shipped design varies
together: desynchronized timing, and simply a lower deferred volume sitting further
from the D9 reversal band. D10 adds two control surcharge-distribution rules that
separate the two candidates:

  - synchronized: every broker gets the SAME surcharge (passthrough / N). Removes
    cross-broker variation (no desynchronization possible) while matching total
    signal strength; deferred volume still falls with broker_count.
  - renormalized: surcharge_b = passthrough * share_b * N. Approximately matches
    deferred volume across broker_count while cross-broker variation (and hence
    desynchronization) still varies.

The discriminating logic (pre-registered in D10, applied here and not before):
  - Gradient PERSISTS under synchronized -> channel is deferred volume, not
    desynchronization (synchronized has nothing left to desynchronize).
  - Gradient WEAKENS or VANISHES under synchronized while SURVIVING under
    renormalized -> desynchronization is the channel.
  - Any other pattern is reported as observed; the claim is downgraded to match.

Data sources (read-only; this script writes only the two named outputs below):
  - results/sweep_surcharge_mode.parquet (720 rows): synchronized and renormalized,
    k in {0.5, 1.0} x broker_count in {2, 3, 5} x ablation in {capacity_disabled,
    capacity_both} x 30 seeds (20260704..20260733).
  - results/sweep_raw.parquet (the shipped sweep): filtered to the same k,
    broker_count and ablation values and relabeled capacity_surcharge_mode =
    "proportional". This is the SAME data the thesis already reports; it is not
    re-run. Seeds are the same 30 in both files.

Writes:
  - results/surcharge_mode_comparison.csv (24 rows: 18 scope="mode_cell" rows, 3
    modes x 2 k x 3 broker_count, one row per cell, plus 6 scope="gradient_bc2_to_bc5"
    rows, 3 modes x 2 k, one row per (mode, k) gradient test -- see the
    "Correction lifted" paragraph below for why the gradient rows were added).
    Each mode_cell row reports, for that cell, capacity_both vs capacity_disabled
    paired by seed on both metric-3 sub-measures (feeder_peak_to_average_ratio,
    feeder_coefficient_of_variation): paired mean percent change, paired Cohen's
    d_z, raw paired-t p-value, and Holm/BH-corrected p-values, plus mean
    total_deferred_kwh under capacity_both and capacity_disabled so deferred-volume
    matching is visible directly alongside every effect. Each gradient_bc2_to_bc5
    row reports the broker_count 2-to-5 gradient in peak-to-average ratio's paired
    percent change for that (mode, k): the two endpoint means, the gradient itself
    in percentage points, its paired d_z, and its raw and Holm/BH-corrected
    p-values. correction_family, metric3_sign_convention and notes (verdict text)
    are shared columns across both scopes and are identical on every row within a
    scope, matching this repo's existing convention of repeating whole-table
    context (see e.g. metric3_sign_convention in results/summary_stats.csv); a
    column that does not apply to a given scope (broker_count on a
    gradient_bc2_to_bc5 row, metric and the gradient_* columns on a mode_cell row)
    is left empty on that row rather than given a placeholder value.
  - results/plots/fig_surcharge_mode.png: the broker-count gradient in
    peak-to-average ratio under all three modes, one panel per k, all three modes
    overlaid so the gradients can be compared directly, with 95 percent CI bands
    across the 30 seeds.

The broker_count=2-to-5 GRADIENT test that Task 3's decision rule needs (is the
gradient significant, does it persist/weaken/vanish across modes) is computed and
printed to stdout by this script (see compute_gradient_tests()/determine_verdict()
below) and its conclusion is written into the notes column of the CSV above. It was
ORIGINALLY not persisted as extra CSV rows, on the reasoning that the task's CSV
spec was exactly the 18 per-cell rows above; re-running this script reproduces the
gradient numbers identically (no randomness involved), so nothing about them was
ever a one-off, unreproducible computation, only a not-yet-persisted one.

Correction lifted: that 18-row CSV spec was a Phase 17 TASK constraint, not a
principle, and a later mutation-testing pass over the test suite found the gap it
left open: the six gradient statistics this package's own headline sentences quote
(docs/DECISIONS.md's "D10 result" and 06_results.md Section 6.9's "-7.93 and -8.37
percentage points... at p_holm 8.4e-28 and 8.3e-15" for synchronized, and the
renormalized figures alongside them) sat in no machine-readable column anywhere.
Two numbers per test, the gradient in percentage points and its Holm-corrected
p-value, did survive as PROSE at 3 to 4 significant figures inside the notes
string this script already wrote on all 18 rows ("synchronized -7.934 pp
(p_holm=8.36e-28, sig=True)", and the -8.367 / 8.34e-15 pair at the other k),
which is exactly where the thesis sentences above were read off; the rest
(pct_at_bc2, pct_at_bc5, dz, p_uncorrected and p_bh) had no home of any kind. An
earlier draft of this paragraph said the statistics were "reproducible only by
re-running this script, in no artifact at all". The second half of that was wrong,
and contradicted this docstring's own description of the notes column a dozen
lines above it. The constraint is lifted from here on: the six
gradient tests compute_gradient_tests() computes are now ALSO written as CSV rows,
under scope="gradient_bc2_to_bc5" (see build_gradient_table() below), distinct from
the 18 rows' scope="mode_cell". The 18 mode_cell rows are unchanged in every field
by this; the gradient rows are strictly additional.

Statistical methods are REUSED from scripts/analyze_sweep.py, not reimplemented:
paired_comparison (paired Cohen's d_z and scipy.stats.ttest_rel), paired_pct_change
(paired mean of per-seed percent changes), holm_bonferroni, benjamini_hochberg,
t_ci_halfwidth, and the metric-3 sign convention (negative paired diff = improved
stability). See scripts/analyze_sweep.py's module docstring and D6/D7/D8 notes for
the full reasoning behind those helpers; this script does not alter them.

Correction families (mirrors scripts/analyze_sweep.py's and
scripts/analyze_structural_sweep.py's precedent: the primary family is the finest-
grained set of REAL, non-overlapping paired tests this sweep produces; a coarser,
overlapping secondary view is corrected separately, never pooled with the primary):

  - "surcharge_mode_primary_real" (36 tests): capacity_both vs capacity_disabled,
    paired by seed, for both metric-3 sub-measures, at every (mode, k,
    broker_count) cell (3 x 2 x 3 x 2 = 36). This is a NEW experimental arm
    (D10), analyzed in its own output file, so it gets its own primary family --
    it is not pooled with the main D8 sweep's 120-test family or the D9 sweep's
    96-test family, exactly as the D9 structural sweep was not pooled with D8's.
    None of these 36 tests is a deterministic identity (unlike the D8 sweep's
    capacity_pnl_only contrasts): capacity_both is expected to differ from
    capacity_disabled under every surcharge mode, so nothing is excluded here.
  - "gradient_bc2_to_bc5_secondary" (6 tests): the broker_count=2-to-5 gradient
    in the peak-to-average ratio's per-seed paired percent change, one test per
    (mode, k). This is D10's own discriminating question -- "does the gradient
    persist, weaken, or vanish" -- quantified explicitly rather than eyeballed
    off the figure, per the task brief. It is corrected as its own secondary
    family because it re-uses the same 36 primary comparisons' underlying seed
    draws under a different (differenced) aggregation, so pooling it with the
    primary family would double-count those draws, the same reasoning
    scripts/analyze_structural_sweep.py gives for keeping its own marginal
    family separate from its cell family.

Run from the repository root:

    python scripts/analyze_surcharge_mode.py

Requires: Python 3.12+, pandas, numpy, scipy (via scripts/analyze_sweep.py),
matplotlib (no seaborn). ASCII only throughout (source and all written CSV/PNG
artifacts). No randomness, no network access, no re-running of any simulation,
no tuning of any analysis choice against a desired result. Effects are reported
as observed; if the observed pattern does not fit either of D10's two clean
decision branches, that is reported plainly instead of forced into one.
"""

from __future__ import annotations

import json
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

import analyze_sweep as base  # noqa: E402 (path set above)
# Reused from scripts/analyze_sweep.py rather than reimplemented: paired_comparison,
# paired_pct_change, holm_bonferroni, benjamini_hochberg, t_ci_halfwidth, ALPHA,
# SIGN_CONVENTION_NOTE.

# ---------------------------------------------------------------------------
# Paths and grid constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SURCHARGE_MODE_PATH = REPO_ROOT / "results" / "sweep_surcharge_mode.parquet"
MAIN_SWEEP_RAW_PATH = REPO_ROOT / "results" / "sweep_raw.parquet"
OUTPUT_CSV_PATH = REPO_ROOT / "results" / "surcharge_mode_comparison.csv"
PLOTS_DIR = REPO_ROOT / "results" / "plots"
FIG_PATH = PLOTS_DIR / "fig_surcharge_mode.png"

K_VALUES = (0.5, 1.0)
BROKER_COUNTS = (2, 3, 5)
NEW_MODES = ("synchronized", "renormalized")
ALL_MODES = ("proportional",) + NEW_MODES
ABLATIONS = ("capacity_disabled", "capacity_both")
SEED_BASE = 20260704
SEEDS = tuple(SEED_BASE + i for i in range(30))
METRIC3_KEYS = ("feeder_peak_to_average_ratio", "feeder_coefficient_of_variation")
GRADIENT_METRIC = "feeder_peak_to_average_ratio"  # the metric D10's decision rule is about

MODE_COLOR = {"proportional": "#1f77b4", "synchronized": "#ff7f0e", "renormalized": "#2ca02c"}

EXPECTED_MODE_CELL_ROWS = len(ALL_MODES) * len(K_VALUES) * len(BROKER_COUNTS)  # 18
assert EXPECTED_MODE_CELL_ROWS == 18
FAMILY_SIZE_PRIMARY = EXPECTED_MODE_CELL_ROWS * len(METRIC3_KEYS)  # 36
FAMILY_SIZE_GRADIENT_SECONDARY = len(ALL_MODES) * len(K_VALUES)  # 6
EXPECTED_COMBINED_ROWS = len(ALL_MODES) * len(K_VALUES) * len(BROKER_COUNTS) * len(ABLATIONS) * len(SEEDS)  # 1080
# results/surcharge_mode_comparison.csv's own row count: the 18 scope="mode_cell"
# rows plus the 6 scope="gradient_bc2_to_bc5" rows (see build_gradient_table() and
# the module docstring's "Correction lifted" paragraph).
EXPECTED_OUTPUT_CSV_ROWS = EXPECTED_MODE_CELL_ROWS + FAMILY_SIZE_GRADIENT_SECONDARY  # 24


# ---------------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------------


def load_combined() -> pd.DataFrame:
    sm = pd.read_parquet(SURCHARGE_MODE_PATH)
    raw = pd.read_parquet(MAIN_SWEEP_RAW_PATH)

    prop = raw[
        raw["k"].isin(K_VALUES) & raw["broker_count"].isin(BROKER_COUNTS) & raw["ablation"].isin(ABLATIONS)
    ].copy()
    prop["capacity_surcharge_mode"] = "proportional"

    combined = pd.concat([sm, prop], ignore_index=True, sort=False)
    return combined


def validate_combined(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []

    if len(df) != EXPECTED_COMBINED_ROWS:
        problems.append(f"expected {EXPECTED_COMBINED_ROWS} combined rows, found {len(df)}")

    observed_modes = set(df["capacity_surcharge_mode"].unique())
    if observed_modes != set(ALL_MODES):
        problems.append(f"capacity_surcharge_mode values {sorted(observed_modes)} != expected {sorted(ALL_MODES)}")

    observed_k = set(df["k"].astype(float).unique())
    if observed_k != set(K_VALUES):
        problems.append(f"k values {sorted(observed_k)} != expected {sorted(K_VALUES)}")

    observed_bc = set(df["broker_count"].astype(int).unique())
    if observed_bc != set(BROKER_COUNTS):
        problems.append(f"broker_count values {sorted(observed_bc)} != expected {sorted(BROKER_COUNTS)}")

    observed_ablation = set(df["ablation"].unique())
    if observed_ablation != set(ABLATIONS):
        problems.append(f"ablation values {sorted(observed_ablation)} != expected {sorted(ABLATIONS)}")

    observed_seeds = set(df["seed"].astype(int).unique())
    if observed_seeds != set(SEEDS):
        problems.append(
            f"seed set does not match the project's 30 seeds: missing "
            f"{sorted(set(SEEDS) - observed_seeds)}, unexpected {sorted(observed_seeds - set(SEEDS))}"
        )

    key_cols = ["capacity_surcharge_mode", "k", "broker_count", "ablation", "seed"]
    dup_counts = df.groupby(key_cols).size()
    duplicates = dup_counts[dup_counts > 1]
    if len(duplicates):
        problems.append(f"{len(duplicates)} duplicate keys found, e.g. {duplicates.index[:5].tolist()}")

    expected_keys = {
        (m, k, bc, a, s) for m in ALL_MODES for k in K_VALUES for bc in BROKER_COUNTS for a in ABLATIONS for s in SEEDS
    }
    actual_keys = set(df[key_cols].itertuples(index=False, name=None))
    missing = expected_keys - actual_keys
    if missing:
        problems.append(f"{len(missing)} missing cells, e.g. {sorted(missing)[:5]}")
    extra = actual_keys - expected_keys
    if extra:
        problems.append(f"{len(extra)} unexpected cells, e.g. {sorted(extra)[:5]}")

    return problems


def _cell_frames(df: pd.DataFrame, mode: str, k: float, bc: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = df[(df["capacity_surcharge_mode"] == mode) & (df["k"] == k) & (df["broker_count"] == bc)]
    both = cell[cell["ablation"] == "capacity_both"].set_index("seed").sort_index()
    disabled = cell[cell["ablation"] == "capacity_disabled"].set_index("seed").sort_index()
    if len(both) != len(SEEDS) or len(disabled) != len(SEEDS):
        raise AssertionError(
            f"cell (mode={mode}, k={k}, broker_count={bc}) has {len(both)} capacity_both and "
            f"{len(disabled)} capacity_disabled rows; expected {len(SEEDS)} each"
        )
    return both, disabled


def per_seed_pct_change(both_vals: np.ndarray, disabled_vals: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(disabled_vals != 0, (both_vals - disabled_vals) / disabled_vals * 100.0, np.nan)


# ---------------------------------------------------------------------------
# Primary family: 36 real paired tests (3 modes x 2 k x 3 broker_count x 2 metrics)
# ---------------------------------------------------------------------------


def compute_primary_tests(df: pd.DataFrame) -> list[dict]:
    tests = []
    for mode in ALL_MODES:
        for k in K_VALUES:
            for bc in BROKER_COUNTS:
                both, disabled = _cell_frames(df, mode, k, bc)
                for metric in METRIC3_KEYS:
                    both_vals = both[metric].to_numpy(dtype=float)
                    disabled_vals = disabled[metric].reindex(both.index).to_numpy(dtype=float)
                    paired = base.paired_comparison(both_vals, disabled_vals)
                    pct_paired, _ = base.paired_pct_change(both_vals, disabled_vals)
                    tests.append(
                        {
                            "mode": mode,
                            "k": k,
                            "broker_count": bc,
                            "metric": metric,
                            "n_seeds": len(both_vals),
                            "paired_mean_pct_change": pct_paired,
                            "paired_dz": paired["d_z"],
                            "p_uncorrected": paired["p_value"],
                            "degenerate": paired["degenerate"],
                            "total_deferred_kwh_both_mean": float(both["total_deferred_kwh"].mean()),
                            "total_deferred_kwh_disabled_mean": float(disabled["total_deferred_kwh"].mean()),
                        }
                    )
    assert len(tests) == FAMILY_SIZE_PRIMARY, f"expected {FAMILY_SIZE_PRIMARY} primary tests, found {len(tests)}"

    pvals = [t["p_uncorrected"] for t in tests]
    holm_adj = base.holm_bonferroni(pvals)
    bh_adj = base.benjamini_hochberg(pvals)
    for i, t in enumerate(tests):
        t["p_holm"] = float(holm_adj[i])
        t["p_bh"] = float(bh_adj[i])
        t["survives_holm_alpha05"] = bool((not t["degenerate"]) and t["p_holm"] <= base.ALPHA)
        t["survives_bh_alpha05"] = bool((not t["degenerate"]) and t["p_bh"] <= base.ALPHA)
        t["correction_family"] = "surcharge_mode_primary_real"
        t["correction_family_size"] = FAMILY_SIZE_PRIMARY
    return tests


def build_mode_cell_table(tests: list[dict], notes_by_mode_k: dict) -> pd.DataFrame:
    by_cell: dict[tuple, dict] = {}
    for t in tests:
        by_cell.setdefault((t["mode"], t["k"], t["broker_count"]), {})[t["metric"]] = t

    rows = []
    for mode in ALL_MODES:
        for k in K_VALUES:
            for bc in BROKER_COUNTS:
                cell = by_cell[(mode, k, bc)]
                anchor = cell[METRIC3_KEYS[0]]
                row = {
                    "scope": "mode_cell",
                    "capacity_surcharge_mode": mode,
                    "k": k,
                    "broker_count": bc,
                    "n_seeds": anchor["n_seeds"],
                }
                for metric in METRIC3_KEYS:
                    t = cell[metric]
                    row[f"{metric}_paired_mean_pct_change"] = t["paired_mean_pct_change"]
                    row[f"{metric}_paired_dz"] = t["paired_dz"]
                    row[f"{metric}_paired_p"] = t["p_uncorrected"]
                    row[f"{metric}_p_holm"] = t["p_holm"]
                    row[f"{metric}_p_bh"] = t["p_bh"]
                    row[f"{metric}_survives_holm_alpha05"] = t["survives_holm_alpha05"]
                row["total_deferred_kwh_capacity_both_mean"] = anchor["total_deferred_kwh_both_mean"]
                row["total_deferred_kwh_capacity_disabled_mean"] = anchor["total_deferred_kwh_disabled_mean"]
                row["correction_family"] = (
                    "surcharge_mode_primary_real (36 real paired tests: 3 modes x 2 k x 3 broker_count x "
                    "2 metric3 sub-measures; new D10 arm, its own family, not pooled with the D8 (120) or "
                    "D9 (96) families)"
                )
                row["metric3_sign_convention"] = base.SIGN_CONVENTION_NOTE
                row["notes"] = notes_by_mode_k.get(mode, "")
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Secondary family: 6 gradient tests (broker_count 2 -> 5, peak-to-average ratio)
# ---------------------------------------------------------------------------


def compute_gradient_tests(df: pd.DataFrame) -> list[dict]:
    tests = []
    for mode in ALL_MODES:
        for k in K_VALUES:
            both2, disabled2 = _cell_frames(df, mode, k, 2)
            both5, disabled5 = _cell_frames(df, mode, k, 5)
            pct2 = per_seed_pct_change(
                both2[GRADIENT_METRIC].to_numpy(dtype=float),
                disabled2[GRADIENT_METRIC].reindex(both2.index).to_numpy(dtype=float),
            )
            pct5 = per_seed_pct_change(
                both5[GRADIENT_METRIC].to_numpy(dtype=float),
                disabled5[GRADIENT_METRIC].reindex(both5.index).to_numpy(dtype=float),
            )
            # both2.index == both5.index == sorted(SEEDS) by construction of _cell_frames.
            paired = base.paired_comparison(pct5, pct2)  # mean_diff = gradient (bc5 minus bc2), in percentage points
            tests.append(
                {
                    "mode": mode,
                    "k": k,
                    "metric": GRADIENT_METRIC,
                    "pct_at_bc2": float(np.mean(pct2)),
                    "pct_at_bc5": float(np.mean(pct5)),
                    "gradient_pct_points_bc2_to_bc5": paired["mean_diff"],
                    "gradient_dz": paired["d_z"],
                    "p_uncorrected": paired["p_value"],
                    "degenerate": paired["degenerate"],
                }
            )
    assert len(tests) == FAMILY_SIZE_GRADIENT_SECONDARY

    pvals = [t["p_uncorrected"] for t in tests]
    holm_adj = base.holm_bonferroni(pvals)
    bh_adj = base.benjamini_hochberg(pvals)
    for i, t in enumerate(tests):
        t["p_holm"] = float(holm_adj[i])
        t["p_bh"] = float(bh_adj[i])
        t["survives_holm_alpha05"] = bool((not t["degenerate"]) and t["p_holm"] <= base.ALPHA)
        t["survives_bh_alpha05"] = bool((not t["degenerate"]) and t["p_bh"] <= base.ALPHA)
    return tests


def build_gradient_table(gradient_tests: list[dict], notes_by_mode_k: dict) -> pd.DataFrame:
    """The six compute_gradient_tests() results as CSV rows, under
    scope="gradient_bc2_to_bc5" (distinct from build_mode_cell_table()'s 18
    scope="mode_cell" rows). See this module's docstring "Correction lifted"
    paragraph for why these are persisted as rows now rather than only printed
    and folded into the mode_cell rows' notes column.

    broker_count and metric are the two columns whose applicability differs by
    scope (mode_cell rows have a real broker_count and no single metric column,
    since they carry both metric3 sub-measures side by side; gradient rows span
    broker_count 2 to 5 by construction and are always about GRADIENT_METRIC).
    broker_count is simply omitted here, exactly like every other mode_cell-only
    column: pd.concat in main() fills it in as empty for these rows, the same
    way it fills the gradient-only columns in as empty for the mode_cell rows.
    """
    rows = []
    for t in gradient_tests:
        rows.append(
            {
                "scope": "gradient_bc2_to_bc5",
                "capacity_surcharge_mode": t["mode"],
                "k": t["k"],
                "n_seeds": len(SEEDS),
                "metric": t["metric"],
                "gradient_pct_at_bc2": t["pct_at_bc2"],
                "gradient_pct_at_bc5": t["pct_at_bc5"],
                "gradient_pct_points_bc2_to_bc5": t["gradient_pct_points_bc2_to_bc5"],
                "gradient_dz": t["gradient_dz"],
                "gradient_p_uncorrected": t["p_uncorrected"],
                "gradient_p_holm": t["p_holm"],
                "gradient_p_bh": t["p_bh"],
                "gradient_survives_holm_alpha05": t["survives_holm_alpha05"],
                "gradient_survives_bh_alpha05": t["survives_bh_alpha05"],
                "gradient_degenerate": t["degenerate"],
                "correction_family": (
                    "gradient_bc2_to_bc5_secondary (6 tests: broker_count 2->5 gradient in "
                    "peak-to-average ratio's paired percent change, one per (mode, k); D10's own "
                    "secondary family, corrected separately from the primary 36-test family since "
                    "it reuses the same underlying paired seed draws under a different, differenced "
                    "aggregation, the same reasoning scripts/analyze_structural_sweep.py gives for "
                    "keeping its own marginal family separate from its cell family)"
                ),
                "metric3_sign_convention": base.SIGN_CONVENTION_NOTE,
                "notes": notes_by_mode_k.get(t["mode"], ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D10 verdict
# ---------------------------------------------------------------------------


def determine_verdict(gradient_tests: list[dict]) -> tuple[str, dict]:
    """Apply D10's pre-registered decision rule using only the computed numbers
    above: statistical significance (Holm, alpha=0.05) of the broker_count 2->5
    gradient in peak-to-average ratio, under each mode, at each k. No magnitude
    threshold is invented here; the actual percentage-point numbers are always
    quoted alongside the verdict so the reader judges scale directly.
    """
    by_mode_k = {(t["mode"], t["k"]): t for t in gradient_tests}

    lines = []
    branch_votes = []
    for k in K_VALUES:
        prop = by_mode_k[("proportional", k)]
        sync = by_mode_k[("synchronized", k)]
        renorm = by_mode_k[("renormalized", k)]

        prop_sig = prop["survives_holm_alpha05"]
        sync_sig = sync["survives_holm_alpha05"]
        renorm_sig = renorm["survives_holm_alpha05"]

        if sync_sig and np.sign(sync["gradient_pct_points_bc2_to_bc5"]) == np.sign(prop["gradient_pct_points_bc2_to_bc5"]):
            branch = "PERSISTS_UNDER_SYNCHRONIZED"
        elif (not sync_sig) and renorm_sig:
            branch = "WEAKENS_VANISHES_UNDER_SYNCHRONIZED_SURVIVES_UNDER_RENORMALIZED"
        else:
            branch = "AMBIGUOUS_OR_OTHER"
        branch_votes.append(branch)

        lines.append(
            f"k={k:g}: proportional {prop['gradient_pct_points_bc2_to_bc5']:+.3f} pp "
            f"(p_holm={prop['p_holm']:.3g}, sig={prop_sig}); "
            f"synchronized {sync['gradient_pct_points_bc2_to_bc5']:+.3f} pp "
            f"(p_holm={sync['p_holm']:.3g}, sig={sync_sig}); "
            f"renormalized {renorm['gradient_pct_points_bc2_to_bc5']:+.3f} pp "
            f"(p_holm={renorm['p_holm']:.3g}, sig={renorm_sig}) -> branch: {branch}"
        )

    if len(set(branch_votes)) == 1:
        overall_branch = branch_votes[0]
    else:
        overall_branch = "K_DEPENDENT_MIXED_PATTERN"

    if overall_branch == "PERSISTS_UNDER_SYNCHRONIZED":
        verdict = (
            "D10 VERDICT: the broker-count gradient in peak-to-average ratio PERSISTS under synchronized "
            "(same sign as proportional, significant after Holm correction) at every tested k. Per D10's "
            "pre-registered rule this fires the FIRST branch: the channel behind the peak-to-average "
            "improvement is deferred VOLUME, not desynchronized timing, because synchronized removes all "
            "cross-broker variation (nothing left to desynchronize) yet the gradient survives anyway. "
            "This control refutes the desynchronization account, and the thesis's explanation is the "
            "volume-based one, revised in response to this arm."
        )
    elif overall_branch == "WEAKENS_VANISHES_UNDER_SYNCHRONIZED_SURVIVES_UNDER_RENORMALIZED":
        verdict = (
            "D10 VERDICT: the broker-count gradient in peak-to-average ratio WEAKENS/VANISHES under "
            "synchronized while SURVIVING under renormalized, at every tested k. Per D10's pre-registered "
            "rule this fires the SECOND branch: desynchronization is the channel behind the gradient. This "
            "branch does not fire on the shipped grid; if it ever fires on a different one, it contradicts "
            "the volume-based explanation the thesis gives for the gradient, and on that grid the "
            "volume-based explanation would have to be withdrawn and a timing-based one put in its place."
        )
    else:
        verdict = (
            "D10 VERDICT: the pattern does not fit either pre-registered clean branch cleanly at every k "
            "(see the per-k breakdown below) -- reported as observed, per D10's own instruction for this "
            "case. On such a pattern this arm establishes no channel at all, so the volume-based "
            "explanation the thesis gives for the gradient would have to be stated as unresolved on that "
            "grid rather than as controlled for: the control built to separate deferred volume from "
            "deferral timing would not have separated them there."
        )

    full_text = verdict + " Per-k detail: " + " | ".join(lines)
    return full_text, {"branch_votes": branch_votes, "overall_branch": overall_branch, "lines": lines}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_surcharge_mode(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(12, 5.5), sharey=True, layout="constrained")

    for ax, k in zip(axes, K_VALUES):
        for mode in ALL_MODES:
            xs, ys, halfwidths = [], [], []
            for bc in BROKER_COUNTS:
                both, disabled = _cell_frames(df, mode, k, bc)
                both_vals = both[GRADIENT_METRIC].to_numpy(dtype=float)
                disabled_vals = disabled[GRADIENT_METRIC].reindex(both.index).to_numpy(dtype=float)
                per_seed = per_seed_pct_change(both_vals, disabled_vals)
                mean_pct = float(np.nanmean(per_seed))
                std_pct = float(np.nanstd(per_seed, ddof=1))
                halfwidth = base.t_ci_halfwidth(std_pct, len(per_seed))
                xs.append(bc)
                ys.append(mean_pct)
                halfwidths.append(halfwidth)
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            halfwidths = np.array(halfwidths, dtype=float)
            color = MODE_COLOR[mode]
            ax.plot(xs, ys, marker="o", markersize=8, linewidth=2.2, color=color, label=mode)
            ax.fill_between(xs, ys - halfwidths, ys + halfwidths, color=color, alpha=0.15)

        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(list(BROKER_COUNTS))
        ax.set_xlabel("broker_count")
        ax.set_title(f"k = {k:g}")

    axes[0].set_ylabel(
        "Feeder peak-to-average ratio,\npaired mean percent change\n(capacity_both vs capacity_disabled, %)"
    )
    axes[0].annotate(
        "Lower (more negative) = more stable",
        xy=(0.02, 0.03),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=8,
        style="italic",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(ALL_MODES), frameon=False)

    fig.suptitle(
        "D10: broker-count gradient in peak-to-average ratio under three surcharge-distribution rules\n"
        "proportional = shipped rule (reused from results/sweep_raw.parquet, not re-run); synchronized removes\n"
        "cross-broker variation while matching signal strength; renormalized approximately matches deferred\n"
        "volume across broker_count while cross-broker variation still varies. Shaded band = 95 percent CI\n"
        "across 30 seeds. See results/surcharge_mode_comparison.csv for the tested gradient numbers and verdict.",
        fontsize=10,
    )
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    return FIG_PATH


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def print_volume_match_check(primary_tests: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("DEFERRED VOLUME MATCH CHECK (mean total_deferred_kwh under capacity_both)")
    print("=" * 78)
    seen = set()
    rows = []
    for t in primary_tests:
        key = (t["mode"], t["k"], t["broker_count"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "mode": t["mode"],
                "k": t["k"],
                "broker_count": t["broker_count"],
                "mean_total_deferred_kwh_both": t["total_deferred_kwh_both_mean"],
                "mean_total_deferred_kwh_disabled": t["total_deferred_kwh_disabled_mean"],
            }
        )
    vol_df = pd.DataFrame(rows).sort_values(["mode", "k", "broker_count"])
    print(vol_df.to_string(index=False))

    print("\nRange (max/min ratio) of mean deferred kWh across broker_count, by mode and k:")
    for mode in ALL_MODES:
        for k in K_VALUES:
            sub = vol_df[(vol_df["mode"] == mode) & (vol_df["k"] == k)]
            vals = sub["mean_total_deferred_kwh_both"].to_numpy(dtype=float)
            ratio = float(np.max(vals) / np.min(vals)) if np.min(vals) > 0 else float("inf")
            print(
                f"  {mode}, k={k:g}: bc2={vals[0]:.1f}, bc3={vals[1]:.1f}, bc5={vals[2]:.1f} kWh/year, "
                f"max/min ratio={ratio:.3f}"
            )


def print_ordering_check(df: pd.DataFrame, primary_tests: list[dict]) -> None:
    """Rank-order check on what the peak-to-average effect actually tracks.

    Reported within a fixed k rather than pooled. k moves the scarcity
    threshold and therefore the whole volume-response curve, so pooling the
    two k levels compares points on different curves. D9's correction section
    had to make the same within-stratum distinction for payback_cap_fraction.
    """
    from scipy import stats

    cells = []
    for t in primary_tests:
        if t["metric"] != GRADIENT_METRIC:
            continue
        both, _ = _cell_frames(df, t["mode"], t["k"], t["broker_count"])
        spreads = both["mean_broker_surcharge_json"].apply(
            lambda blob: (lambda vals: max(vals) - min(vals))(list(json.loads(blob).values()))
        )
        cells.append(
            {
                "mode": t["mode"],
                "k": t["k"],
                "effect": t["paired_mean_pct_change"],
                "volume": t["total_deferred_kwh_both_mean"],
                "spread": float(spreads.mean()),
            }
        )
    cell_df = pd.DataFrame(cells)

    print("\n" + "=" * 78)
    print("ORDERING CHECK: does the effect track deferred volume or cross-broker spread?")
    print("=" * 78)
    for k in K_VALUES:
        sub = cell_df[cell_df["k"] == k]
        rho_v, p_v = stats.spearmanr(sub["volume"], sub["effect"])
        rho_s, p_s = stats.spearmanr(sub["spread"], sub["effect"])
        print(
            f"  k={k:g} (n={len(sub)} cells): "
            f"volume rho={rho_v:+.3f} p={p_v:.3f} | cross-broker spread rho={rho_s:+.3f} p={p_s:.3f}"
        )
    rho_all, p_all = stats.spearmanr(cell_df["volume"], cell_df["effect"])
    print(
        f"  pooled over both k (n={len(cell_df)}): volume rho={rho_all:+.3f} p={p_all:.3f}, "
        "reported only to show pooling across k does NOT order the effect"
    )
    print("  n is 9 cells per k, so this corroborates the control-arm result rather than carrying it.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    if not SURCHARGE_MODE_PATH.exists():
        print(f"ERROR: {SURCHARGE_MODE_PATH} not found.", file=sys.stderr)
        return 1
    if not MAIN_SWEEP_RAW_PATH.exists():
        print(f"ERROR: {MAIN_SWEEP_RAW_PATH} not found.", file=sys.stderr)
        return 1

    df = load_combined()
    problems = validate_combined(df)
    if problems:
        print("VALIDATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"VALIDATION OK: {len(df)} combined rows across {len(ALL_MODES)} modes.")

    primary_tests = compute_primary_tests(df)
    gradient_tests = compute_gradient_tests(df)

    verdict_text, verdict_detail = determine_verdict(gradient_tests)
    notes_by_mode_k = {mode: verdict_text for mode in ALL_MODES}  # same whole-analysis verdict on every row

    mode_cell_df = build_mode_cell_table(primary_tests, notes_by_mode_k)
    assert len(mode_cell_df) == EXPECTED_MODE_CELL_ROWS, f"expected {EXPECTED_MODE_CELL_ROWS} rows, got {len(mode_cell_df)}"

    gradient_df = build_gradient_table(gradient_tests, notes_by_mode_k)
    assert len(gradient_df) == FAMILY_SIZE_GRADIENT_SECONDARY, (
        f"expected {FAMILY_SIZE_GRADIENT_SECONDARY} gradient rows, got {len(gradient_df)}"
    )

    # sort=False: keeps mode_cell_df's own column order first, with gradient_df's
    # columns that mode_cell_df does not already have (metric, the gradient_*
    # columns) appended after it, rather than an alphabetical reshuffle of the
    # whole header. broker_count (real on mode_cell rows, not applicable on
    # gradient rows) and metric (the reverse) each pick up a float NaN from this
    # concat on the scope that does not carry them; the explicit fix below turns
    # that back into a real int / plain empty cell so the 18 mode_cell rows'
    # broker_count column renders exactly as it always did ("2", not "2.0"),
    # rather than silently changing those rows' own text to satisfy a brand new
    # column that has nothing to do with them.
    output_df = pd.concat([mode_cell_df, gradient_df], ignore_index=True, sort=False)
    output_df["broker_count"] = output_df["broker_count"].apply(lambda v: "" if pd.isna(v) else int(v))
    assert len(output_df) == EXPECTED_OUTPUT_CSV_ROWS, f"expected {EXPECTED_OUTPUT_CSV_ROWS} rows, got {len(output_df)}"

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nwrote {OUTPUT_CSV_PATH} ({len(output_df)} rows: {len(mode_cell_df)} mode_cell + {len(gradient_df)} gradient_bc2_to_bc5)")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = plot_surcharge_mode(df)
    print(f"wrote {fig_path}")

    print_volume_match_check(primary_tests)
    print_ordering_check(df, primary_tests)

    print("\n" + "=" * 78)
    print("GRADIENT TEST DETAIL (broker_count 2 -> 5, peak-to-average ratio)")
    print("=" * 78)
    for line in verdict_detail["lines"]:
        print(line)
    print(f"\nBranch per k: {verdict_detail['branch_votes']}")
    print(f"Overall branch: {verdict_detail['overall_branch']}")
    print("\n" + verdict_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
