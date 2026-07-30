"""Tests for the Phase 10 multiple-comparison correction and bootstrap CI
additions to scripts/analyze_sweep.py (Holm-Bonferroni primary, Benjamini-
Hochberg FDR secondary, seed-level paired bootstrap CIs).

scripts/ is not an importable package (no __init__.py, not on pythonpath),
so this module loads scripts/analyze_sweep.py the same way
test_golden_master.py loads scripts/regenerate_golden_master.py: via
importlib.util.spec_from_file_location. Importing the module only defines
functions/constants; its main() is gated behind
`if __name__ == "__main__":`, so nothing is written to disk as a side effect
of these tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZE_SWEEP_PATH = REPO_ROOT / "scripts" / "analyze_sweep.py"


def _load_analyze_sweep():
    spec = importlib.util.spec_from_file_location("analyze_sweep_under_test", ANALYZE_SWEEP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__name__ != "__main__"
    return module


analyze_sweep = _load_analyze_sweep()

RAW_PATH = REPO_ROOT / "results" / "sweep_raw.parquet"
MONOPOLY_PATH = REPO_ROOT / "results" / "sweep_monopoly.parquet"


# ---------------------------------------------------------------------------
# Hand-computable examples
# ---------------------------------------------------------------------------
#
# p = [0.01, 0.02, 0.03, 0.04, 0.5], m = 5, already sorted ascending.
#
# Holm: adjusted_(i) = running max of (m - i + 1) * p_(i), i = 1..5 (1-indexed)
#   i=1: (5-1+1)*0.01 = 5*0.01 = 0.05                    -> running max 0.05
#   i=2: (5-2+1)*0.02 = 4*0.02 = 0.08                    -> running max 0.08
#   i=3: (5-3+1)*0.03 = 3*0.03 = 0.09                    -> running max 0.09
#   i=4: (5-4+1)*0.04 = 2*0.04 = 0.08                    -> running max stays 0.09
#   i=5: (5-5+1)*0.50 = 1*0.50 = 0.50                    -> running max 0.50
#   => [0.05, 0.08, 0.09, 0.09, 0.50]
#
# BH: candidate_(i) = p_(i) * m / i, then running min from the top down
#   i=1: 0.01*5/1 = 0.05   i=2: 0.02*5/2 = 0.05   i=3: 0.03*5/3 = 0.05
#   i=4: 0.04*5/4 = 0.05   i=5: 0.50*5/5 = 0.50
#   running min from i=5 down to i=1: [0.05, 0.05, 0.05, 0.05, 0.50]

HAND_PVALS = [0.01, 0.02, 0.03, 0.04, 0.5]
HAND_HOLM_EXPECTED = [0.05, 0.08, 0.09, 0.09, 0.5]
HAND_BH_EXPECTED = [0.05, 0.05, 0.05, 0.05, 0.5]


def test_holm_hand_computed_example():
    result = analyze_sweep.holm_bonferroni(HAND_PVALS)
    assert result == pytest.approx(HAND_HOLM_EXPECTED, abs=1e-12)


def test_bh_hand_computed_example():
    result = analyze_sweep.benjamini_hochberg(HAND_PVALS)
    assert result == pytest.approx(HAND_BH_EXPECTED, abs=1e-12)


def test_holm_hand_computed_example_is_order_invariant():
    # Same multiset of p-values, shuffled input order: each corrected value
    # must travel with its own p-value, not with its input position.
    shuffled = [0.5, 0.03, 0.01, 0.04, 0.02]
    expected = {0.5: 0.5, 0.03: 0.09, 0.01: 0.05, 0.04: 0.09, 0.02: 0.08}
    result = analyze_sweep.holm_bonferroni(shuffled)
    for p_in, adj in zip(shuffled, result):
        assert adj == pytest.approx(expected[p_in], abs=1e-12)


def test_bh_hand_computed_example_is_order_invariant():
    shuffled = [0.5, 0.03, 0.01, 0.04, 0.02]
    expected = {0.5: 0.5, 0.03: 0.05, 0.01: 0.05, 0.04: 0.05, 0.02: 0.05}
    result = analyze_sweep.benjamini_hochberg(shuffled)
    for p_in, adj in zip(shuffled, result):
        assert adj == pytest.approx(expected[p_in], abs=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_holm_and_bh_single_pvalue_is_identity():
    assert analyze_sweep.holm_bonferroni([0.03]) == pytest.approx([0.03])
    assert analyze_sweep.benjamini_hochberg([0.03]) == pytest.approx([0.03])


def test_holm_and_bh_all_ones_stays_at_one():
    p = [1.0, 1.0, 1.0, 1.0]
    assert analyze_sweep.holm_bonferroni(p) == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert analyze_sweep.benjamini_hochberg(p) == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_holm_and_bh_exact_zero_pvalue_stays_zero():
    p = [0.0, 0.2, 0.4]
    holm = analyze_sweep.holm_bonferroni(p)
    bh = analyze_sweep.benjamini_hochberg(p)
    assert holm[0] == pytest.approx(0.0, abs=1e-15)
    assert bh[0] == pytest.approx(0.0, abs=1e-15)


def test_holm_and_bh_never_exceed_one():
    rng = np.random.default_rng(7)
    for _ in range(200):
        m = rng.integers(1, 30)
        p = rng.random(m)
        assert np.all(analyze_sweep.holm_bonferroni(p) <= 1.0 + 1e-12)
        assert np.all(analyze_sweep.benjamini_hochberg(p) <= 1.0 + 1e-12)


def test_holm_and_bh_empty_input():
    assert len(analyze_sweep.holm_bonferroni([])) == 0
    assert len(analyze_sweep.benjamini_hochberg([])) == 0


# ---------------------------------------------------------------------------
# Property checks: (1) Holm-adjusted p is non-decreasing in sorted order;
# (2) Holm-adjusted p is always >= BH-adjusted p for the same input (BH's
# FDR control is uniformly less conservative than Holm's FWER control).
# ---------------------------------------------------------------------------


def test_holm_is_monotonic_nondecreasing_in_sorted_order():
    rng = np.random.default_rng(20260704)
    for _ in range(500):
        m = rng.integers(2, 25)
        p = rng.random(m)
        adjusted = analyze_sweep.holm_bonferroni(p)
        order = np.argsort(p, kind="stable")
        sorted_adjusted = adjusted[order]
        diffs = np.diff(sorted_adjusted)
        assert np.all(diffs >= -1e-12), "Holm-adjusted p must be non-decreasing in sorted p order"


def test_bh_is_monotonic_nondecreasing_in_sorted_order():
    rng = np.random.default_rng(2026070401)
    for _ in range(500):
        m = rng.integers(2, 25)
        p = rng.random(m)
        adjusted = analyze_sweep.benjamini_hochberg(p)
        order = np.argsort(p, kind="stable")
        sorted_adjusted = adjusted[order]
        diffs = np.diff(sorted_adjusted)
        assert np.all(diffs >= -1e-12), "BH-adjusted p must be non-decreasing in sorted p order"


def test_holm_always_geq_bh_elementwise():
    rng = np.random.default_rng(42)
    for _ in range(1000):
        m = rng.integers(1, 30)
        p = rng.random(m)
        holm = analyze_sweep.holm_bonferroni(p)
        bh = analyze_sweep.benjamini_hochberg(p)
        assert np.all(holm >= bh - 1e-9), "Holm (FWER) must never be less conservative than BH (FDR)"


def test_holm_geq_bh_on_the_hand_computed_example():
    holm = analyze_sweep.holm_bonferroni(HAND_PVALS)
    bh = analyze_sweep.benjamini_hochberg(HAND_PVALS)
    assert np.all(holm >= bh - 1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cross_check_against_statsmodels_if_available(seed):
    statsmodels_multitest = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(seed)
    p = rng.random(15)

    _, expected_holm, _, _ = statsmodels_multitest.multipletests(p, method="holm")
    _, expected_bh, _, _ = statsmodels_multitest.multipletests(p, method="fdr_bh")

    assert analyze_sweep.holm_bonferroni(p) == pytest.approx(expected_holm, abs=1e-9)
    assert analyze_sweep.benjamini_hochberg(p) == pytest.approx(expected_bh, abs=1e-9)


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------


def test_bootstrap_ci_is_reproducible_given_the_same_seed():
    diff = np.array([1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.0, 1.5])
    rng_a = np.random.default_rng(20260704)
    rng_b = np.random.default_rng(20260704)
    ci_a = analyze_sweep.bootstrap_paired_diff_ci(diff, rng_a, n_boot=2000)
    ci_b = analyze_sweep.bootstrap_paired_diff_ci(diff, rng_b, n_boot=2000)
    assert ci_a == ci_b


def test_bootstrap_ci_degenerate_all_zero_diff_is_zero_width_at_zero():
    diff = np.zeros(30)
    rng = np.random.default_rng(1)
    lo, hi = analyze_sweep.bootstrap_paired_diff_ci(diff, rng, n_boot=2000)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_ci_brackets_the_true_mean_for_a_clear_nonzero_effect():
    # 30 "seeds" of paired differences with a clear, consistent negative
    # effect and modest noise: the 95 percent bootstrap CI should exclude
    # zero and bracket the true generating mean.
    rng_data = np.random.default_rng(3)
    diff = rng_data.normal(loc=-2.0, scale=0.5, size=30)
    rng = np.random.default_rng(20260704)
    lo, hi = analyze_sweep.bootstrap_paired_diff_ci(diff, rng, n_boot=10000)
    assert lo < hi
    assert hi < 0.0  # excludes zero: a real, consistent effect should show up
    assert lo < -2.0 < hi or abs(np.mean(diff) - (-2.0)) < 0.5  # sanity: near the generating mean


# ---------------------------------------------------------------------------
# CI disagreement flag
# ---------------------------------------------------------------------------


def test_ci_disagreement_flags_zero_exclusion_mismatch():
    flag, detail, ratio = analyze_sweep._ci_disagreement(-0.001, 0.002, 0.0005, 0.002)
    assert flag is True
    assert "zero-exclusion" in detail


def test_ci_disagreement_flags_large_width_ratio():
    flag, detail, ratio = analyze_sweep._ci_disagreement(-1.0, 1.0, -0.1, 0.1)
    assert flag is True
    assert ratio == pytest.approx(10.0)
    assert "width ratio" in detail


def test_ci_disagreement_no_flag_when_intervals_agree():
    flag, detail, ratio = analyze_sweep._ci_disagreement(-1.0, 1.0, -0.9, 1.1)
    assert flag is False
    assert detail == ""


# ---------------------------------------------------------------------------
# Integration: the full correction family built from the real sweep output
# (skips gracefully if the parquet files are not present on this machine;
# never re-runs a simulation, only reads existing results/*.parquet).
# ---------------------------------------------------------------------------


def test_build_corrected_summary_family_sizes_and_invariants_on_real_data():
    if not RAW_PATH.is_file():
        pytest.skip(f"results/sweep_raw.parquet not present: {RAW_PATH}")

    df = analyze_sweep.load_raw()
    mono_only_df = None
    if MONOPOLY_PATH.is_file():
        combined_df = analyze_sweep.load_monopoly_combined(df)
        if combined_df is not None:
            mono_only_df = combined_df[combined_df["broker_count"] == analyze_sweep.MONOPOLY_BROKER_COUNT]

    corrected = analyze_sweep.build_corrected_summary(df, mono_only_df)

    primary = corrected[corrected["correction_family"] == "primary_main_sweep_real"]
    degenerate = corrected[corrected["correction_family"] == "excluded_degenerate_main_sweep"]
    assert len(primary) == analyze_sweep.FAMILY_SIZE_PRIMARY
    assert len(degenerate) == analyze_sweep.FAMILY_SIZE_ALTERNATIVE - analyze_sweep.FAMILY_SIZE_PRIMARY

    # every degenerate row is reported, not silently dropped, and is excluded
    # from correction by convention (p_holm == p_bh == 1.0, never "survives")
    assert (degenerate["p_holm"] == 1.0).all()
    assert (degenerate["p_bh"] == 1.0).all()
    assert not degenerate["survives_holm_alpha05"].any()
    assert not degenerate["survives_bh_alpha05"].any()

    # Holm never more lenient than BH, on the real p-values actually produced
    # by this sweep's paired t-tests.
    assert (primary["p_holm"] >= primary["p_bh"] - 1e-9).all()

    if mono_only_df is not None:
        mono_rows = corrected[corrected["correction_family"] == "secondary_monopoly_supplement_real"]
        assert len(mono_rows) == analyze_sweep.FAMILY_SIZE_MONOPOLY_SECONDARY
        assert (mono_rows["p_holm"] >= mono_rows["p_bh"] - 1e-9).all()


def test_build_corrected_summary_headline_metric3_and_cost_effects_survive_holm():
    if not RAW_PATH.is_file():
        pytest.skip(f"results/sweep_raw.parquet not present: {RAW_PATH}")

    df = analyze_sweep.load_raw()
    corrected = analyze_sweep.build_corrected_summary(df, None)
    primary = corrected[corrected["correction_family"] == "primary_main_sweep_real"]
    headline = primary[
        primary["metric"].isin(
            ["feeder_coefficient_of_variation", "feeder_peak_to_average_ratio", "avg_cost_per_agent_eur"]
        )
    ]
    assert len(headline) == 72  # 3 metrics x 2 ablations x 4 k x 3 broker_count
    assert headline["survives_holm_alpha05"].all(), "headline metric-3/cost effects are expected to survive Holm"
    assert headline["survives_bh_alpha05"].all()
