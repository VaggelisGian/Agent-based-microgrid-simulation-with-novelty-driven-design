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
# The numbers the family-size assertions are checked against, written out here
# LITERALLY rather than imported from scripts/analyze_sweep.py.
#
# Those assertions need an expectation that does not come from the module under
# test. Comparing len(primary) against analyze_sweep.FAMILY_SIZE_PRIMARY
# compares the module's output against the module's own constant, so a wrong
# constant and a wrong row count agree with each other and the test still
# passes.
#
# The constants below are NOT all the same kind of thing, and an earlier
# version of this comment said they were: it claimed they all "come from
# docs/DECISIONS.md D8 and scripts/run_sensitivity_sweep.py's grid, i.e. from
# the experiment's definition". That is true of five of them and false of
# three, so it is split out here per constant. (docs/DECISIONS.md is this
# repo's local, untracked design log; it is deliberately not in the tracked
# tree, so a fresh clone will not have it.)
#
# FROM THE EXPERIMENT'S DEFINITION. GRID_K_VALUES, GRID_BROKER_COUNTS,
# GRID_MONOPOLY_BROKER_COUNT, GRID_REAL_CONTRASTS, GRID_DEGENERATE_CONTRASTS.
# D8 fixes the four k values, the three broker counts and the four ablations;
# the Phase 8 addendum fixes broker_count=1 for the monopoly arm; and which of
# the three non-disabled contrasts is a deterministic identity rather than an
# estimated test follows from D6's channel definition, not from an analysis
# choice. Drift in scripts/analyze_sweep.py away from any of these fails
# against a number the experiment really does pin.
#
# AN ANALYSIS-DESIGN CHOICE THE EXPERIMENT DOES NOT FIX. GRID_METRICS,
# GRID_METRIC3_KEYS, GRID_MONOPOLY_CONTRASTS. D8 describes eight metric columns
# per run (see ROW_COLUMNS in scripts/run_sensitivity_sweep.py: cost per agent
# AND per kWh, load concentration AND broker shares, CoV, peak-to-average AND
# mean ramp, self-sufficiency), and nothing in the experiment's definition says
# five of them make up the corrected family. D8 lists three metric-3
# sub-measures, and dropping the mean ramp is not something the experiment
# asks for. D8's Phase 8 addendum ran the monopoly arm over all four ablations
# (480 runs), and restricting the corrected monopoly family to
# capacity_pricing_only is not something the experiment asks for either. All
# three are choices scripts/analyze_sweep.py makes for itself: METRICS
# (analyze_sweep.py:155-161), METRIC3_KEYS (:162), and the monopoly
# restriction argued in the family-size comment at :194-203 and applied at
# :1145-1153.
#
# So those three tuples are a deliberate independent DUPLICATE of an analysis
# design choice, not a derivation of it. There is no derivation to be had, and
# inventing one would be dishonest; duplicating rather than importing IS the
# point, because it means a silent change to the analysis design fails here
# instead of passing unnoticed.
#
# What the duplication buys, concretely, since that is the justification and
# not the tidiness: a CONSISTENT drift now fails. Moving analyze_sweep to four
# metrics and updating its own FAMILY_SIZE_* constants to 96/144 to match keeps
# every check internal to that module happy, and is caught here at the
# row-count assertion, because the expected count is computed from these tuples
# and not from the module. The previous form of this file, which took the
# metric list from the module under test, structurally could not catch that.
#
# What it does NOT catch, stated so the next reader does not overestimate it: a
# monopoly family built by some other route that still totals 8 rows passes.
# These tuples pin the product, not the construction.
# ---------------------------------------------------------------------------

GRID_K_VALUES = (0.5, 1.0, 1.5, 2.0)  # 4 scarcity levels
GRID_BROKER_COUNTS = (2, 3, 5)  # 3 competitive broker counts ("the sweep" per D8)
GRID_MONOPOLY_BROKER_COUNT = 1  # the Phase 8 supplementary arm, outside D8's sweep
# Analysis design, duplicating analyze_sweep.METRICS' keys and their order: the
# five of the run's eight recorded metrics that enter the corrected family.
GRID_METRICS = (
    "avg_cost_per_agent_eur",
    "load_concentration_hhi",
    "feeder_peak_to_average_ratio",
    "feeder_coefficient_of_variation",
    "prosumer_self_sufficiency",
)  # 5
# Analysis design, duplicating analyze_sweep.METRIC3_KEYS: two of the three
# metric-3 stability sub-measures D8 names, the mean ramp being left out there.
GRID_METRIC3_KEYS = (
    "feeder_peak_to_average_ratio",
    "feeder_coefficient_of_variation",
)  # 2

# Each non-disabled ablation contributes one contrast against capacity_disabled.
# capacity_pnl_only is split out because the P&L channel debits a ledger and
# writes nothing into any price or physical quantity (D6), so its paired
# difference is exactly 0 on every metric for every seed: a deterministic
# identity, excluded from the primary family and reported separately.
GRID_REAL_CONTRASTS = ("capacity_pricing_only", "capacity_both")  # 2
GRID_DEGENERATE_CONTRASTS = ("capacity_pnl_only",)  # 1
# Analysis design, duplicating the restriction analyze_sweep.py argues at
# :194-203 and applies at :1145-1153: the monopoly secondary family quotes only
# capacity_pricing_only vs disabled, though the arm itself ran all four
# ablations.
GRID_MONOPOLY_CONTRASTS = ("capacity_pricing_only",)  # 1

# Derived family sizes: written as explicit products so the multiplication can
# be checked by eye against the grid above. Every factor is a length of one of
# those constants, including the monopoly arm's single-valued broker_count
# axis, so there is no bare number in any of the three products.
EXPECTED_PRIMARY_SIZE = (
    len(GRID_REAL_CONTRASTS) * len(GRID_METRICS) * len(GRID_K_VALUES) * len(GRID_BROKER_COUNTS)
)  # 2 x 5 x 4 x 3 = 120
EXPECTED_DEGENERATE_SIZE = (
    len(GRID_DEGENERATE_CONTRASTS) * len(GRID_METRICS) * len(GRID_K_VALUES) * len(GRID_BROKER_COUNTS)
)  # 1 x 5 x 4 x 3 = 60
EXPECTED_ALTERNATIVE_SIZE = EXPECTED_PRIMARY_SIZE + EXPECTED_DEGENERATE_SIZE  # 120 + 60 = 180
EXPECTED_MONOPOLY_SECONDARY_SIZE = (
    len(GRID_MONOPOLY_CONTRASTS)
    * len(GRID_METRIC3_KEYS)
    * len(GRID_K_VALUES)
    * len((GRID_MONOPOLY_BROKER_COUNT,))
)  # 1 x 2 x 4 x 1 broker_count = 8


def test_grid_derived_family_sizes_match_the_numbers_the_thesis_reports():
    """Pin the three derived sizes to the numbers this thesis actually ships.

    This lived as a bare module-level `assert ... == (120, 60, 8)` with no
    message. It ran at import, so any legitimate future change to the grid
    tuples above turned every test in this file into a single unexplained
    collection error. A message alone would not have fixed that, so it is a
    named test instead: a grid change now fails once, here, by name, and the
    rest of the file still runs and reports.
    """
    assert (EXPECTED_PRIMARY_SIZE, EXPECTED_DEGENERATE_SIZE, EXPECTED_MONOPOLY_SECONDARY_SIZE) == (120, 60, 8), (
        f"the grid tuples above now derive {EXPECTED_PRIMARY_SIZE} primary, {EXPECTED_DEGENERATE_SIZE} "
        f"degenerate and {EXPECTED_MONOPOLY_SECONDARY_SIZE} monopoly-secondary tests, not the 120/60/8 "
        "this thesis reports throughout. If the grid or the analysis design genuinely changed, update "
        "these numbers together with everything that quotes them; if neither changed, one of the tuples "
        "above is wrong"
    )


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
    sample_mean = float(np.mean(diff))
    rng = np.random.default_rng(20260704)
    lo, hi = analyze_sweep.bootstrap_paired_diff_ci(diff, rng, n_boot=10000)
    assert lo < hi
    assert hi < 0.0  # excludes zero: a real, consistent effect should show up

    # The assertion with teeth: a PERCENTILE bootstrap CI of the mean brackets
    # the SAMPLE mean of the data it resampled, STRICTLY, as long as that data
    # has nonzero variance. Every resample is drawn with replacement from
    # `diff` itself, so the resample means concentrate around sample_mean; the
    # 2.5th percentile lies below it and the 97.5th above it.
    #
    # The non-degeneracy qualifier is load-bearing, not pedantry. On a
    # zero-variance input every resample mean is the same number, so
    # lo == hi == sample_mean and the strict inequality is FALSE. That is not a
    # hypothetical input: it is exactly what
    # test_bootstrap_ci_degenerate_all_zero_diff_is_zero_width_at_zero above
    # feeds this function (np.zeros(30)), and it is the real shape of the 60
    # deterministic capacity_pnl_only rows the same module bootstraps in
    # build_corrected_summary. `diff` here is 30 draws from N(-2.0, 0.5), so
    # the qualifier holds and the assertion is meaningful.
    #
    # This replaces a disjunct that had no teeth at all:
    #     assert lo < -2.0 < hi or abs(np.mean(diff) - (-2.0)) < 0.5
    # whose second branch only says "30 draws from N(-2.0, 0.5) have a sample
    # mean within 0.5 of -2.0", a fact about the GENERATED DATA at this fixed
    # seed (it is 0.029 away) that holds no matter what the function returns.
    # Because that branch is unconditionally true, the whole `or` was
    # unconditionally true: stubbing the return to (-1e-6, -1e-9) still passed
    # every assertion in this test.
    #
    # Which assertion is principled and which one detects more faults are two
    # different questions with two different answers, and an earlier version of
    # this comment ran them together. Statistically, the sample-mean bracketing
    # is the principled check: it is a property of the estimator this function
    # implements, whereas `lo < -2.0 < hi` is a statement about the
    # data-generating process that a correct 95 percent interval is allowed to
    # miss. On fault detection the ranking is the other way round, measured
    # rather than assumed: over a 12-mutant set the -2.0 check caught a strict
    # superset of what the sample-mean check caught, and two mutants were
    # caught by it alone (the percentiles moved to a still-central 40th/60th,
    # and the resample mean taken over the wrong axis). Both are kept, each for
    # its own reason.
    #
    # Known gaps, recorded rather than left for the next reviewer to
    # rediscover. An earlier version of this comment claimed of the sample-mean
    # check that "a wrong implementation (wrong resampling unit, wrong
    # percentiles, an unrelated constant) fails it". It overclaimed: a wrong
    # resampling unit is caught only by the -2.0 check below. And three mutants
    # pass this whole file, so nothing here rules them out: substituting a
    # classical t-interval with no bootstrap at all, taking the percentiles of
    # the raw resampled VALUES instead of of the resample means, and a resample
    # that breaks the seed-level pairing. Closing those needs a different test,
    # one that pins how the interval depends on the resampling stream, not a
    # stronger assertion on this one input.
    assert lo < sample_mean < hi, f"bootstrap CI ({lo}, {hi}) must bracket the sample mean {sample_mean}"

    # Additionally true at this seed, checked numerically before being asserted
    # rather than assumed: with n=30 and sd 0.5 the CI is roughly +-0.2 wide on
    # each side of a sample mean 0.029 away from -2.0, so it also covers the
    # generating mean.
    assert lo < -2.0 < hi


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
            # Filtered on this module's own GRID_MONOPOLY_BROKER_COUNT, not on
            # analyze_sweep.MONOPOLY_BROKER_COUNT: the frame whose row count
            # the expectation below is derived for must not be selected by the
            # module under test.
            mono_only_df = combined_df[combined_df["broker_count"] == GRID_MONOPOLY_BROKER_COUNT]

    corrected = analyze_sweep.build_corrected_summary(df, mono_only_df)

    primary = corrected[corrected["correction_family"] == "primary_main_sweep_real"]
    degenerate = corrected[corrected["correction_family"] == "excluded_degenerate_main_sweep"]

    # Two separate checks, deliberately not one. First, immediately below: the
    # produced row counts against the sizes derived from the grid at the top of
    # this module. Asserting only the module's constants (as this test used to)
    # lets a wrong constant and a wrong row count agree with each other and
    # pass.
    assert len(primary) == EXPECTED_PRIMARY_SIZE, (
        f"primary family has {len(primary)} rows; grid says "
        f"{len(GRID_REAL_CONTRASTS)} contrasts x {len(GRID_METRICS)} metrics x "
        f"{len(GRID_K_VALUES)} k x {len(GRID_BROKER_COUNTS)} broker_count = {EXPECTED_PRIMARY_SIZE}"
    )
    assert len(degenerate) == EXPECTED_DEGENERATE_SIZE, (
        f"degenerate family has {len(degenerate)} rows; grid says "
        f"{len(GRID_DEGENERATE_CONTRASTS)} contrast x {len(GRID_METRICS)} metrics x "
        f"{len(GRID_K_VALUES)} k x {len(GRID_BROKER_COUNTS)} broker_count = {EXPECTED_DEGENERATE_SIZE}"
    )
    # Second: the module's own constants against those same derived sizes. An
    # earlier comment here said this pair means "drift in either direction
    # fails and names itself". It does not, and these two normally never fire:
    # build_corrected_summary already asserts its produced row counts against
    # these same constants before it returns (scripts/analyze_sweep.py:1127 for
    # FAMILY_SIZE_PRIMARY and :1139 for FAMILY_SIZE_ALTERNATIVE), so a wrong
    # constant with a right row count raises inside the module, at the
    # build_corrected_summary() call above, not here; and a constant that
    # drifts consistently with the row count trips the row-count assertions
    # above first. They are kept as a genuine backstop, not as the check that
    # names the constant-drift direction: if those internal asserts are ever
    # removed or weakened, these are the only thing left holding the constants
    # to the experiment.
    assert analyze_sweep.FAMILY_SIZE_PRIMARY == EXPECTED_PRIMARY_SIZE, (
        f"analyze_sweep.FAMILY_SIZE_PRIMARY is {analyze_sweep.FAMILY_SIZE_PRIMARY}, but the sweep grid "
        f"implies {EXPECTED_PRIMARY_SIZE}: the constant has drifted from the experiment it describes"
    )
    assert analyze_sweep.FAMILY_SIZE_ALTERNATIVE == EXPECTED_ALTERNATIVE_SIZE, (
        f"analyze_sweep.FAMILY_SIZE_ALTERNATIVE is {analyze_sweep.FAMILY_SIZE_ALTERNATIVE}, but the sweep "
        f"grid implies {EXPECTED_PRIMARY_SIZE} real + {EXPECTED_DEGENERATE_SIZE} degenerate = "
        f"{EXPECTED_ALTERNATIVE_SIZE}"
    )

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
        # Same split as above: row count against the grid-derived size, then the
        # module constant against the same number. The monopoly family is
        # narrower than the main sweep's: only capacity_pricing_only vs
        # disabled, only the two metric-3 sub-measures, only broker_count=1 (the
        # first two of those are analysis-design choices, not the experiment's;
        # see the header comment).
        assert len(mono_rows) == EXPECTED_MONOPOLY_SECONDARY_SIZE, (
            f"monopoly secondary family has {len(mono_rows)} rows; grid says "
            f"{len(GRID_MONOPOLY_CONTRASTS)} contrast x {len(GRID_METRIC3_KEYS)} metric3 sub-measures x "
            f"{len(GRID_K_VALUES)} k x {len((GRID_MONOPOLY_BROKER_COUNT,))} broker_count = "
            f"{EXPECTED_MONOPOLY_SECONDARY_SIZE}"
        )
        # Same backstop status as the two constant checks above: unreachable
        # while build_corrected_summary's own assert at
        # scripts/analyze_sweep.py:1155 stands, kept for when it does not.
        assert analyze_sweep.FAMILY_SIZE_MONOPOLY_SECONDARY == EXPECTED_MONOPOLY_SECONDARY_SIZE, (
            f"analyze_sweep.FAMILY_SIZE_MONOPOLY_SECONDARY is "
            f"{analyze_sweep.FAMILY_SIZE_MONOPOLY_SECONDARY}, but the monopoly arm's grid implies "
            f"{EXPECTED_MONOPOLY_SECONDARY_SIZE}"
        )
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
