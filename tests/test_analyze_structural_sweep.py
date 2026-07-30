"""Tests for scripts/analyze_structural_sweep.py (docs/DECISIONS.md D9 analyzer).

scripts/ is not an importable package, so this module loads the script the
same way tests/test_multiple_comparison_correction.py loads
scripts/analyze_sweep.py: via importlib.util.spec_from_file_location. The
module's own main() is gated behind `if __name__ == "__main__":`, so
importing it never writes anything to disk.

Two kinds of tests:
  1. Pure statistical/aggregation helpers (validation, marginal averaging,
     the interaction/sign-flip scan, the per-coefficient verdict, the
     correction-family and bootstrap wiring) exercised with small,
     hand-constructed frames -- fast, no simulation, no network.
  2. End-to-end runs of the full pipeline against two synthetic 2880-row
     fixtures built in-memory with the exact D9 schema: one "realistic"
     fixture where the metric-3 effect never changes sign, and one
     "signflip" fixture that deliberately reproduces D9's own named
     plausible failure mode (large deferrable_fraction x tight
     payback_cap_fraction inverting the peak-to-average effect) so the test
     suite proves the analyzer actually catches it, not just that it runs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_structural_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_structural_sweep_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__name__ != "__main__"
    return module


mod = _load_module()


# ---------------------------------------------------------------------------
# Fixture builder: a full, schema-correct 2880-row synthetic frame, built
# in-memory (no simulation, no I/O). Shared by the end-to-end tests below.
# ---------------------------------------------------------------------------

_BROKERS = ("premium_green", "regulated_utility", "volatile_low_cost")


def _seed_baseline(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "avg_cost_per_agent_eur": 580.0 + rng.normal(0, 5.0),
        "avg_cost_per_kwh_eur": 0.170 + rng.normal(0, 0.002),
        "load_concentration_hhi": 0.352 + rng.normal(0, 0.003),
        "feeder_coefficient_of_variation": 0.703 + rng.normal(0, 0.004),
        "feeder_peak_to_average_ratio": 3.564 + rng.normal(0, 0.03),
        "feeder_mean_hourly_ramp_kwh": 21.2 + rng.normal(0, 0.2),
        "prosumer_self_sufficiency": 0.75 + rng.normal(0, 0.01),
    }


def _cov_pct_effect(dfrac: float, rref: float, pcap: float) -> float:
    return -(dfrac / 0.20) * (0.05 / rref) ** 0.15 * 7.2 * (0.7 + 0.3 * pcap)


def _p2a_pct_effect(dfrac: float, rref: float, pcap: float) -> float:
    return -(dfrac / 0.20) * (0.05 / rref) ** 0.1 * 9.8 * pcap


def build_synthetic_grid(flip_corner: bool) -> pd.DataFrame:
    """A schema-correct 2880-row frame. Disabled-arm rows are identical
    across all 48 cells for a given seed (the D9 materialized-baseline
    structure), split into "verification" (the 8 real 3-axis corners x the
    first 5 seeds, matching scripts/run_structural_sweep.py exactly) and
    "materialized_from_canonical_seed_run" for the rest. capacity_both rows
    are all "direct". When flip_corner is True, the peak-to-average effect is
    forced positive (worsened) at D9's own named failure corner
    (deferrable_fraction=0.30, payback_cap_fraction=0.25) across every
    response_reference level.
    """
    verification_cells = {
        (d, r, p)
        for d in (min(mod.DEFERRABLE_FRACTIONS), max(mod.DEFERRABLE_FRACTIONS))
        for r in (min(mod.RESPONSE_REFERENCES_EUR_PER_KWH), max(mod.RESPONSE_REFERENCES_EUR_PER_KWH))
        for p in (min(mod.PAYBACK_CAP_FRACTIONS), max(mod.PAYBACK_CAP_FRACTIONS))
    }
    verification_seeds = set(mod.SEEDS[:5])

    baselines = {seed: _seed_baseline(seed) for seed in mod.SEEDS}
    rows = []
    for dfrac in mod.DEFERRABLE_FRACTIONS:
        for rref in mod.RESPONSE_REFERENCES_EUR_PER_KWH:
            for pcap in mod.PAYBACK_CAP_FRACTIONS:
                is_corner = (dfrac, rref, pcap) in verification_cells
                for seed in mod.SEEDS:
                    base = baselines[seed]
                    cell_rng = np.random.default_rng(abs(hash((seed, dfrac, rref, pcap))) % (2**32))

                    disabled_row = dict(base)
                    disabled_row.update(
                        deferrable_fraction=dfrac, response_reference_eur_per_kwh=rref, payback_cap_fraction=pcap,
                        ablation="capacity_disabled", seed=seed, k=1.0, broker_count=3,
                        broker_load_share_json=json.dumps({"premium_green": 0.23, "regulated_utility": 0.35, "volatile_low_cost": 0.42}),
                        total_capacity_charge_eur=0.0, capacity_fire_rate=0.0,
                        mean_broker_surcharge_json=json.dumps({b: 0.0 for b in _BROKERS}),
                        broker_load_share_gini=0.129, total_deferred_kwh=0.0, switching_rate=0.0, n_brokers=3,
                        run_wallclock_sec=(370.0 if is_corner and seed in verification_seeds else 0.0),
                        peak_rss_mb=(130.0 if is_corner and seed in verification_seeds else float("nan")),
                        row_provenance=(
                            "verification" if is_corner and seed in verification_seeds
                            else "materialized_from_canonical_seed_run"
                        ),
                    )
                    rows.append(disabled_row)

                    cov_pct = _cov_pct_effect(dfrac, rref, pcap) + cell_rng.normal(0, 0.25)
                    p2a_pct = _p2a_pct_effect(dfrac, rref, pcap) + cell_rng.normal(0, 0.3)
                    if flip_corner and np.isclose(dfrac, mod.FLAGGED_CORNER[0]) and np.isclose(pcap, mod.FLAGGED_CORNER[1]):
                        p2a_pct = 4.0 + cell_rng.normal(0, 0.3)

                    both_row = dict(base)
                    both_row["feeder_coefficient_of_variation"] = base["feeder_coefficient_of_variation"] * (1 + cov_pct / 100.0)
                    both_row["feeder_peak_to_average_ratio"] = base["feeder_peak_to_average_ratio"] * (1 + p2a_pct / 100.0)
                    both_row["avg_cost_per_agent_eur"] = base["avg_cost_per_agent_eur"] * (1 + 0.06 * (dfrac / 0.2))
                    both_row.update(
                        deferrable_fraction=dfrac, response_reference_eur_per_kwh=rref, payback_cap_fraction=pcap,
                        ablation="capacity_both", seed=seed, k=1.0, broker_count=3,
                        broker_load_share_json=disabled_row["broker_load_share_json"],
                        total_capacity_charge_eur=6000.0 + cell_rng.normal(0, 100),
                        capacity_fire_rate=0.20 + cell_rng.normal(0, 0.01),
                        mean_broker_surcharge_json=json.dumps({b: 0.006 for b in _BROKERS}),
                        broker_load_share_gini=0.129, total_deferred_kwh=38000.0 * (dfrac / 0.2) + cell_rng.normal(0, 500),
                        switching_rate=0.006, n_brokers=3, row_provenance="direct",
                        run_wallclock_sec=400.0, peak_rss_mb=129.0,
                    )
                    rows.append(both_row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def realistic_grid() -> pd.DataFrame:
    return build_synthetic_grid(flip_corner=False)


@pytest.fixture(scope="module")
def signflip_grid() -> pd.DataFrame:
    return build_synthetic_grid(flip_corner=True)


# ---------------------------------------------------------------------------
# validate_structural_sweep
# ---------------------------------------------------------------------------


def test_validate_structural_sweep_passes_on_well_formed_grid(realistic_grid):
    assert mod.validate_structural_sweep(realistic_grid) == []


def test_validate_structural_sweep_flags_wrong_row_count(realistic_grid):
    truncated = realistic_grid.iloc[:-1]
    problems = mod.validate_structural_sweep(truncated)
    assert any("2880" in p for p in problems)
    assert any("missing" in p and "cells" in p for p in problems)


def test_validate_structural_sweep_flags_duplicate_tuple(realistic_grid):
    duplicated = pd.concat([realistic_grid, realistic_grid.iloc[[0]]], ignore_index=True)
    problems = mod.validate_structural_sweep(duplicated)
    assert any("duplicate" in p for p in problems)


def test_validate_structural_sweep_flags_bad_axis_value(realistic_grid):
    bad = realistic_grid.copy()
    bad.loc[bad.index[0], "deferrable_fraction"] = 0.15  # not one of the 4 swept levels
    problems = mod.validate_structural_sweep(bad)
    assert any("deferrable_fraction values" in p for p in problems)


def test_validate_structural_sweep_flags_bad_k(realistic_grid):
    bad = realistic_grid.copy()
    bad.loc[bad.index[0], "k"] = 0.5
    problems = mod.validate_structural_sweep(bad)
    assert any(p.startswith("k must be") for p in problems)


def test_validate_structural_sweep_flags_bad_seed(realistic_grid):
    bad = realistic_grid.copy()
    bad.loc[bad.index[0], "seed"] = 999999
    problems = mod.validate_structural_sweep(bad)
    assert any("seed set does not match" in p for p in problems)


def test_validate_structural_sweep_flags_invalid_provenance(realistic_grid):
    bad = realistic_grid.copy()
    disabled_idx = bad.index[bad["ablation"] == "capacity_disabled"][0]
    bad.loc[disabled_idx, "row_provenance"] = "bogus_value"
    problems = mod.validate_structural_sweep(bad)
    assert any("invalid row_provenance" in p for p in problems)


def test_validate_structural_sweep_flags_capacity_both_not_direct(realistic_grid):
    bad = realistic_grid.copy()
    both_idx = bad.index[bad["ablation"] == "capacity_both"][0]
    bad.loc[both_idx, "row_provenance"] = "verification"
    problems = mod.validate_structural_sweep(bad)
    assert any("capacity_both rows must all have row_provenance" in p for p in problems)


def test_validate_structural_sweep_flags_missing_columns():
    bad = pd.DataFrame({"deferrable_fraction": [0.05]})
    problems = mod.validate_structural_sweep(bad)
    assert any("missing expected columns" in p for p in problems)


# ---------------------------------------------------------------------------
# provenance_breakdown / disabled_arm_within_seed_spread
# ---------------------------------------------------------------------------


def test_provenance_breakdown_matches_expected_counts(realistic_grid):
    df = mod._snap_axes(realistic_grid)
    breakdown = mod.provenance_breakdown(df)
    counts = dict(zip(zip(breakdown["ablation"], breakdown["row_provenance"]), breakdown["n_rows"]))
    assert counts[("capacity_both", "direct")] == 1440
    assert counts[("capacity_disabled", "verification")] == 40
    assert counts[("capacity_disabled", "materialized_from_canonical_seed_run")] == 1400


def test_disabled_arm_within_seed_spread_is_zero_on_well_formed_grid(realistic_grid):
    df = mod._snap_axes(realistic_grid)
    spread = mod.disabled_arm_within_seed_spread(df)
    assert float(spread.to_numpy().max()) == pytest.approx(0.0, abs=1e-9)


def test_disabled_arm_within_seed_spread_detects_a_planted_divergence(realistic_grid):
    df = mod._snap_axes(realistic_grid).copy()
    disabled_mask = df["ablation"] == "capacity_disabled"
    one_row = df.index[disabled_mask][0]
    df.loc[one_row, "feeder_coefficient_of_variation"] += 0.5  # plant a divergence for one seed/cell
    spread = mod.disabled_arm_within_seed_spread(df)
    assert float(spread["feeder_coefficient_of_variation"].max()) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# _marginal_arrays: hand-computable averaging correctness
# ---------------------------------------------------------------------------


def test_marginal_arrays_averages_over_the_other_two_axes_correctly():
    seeds = mod.SEEDS
    cells = [(0.05, 0.0125), (0.30, 0.20)]  # both share payback_cap_fraction = 0.5
    both_value_by_cell = {cells[0]: 10.0, cells[1]: 20.0}
    rows = []
    for dfrac, rref in cells:
        for ablation, value in (("capacity_both", both_value_by_cell[(dfrac, rref)]), ("capacity_disabled", 0.0)):
            for seed in seeds:
                rows.append(
                    {
                        "deferrable_fraction": dfrac,
                        "response_reference_eur_per_kwh": rref,
                        "payback_cap_fraction": 0.5,
                        "ablation": ablation,
                        "seed": seed,
                        "metric_x": value,
                    }
                )
    df = mod._snap_axes(pd.DataFrame(rows))
    both_vals, disabled_vals = mod._marginal_arrays(df, "payback_cap_fraction", 0.5, "metric_x")
    assert both_vals == pytest.approx(np.full(30, 15.0))  # (10 + 20) / 2
    assert disabled_vals == pytest.approx(np.zeros(30))


def test_marginal_arrays_raises_if_a_seed_is_missing_for_the_slice():
    seeds = mod.SEEDS[:-1]  # deliberately drop one seed
    rows = []
    for ablation, value in (("capacity_both", 10.0), ("capacity_disabled", 0.0)):
        for seed in seeds:
            rows.append(
                {
                    "deferrable_fraction": 0.05,
                    "response_reference_eur_per_kwh": 0.0125,
                    "payback_cap_fraction": 0.5,
                    "ablation": ablation,
                    "seed": seed,
                    "metric_x": value,
                }
            )
    df = mod._snap_axes(pd.DataFrame(rows))
    with pytest.raises(AssertionError):
        mod._marginal_arrays(df, "payback_cap_fraction", 0.5, "metric_x")


# ---------------------------------------------------------------------------
# interaction_report: sign-flip detection
# ---------------------------------------------------------------------------


def _synthetic_cell_df(diff_fn) -> pd.DataFrame:
    """A 48-row cell_df stand-in for feeder_peak_to_average_ratio only, with
    paired_mean_diff set by diff_fn(dfrac, rref, pcap)."""
    rows = []
    for dfrac in mod.DEFERRABLE_FRACTIONS:
        for rref in mod.RESPONSE_REFERENCES_EUR_PER_KWH:
            for pcap in mod.PAYBACK_CAP_FRACTIONS:
                diff = diff_fn(dfrac, rref, pcap)
                rows.append(
                    {
                        "metric": "feeder_peak_to_average_ratio",
                        "deferrable_fraction": dfrac,
                        "response_reference_eur_per_kwh": rref,
                        "payback_cap_fraction": pcap,
                        "paired_mean_diff": diff,
                        "paired_mean_pct_change": diff,
                        "p_holm": 0.001,
                        "worsened_vs_disabled": diff > 0,
                    }
                )
    return pd.DataFrame(rows)


def test_interaction_report_detects_no_flip_when_everything_improves():
    cell_df = _synthetic_cell_df(lambda d, r, p: -1.0)
    report = mod.interaction_report(cell_df)
    assert report["any_flip_anywhere"] is False
    assert report["flip_at_flagged_corner"] is False
    assert len(report["all_flip_cells"]) == 0


def test_interaction_report_detects_flip_confined_to_the_flagged_corner():
    def diff_fn(d, r, p):
        is_flagged = np.isclose(d, mod.FLAGGED_CORNER[0]) and np.isclose(p, mod.FLAGGED_CORNER[1])
        return 1.0 if is_flagged else -1.0

    cell_df = _synthetic_cell_df(diff_fn)
    report = mod.interaction_report(cell_df)
    assert report["any_flip_anywhere"] is True
    assert report["flip_at_flagged_corner"] is True
    # exactly the 4 response_reference levels at the flagged (dfrac, pcap) corner
    assert len(report["all_flip_cells"]) == len(mod.RESPONSE_REFERENCES_EUR_PER_KWH)
    assert len(report["corner_flip_cells"]) == len(mod.RESPONSE_REFERENCES_EUR_PER_KWH)


def test_interaction_report_detects_flip_elsewhere_not_at_the_flagged_corner():
    def diff_fn(d, r, p):
        # flip only at the SMALLEST deferrable_fraction / loosest payback cap,
        # nowhere near D9's flagged corner
        is_other_corner = np.isclose(d, min(mod.DEFERRABLE_FRACTIONS)) and np.isclose(p, max(mod.PAYBACK_CAP_FRACTIONS))
        return 1.0 if is_other_corner else -1.0

    cell_df = _synthetic_cell_df(diff_fn)
    report = mod.interaction_report(cell_df)
    assert report["any_flip_anywhere"] is True
    assert report["flip_at_flagged_corner"] is False


# ---------------------------------------------------------------------------
# build_verdict_table
# ---------------------------------------------------------------------------


def _make_marginal_row(coefficient, level, metric, diff, survives_holm):
    return {
        "coefficient": coefficient,
        "coefficient_level": level,
        "metric": metric,
        "paired_mean_diff": diff,
        "paired_mean_pct_change": diff,
        "survives_holm_alpha05": survives_holm,
    }


def test_verdict_is_robust_when_sign_significance_and_cells_all_agree():
    metric = "feeder_peak_to_average_ratio"
    marginal_rows = [_make_marginal_row("deferrable_fraction", lvl, metric, -1.0, True) for lvl in mod.DEFERRABLE_FRACTIONS]
    marginal_rows += [_make_marginal_row(c, lvl, m, -1.0, True) for c, levels in mod.COEFFICIENT_LEVELS.items()
                       for lvl in levels for m in mod.METRIC3_KEYS if not (c == "deferrable_fraction" and m == metric)]
    marginal_df = pd.DataFrame(marginal_rows)
    cell_df = _synthetic_cell_df(lambda d, r, p: -1.0)
    cell_df["response_reference_eur_per_kwh"] = cell_df["response_reference_eur_per_kwh"].astype(float)
    other_metric_cells = cell_df.copy()
    other_metric_cells["metric"] = "feeder_coefficient_of_variation"
    full_cell_df = pd.concat([cell_df, other_metric_cells], ignore_index=True)

    verdict_df = mod.build_verdict_table(full_cell_df, marginal_df)
    row = verdict_df[(verdict_df["coefficient"] == "deferrable_fraction") & (verdict_df["metric"] == metric)].iloc[0]
    assert row["verdict"] == "ROBUST across the swept range"
    # values retrieved via DataFrame.iloc come back as numpy.bool_, so assert
    # truthiness rather than Python "is" identity against the plain bool True/False.
    assert row["sign_consistent_across_marginal_levels"]
    assert row["cell_level_sign_consistent_within_every_level"]


def test_verdict_flags_marginal_sign_change_as_fragile():
    metric = "feeder_peak_to_average_ratio"
    diffs = [-1.0, -1.0, -1.0, 1.0]  # last level flips sign at the MARGINAL level itself
    marginal_rows = [
        _make_marginal_row("deferrable_fraction", lvl, metric, diff, True)
        for lvl, diff in zip(mod.DEFERRABLE_FRACTIONS, diffs)
    ]
    marginal_rows += [_make_marginal_row(c, lvl, m, -1.0, True) for c, levels in mod.COEFFICIENT_LEVELS.items()
                       for lvl in levels for m in mod.METRIC3_KEYS if not (c == "deferrable_fraction" and m == metric)]
    marginal_df = pd.DataFrame(marginal_rows)
    cell_df = _synthetic_cell_df(lambda d, r, p: -1.0)
    other_metric_cells = cell_df.copy()
    other_metric_cells["metric"] = "feeder_coefficient_of_variation"
    full_cell_df = pd.concat([cell_df, other_metric_cells], ignore_index=True)

    verdict_df = mod.build_verdict_table(full_cell_df, marginal_df)
    row = verdict_df[(verdict_df["coefficient"] == "deferrable_fraction") & (verdict_df["metric"] == metric)].iloc[0]
    assert "FRAGILE" in row["verdict"]
    assert not row["sign_consistent_across_marginal_levels"]


def test_verdict_flags_interaction_fragility_when_marginal_stays_robust():
    # Marginal effect for deferrable_fraction is negative (improving) at every
    # level, but the underlying cells at deferrable_fraction=0.30 disagree in
    # sign -- exactly the dilution scenario the interaction check exists for.
    metric = "feeder_peak_to_average_ratio"
    marginal_rows = [_make_marginal_row("deferrable_fraction", lvl, metric, -1.0, True) for lvl in mod.DEFERRABLE_FRACTIONS]
    marginal_rows += [_make_marginal_row(c, lvl, m, -1.0, True) for c, levels in mod.COEFFICIENT_LEVELS.items()
                       for lvl in levels for m in mod.METRIC3_KEYS if not (c == "deferrable_fraction" and m == metric)]
    marginal_df = pd.DataFrame(marginal_rows)

    def diff_fn(d, r, p):
        if np.isclose(d, max(mod.DEFERRABLE_FRACTIONS)) and np.isclose(p, min(mod.PAYBACK_CAP_FRACTIONS)):
            return 1.0  # worsens at this corner only
        return -1.0

    cell_df = _synthetic_cell_df(diff_fn)
    other_metric_cells = _synthetic_cell_df(lambda d, r, p: -1.0)
    other_metric_cells["metric"] = "feeder_coefficient_of_variation"
    full_cell_df = pd.concat([cell_df, other_metric_cells], ignore_index=True)

    verdict_df = mod.build_verdict_table(full_cell_df, marginal_df)
    row = verdict_df[(verdict_df["coefficient"] == "deferrable_fraction") & (verdict_df["metric"] == metric)].iloc[0]
    assert row["sign_consistent_across_marginal_levels"]  # marginal alone would call this robust
    assert not row["cell_level_sign_consistent_within_every_level"]
    assert "INTERACTION-FRAGILE" in row["verdict"]


# ---------------------------------------------------------------------------
# _apply_correction_family / _apply_bootstrap wiring
# ---------------------------------------------------------------------------


def test_apply_correction_family_asserts_on_wrong_size():
    rows = [{"p_uncorrected": 0.01, "degenerate": False}]
    with pytest.raises(AssertionError):
        mod._apply_correction_family(rows, "test_family", family_size=2)


def test_apply_correction_family_matches_hand_computed_holm_and_survival():
    # p = [0.01, 0.02, 0.03, 0.04, 0.5]; Holm-adjusted (see
    # tests/test_multiple_comparison_correction.py's identical hand-worked
    # example): [0.05, 0.08, 0.09, 0.09, 0.5]
    pvals = [0.01, 0.02, 0.03, 0.04, 0.5]
    rows = [{"p_uncorrected": p, "degenerate": False} for p in pvals]
    mod._apply_correction_family(rows, "test_family", family_size=5)
    holm = [r["p_holm"] for r in rows]
    assert holm == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.5], abs=1e-12)
    # only p_holm=0.05 is <= alpha=0.05; the rest (0.08, 0.09, 0.09, 0.5) do not survive
    assert [r["survives_holm_alpha05"] for r in rows] == [True, False, False, False, False]
    assert all(r["correction_family"] == "test_family" and r["correction_family_size"] == 5 for r in rows)


def test_apply_correction_family_degenerate_row_never_survives_even_with_small_p():
    # A degenerate row should never be reported as surviving, regardless of
    # what its (nominally meaningless) p_uncorrected happens to be.
    rows = [{"p_uncorrected": 1e-12, "degenerate": True}, {"p_uncorrected": 0.9, "degenerate": False}]
    mod._apply_correction_family(rows, "test_family", family_size=2)
    assert rows[0]["survives_holm_alpha05"] is False
    assert rows[0]["survives_bh_alpha05"] is False


def test_apply_bootstrap_is_reproducible_given_the_same_seed():
    rows_a = [{"ci95_t_lo": -1.0, "ci95_t_hi": 1.0, "_diff": np.array([1.0, -1.0, 0.5, -0.5, 2.0] * 6)} for _ in range(3)]
    rows_b = [{"ci95_t_lo": -1.0, "ci95_t_hi": 1.0, "_diff": np.array([1.0, -1.0, 0.5, -0.5, 2.0] * 6)} for _ in range(3)]
    mod._apply_bootstrap(rows_a, np.random.default_rng(20260704))
    mod._apply_bootstrap(rows_b, np.random.default_rng(20260704))
    for a, b in zip(rows_a, rows_b):
        assert a["ci95_boot_lo"] == pytest.approx(b["ci95_boot_lo"])
        assert a["ci95_boot_hi"] == pytest.approx(b["ci95_boot_hi"])
    assert all("_diff" not in r for r in rows_a)  # popped, not left behind


# ---------------------------------------------------------------------------
# shipped_cell_anchor: gracefully absent when the main sweep parquet is missing
# ---------------------------------------------------------------------------


def test_shipped_cell_anchor_returns_none_when_file_absent(tmp_path):
    assert mod.shipped_cell_anchor(tmp_path / "does_not_exist.parquet") is None


def test_shipped_cell_anchor_reads_real_main_sweep_if_present():
    if not mod.MAIN_SWEEP_RAW_PATH.exists():
        pytest.skip(f"{mod.MAIN_SWEEP_RAW_PATH} not present on this checkout")
    anchor = mod.shipped_cell_anchor()
    assert anchor is not None
    assert anchor["n_seeds"] == 30
    for metric in mod.METRIC3_KEYS:
        assert np.isfinite(anchor[f"{metric}_paired_mean_pct_change"])


# ---------------------------------------------------------------------------
# End-to-end: the two required fixtures, run through the real pipeline
# ---------------------------------------------------------------------------


def test_end_to_end_missing_parquet_is_graceful_not_a_traceback(tmp_path, capsys):
    result = mod.run_analysis(tmp_path / "nope.parquet", tmp_path / "out")
    assert result == {"ok": False}
    captured = capsys.readouterr()
    assert "not found" in captured.err
    assert "has not finished yet" in captured.err


def test_end_to_end_realistic_fixture_is_fully_robust(tmp_path, realistic_grid):
    parquet_path = tmp_path / "realistic.parquet"
    realistic_grid.to_parquet(parquet_path)
    result = mod.run_analysis(parquet_path, tmp_path / "out")

    assert result["ok"] is True
    assert result["csv_path"].exists()
    assert result["fig_path"].exists()
    assert len(result["cell_df"]) == 96
    assert len(result["marginal_df"]) == 22
    assert len(result["verdict_df"]) == 6

    assert result["interaction"]["any_flip_anywhere"] is False
    assert result["interaction"]["flip_at_flagged_corner"] is False
    assert (result["verdict_df"]["verdict"] == "ROBUST across the swept range").all()


def test_end_to_end_signflip_fixture_is_detected_and_reported_loudly(tmp_path, signflip_grid, capsys):
    parquet_path = tmp_path / "signflip.parquet"
    signflip_grid.to_parquet(parquet_path)
    result = mod.run_analysis(parquet_path, tmp_path / "out")

    assert result["ok"] is True
    interaction = result["interaction"]
    assert interaction["any_flip_anywhere"] is True
    assert interaction["flip_at_flagged_corner"] is True
    # exactly the 4 planted cells (one per response_reference level) at the flagged corner
    assert len(interaction["corner_flip_cells"]) == len(mod.RESPONSE_REFERENCES_EUR_PER_KWH)

    verdict_df = result["verdict_df"]
    p2a_verdicts = verdict_df[verdict_df["metric"] == "feeder_peak_to_average_ratio"]["verdict"]
    assert p2a_verdicts.str.contains("INTERACTION-FRAGILE").all()
    # the companion metric (CoV), never perturbed by the fixture, must stay robust
    cov_verdicts = verdict_df[verdict_df["metric"] == "feeder_coefficient_of_variation"]["verdict"]
    assert (cov_verdicts == "ROBUST across the swept range").all()

    captured = capsys.readouterr()
    assert "SIGN FLIP DETECTED AT THE D9-FLAGGED CORNER" in captured.out


def test_end_to_end_csv_columns_match_declared_schema(tmp_path, realistic_grid):
    parquet_path = tmp_path / "realistic.parquet"
    realistic_grid.to_parquet(parquet_path)
    result = mod.run_analysis(parquet_path, tmp_path / "out")
    on_disk = pd.read_csv(result["csv_path"])
    assert list(on_disk.columns) == mod.CSV_COLUMNS
    assert set(on_disk["scope"].unique()) == {"cell_full_factorial", "marginal_per_coefficient", "coefficient_verdict"}
