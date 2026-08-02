"""Deliberate, explicit regenerator for the tests/golden/*.json fixtures used
by tests/test_golden_master.py.

This script is NEVER imported or run by tests/test_golden_master.py, and it
lives outside pyproject.toml's testpaths ("tests"), so pytest's collector
never even looks at it. A plain `pytest` invocation cannot trigger it: the
only way these golden files change is by a human running this script on
purpose, from the repository root:

    python scripts/regenerate_golden_master.py

Run this ONLY after confirming, by hand, that a shift in the pinned numbers
below is the intended consequence of a real change (model, config, or a
rerun of the Phase 5/6 sweep), never to make a failing golden test pass
without understanding why it failed.

What this writes:

  - tests/golden/short_deterministic_run.json: the four thesis metrics from a
    short, fixed-seed, fixed-config run under capacity_both and
    capacity_disabled (see SHORT_RUN_HORIZON_HOURS / SHORT_RUN_NUM_AGENTS /
    SHORT_RUN_SEED below; these must match the same-named constants in
    tests/test_golden_master.py or the pinned values stop corresponding to
    what that test actually runs).
  - tests/golden/summary_stats_pins.json: a field-by-field snapshot of EVERY
    data row of results/summary_stats.csv (the Phase 5/6 capacity-mechanism
    sweep), keyed by (k, broker_count, ablation, metric), plus the row count
    and every non-identity column the header carries. Reads that CSV
    read-only; if it is not present (e.g. the sweep has not been run on this
    machine) this part of the regeneration is skipped with a printed note and
    the short-run golden file is still written.
  - tests/golden/structural_sensitivity_pins.json: every data row of
    results/structural_sensitivity.csv (the D9 structural coefficient
    sensitivity analysis).
  - tests/golden/demand_source_comparison_pins.json: every data row of
    results/demand_source_comparison.csv (the Phase 12 synthetic-versus-OPSD
    headline comparison).
  - tests/golden/monopoly_comparison_pins.json: every data row of
    results/monopoly_comparison.csv (the broker_count=1 monopoly arm).
  - tests/golden/effect_sizes_pins.json: every data row of
    results/effect_sizes.csv (Chapter 1's headline figures).
  - tests/golden/summary_stats_corrected_pins.json: every data row of
    results/summary_stats_corrected.csv (Section 6.10's per-comparison
    multiple-comparison correction and bootstrap-CI detail).
  - tests/golden/family_sensitivity_pins.json: a CLAIM-BEARING SUBSET (not
    every row; see FAMILY_SENSITIVITY_IDENTITY's comment below) of
    results/family_sensitivity.csv, plus that file's total row count.
  - tests/golden/dose_matched_monopoly_pins.json: every data row of
    results/dose_matched_monopoly.csv (D11's dose-matched-monopoly arm,
    Phase 18.1), pinned whole (46 rows).
  - tests/golden/sweep_dose_matched_monopoly_parquet_cell_pins.json: per-(k,
    capacity_surcharge_divisor, ablation) cell aggregates (row count plus six
    metric means) over results/sweep_dose_matched_monopoly.parquet, the
    claim-bearing numbers behind D11 result that no CSV covers directly.

Every CSV or parquet named above is READ-ONLY input here. This script writes
nothing but tests/golden/*.json, and each derived snapshot is skipped with a
printed note, rather than failing, when its source file is absent on this
machine.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from microgrid_sim.config.loader import load_config  # noqa: E402  (path set above)
from microgrid_sim.environment.model import MicrogridModel  # noqa: E402

warnings.filterwarnings(
    "ignore", message="The use of the `seed` keyword argument is deprecated", category=FutureWarning
)

GOLDEN_DIR = _REPO_ROOT / "tests" / "golden"
_RESULTS_DIR = _REPO_ROOT / "results"
SHORT_RUN_GOLDEN_PATH = GOLDEN_DIR / "short_deterministic_run.json"
SUMMARY_STATS_GOLDEN_PATH = GOLDEN_DIR / "summary_stats_pins.json"
SUMMARY_STATS_CSV_PATH = _RESULTS_DIR / "summary_stats.csv"
STRUCTURAL_SENSITIVITY_GOLDEN_PATH = GOLDEN_DIR / "structural_sensitivity_pins.json"
STRUCTURAL_SENSITIVITY_CSV_PATH = _RESULTS_DIR / "structural_sensitivity.csv"
DEMAND_SOURCE_GOLDEN_PATH = GOLDEN_DIR / "demand_source_comparison_pins.json"
DEMAND_SOURCE_CSV_PATH = _RESULTS_DIR / "demand_source_comparison.csv"
MONOPOLY_GOLDEN_PATH = GOLDEN_DIR / "monopoly_comparison_pins.json"
MONOPOLY_CSV_PATH = _RESULTS_DIR / "monopoly_comparison.csv"
EFFECT_SIZES_GOLDEN_PATH = GOLDEN_DIR / "effect_sizes_pins.json"
EFFECT_SIZES_CSV_PATH = _RESULTS_DIR / "effect_sizes.csv"
FAMILY_SENSITIVITY_GOLDEN_PATH = GOLDEN_DIR / "family_sensitivity_pins.json"
FAMILY_SENSITIVITY_CSV_PATH = _RESULTS_DIR / "family_sensitivity.csv"
SUMMARY_STATS_CORRECTED_GOLDEN_PATH = GOLDEN_DIR / "summary_stats_corrected_pins.json"
SUMMARY_STATS_CORRECTED_CSV_PATH = _RESULTS_DIR / "summary_stats_corrected.csv"
DOSE_MATCHED_MONOPOLY_GOLDEN_PATH = GOLDEN_DIR / "dose_matched_monopoly_pins.json"
DOSE_MATCHED_MONOPOLY_CSV_PATH = _RESULTS_DIR / "dose_matched_monopoly.csv"
SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_CELL_GOLDEN_PATH = (
    GOLDEN_DIR / "sweep_dose_matched_monopoly_parquet_cell_pins.json"
)
SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH = _RESULTS_DIR / "sweep_dose_matched_monopoly.parquet"

# Must match tests/test_golden_master.py's SHORT_HORIZON_HOURS / SHORT_NUM_AGENTS
# / SHORT_SEED exactly. Horizon comfortably exceeds the 168h rolling window
# (three window-widths) so the capacity mechanism actually fires; agent count
# and horizon both sit inside the fix brief's suggested envelope.
SHORT_RUN_HORIZON_HOURS = 504
SHORT_RUN_NUM_AGENTS = 50
SHORT_RUN_SEED = 101  # fixed for reproducibility; arbitrary, not tuned

# Identity columns and pinned fields for the row-keyed CSV snapshots. Each of
# these must match the same-named list in tests/test_golden_master.py exactly,
# or the snapshot written here stops describing what those tests check.
_MISSING_TOKEN = "__NA__"

SUMMARY_STATS_IDENTITY = ("k", "broker_count", "ablation", "metric")
# Every non-identity column results/summary_stats.csv carries. The four
# identity columns plus these fifteen are the whole header, so no column of
# that file is left unpinned.
SUMMARY_STATS_PINNED_FIELDS = {
    "metric_label": "str",
    "n_seeds": "int",
    "mean": "float",
    "std": "float",
    "ci95_lo": "float",
    "ci95_hi": "float",
    "ci95_halfwidth": "float",
    "vs_disabled_cohens_d_independent": "float",
    "vs_disabled_paired_dz": "float",
    "vs_disabled_paired_p": "pvalue",
    "vs_disabled_paired_mean_diff": "float",
    "vs_disabled_paired_mean_pct_change": "float",
    "vs_disabled_degenerate_paired_diff": "bool",
    "metric3_sign_convention": "str",
    "notes": "str",
}

STRUCTURAL_SENSITIVITY_IDENTITY = (
    "scope",
    "deferrable_fraction",
    "response_reference_eur_per_kwh",
    "payback_cap_fraction",
    "coefficient",
    "coefficient_level",
    "metric",
)
STRUCTURAL_SENSITIVITY_PINNED_FIELDS = {
    "n_seeds": "int",
    "mean_disabled": "float",
    "mean_both": "float",
    "paired_mean_diff": "float",
    "paired_mean_pct_change": "float",
    "paired_dz": "float",
    "p_uncorrected": "pvalue",
    "p_holm": "pvalue",
    "p_bh": "pvalue",
    "survives_holm_alpha05": "bool",
    "survives_bh_alpha05": "bool",
    "worsened_vs_disabled": "bool",
    "ci95_t_lo": "float",
    "ci95_t_hi": "float",
    "ci95_boot_lo": "float",
    "ci95_boot_hi": "float",
    "ci_disagreement_flag": "bool",
    "correction_family": "str",
    "correction_family_size": "int",
    # The six coefficient_verdict rows' supporting evidence. The verdict string
    # alone does not carry it: the sign flip the FRAGILE verdict rests on lives
    # in min/max_pct_change_across_levels, and no verdict string mentions
    # significance at all, so significant_at_every_level_holm could flip
    # without a word of the verdict changing.
    "min_pct_change_across_levels": "float",
    "max_pct_change_across_levels": "float",
    "sign_consistent_across_marginal_levels": "bool",
    "significant_at_every_level_holm": "bool",
    "cell_level_sign_consistent_within_every_level": "bool",
    "levels_with_within_level_cell_sign_disagreement": "str",
    "verdict": "str",
}
# Constant across every row of structural_sensitivity.csv, so pinned once at
# the top level and checked against every row by the test. Same column, same
# reason as in monopoly_comparison.csv below: rewording it reverses how every
# metric-3 number in the file is read without changing any number.
STRUCTURAL_SENSITIVITY_CONSTANT_FIELDS = ("metric3_sign_convention",)

DEMAND_SOURCE_IDENTITY = ("k", "broker_count", "ablation", "metric")
DEMAND_SOURCE_PINNED_FIELDS = {
    "n_seeds_synthetic": "int",
    "n_seeds_opsd": "int",
    "mean_synthetic": "float",
    "mean_opsd": "float",
    "std_synthetic": "float",
    "std_opsd": "float",
    "delta_opsd_minus_synthetic": "float",
}

MONOPOLY_IDENTITY = ("broker_count", "k")
MONOPOLY_PINNED_FIELDS = {
    "is_monopoly": "bool",
    "n_seeds": "int",
    "feeder_peak_to_average_ratio_paired_mean_pct_change": "float",
    "feeder_peak_to_average_ratio_paired_dz": "float",
    "feeder_peak_to_average_ratio_paired_p": "pvalue",
    "feeder_peak_to_average_ratio_n_improved_of_30": "int",
    "feeder_coefficient_of_variation_paired_mean_pct_change": "float",
    "feeder_coefficient_of_variation_paired_dz": "float",
    "feeder_coefficient_of_variation_paired_p": "pvalue",
    "feeder_coefficient_of_variation_n_improved_of_30": "int",
    "capacity_both_mean_fire_rate": "float",
    "capacity_both_mean_total_deferred_kwh": "float",
    "avg_cost_per_agent_eur_paired_mean_pct_change": "float",
}
# Constant across every row of monopoly_comparison.csv, so pinned once at the
# top level and checked against every row by the test.
MONOPOLY_CONSTANT_FIELDS = ("metric3_sign_convention", "note")

EFFECT_SIZES_IDENTITY = ("block", "k", "broker_count", "ablation", "metric")
# Every non-identity column results/effect_sizes.csv carries. The five
# identity columns plus these fifteen are the whole header, so no column of
# that file is left unpinned. Must match the same-named list in
# tests/test_golden_master.py exactly.
EFFECT_SIZES_PINNED_FIELDS = {
    "metric_label": "str",
    "n_seeds": "int",
    "mean_disabled": "float",
    "mean_ablation": "float",
    "paired_mean_diff": "float",
    "paired_mean_pct_change": "float",
    "naive_pct_change_of_means": "float",
    "paired_dz": "float",
    "paired_p": "pvalue",
    "cohens_d_independent": "float",
    "degenerate_paired_diff": "bool",
    "max_abs_paired_diff": "float",
    "pnl_physical_leak": "str",
    "metric3_sign_convention": "str",
    "note": "str",
}

SUMMARY_STATS_CORRECTED_IDENTITY = ("scope", "k", "broker_count", "ablation", "metric")
# Every non-identity column results/summary_stats_corrected.csv carries. The
# five identity columns plus these twenty-eight are the whole header. Must
# match the same-named list in tests/test_golden_master.py exactly.
SUMMARY_STATS_CORRECTED_PINNED_FIELDS = {
    "metric_label": "str",
    "n_seeds": "int",
    "paired_mean_diff": "float",
    "paired_mean_pct_change": "float",
    "paired_dz": "float",
    "p_uncorrected": "pvalue",
    "p_holm": "pvalue",
    "p_bh": "pvalue",
    "survives_holm_alpha05": "bool",
    "survives_bh_alpha05": "bool",
    "ci95_t_lo": "float",
    "ci95_t_hi": "float",
    "ci95_boot_lo": "float",
    "ci95_boot_hi": "float",
    "ci_disagreement_flag": "bool",
    "ci_disagreement_detail": "str",
    "ci_width_ratio": "float",
    "degenerate": "bool",
    "correction_family": "str",
    "correction_family_size": "int",
    "family_size_primary": "int",
    "family_size_alternative": "int",
    "family_size_monopoly_secondary": "int",
    "bootstrap_seed": "int",
    "n_bootstrap": "int",
    "alpha": "float",
    "metric3_sign_convention": "str",
    "notes": "str",
}

# results/family_sensitivity.csv is 1227 rows: a whole-file digest at that
# scale would run to roughly 1.6 MB (about ten times structural_sensitivity's
# 124-row, 177891-byte snapshot), so it is pinned as a named CLAIM-BEARING
# SUBSET instead, plus the file's total row count as a shape guard. See
# FAMILY_SENSITIVITY_IDENTITY's comment in tests/test_golden_master.py for the
# full reasoning; the constants below must match that file's exactly.
FAMILY_SENSITIVITY_IDENTITY = ("family_definition", "test_id")
FAMILY_SENSITIVITY_PINNED_FIELDS = {
    "family_size": "int",
    "n_families_in_definition": "int",
    "ablation": "str",
    "metric": "str",
    "is_headline_72": "bool",
    "degenerate": "bool",
    "p_uncorrected": "pvalue",
    "p_holm": "pvalue",
    "p_bh": "pvalue",
    "survives_holm_alpha05": "bool",
    "survives_bh_alpha05": "bool",
    "alpha": "float",
    "n_headline_tests": "int",
    "n_headline_surviving_holm": "int",
    "n_headline_surviving_bh": "int",
    "worst_headline_p_uncorrected": "pvalue",
    "worst_headline_p_holm": "pvalue",
    "worst_headline_p_bh": "pvalue",
    "bonferroni_alpha_over_family_size": "float",
    "notes": "str",
}
# The explicit, named claim-bearing subset: the seven family_summary rows
# (one per family definition) plus the one real per-test row that backs
# Section 6.2's own quoted "d_z = -22.2 (p = 7.4e-41)" figure and is small
# enough in magnitude to prove the relative-only p-value rule earns its keep
# on this file too. Must match the same-named constant in
# tests/test_golden_master.py exactly.
FAMILY_SENSITIVITY_PINNED_ROW_KEYS = (
    ("F1_main_sweep_primary_120", "family_summary|F1_main_sweep_primary_120"),
    ("F2_main_sweep_with_degenerates_180", "family_summary|F2_main_sweep_with_degenerates_180"),
    ("F3_per_metric_24", "family_summary|F3_per_metric_24"),
    ("F4_pooled_with_monopoly", "family_summary|F4_pooled_with_monopoly"),
    ("F5_everything_pooled", "family_summary|F5_everything_pooled"),
    ("F6_per_cell_k_by_broker_count", "family_summary|F6_per_cell_k_by_broker_count"),
    ("F7_everything_pooled_with_degenerates", "family_summary|F7_everything_pooled_with_degenerates"),
    ("F1_main_sweep_primary_120", "main|k=1|bc=3|capacity_both|feeder_coefficient_of_variation"),
)

# results/dose_matched_monopoly.csv (D11's dose-matched-monopoly arm, Phase
# 18.1): 46 rows, small enough to pin whole rather than as a subset. Each of
# the three arm families (monopoly at divisor 1/2/3/5, broker_count 2/3/5,
# and the divisor-3-vs-broker_count-3 decisive contrast) reports at most one
# row per (scope, k, metric) combination, so (scope, arm, k, metric) is
# unique over the whole file; verified in tests/test_golden_master.py rather
# than assumed. Must match the same-named constants in that file exactly.
DOSE_MATCHED_MONOPOLY_IDENTITY = ("scope", "arm", "k", "metric")
DOSE_MATCHED_MONOPOLY_PINNED_FIELDS = {
    "arm_type": "str",
    "capacity_surcharge_divisor": "float",
    "broker_count": "int",
    "comparator_arm": "str",
    "data_source": "str",
    "n_seeds": "int",
    "paired_mean_pct_change": "float",
    "pct_point_diff_mono_minus_comparator": "float",
    "paired_mean_diff_raw": "float",
    "paired_dz": "float",
    "p_uncorrected": "pvalue",
    "p_holm": "pvalue",
    "p_bh": "pvalue",
    "survives_holm_alpha05": "bool",
    "survives_bh_alpha05": "bool",
    "ci95_t_lo": "float",
    "ci95_t_hi": "float",
    "ci95_boot_lo": "float",
    "ci95_boot_hi": "float",
    "n_favorable_of_n_seeds": "int",
    "total_deferred_kwh_capacity_both_mean": "float",
    "total_deferred_kwh_capacity_disabled_mean": "float",
    "comparator_total_deferred_kwh_capacity_both_mean": "float",
    "deferred_volume_ratio_mono_over_comparator": "float",
    "correction_family": "str",
    "correction_family_size": "int",
    "metric3_sign_convention": "str",
    "notes": "str",
}

# Per-(k, capacity_surcharge_divisor, ablation) cell aggregates over
# results/sweep_dose_matched_monopoly.parquet: the row count (30 in every one
# of the 12 cells) and the mean of the six metrics docs/DECISIONS.md's "D11
# result" section quotes (the peak-to-average table and the guardrail
# paragraph). Must match the same-named constants in
# tests/test_golden_master.py exactly.
DOSE_MATCHED_MONOPOLY_PARQUET_CELL_IDENTITY = ("k", "capacity_surcharge_divisor", "ablation")
DOSE_MATCHED_MONOPOLY_PARQUET_CELL_FIELDS = {
    "n_rows": "int",
    "mean_total_deferred_kwh": "float",
    "mean_feeder_peak_to_average_ratio": "float",
    "mean_feeder_coefficient_of_variation": "float",
    "mean_avg_cost_per_agent_eur": "float",
    "mean_capacity_fire_rate": "float",
    "mean_total_capacity_charge_eur": "float",
}


def _dose_matched_monopoly_parquet_cell_aggregates(df):
    """One row per (k, capacity_surcharge_divisor, ablation) cell of
    results/sweep_dose_matched_monopoly.parquet (12 cells, 30 seeds each).

    Shaped exactly like any other pinned CSV here (identity columns plus
    pinned fields), on purpose: the parquet's own rows are one per
    (cell, seed), not one per cell, so this aggregates first and then hands
    the result to the SAME _row_keyed_rows helper every whole-file CSV digest
    in this module already uses, rather than inventing a second snapshot
    shape. Twin of the same-named function in tests/test_golden_master.py;
    must compute identically or the golden snapshot stops describing what
    that test checks.
    """
    return df.groupby(list(DOSE_MATCHED_MONOPOLY_PARQUET_CELL_IDENTITY), as_index=False).agg(
        n_rows=("seed", "size"),
        mean_total_deferred_kwh=("total_deferred_kwh", "mean"),
        mean_feeder_peak_to_average_ratio=("feeder_peak_to_average_ratio", "mean"),
        mean_feeder_coefficient_of_variation=("feeder_coefficient_of_variation", "mean"),
        mean_avg_cost_per_agent_eur=("avg_cost_per_agent_eur", "mean"),
        mean_capacity_fire_rate=("capacity_fire_rate", "mean"),
        mean_total_capacity_charge_eur=("total_capacity_charge_eur", "mean"),
    )


_REGENERATION_COMMAND = "python scripts/regenerate_golden_master.py"


def _scenario_config(name: str) -> dict:
    config = load_config(f"config/scenarios/{name}.yaml")
    config = copy.deepcopy(config)
    config["simulation"]["horizon_hours"] = SHORT_RUN_HORIZON_HOURS
    config["simulation"]["seed"] = SHORT_RUN_SEED
    config["population"]["num_agents"] = SHORT_RUN_NUM_AGENTS
    return config


def _metrics_dict(result) -> dict:
    return {
        "avg_cost_per_agent_eur": result.avg_cost_per_agent_eur,
        "avg_cost_per_kwh_eur": result.avg_cost_per_kwh_eur,
        "broker_load_share": dict(result.broker_load_share),
        "load_concentration_hhi": result.load_concentration_hhi,
        "feeder_coefficient_of_variation": result.feeder_coefficient_of_variation,
        "feeder_peak_to_average_ratio": result.feeder_peak_to_average_ratio,
        "feeder_mean_hourly_ramp_kwh": result.feeder_mean_hourly_ramp_kwh,
        "prosumer_self_sufficiency": result.prosumer_self_sufficiency,
    }


def _compute_short_run_golden() -> dict:
    golden = {
        "_comment": (
            "Golden-master pins for tests/test_golden_master.py's short deterministic "
            "run (item 1 of the harness). Regenerate DELIBERATELY, never automatically, "
            f"by running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "scenario_horizon_hours": SHORT_RUN_HORIZON_HOURS,
        "scenario_num_agents": SHORT_RUN_NUM_AGENTS,
        "scenario_seed": SHORT_RUN_SEED,
    }
    for name in ("capacity_both", "capacity_disabled"):
        model = MicrogridModel(_scenario_config(name))
        model.run()
        golden[name] = _metrics_dict(model.compute_metrics())
    return golden


def _plain(value):
    """A numpy/pandas cell as a plain Python object, with missing as None."""
    if value is None:
        return None
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _identity_token(value) -> str:
    """Deterministic, collision-free ordering token for one identity cell.

    Missing gets an explicit token of its own, kept distinct from the literal
    string "ALL" that some of these columns use for a genuinely aggregated
    marginal row. Used only to sort the snapshot's rows into a stable order;
    the tests key on the identity values themselves.
    """
    plain = _plain(value)
    if plain is None:
        return _MISSING_TOKEN
    if isinstance(plain, str):
        return plain
    return repr(plain)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("True", "False"):
        return value == "True"
    raise ValueError(f"not a boolean literal: {value!r}")


def _pinned_value(raw, kind: str):
    value = _plain(raw)
    if value is None:
        return None
    # "pvalue" is stored exactly like "float"; the two differ only in how
    # tests/test_golden_master.py COMPARES them (p-values get a relative-only
    # tolerance, because an absolute floor of 1e-9 cannot tell 1e-55 from
    # 1e-20). Keeping the stored value identical is why splitting the kind out
    # does not change these snapshots.
    if kind in ("float", "pvalue"):
        return float(value)
    if kind == "int":
        return int(value)
    if kind == "bool":
        return _as_bool(value)
    return str(value)


def _row_keyed_rows(df, identity_columns, pinned_fields) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        entry = {column: _plain(row[column]) for column in identity_columns}
        for field, kind in pinned_fields.items():
            entry[field] = _pinned_value(row[field], kind)
        rows.append(entry)
    rows.sort(key=lambda entry: tuple(_identity_token(entry[column]) for column in identity_columns))
    return rows


def _constant_field_pins(df, constant_fields, csv_path: Path) -> dict:
    """Top-level pins for columns that are constant across a whole CSV.

    Stored once rather than per row, and checked against every row by the
    test, so a value that is meant to describe the whole file cannot start
    disagreeing with itself unnoticed either.
    """
    pins = {}
    for field in constant_fields:
        distinct = sorted({str(_plain(value)) for value in df[field]})
        if len(distinct) != 1:
            print(f"note: {field} is not constant across {csv_path}; pinning the first value only")
        pins[field] = str(_plain(df[field].iloc[0]))
    return pins


def _compute_summary_stats_pins() -> dict | None:
    if not SUMMARY_STATS_CSV_PATH.is_file():
        print(f"note: {SUMMARY_STATS_CSV_PATH} not present; skipping summary_stats_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(SUMMARY_STATS_CSV_PATH)
    return {
        "_comment": (
            "Pinned digest of results/summary_stats.csv (the Phase 5/6 capacity-mechanism "
            "sweep): every data row and every non-identity column, keyed by "
            "(k, broker_count, ablation, metric), plus the row count. That key is unique "
            "over the whole file because the sweep emits exactly one row per "
            "(k, broker_count, ablation, metric) combination. results/summary_stats.csv is "
            "read-only input here and is never written by this script; only this pinned "
            "snapshot is written. Regenerate DELIBERATELY, never automatically, only after "
            f"a verified intentional sweep rerun, by running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "n_rows": int(len(df)),
        "rows": _row_keyed_rows(df, SUMMARY_STATS_IDENTITY, SUMMARY_STATS_PINNED_FIELDS),
    }


def _compute_structural_sensitivity_pins() -> dict | None:
    if not STRUCTURAL_SENSITIVITY_CSV_PATH.is_file():
        print(f"note: {STRUCTURAL_SENSITIVITY_CSV_PATH} not present; skipping structural_sensitivity_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(STRUCTURAL_SENSITIVITY_CSV_PATH)
    pins = {
        "_comment": (
            "Pinned digest of results/structural_sensitivity.csv (the D9 structural "
            "coefficient sensitivity analysis): every data row, keyed by "
            "(scope, deferrable_fraction, response_reference_eur_per_kwh, "
            "payback_cap_fraction, coefficient, coefficient_level, metric), plus the "
            "row count. An empty identity cell (a coefficient_verdict row has no "
            "deferrable_fraction, a cell_full_factorial row has no coefficient) is "
            "stored as null and is kept distinct from the literal string ALL. The six "
            "coefficient_verdict rows pin the evidence behind the verdict string, not "
            "just the string: the percent-change range whose sign flip IS the fragility "
            "finding, and the per-level significance and cell-level agreement flags, "
            "none of which the verdict wording mentions. metric3_sign_convention is "
            "constant across the file, so it is pinned once here and checked against "
            "every row. results/structural_sensitivity.csv is read-only input here and "
            "is never written by this script; only this pinned snapshot is written. "
            "Regenerate DELIBERATELY, never automatically, only after a verified "
            "intentional rerun of that analysis, by running from the repository root: "
            f"{_REGENERATION_COMMAND}"
        ),
        "n_rows": int(len(df)),
        "rows": _row_keyed_rows(df, STRUCTURAL_SENSITIVITY_IDENTITY, STRUCTURAL_SENSITIVITY_PINNED_FIELDS),
    }
    pins.update(_constant_field_pins(df, STRUCTURAL_SENSITIVITY_CONSTANT_FIELDS, STRUCTURAL_SENSITIVITY_CSV_PATH))
    return pins


def _compute_demand_source_comparison_pins() -> dict | None:
    if not DEMAND_SOURCE_CSV_PATH.is_file():
        print(f"note: {DEMAND_SOURCE_CSV_PATH} not present; skipping demand_source_comparison_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(DEMAND_SOURCE_CSV_PATH)
    return {
        "_comment": (
            "Pinned digest of results/demand_source_comparison.csv (the Phase 12 "
            "synthetic-versus-OPSD headline comparison): every data row, keyed by "
            "(k, broker_count, ablation, metric), plus the row count. "
            "results/demand_source_comparison.csv is read-only input here and is never "
            "written by this script; only this pinned snapshot is written. Regenerate "
            "DELIBERATELY, never automatically, only after a verified intentional "
            f"rerun of that comparison, by running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "n_rows": int(len(df)),
        "rows": _row_keyed_rows(df, DEMAND_SOURCE_IDENTITY, DEMAND_SOURCE_PINNED_FIELDS),
    }


def _compute_monopoly_comparison_pins() -> dict | None:
    if not MONOPOLY_CSV_PATH.is_file():
        print(f"note: {MONOPOLY_CSV_PATH} not present; skipping monopoly_comparison_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(MONOPOLY_CSV_PATH)
    pins = {
        "_comment": (
            "Pinned digest of results/monopoly_comparison.csv (the broker_count=1 "
            "monopoly arm): every data row and every numeric column, keyed by "
            "(broker_count, k), plus the row count. metric3_sign_convention and note "
            "are constant across the file, so they are pinned once here and checked "
            "against every row: rewording the sign convention would reverse how every "
            "metric-3 number in the file is read without changing any number. "
            "results/monopoly_comparison.csv is read-only input here and is never "
            "written by this script; only this pinned snapshot is written. Regenerate "
            "DELIBERATELY, never automatically, only after a verified intentional "
            f"rerun of that comparison, by running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "n_rows": int(len(df)),
        "rows": _row_keyed_rows(df, MONOPOLY_IDENTITY, MONOPOLY_PINNED_FIELDS),
    }
    pins.update(_constant_field_pins(df, MONOPOLY_CONSTANT_FIELDS, MONOPOLY_CSV_PATH))
    return pins


def _compute_effect_sizes_pins() -> dict | None:
    if not EFFECT_SIZES_CSV_PATH.is_file():
        print(f"note: {EFFECT_SIZES_CSV_PATH} not present; skipping effect_sizes_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(EFFECT_SIZES_CSV_PATH)
    return {
        "_comment": (
            "Pinned digest of results/effect_sizes.csv (Chapter 1's headline figures and "
            "the per-cell effect sizes behind Sections 6.1 to 6.9): every data row and every "
            "column, keyed by (block, k, broker_count, ablation, metric), plus the row "
            "count. That key is unique over the whole file because the analysis emits "
            "exactly one row per combination of those five. Pinned whole rather than as a "
            "subset: 185 rows is small enough that a claim-bearing subset would save little. "
            "results/effect_sizes.csv is read-only input here and is never written by this "
            "script; only this pinned snapshot is written. Regenerate DELIBERATELY, never "
            "automatically, only after a verified intentional rerun of that analysis, by "
            f"running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "n_rows": int(len(df)),
        "rows": _row_keyed_rows(df, EFFECT_SIZES_IDENTITY, EFFECT_SIZES_PINNED_FIELDS),
    }


def _compute_summary_stats_corrected_pins() -> dict | None:
    if not SUMMARY_STATS_CORRECTED_CSV_PATH.is_file():
        print(
            f"note: {SUMMARY_STATS_CORRECTED_CSV_PATH} not present; "
            "skipping summary_stats_corrected_pins.json regeneration"
        )
        return None

    import pandas as pd

    df = pd.read_csv(SUMMARY_STATS_CORRECTED_CSV_PATH)
    return {
        "_comment": (
            "Pinned digest of results/summary_stats_corrected.csv (Section 6.10's "
            "multiple-comparison correction and bootstrap-CI detail, one row per "
            "comparison): every data row and every column, keyed by (scope, k, "
            "broker_count, ablation, metric), plus the row count. Pinned whole (188 rows, "
            "the same scale as summary_stats.csv's 240). "
            "results/summary_stats_corrected.csv is read-only input here and is never "
            "written by this script; only this pinned snapshot is written. Regenerate "
            "DELIBERATELY, never automatically, only after a verified intentional rerun of "
            f"that analysis, by running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "n_rows": int(len(df)),
        "rows": _row_keyed_rows(df, SUMMARY_STATS_CORRECTED_IDENTITY, SUMMARY_STATS_CORRECTED_PINNED_FIELDS),
    }


# See tests/test_golden_master.py's FAMILY_SENSITIVITY_AGGREGATE_FIELDS
# comment for the full rationale (why min/max alone miss a mid-distribution
# p-value shift, why a plain sum of p is the wrong fix, and why this floor and
# this value). Must match that file's P_VALUE_LOG_FLOOR exactly.
P_VALUE_LOG_FLOOR = 1e-300


def _log10_with_floor(value: float) -> float:
    return math.log10(value) if value > P_VALUE_LOG_FLOOR else math.log10(P_VALUE_LOG_FLOOR)


# See tests/test_golden_master.py's LABEL_COMBO_COLUMNS comment for the full
# rule: every column of results/family_sensitivity.csv that is an identifier
# or a label rather than a computed statistical output, not a hand-picked
# subset. Must match that file's LABEL_COMBO_COLUMNS exactly.
LABEL_COMBO_COLUMNS = (
    "family_key",
    "scope",
    "source",
    "k",
    "broker_count",
    "ablation",
    "cell_is_stamped_not_native",
    "metric",
    "test_id",
    "is_headline_72",
)


def _label_combo_key(row) -> str:
    return "|".join(f"{column}={_identity_token(row[column])}" for column in LABEL_COMBO_COLUMNS)


def _label_combo_counts(group) -> dict:
    counts: dict = {}
    for _, row in group.iterrows():
        key = _label_combo_key(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _family_sensitivity_aggregate_row(group) -> dict:
    """Whole-group counts, p-value extrema, log-magnitude sums, and the
    identity-adjacent label multiset. Twin of the same-named function in
    tests/test_golden_master.py; must compute identically or the snapshot
    written here stops describing what that file checks.
    """
    p_columns = ("p_uncorrected", "p_holm", "p_bh")
    row = {
        "n_rows": int(len(group)),
        "n_survives_holm_true": int(group["survives_holm_alpha05"].eq(True).sum()),
        "n_survives_holm_false": int(group["survives_holm_alpha05"].eq(False).sum()),
        "n_survives_bh_true": int(group["survives_bh_alpha05"].eq(True).sum()),
        "n_survives_bh_false": int(group["survives_bh_alpha05"].eq(False).sum()),
        "n_degenerate_true": int(group["degenerate"].eq(True).sum()),
        "p_uncorrected_min": float(group["p_uncorrected"].min()),
        "p_uncorrected_max": float(group["p_uncorrected"].max()),
        "p_holm_min": float(group["p_holm"].min()),
        "p_holm_max": float(group["p_holm"].max()),
        "p_bh_min": float(group["p_bh"].min()),
        "p_bh_max": float(group["p_bh"].max()),
        "n_rows_contributing": int(group["p_uncorrected"].notna().sum()),
    }
    for column in p_columns:
        values = group[column].dropna()
        row[f"sum_log10_{column}"] = float(sum(_log10_with_floor(float(value)) for value in values))
        row[f"n_at_floor_{column}"] = int((values <= P_VALUE_LOG_FLOOR).sum())
    row["label_counts"] = _label_combo_counts(group)
    return row


def _family_sensitivity_aggregates(df) -> dict:
    """Aggregates computed over the FULL dataframe (all 1227 rows), not the
    claim-bearing subset: this is the whole-file guard that closes the gap
    the eight individually pinned rows leave open on their own. See
    tests/test_golden_master.py's comment above FAMILY_SENSITIVITY_AGGREGATE_FIELDS
    for exactly what this does and does not cover.

    label_counts is dropped from the overall dict before it is returned: with
    test_id in LABEL_COMBO_COLUMNS, every row's combination is already unique
    within its own family_definition, so the overall multiset would be the
    identical 1227 entries the seven per-family dicts already hold between
    them, merged into one dict (the overall figure IS the sum of the
    per-family ones). That is a pure duplicate carrying strictly less
    information than the per-family breakdown, which can say WHICH family a
    mislabel landed in; only the per-family breakdown is kept.
    """
    overall = _family_sensitivity_aggregate_row(df)
    overall["n_distinct_family_definitions"] = int(df["family_definition"].nunique())
    overall["n_distinct_test_ids"] = int(df["test_id"].nunique())
    del overall["label_counts"]
    by_family_definition = {
        family_definition: _family_sensitivity_aggregate_row(group)
        for family_definition, group in df.groupby("family_definition", sort=True)
    }
    return {"overall": overall, "by_family_definition": by_family_definition}


def _compute_family_sensitivity_pins() -> dict | None:
    if not FAMILY_SENSITIVITY_CSV_PATH.is_file():
        print(f"note: {FAMILY_SENSITIVITY_CSV_PATH} not present; skipping family_sensitivity_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(FAMILY_SENSITIVITY_CSV_PATH)
    wanted = set(FAMILY_SENSITIVITY_PINNED_ROW_KEYS)
    mask = df.apply(lambda row: (row["family_definition"], row["test_id"]) in wanted, axis=1)
    subset = df[mask]
    assert len(subset) == len(wanted), (
        f"expected exactly {len(wanted)} claim-bearing rows in {FAMILY_SENSITIVITY_CSV_PATH}, "
        f"found {len(subset)}; FAMILY_SENSITIVITY_PINNED_ROW_KEYS no longer matches the file"
    )
    return {
        "_comment": (
            "Pinned digest of a CLAIM-BEARING SUBSET of results/family_sensitivity.csv "
            "(1227 rows; a whole-file digest at this scale would run to roughly 1.6 MB, so "
            "only the rows this package actually quotes a number from are pinned): the "
            "seven family_summary rows (one per family definition, carrying every number "
            "Section 6.10, Slide 13a/14, DECISIONS and qa_prep quote about the "
            "family-sensitivity check) plus the one real per-test row backing Section 6.2's "
            "own quoted 'd_z = -22.2 (p = 7.4e-41)' figure and Section 6.10's "
            "p_holm-reproduction claim. Keyed by (family_definition, test_id). "
            "n_total_rows is the file's real row count (a shape guard covering the 1219 "
            "rows this subset does not otherwise look at); n_rows is the size of the "
            "pinned subset itself. 'aggregates' closes that same gap, over EVERY row, not "
            "just the pinned eight, in three parts: (1) whole-file counts and p-value "
            "extrema on six value columns (p_uncorrected, p_holm, p_bh, "
            "survives_holm_alpha05, survives_bh_alpha05, degenerate); (2) log-magnitude sums "
            "on the three p-value columns (sum_log10_p_uncorrected/p_holm/p_bh, floored at "
            "P_VALUE_LOG_FLOOR=1e-300), closing the blind spot parts (1) leaves for a "
            "p-value that moves in the middle of the distribution, where it flips no flag "
            "and sets no new min or max; (3) an exact multiset over EVERY identifier and "
            "label column the file has (family_key, scope, source, k, broker_count, "
            "ablation, cell_is_stamped_not_native, metric, test_id, is_headline_72; every "
            "other column is a computed statistical output, a constant setting, free text, "
            "or the grouping key itself, none of which needed a separate label check), "
            "computed and pinned PER family_definition ONLY, not also at the overall level, "
            "closing a DIFFERENT blind spot that (1) and (2) cannot touch, since they never "
            "read those columns: a mislabelled metric, a swapped ablation, a flipped "
            "is_headline_72, a mislabelled source and so on all move no count, no extremum "
            "and no log-sum at all. Including test_id pushes this multiset close to one "
            "entry per row (313 distinct values across 1227), which is deliberate: every "
            "label is pinned exactly, every value column is pinned in aggregate. The overall "
            "level deliberately has no label_counts of its own: with test_id included, every "
            "row's combination is already unique within its own family_definition, so an "
            "overall multiset would be the identical 1227 entries the seven per-family dicts "
            "already hold, merged into one (the overall count IS the sum of the per-family "
            "ones), a pure duplicate that is strictly less diagnostic, since it cannot say "
            "WHICH family a mislabel landed in the way the per-family breakdown can. An "
            "adversarial review reproduced the original gap three separate ways, and a "
            "follow-up review then found the first fix was itself a hand-picked subset of "
            "label columns rather than all of them; both are why this is the rule now "
            "rather than a narrower one. What remains unpinned even now: a change to an "
            "unpinned row's individual p-value that preserves its family's log-sum exactly, "
            "which needs two compensating shifts in exact reciprocal proportion, a coincidence "
            "rather than a realistic regression, but a real and disclosed limit of a sum "
            "rather than a per-row pin. Deliberately not a bit-exact whole-file digest: "
            "Phase 11 already found that check false-alarms on cross-process 1-ULP "
            "floating-point drift on a grid this size. "
            "results/family_sensitivity.csv is read-only input here "
            "and is never written by this script; only this pinned snapshot is written. "
            "Regenerate DELIBERATELY, never automatically, only after a verified "
            "intentional rerun of that analysis, by running from the repository root: "
            f"{_REGENERATION_COMMAND}"
        ),
        "n_total_rows": int(len(df)),
        "n_rows": int(len(subset)),
        "rows": _row_keyed_rows(subset, FAMILY_SENSITIVITY_IDENTITY, FAMILY_SENSITIVITY_PINNED_FIELDS),
        "aggregates": _family_sensitivity_aggregates(df),
    }


def _compute_dose_matched_monopoly_pins() -> dict | None:
    if not DOSE_MATCHED_MONOPOLY_CSV_PATH.is_file():
        print(f"note: {DOSE_MATCHED_MONOPOLY_CSV_PATH} not present; skipping dose_matched_monopoly_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(DOSE_MATCHED_MONOPOLY_CSV_PATH)
    return {
        "_comment": (
            "Pinned digest of results/dose_matched_monopoly.csv (D11's dose-matched-"
            "monopoly arm, Phase 18.1): every data row and every column, keyed by "
            "(scope, arm, k, metric), plus the row count. Pinned whole rather than as a "
            "subset: 46 rows is small enough that a claim-bearing subset would save "
            "nothing. results/dose_matched_monopoly.csv is read-only input here and is "
            "never written by this script; only this pinned snapshot is written. "
            "Regenerate DELIBERATELY, never automatically, only after a verified "
            "intentional rerun of that analysis, by running from the repository root: "
            f"{_REGENERATION_COMMAND}"
        ),
        "n_rows": int(len(df)),
        "rows": _row_keyed_rows(df, DOSE_MATCHED_MONOPOLY_IDENTITY, DOSE_MATCHED_MONOPOLY_PINNED_FIELDS),
    }


def _compute_dose_matched_monopoly_parquet_cell_pins() -> dict | None:
    if not SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH.is_file():
        print(
            f"note: {SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH} not present; skipping "
            "sweep_dose_matched_monopoly_parquet_cell_pins.json regeneration"
        )
        return None

    import pandas as pd

    df = pd.read_parquet(SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_PATH)
    grouped = _dose_matched_monopoly_parquet_cell_aggregates(df)
    return {
        "_comment": (
            "Per-(k, capacity_surcharge_divisor, ablation) cell aggregates over "
            "results/sweep_dose_matched_monopoly.parquet (D11's dose-matched-monopoly "
            "arm, Phase 18.1): the row count (30 in every one of the 12 cells) and the "
            "mean of total_deferred_kwh, feeder_peak_to_average_ratio, "
            "feeder_coefficient_of_variation, avg_cost_per_agent_eur, "
            "capacity_fire_rate and total_capacity_charge_eur, keyed by (k, "
            "capacity_surcharge_divisor, ablation). These are claim-bearing aggregates "
            "no CSV covers directly: docs/DECISIONS.md's 'D11 result' section quotes "
            "the peak-to-average and deferred-volume figures straight from this "
            "parquet, filtered the same way. "
            "results/sweep_dose_matched_monopoly.parquet is read-only input here and is "
            "never written by this script; only this pinned snapshot is written. "
            "Regenerate DELIBERATELY, never automatically, only after a verified "
            "intentional rerun of that sweep, by running from the repository root: "
            f"{_REGENERATION_COMMAND}"
        ),
        "n_rows": int(len(grouped)),
        "rows": _row_keyed_rows(
            grouped, DOSE_MATCHED_MONOPOLY_PARQUET_CELL_IDENTITY, DOSE_MATCHED_MONOPOLY_PARQUET_CELL_FIELDS
        ),
    }


def _write_golden(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {path}")


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    _write_golden(SHORT_RUN_GOLDEN_PATH, _compute_short_run_golden())

    for golden_path, compute in (
        (SUMMARY_STATS_GOLDEN_PATH, _compute_summary_stats_pins),
        (STRUCTURAL_SENSITIVITY_GOLDEN_PATH, _compute_structural_sensitivity_pins),
        (DEMAND_SOURCE_GOLDEN_PATH, _compute_demand_source_comparison_pins),
        (MONOPOLY_GOLDEN_PATH, _compute_monopoly_comparison_pins),
        (EFFECT_SIZES_GOLDEN_PATH, _compute_effect_sizes_pins),
        (SUMMARY_STATS_CORRECTED_GOLDEN_PATH, _compute_summary_stats_corrected_pins),
        (FAMILY_SENSITIVITY_GOLDEN_PATH, _compute_family_sensitivity_pins),
        (DOSE_MATCHED_MONOPOLY_GOLDEN_PATH, _compute_dose_matched_monopoly_pins),
        (SWEEP_DOSE_MATCHED_MONOPOLY_PARQUET_CELL_GOLDEN_PATH, _compute_dose_matched_monopoly_parquet_cell_pins),
    ):
        payload = compute()
        if payload is not None:
            _write_golden(golden_path, payload)


if __name__ == "__main__":
    main()
