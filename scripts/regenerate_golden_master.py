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
  - tests/golden/summary_stats_pins.json: a field-by-field snapshot of
    results/summary_stats.csv at k=1.0, broker_count=3, across all four
    capacity ablations (the Phase 5/6 sweep's headline configuration). Reads
    that CSV read-only; if it is not present (e.g. the sweep has not been run
    on this machine) this half of the regeneration is skipped with a printed
    note and the short-run golden file is still written.
  - tests/golden/structural_sensitivity_pins.json: every data row of
    results/structural_sensitivity.csv (the D9 structural coefficient
    sensitivity analysis).
  - tests/golden/demand_source_comparison_pins.json: every data row of
    results/demand_source_comparison.csv (the Phase 12 synthetic-versus-OPSD
    headline comparison).
  - tests/golden/monopoly_comparison_pins.json: every data row of
    results/monopoly_comparison.csv (the broker_count=1 monopoly arm).

Every CSV named above is READ-ONLY input here. This script writes nothing but
tests/golden/*.json, and each CSV-derived snapshot is skipped with a printed
note, rather than failing, when its source CSV is absent on this machine.
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

# Must match tests/test_golden_master.py's SHORT_HORIZON_HOURS / SHORT_NUM_AGENTS
# / SHORT_SEED exactly. Horizon comfortably exceeds the 168h rolling window
# (three window-widths) so the capacity mechanism actually fires; agent count
# and horizon both sit inside the fix brief's suggested envelope.
SHORT_RUN_HORIZON_HOURS = 504
SHORT_RUN_NUM_AGENTS = 50
SHORT_RUN_SEED = 101  # fixed for reproducibility; arbitrary, not tuned

# Must match tests/test_golden_master.py's pinned sweep coordinates.
PINNED_K = 1.0
PINNED_BROKER_COUNT = 3

# Identity columns and pinned fields for the row-keyed CSV snapshots. Each of
# these must match the same-named list in tests/test_golden_master.py exactly,
# or the snapshot written here stops describing what those tests check.
_MISSING_TOKEN = "__NA__"

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
    "correction_family": "str",
    "correction_family_size": "int",
    "verdict": "str",
}

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


def _compute_summary_stats_pins() -> dict | None:
    if not SUMMARY_STATS_CSV_PATH.is_file():
        print(f"note: {SUMMARY_STATS_CSV_PATH} not present; skipping summary_stats_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(SUMMARY_STATS_CSV_PATH)
    subset = df[(df["k"] == PINNED_K) & (df["broker_count"] == PINNED_BROKER_COUNT)]
    if subset.empty:
        print(f"note: no rows at k={PINNED_K}, broker_count={PINNED_BROKER_COUNT} in {SUMMARY_STATS_CSV_PATH}")
        return None

    rows = []
    for _, row in subset.sort_values(["ablation", "metric"]).iterrows():
        rows.append(
            {
                "ablation": row["ablation"],
                "metric": row["metric"],
                "n_seeds": int(row["n_seeds"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
        )

    return {
        "_comment": (
            "Pinned digest of results/summary_stats.csv (the Phase 5/6 capacity-mechanism "
            f"sweep) at k={PINNED_K}, broker_count={PINNED_BROKER_COUNT}, across all four "
            "ablations. results/summary_stats.csv is read-only input here and is never "
            "written by this script; only this pinned snapshot is written. Regenerate "
            "DELIBERATELY, never automatically, only after a verified intentional sweep "
            f"rerun, by running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "k": PINNED_K,
        "broker_count": PINNED_BROKER_COUNT,
        "rows": rows,
    }


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


def _compute_structural_sensitivity_pins() -> dict | None:
    if not STRUCTURAL_SENSITIVITY_CSV_PATH.is_file():
        print(f"note: {STRUCTURAL_SENSITIVITY_CSV_PATH} not present; skipping structural_sensitivity_pins.json regeneration")
        return None

    import pandas as pd

    df = pd.read_csv(STRUCTURAL_SENSITIVITY_CSV_PATH)
    return {
        "_comment": (
            "Pinned digest of results/structural_sensitivity.csv (the D9 structural "
            "coefficient sensitivity analysis): every data row, keyed by "
            "(scope, deferrable_fraction, response_reference_eur_per_kwh, "
            "payback_cap_fraction, coefficient, coefficient_level, metric), plus the "
            "row count. An empty identity cell (a coefficient_verdict row has no "
            "deferrable_fraction, a cell_full_factorial row has no coefficient) is "
            "stored as null and is kept distinct from the literal string ALL. "
            "results/structural_sensitivity.csv is read-only input here and is never "
            "written by this script; only this pinned snapshot is written. Regenerate "
            "DELIBERATELY, never automatically, only after a verified intentional "
            f"rerun of that analysis, by running from the repository root: {_REGENERATION_COMMAND}"
        ),
        "n_rows": int(len(df)),
        "rows": _row_keyed_rows(df, STRUCTURAL_SENSITIVITY_IDENTITY, STRUCTURAL_SENSITIVITY_PINNED_FIELDS),
    }


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
    for field in MONOPOLY_CONSTANT_FIELDS:
        distinct = sorted({str(_plain(value)) for value in df[field]})
        if len(distinct) != 1:
            print(f"note: {field} is not constant across {MONOPOLY_CSV_PATH}; pinning the first value only")
        pins[field] = str(_plain(df[field].iloc[0]))
    return pins


def _write_golden(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {path}")


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    _write_golden(SHORT_RUN_GOLDEN_PATH, _compute_short_run_golden())

    summary_stats_golden = _compute_summary_stats_pins()
    if summary_stats_golden is not None:
        _write_golden(SUMMARY_STATS_GOLDEN_PATH, summary_stats_golden)

    for golden_path, compute in (
        (STRUCTURAL_SENSITIVITY_GOLDEN_PATH, _compute_structural_sensitivity_pins),
        (DEMAND_SOURCE_GOLDEN_PATH, _compute_demand_source_comparison_pins),
        (MONOPOLY_GOLDEN_PATH, _compute_monopoly_comparison_pins),
    ):
        payload = compute()
        if payload is not None:
            _write_golden(golden_path, payload)


if __name__ == "__main__":
    main()
