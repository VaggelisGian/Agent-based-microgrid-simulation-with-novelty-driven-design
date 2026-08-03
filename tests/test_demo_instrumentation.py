"""Pins scripts/demo_instrumentation.py's AND scripts/feeder_statistics.py's
measured output against their tests/golden/*_pins.json files, so a later
change to the model, the dashboard's DemoSession, or the thesis prose that
quotes either script's numbers (docs/thesis/demo_instructions.md,
docs/thesis/appendix_validation.md) fails a test instead of needing another
manual remeasurement (docs/PROGRESS.md, "19.4 A finding that did not
reproduce", and the closing note naming both gaps this file closes).

Both scripts' pins live in one test module because Phase 20 item 20.7's
allowed-new-files list names exactly one new test file
(tests/test_demo_instrumentation.py) and forbids editing any existing one;
this module therefore covers scripts/feeder_statistics.py too, in its own
clearly separated section below, rather than skipping that coverage.

Both golden files share the same shape: two top-level runs, "full" (the
horizon the relevant thesis document's own numbers are stated over) and
"smoke" (a short run of the identical code path, fast, always on by
default), both built by each script's own build_pins() from a single
per-field-computing function, so there is exactly one definition of every
field. The two scripts' "full" runs are NOT treated the same way: demo_
instrumentation's full (2000h) runs by default, 43.6s on a quiet machine, so
that a stale demo_instructions.md figure fails on a plain `pytest`
invocation; feeder_statistics's full (8760h x 2 configs, 532.65s measured)
is marked slow and skipped unless RUN_SLOW_TESTS=1, since that cost is
prohibitive for a default run, and only its cheap smoke subset runs by
default (see each section below for the full reasoning, and
docs/verification/demo_instrumentation.md section 3 for the cost accounting
that motivated ungating the demo_instrumentation full tests specifically).

Two hazards this project was explicitly warned about, both avoided here:

  - No pytest.approx with both rel and abs set. demo_instrumentation's
    same_hour_max_gap is exactly 0.0 by construction; an abs tolerance would
    make that assertion vacuous. Each script's own EXACT_FIELDS (imported,
    never redefined here) are compared by `==`; every other field is
    compared by math.isclose(..., rel_tol=..., the default abs_tol=0.0),
    i.e. relative tolerance only. feeder_statistics can also emit JSON null
    for a season with no hours in a short run (see its section below);
    _mismatched_fields treats None specially so that comparison never calls
    math.isclose on it.
  - Comparisons iterate the KEY-SET UNION of measured and expected, never
    expected.items() alone (that asymmetry is exactly what let a vanished pin
    pass silently in tests/test_golden_master.py's _metrics_dict twin,
    docs/PROGRESS.md "19.3"). _mismatched_fields below checks the key sets
    first and reports a missing key on EITHER side as its own mismatch.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import demo_instrumentation as di  # noqa: E402
import feeder_statistics as fs  # noqa: E402

DI_GOLDEN_PATH = _REPO_ROOT / "tests" / "golden" / "demo_instrumentation_pins.json"
FS_GOLDEN_PATH = _REPO_ROOT / "tests" / "golden" / "feeder_statistics_pins.json"

_RUN_SLOW = bool(os.environ.get("RUN_SLOW_TESTS"))
slow = pytest.mark.skipif(
    not _RUN_SLOW,
    reason="heavy model run (see this module's docstring); set RUN_SLOW_TESTS=1 to include it",
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mismatched_fields(measured: dict, expected: dict, exact_fields: frozenset) -> list[str]:
    """Field names where measured and expected disagree, checked over the
    KEY-SET UNION (see module docstring). Fields in `exact_fields` compare
    with `==`; everything else compares with a relative-tolerance-only
    isclose, except that a None on either side (feeder_statistics: a season
    absent from a short run) always compares with `==` too, since
    math.isclose has no meaning for None."""
    mismatches = []
    for key in sorted(set(measured) | set(expected)):
        if key not in measured:
            mismatches.append(key)  # missing from measured: a vanished field
            continue
        if key not in expected:
            mismatches.append(key)  # missing from the pin: an unpinned field
            continue
        m, e = measured[key], expected[key]
        if key in exact_fields or m is None or e is None:
            if m != e:
                mismatches.append(key)
        else:
            if not math.isclose(m, e, rel_tol=1e-9):
                mismatches.append(key)
    return mismatches


def _perturb(value):
    """A different-enough value of the same type, used only to prove a pin
    bites (task 5): perturb the loaded PIN in memory, never the file, never
    the model."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0 if value == 0.0 else value * 1.01
    if isinstance(value, str):
        return value + "_perturbed"
    raise TypeError(f"no perturbation rule for value {value!r} of type {type(value)}")


# =============================================================================
# scripts/demo_instrumentation.py
# =============================================================================
#
#   "full"  -- 2000 hours, the horizon docs/thesis/demo_instructions.md's own
#              numbers are stated over. Runs by DEFAULT (not marked slow):
#              measured 40-48s on a busy machine (idle timing closer to
#              13-15s) and 43.6s for all three di_full tests together on a
#              quieter one, since they share the measured_di_full fixture and
#              the 2000-hour run happens once. This is the one test that
#              constrains demo_instructions.md's own quoted figures against
#              the shipped model, so it has to run on a plain `pytest`
#              invocation, not just under an opt-in flag, or a stale figure
#              can drift for a full suite cycle before anyone notices, the
#              exact gap docs/PROGRESS.md's closing note names. See
#              docs/verification/demo_instrumentation.md section 3 for the
#              full timing history and cost accounting.
#   "smoke" -- 250 hours (past the 168-hour window fill, so every field is
#              well-defined, none of the doc's own numbers), fast enough
#              (a few seconds) to run in the default suite. It does NOT
#              reproduce demo_instructions.md's numbers and is not meant to;
#              it exists so the code path and the pin-comparison machinery
#              itself stay covered even faster than the full run does.

# ---------------------------------------------------------------------------
# field-list hygiene (no model run needed; always runs)
# ---------------------------------------------------------------------------


def test_di_field_kind_partition_is_total_and_disjoint():
    exact = di.EXACT_FIELDS
    rel = di.REL_TOLERANCE_FIELDS
    assert exact & rel == set()
    assert exact | rel == set(di.FIELD_NAMES)


def test_di_golden_file_has_full_and_smoke_with_the_full_field_set():
    golden = _load_json(DI_GOLDEN_PATH)
    assert set(golden) == {"full", "smoke"}
    for run_name in ("full", "smoke"):
        assert set(golden[run_name]) == set(di.FIELD_NAMES), (
            f"tests/golden/demo_instrumentation_pins.json's {run_name!r} run "
            "does not have exactly FIELD_NAMES's keys"
        )


# ---------------------------------------------------------------------------
# smoke run: fast, default, exercises the same code path as "full"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def measured_di_smoke():
    expected_hours = _load_json(DI_GOLDEN_PATH)["smoke"]["hours_measured"]
    return di.run_instrumentation(hours=expected_hours)


def test_di_smoke_matches_pinned_values(measured_di_smoke):
    expected = _load_json(DI_GOLDEN_PATH)["smoke"]
    mismatches = _mismatched_fields(measured_di_smoke, expected, di.EXACT_FIELDS)
    assert not mismatches, f"smoke run disagrees with the pinned fields: {mismatches}"


def test_di_smoke_every_pin_bites(measured_di_smoke):
    """Perturbing any one pinned field, alone, must make the comparison fail.
    Also proves a pin that vanishes entirely (Phase 19's exact defect) is
    caught, not silently ignored."""
    expected = _load_json(DI_GOLDEN_PATH)["smoke"]
    bitten = 0
    for key in di.FIELD_NAMES:
        perturbed = dict(expected)
        perturbed[key] = _perturb(perturbed[key])
        mismatches = _mismatched_fields(measured_di_smoke, perturbed, di.EXACT_FIELDS)
        assert key in mismatches, f"perturbing pinned field {key!r} did not make the comparison fail"
        bitten += 1

    dropped = dict(expected)
    del dropped[di.FIELD_NAMES[0]]
    assert _mismatched_fields(measured_di_smoke, dropped, di.EXACT_FIELDS), "a pin that vanished entirely must be caught"
    bitten += 1

    assert bitten == len(di.FIELD_NAMES) + 1


# ---------------------------------------------------------------------------
# full run: the actual 2000-hour, thesis-headline-cell measurement.
# Runs by default (not marked slow): this is the one test that constrains
# demo_instructions.md's own quoted figures (380 / 1833 / 1453 / 111 /
# 3.815e-03 / ...), so it must run on a plain `pytest` invocation or a stale
# figure can drift for a full suite cycle before anyone notices, exactly the
# gap docs/PROGRESS.md's closing note names. Measured 43.6s for all three
# di_full tests together on a quieter machine (they share measured_di_full,
# so the 2000-hour run happens once, not three times); see
# docs/verification/demo_instrumentation.md section 3 for the full timing
# history and the decision to ungate these three specifically.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def measured_di_full():
    expected_hours = _load_json(DI_GOLDEN_PATH)["full"]["hours_measured"]
    return di.run_instrumentation(hours=expected_hours)


def test_di_full_matches_pinned_values(measured_di_full):
    expected = _load_json(DI_GOLDEN_PATH)["full"]
    mismatches = _mismatched_fields(measured_di_full, expected, di.EXACT_FIELDS)
    assert not mismatches, f"full 2000-hour run disagrees with the pinned fields: {mismatches}"


def test_di_full_every_pin_bites(measured_di_full):
    expected = _load_json(DI_GOLDEN_PATH)["full"]
    bitten = 0
    for key in di.FIELD_NAMES:
        perturbed = dict(expected)
        perturbed[key] = _perturb(perturbed[key])
        mismatches = _mismatched_fields(measured_di_full, perturbed, di.EXACT_FIELDS)
        assert key in mismatches, f"perturbing pinned field {key!r} did not make the comparison fail"
        bitten += 1

    dropped = dict(expected)
    del dropped[di.FIELD_NAMES[0]]
    assert _mismatched_fields(measured_di_full, dropped, di.EXACT_FIELDS), "a pin that vanished entirely must be caught"
    bitten += 1

    assert bitten == len(di.FIELD_NAMES) + 1


def test_di_full_matches_demo_instructions_headline_numbers(measured_di_full):
    """Cross-check directly against docs/thesis/demo_instructions.md's own
    prose (lines 43-71), independent of the golden file, so a golden file
    that was regenerated from a since-changed script cannot silently drift
    away from what the thesis text itself claims without a test noticing.
    Rounded doc figures get a loose relative tolerance matching their own
    printed precision (3-4 significant figures); exact counts compare exactly.
    """
    m = measured_di_full
    assert m["firing_hours"] == 380  # line 43
    assert m["same_hour_max_gap"] == 0.0  # lines 43-44
    assert math.isclose(m["prev_hour_max_gap_documented_variant"], 3.815e-03, rel_tol=2e-3)  # lines 44-45
    assert math.isclose(m["prev_hour_max_gap_previous_firing_hour_variant"], 3.896e-03, rel_tol=2e-3)
    assert math.isclose(m["prev_hour_max_gap_all_hours_variant"], 6.497e-02, rel_tol=2e-3)
    assert m["post_window_hours"] == 1833  # line 57
    assert m["all_zero_bar_hours"] == 1453  # line 58
    assert math.isclose(m["all_zero_bar_fraction"], 0.793, rel_tol=2e-3)  # line 58
    assert m["net_export_hours"] == 111  # lines 65-66
    assert m["net_export_and_firing_hours"] == 0  # lines 65-66
    assert math.isclose(m["area_share_final_hour_regulated_utility"], 0.386, rel_tol=3e-3)  # lines 70-71
    assert math.isclose(m["area_share_final_hour_volatile_low_cost"], 0.396, rel_tol=3e-3)
    assert math.isclose(m["area_share_final_hour_premium_green"], 0.218, rel_tol=3e-3)
    assert math.isclose(m["card_load_share_regulated_utility"], 0.391, rel_tol=3e-3)
    assert math.isclose(m["card_load_share_volatile_low_cost"], 0.387, rel_tol=3e-3)
    assert math.isclose(m["card_load_share_premium_green"], 0.222, rel_tol=3e-3)


# =============================================================================
# scripts/feeder_statistics.py
# =============================================================================
#
#   "full"  -- 8760 hours (the mandated full year, D8) x 2 configs (baseline,
#              capacity_both), appendix_validation.md A.2's own methodology.
#              Measured well over 60s combined on this machine (see
#              docs/verification/demo_instrumentation.md), so marked slow and
#              skipped by default (set RUN_SLOW_TESTS=1 to run it).
#   "smoke" -- 240 hours (10 days from hour 0, i.e. all of early January) x 2
#              configs, a few seconds total, on by default. Every field
#              except month_of_peak_hour, winter_mean_kwh, summer_mean_kwh
#              and winter_summer_ratio is well-defined at this horizon and is
#              pinned; the season fields are genuinely undefined this early in
#              the year (no June-July-August hours exist yet) and are pinned
#              as JSON null, which _mismatched_fields compares with `==`
#              rather than calling math.isclose on it.


# ---------------------------------------------------------------------------
# field-list hygiene (no model run needed; always runs)
# ---------------------------------------------------------------------------


def test_fs_field_kind_partition_is_total_and_disjoint():
    exact = fs.EXACT_FIELDS
    rel = fs.REL_TOLERANCE_FIELDS
    assert exact & rel == set()
    assert exact | rel == set(fs.FIELD_NAMES)


def test_fs_golden_file_has_full_and_smoke_with_both_configs_and_the_full_field_set():
    golden = _load_json(FS_GOLDEN_PATH)
    assert set(golden) == {"full", "smoke"}
    for run_name in ("full", "smoke"):
        assert set(golden[run_name]) == set(fs.CONFIG_PATHS), (
            f"tests/golden/feeder_statistics_pins.json's {run_name!r} run "
            "does not have exactly the configured config names"
        )
        for config_name in fs.CONFIG_PATHS:
            assert set(golden[run_name][config_name]) == set(fs.FIELD_NAMES), (
                f"tests/golden/feeder_statistics_pins.json's {run_name!r}/{config_name!r} "
                "does not have exactly FIELD_NAMES's keys"
            )


def _fs_field_mismatches(measured_run: dict, expected_run: dict) -> dict[str, list[str]]:
    """Per-config mismatches over a whole {"baseline": {...}, "capacity_both":
    {...}} run dict; keys with an empty mismatch list are omitted."""
    result = {}
    for config_name in fs.CONFIG_PATHS:
        mismatches = _mismatched_fields(measured_run[config_name], expected_run[config_name], fs.EXACT_FIELDS)
        if mismatches:
            result[config_name] = mismatches
    return result


# ---------------------------------------------------------------------------
# smoke run: fast, default, exercises the same code path as "full"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def measured_fs_smoke():
    expected = _load_json(FS_GOLDEN_PATH)["smoke"]
    horizon_hours = next(iter(expected.values()))["horizon_hours"]
    return fs.run_feeder_statistics(horizon_hours=horizon_hours)


def test_fs_smoke_matches_pinned_values(measured_fs_smoke):
    expected = _load_json(FS_GOLDEN_PATH)["smoke"]
    mismatches = _fs_field_mismatches(measured_fs_smoke, expected)
    assert not mismatches, f"smoke run disagrees with the pinned fields: {mismatches}"


def test_fs_smoke_every_pin_bites(measured_fs_smoke):
    expected = _load_json(FS_GOLDEN_PATH)["smoke"]
    bitten = 0
    for config_name in fs.CONFIG_PATHS:
        for key in fs.FIELD_NAMES:
            original = expected[config_name][key]
            if original is None:
                # A null pin (an undefined season on this short run) bites
                # via the "== None" branch: any non-null perturbation must
                # already disagree with the measured None. Use a concrete
                # value of the field's own usual type instead of _perturb,
                # which has no rule for None.
                perturbed_value = 1.0
            else:
                perturbed_value = _perturb(original)
            perturbed_run = {name: dict(cfg) for name, cfg in expected.items()}
            perturbed_run[config_name] = dict(expected[config_name])
            perturbed_run[config_name][key] = perturbed_value
            mismatches = _fs_field_mismatches(measured_fs_smoke, perturbed_run)
            assert config_name in mismatches and key in mismatches[config_name], (
                f"perturbing pinned field {config_name}.{key} did not make the comparison fail"
            )
            bitten += 1

    dropped_run = {name: dict(cfg) for name, cfg in expected.items()}
    dropped_run["baseline"] = dict(expected["baseline"])
    del dropped_run["baseline"][fs.FIELD_NAMES[0]]
    assert _fs_field_mismatches(measured_fs_smoke, dropped_run), "a pin that vanished entirely must be caught"
    bitten += 1

    assert bitten == len(fs.CONFIG_PATHS) * len(fs.FIELD_NAMES) + 1


# ---------------------------------------------------------------------------
# full run: 8760h x 2 configs, appendix_validation.md's own methodology.
# Marked slow (see section header above); skipped unless RUN_SLOW_TESTS=1.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def measured_fs_full():
    expected = _load_json(FS_GOLDEN_PATH)["full"]
    horizon_hours = next(iter(expected.values()))["horizon_hours"]
    return fs.run_feeder_statistics(horizon_hours=horizon_hours)


@slow
def test_fs_full_matches_pinned_values(measured_fs_full):
    expected = _load_json(FS_GOLDEN_PATH)["full"]
    mismatches = _fs_field_mismatches(measured_fs_full, expected)
    assert not mismatches, f"full 8760-hour run disagrees with the pinned fields: {mismatches}"


@slow
def test_fs_full_every_pin_bites(measured_fs_full):
    expected = _load_json(FS_GOLDEN_PATH)["full"]
    bitten = 0
    for config_name in fs.CONFIG_PATHS:
        for key in fs.FIELD_NAMES:
            original = expected[config_name][key]
            perturbed_value = 1.0 if original is None else _perturb(original)
            perturbed_run = {name: dict(cfg) for name, cfg in expected.items()}
            perturbed_run[config_name] = dict(expected[config_name])
            perturbed_run[config_name][key] = perturbed_value
            mismatches = _fs_field_mismatches(measured_fs_full, perturbed_run)
            assert config_name in mismatches and key in mismatches[config_name], (
                f"perturbing pinned field {config_name}.{key} did not make the comparison fail"
            )
            bitten += 1

    dropped_run = {name: dict(cfg) for name, cfg in expected.items()}
    dropped_run["baseline"] = dict(expected["baseline"])
    del dropped_run["baseline"][fs.FIELD_NAMES[0]]
    assert _fs_field_mismatches(measured_fs_full, dropped_run), "a pin that vanished entirely must be caught"
    bitten += 1

    assert bitten == len(fs.CONFIG_PATHS) * len(fs.FIELD_NAMES) + 1


@slow
def test_fs_full_matches_appendix_validation_headline_numbers(measured_fs_full):
    """Cross-check directly against docs/thesis/appendix_validation.md's own
    A.3 results table, independent of the golden file. docs/PROGRESS.md
    already records two known, accepted imprecisions in that table (four
    significant figures where it prints five, appendix_validation.md:105's
    3.173 for a value of 3.172486), so this only checks agreement to the
    precision the appendix itself actually prints, not further.
    """
    baseline = measured_fs_full["baseline"]
    capacity_both = measured_fs_full["capacity_both"]

    assert math.isclose(baseline["load_factor"], 0.287, rel_tol=5e-3)
    assert math.isclose(capacity_both["load_factor"], 0.315, rel_tol=5e-3)
    assert math.isclose(baseline["coincidence_factor"], 0.951, rel_tol=5e-3)
    assert math.isclose(capacity_both["coincidence_factor"], 0.906, rel_tol=5e-3)
    assert math.isclose(baseline["diversity_factor"], 1.051, rel_tol=5e-3)
    assert math.isclose(capacity_both["diversity_factor"], 1.104, rel_tol=5e-3)
    assert math.isclose(baseline["peak_to_average_ratio"], 3.483, rel_tol=5e-3)
    # appendix_validation.md:105 prints 3.173 (rounds to 4 s.f.); the value
    # already on record (docs/PROGRESS.md) is 3.172486 -- checked against
    # that, the precise figure, not the rounded print.
    assert math.isclose(capacity_both["peak_to_average_ratio"], 3.172486, rel_tol=5e-4)
    assert math.isclose(baseline["mean_demand_per_household_kwh"], 4295.0, rel_tol=5e-3)
    assert math.isclose(capacity_both["mean_demand_per_household_kwh"], 4294.9, rel_tol=5e-3)
    assert math.isclose(baseline["mean_net_import_per_household_kwh"], 3428.5, rel_tol=5e-3)
    assert math.isclose(capacity_both["mean_net_import_per_household_kwh"], 3428.4, rel_tol=5e-3)
    assert baseline["hour_of_daily_peak"] == 20
    assert capacity_both["hour_of_daily_peak"] == 20
    assert baseline["month_of_peak_hour"] == 12  # December
    assert capacity_both["month_of_peak_hour"] == 1  # January
    assert math.isclose(baseline["min_max_ratio"], -0.116, rel_tol=2e-2)
    assert math.isclose(capacity_both["min_max_ratio"], -0.127, rel_tol=2e-2)
    assert math.isclose(baseline["fraction_negative_hours"], 0.0533, rel_tol=5e-3)
    assert math.isclose(capacity_both["fraction_negative_hours"], 0.0531, rel_tol=5e-3)
    assert math.isclose(baseline["night_valley_depth"], 0.709, rel_tol=5e-3)
    assert math.isclose(capacity_both["night_valley_depth"], 0.603, rel_tol=5e-3)
    assert math.isclose(baseline["feeder_coefficient_of_variation"], 0.672, rel_tol=5e-3)
    assert math.isclose(capacity_both["feeder_coefficient_of_variation"], 0.623, rel_tol=5e-3)
    # four significant figures, not five (docs/PROGRESS.md, already recorded).
    assert math.isclose(baseline["total_energy_served_kwh"], 858996.5, rel_tol=5e-5)
    assert math.isclose(capacity_both["total_energy_served_kwh"], 858971.9, rel_tol=5e-5)
