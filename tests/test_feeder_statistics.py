"""Pins scripts/feeder_statistics.py's measured output against
tests/golden/feeder_statistics_pins.json, so a later change to the model or
to the thesis prose that quotes this script's numbers
(docs/thesis/appendix_validation.md) fails a test instead of needing
another manual remeasurement (docs/PROGRESS.md, "19.4 A finding that did
not reproduce").

The golden file has two top-level runs, "full" (8760 hours, the mandated
full year, D8, x 2 configs: baseline and capacity_both, appendix_
validation.md A.2's own methodology) and "smoke" (240 hours, 10 days from
hour 0, i.e. all of early January, x 2 configs, fast, always on by
default), both built by build_pins() from a single per-field-computing
function, so there is exactly one definition of every field.

The "full" run is measured at 532.65s combined on this machine
(docs/PROGRESS.md, Phase 20.7), which is prohibitive for a default run, so
it is marked slow and skipped unless RUN_SLOW_TESTS=1; only the cheap
"smoke" subset runs by default. Every field except month_of_peak_hour,
winter_mean_kwh, summer_mean_kwh and winter_summer_ratio is well-defined at
the smoke horizon and is pinned; the season fields are genuinely undefined
this early in the year (no June-July-August hours exist yet) and are
pinned as JSON null.

Two hazards this project was explicitly warned about, both avoided here:

  - No pytest.approx with both rel and abs set. EXACT_FIELDS (imported from
    feeder_statistics, never redefined here) are compared by `==`; every
    other field is compared by math.isclose(..., rel_tol=1e-9), leaving
    abs_tol at its default of 0.0, i.e. relative tolerance only. A null
    season pin is compared by `==` instead, since math.isclose has no
    meaning for None.
  - Comparisons iterate the KEY-SET UNION of measured and expected, never
    expected.items() alone (that asymmetry is exactly what let a vanished
    pin pass silently in tests/test_golden_master.py's _metrics_dict twin,
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

import feeder_statistics as fs  # noqa: E402

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
    isclose, except that a None on either side (a season absent from a
    short run) always compares with `==` too, since math.isclose has no
    meaning for None."""
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
# Marked slow (see module docstring); skipped unless RUN_SLOW_TESTS=1.
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
