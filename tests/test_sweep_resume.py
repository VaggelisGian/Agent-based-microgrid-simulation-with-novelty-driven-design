"""Tests for the sweep-runner resume/skip logic (Phase 20.5 reproducibility audit).

scripts/ is not an importable package, so this module loads
scripts/run_sensitivity_sweep.py the same way tests/test_analyze_structural_sweep.py
loads scripts/analyze_structural_sweep.py: via importlib.util.spec_from_file_location.
The module's own main() is gated behind `if __name__ == "__main__":`, so importing it
never runs a sweep and never writes anything outside this process.

What is under test is the exact resume decision used by cmd_full/cmd_smoke in
scripts/run_sensitivity_sweep.py:

    existing_rows = _load_existing_rows(paths)
    completed_keys = {_row_key(row) for row in existing_rows}
    remaining_tasks = [task for task in grid if task not in completed_keys]

These tests call _load_existing_rows and _row_key from the real module and then
reproduce that same list-comprehension line, against a synthetic checkpoint CSV
built from scratch in tmp_path (not the real checkpoint CSVs under results/, which
are read only once, elsewhere in this project, to learn the column schema this
fixture mirrors). No simulation is run and nothing is written outside tmp_path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_sensitivity_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_sensitivity_sweep_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__name__ != "__main__"
    return module


mod = _load_module()


def _make_row(k: float, broker_count: int, ablation: str, seed: int, mode: str) -> dict:
    """One synthetic checkpoint row, schema-matched to mod.ROW_COLUMNS. Metric
    values are arbitrary fixed dummies: only the five key columns (k,
    broker_count, ablation, seed, capacity_surcharge_mode) matter to the
    resume decision under test."""
    row = {column: 0.0 for column in mod.ROW_COLUMNS}
    row.update(
        {
            "k": k,
            "broker_count": broker_count,
            "ablation": ablation,
            "seed": seed,
            "capacity_surcharge_mode": mode,
            "broker_load_share_json": "{}",
            "mean_broker_surcharge_json": "{}",
            "n_brokers": broker_count,
        }
    )
    return row


def _write_checkpoint(tmp_path: Path, rows: list[dict]) -> dict:
    """Write rows to the checkpoint path _result_paths would use for
    results_dir=tmp_path, and return that same paths dict."""
    paths = mod._result_paths(tmp_path)
    frame = pd.DataFrame(rows).reindex(columns=mod.ROW_COLUMNS)
    frame.to_csv(paths["checkpoint"], index=False)
    return paths


def _remaining(grid: tuple, paths: dict) -> list:
    """The exact resume expression from cmd_full/cmd_smoke (see module
    docstring above), reproduced here so the test exercises the real
    decision rather than a paraphrase of it."""
    existing_rows = mod._load_existing_rows(paths)
    completed_keys = {mod._row_key(row) for row in existing_rows}
    return [task for task in grid if task not in completed_keys]


# ---------------------------------------------------------------------------
# Binding assertion 1: a (config, seed) pair already present in the
# checkpoint is skipped. Also proves the key is the full (k, broker_count,
# ablation, seed, mode) tuple, not the config alone: two tasks share every
# field except seed, only the checkpointed seed is skipped.
# ---------------------------------------------------------------------------


def test_pair_present_in_checkpoint_is_skipped(tmp_path):
    task_done = (1.0, 3, "capacity_both", 111, "proportional")
    task_pending = (1.0, 3, "capacity_both", 222, "proportional")
    grid = (task_done, task_pending)

    paths = _write_checkpoint(tmp_path, [_make_row(*task_done)])
    remaining = _remaining(grid, paths)

    assert task_done not in remaining
    assert task_pending in remaining


# ---------------------------------------------------------------------------
# Binding assertion 2: a (config, seed) pair absent from the checkpoint is
# not skipped, including the cold-start case where no checkpoint file
# exists at all yet (the state before the very first run).
# ---------------------------------------------------------------------------


def test_pair_absent_from_checkpoint_is_not_skipped(tmp_path):
    grid = (
        (0.5, 2, "capacity_disabled", 111, "proportional"),
        (1.5, 5, "capacity_pricing_only", 999, "proportional"),
    )
    paths = mod._result_paths(tmp_path)  # no checkpoint file written: cold start

    remaining = _remaining(grid, paths)

    assert list(remaining) == list(grid)


# ---------------------------------------------------------------------------
# Binding assertion 3: a partially-complete checkpoint resumes at exactly
# the right place. A 6-task grid with a scattered (non-prefix, non-suffix)
# subset marked done, so the test cannot pass by accident of position, e.g.
# a mutant that only skips a leading run of tasks.
# ---------------------------------------------------------------------------


def test_partial_checkpoint_resumes_at_the_right_place(tmp_path):
    grid = mod._build_grid(
        k_values=(0.5, 1.0, 1.5),
        broker_counts=(3,),
        ablations=("capacity_both",),
        seeds=(111, 222),
        modes=("proportional",),
    )
    assert len(grid) == 6

    done_indices = {1, 3, 4}
    done_rows = [_make_row(*grid[i]) for i in sorted(done_indices)]
    paths = _write_checkpoint(tmp_path, done_rows)

    remaining = _remaining(grid, paths)

    expected = [grid[i] for i in range(len(grid)) if i not in done_indices]
    assert remaining == expected
    assert len(remaining) == 3


# ---------------------------------------------------------------------------
# D10 backfill: a checkpoint row written before capacity_surcharge_mode
# existed (column absent from the CSV entirely) still keys as "proportional"
# and is skipped, exactly as an all-modes-proportional pre-D10 sweep expects.
# ---------------------------------------------------------------------------


def test_pre_d10_row_missing_mode_column_backfills_to_proportional_and_is_skipped(tmp_path):
    task = (1.0, 3, "capacity_both", 111, "proportional")
    grid = (task,)

    row = _make_row(*task)
    del row["capacity_surcharge_mode"]  # simulate a pre-D10 checkpoint file
    paths = mod._result_paths(tmp_path)
    frame = pd.DataFrame([row])  # deliberately not reindexed: column genuinely absent
    frame.to_csv(paths["checkpoint"], index=False)

    remaining = _remaining(grid, paths)

    assert remaining == []
