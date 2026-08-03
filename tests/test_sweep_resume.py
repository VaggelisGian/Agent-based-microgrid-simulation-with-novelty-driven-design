"""Tests for the sweep-runner resume/skip logic (Phase 20.5 reproducibility audit).

scripts/ is not an importable package, so this module loads
scripts/run_sensitivity_sweep.py the same way tests/test_analyze_structural_sweep.py
loads scripts/analyze_structural_sweep.py: via importlib.util.spec_from_file_location.
The module's own main() is gated behind `if __name__ == "__main__":`, so importing it
never runs a sweep and never writes anything outside this process.

What is under test is the resume decision cmd_full and cmd_smoke each make once per
call, at run_sensitivity_sweep.py:821 and :784 respectively:

    existing_rows = _load_existing_rows(paths)
    completed_keys = {_row_key(row) for row in existing_rows}
    remaining_tasks = [task for task in grid if task not in completed_keys]

An earlier version of this file tested that decision by re-typing this same
expression into a local helper and calling _row_key/_load_existing_rows directly.
That is a lookalike, not the thing itself: it exercises the two functions the
expression calls, but never executes the expression as it actually appears inside
cmd_full/cmd_smoke, so a mutation to line 821 or 784 alone (for example inverting
`not in` to `in`) was invisible to it -- confirmed by review, see
docs/verification/reproducibility.md section 4.

This version calls the real cmd_full and cmd_smoke instead, so lines 821 and 784
genuinely execute. The one per-task side effect they make, _run_one_task (which
builds a config and runs a real year-long MicrogridModel simulation), is
monkeypatched to a recorder that appends the task it was called with and returns a
cheap synthetic row; multiprocessing.Pool is monkeypatched to an in-process
stand-in for the same reason. Windows' multiprocessing start method is "spawn":
a real Pool re-imports this module fresh inside each worker process, so a
monkeypatch applied to the module object this test holds (a one-off object from
spec_from_file_location, not even reachable via a normal `import` from a fresh
interpreter) would not exist in the worker's copy, and the worker would run the
real, unpatched _run_one_task, defeating the point and doing real simulation work.
Routing everything through one in-process synchronous stand-in for
multiprocessing.Pool sidesteps that entirely: no subprocess is ever started, so
the patched _run_one_task is what actually runs. Every other line cmd_full and
cmd_smoke execute -- the resume decision itself, _load_existing_rows, _row_key,
_estimate_via_smoke's bookkeeping, _run_main_loop's checkpointing loop -- is the
real, unmodified code. All result/checkpoint output is confined to tmp_path.
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
    """One synthetic row, schema-matched to mod.ROW_COLUMNS. Metric values are
    arbitrary fixed dummies (0.0): only the five key columns (k, broker_count,
    ablation, seed, capacity_surcharge_mode) matter to the resume decision
    under test, and only run_wallclock_sec/peak_rss_mb are read for anything
    else (by _estimate_via_smoke's averaging), where 0.0 is a valid value."""
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


def _write_checkpoint(tmp_path: Path, rows: list[dict]) -> None:
    """Write rows to the checkpoint path _result_paths would use for
    results_dir=tmp_path (mirrors the schema of a real checkpoint file,
    results/sweep_opsd_headline_checkpoint.csv, read once elsewhere in this
    project to learn that schema, never read or written by this test)."""
    paths = mod._result_paths(tmp_path)
    frame = pd.DataFrame(rows).reindex(columns=mod.ROW_COLUMNS)
    frame.to_csv(paths["checkpoint"], index=False)


# ---------------------------------------------------------------------------
# The seam: an in-process stand-in for multiprocessing.Pool, and a recorder
# used as a drop-in replacement for _run_one_task. Both are installed via
# pytest's monkeypatch fixture, which reverts them after each test.
# ---------------------------------------------------------------------------


class _SynchronousPool:
    """Same context-manager/map/imap_unordered surface real Pool objects
    give _run_batch and _run_main_loop, but everything runs synchronously in
    this process instead of a spawned worker (see the module docstring for
    why a real Pool cannot be used here with a patched _run_one_task)."""

    def __init__(self, processes=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, func, iterable, chunksize=1):
        return [func(item) for item in iterable]

    def imap_unordered(self, func, iterable, chunksize=1):
        for item in iterable:
            yield func(item)


class _FakeMultiprocessingModule:
    Pool = _SynchronousPool


def _make_fake_run_one_task(recorder: list):
    """Records the task it was called with instead of building a config and
    running a real MicrogridModel; returns a cheap synthetic row so the
    caller's bookkeeping (averaging run_wallclock_sec, writing a checkpoint)
    has something schema-shaped to work with."""

    def _fake_run_one_task(task: tuple) -> dict:
        recorder.append(task)
        return _make_row(*task)

    return _fake_run_one_task


def _patch_pool_and_recorder(monkeypatch) -> list:
    recorder: list = []
    monkeypatch.setattr(mod, "multiprocessing", _FakeMultiprocessingModule())
    monkeypatch.setattr(mod, "_run_one_task", _make_fake_run_one_task(recorder))
    return recorder


# ---------------------------------------------------------------------------
# Binding assertion 1 (+ seed-matters, + self-sufficient against both trivial
# degenerate implementations: "always treat everything as done" and "never
# treat anything as done" would each fail this test, since it contains one
# checkpointed and one non-checkpointed task and checks both directions).
# ---------------------------------------------------------------------------


def test_cmd_full_pair_present_is_skipped_pair_absent_is_not(tmp_path, monkeypatch):
    task_done = (1.0, 3, "capacity_both", 111, "proportional")
    task_pending = (1.0, 3, "capacity_both", 222, "proportional")
    grid = (task_done, task_pending)
    _write_checkpoint(tmp_path, [_make_row(*task_done)])
    recorder = _patch_pool_and_recorder(monkeypatch)

    mod.cmd_full(
        base_config={},
        workers=1,
        horizon_hours=mod.HORIZON_HOURS,
        smoke_size=5,
        results_dir=tmp_path,
        grid=grid,
        total_runs=len(grid),
        k_values=(1.0,),
        modes=("proportional",),
        output_path=None,
    )

    assert recorder == [task_pending]
    assert task_done not in recorder


def test_cmd_smoke_pair_present_is_skipped_pair_absent_is_not(tmp_path, monkeypatch):
    """Same fixture as above, through cmd_smoke instead, so line 784 (the
    twin of line 821, cmd_smoke's own copy of the same resume decision)
    is exercised too, not just cmd_full's."""
    task_done = (0.5, 3, "capacity_pnl_only", 777, "proportional")
    task_pending = (0.5, 3, "capacity_pnl_only", 888, "proportional")
    grid = (task_done, task_pending)
    _write_checkpoint(tmp_path, [_make_row(*task_done)])
    recorder = _patch_pool_and_recorder(monkeypatch)

    mod.cmd_smoke(
        base_config={},
        workers=1,
        horizon_hours=mod.HORIZON_HOURS,
        smoke_size=5,
        results_dir=tmp_path,
        grid=grid,
        total_runs=len(grid),
        k_values=(0.5,),
        modes=("proportional",),
        output_path=None,
    )

    assert recorder == [task_pending]
    assert task_done not in recorder


# ---------------------------------------------------------------------------
# Binding assertion 2: a (config, seed) pair absent from the checkpoint is
# not skipped, cold start (no checkpoint file at all). Not individually
# self-sufficient: a degenerate "always treat every task as remaining"
# implementation would also pass this one. See test_degenerate_notes below
# for why the suite as a whole still rules that out.
# ---------------------------------------------------------------------------


def test_cmd_full_cold_start_with_no_checkpoint_runs_every_task(tmp_path, monkeypatch):
    grid = (
        (0.5, 2, "capacity_disabled", 111, "proportional"),
        (1.5, 5, "capacity_pricing_only", 999, "proportional"),
    )
    # No checkpoint file written at all: the state before the first run.
    recorder = _patch_pool_and_recorder(monkeypatch)

    mod.cmd_full(
        base_config={},
        workers=1,
        horizon_hours=mod.HORIZON_HOURS,
        smoke_size=5,
        results_dir=tmp_path,
        grid=grid,
        total_runs=len(grid),
        k_values=(0.5, 1.5),
        modes=("proportional",),
        output_path=None,
    )

    assert sorted(recorder) == sorted(grid)


# ---------------------------------------------------------------------------
# Binding assertion 3: a partially-complete checkpoint resumes at exactly
# the right place. A 6-task grid with a scattered (non-prefix, non-suffix)
# subset marked done, and smoke_size set below the remaining count so both
# the smoke-estimate batch and the main resumable loop each run part of the
# remainder -- both of cmd_full's two call sites for _run_one_task are
# exercised, not just one. Self-sufficient against both trivial degenerates
# for the same reason as the pair test above: it contains both done and
# not-done tasks and checks the exact set of each.
# ---------------------------------------------------------------------------


def test_cmd_full_partial_checkpoint_resumes_at_the_right_place(tmp_path, monkeypatch):
    grid = mod._build_grid(
        k_values=(0.5, 1.0, 1.5),
        broker_counts=(3,),
        ablations=("capacity_both",),
        seeds=(111, 222),
        modes=("proportional",),
    )
    assert len(grid) == 6

    done_indices = {1, 3, 4}
    _write_checkpoint(tmp_path, [_make_row(*grid[i]) for i in sorted(done_indices)])
    recorder = _patch_pool_and_recorder(monkeypatch)

    result_rows = mod.cmd_full(
        base_config={},
        workers=1,
        horizon_hours=mod.HORIZON_HOURS,
        smoke_size=1,  # smaller than the 3 remaining: forces the main loop to run too
        results_dir=tmp_path,
        grid=grid,
        total_runs=len(grid),
        k_values=(0.5, 1.0, 1.5),
        modes=("proportional",),
        output_path=None,
    )

    expected_remaining = [grid[i] for i in range(len(grid)) if i not in done_indices]
    assert sorted(recorder) == sorted(expected_remaining)
    assert len(recorder) == 3
    assert len(result_rows) == 6  # 3 pre-existing + 3 newly recorded


# ---------------------------------------------------------------------------
# Fully-complete checkpoint: cmd_full's early-exit branch (nothing left to
# do), the cleanest possible proof that line 821 genuinely ran, since a
# non-empty remaining_tasks here would have gone on to call
# _estimate_via_smoke, which this test's patches never touch (no smoke
# batch happens on this path, patched or not). Not individually
# self-sufficient against "always treat everything as done"; see below.
# ---------------------------------------------------------------------------


def test_cmd_full_all_done_takes_the_early_exit_with_zero_runs(tmp_path, monkeypatch):
    grid = (
        (1.0, 3, "capacity_both", 111, "proportional"),
        (1.0, 3, "capacity_both", 222, "proportional"),
    )
    _write_checkpoint(tmp_path, [_make_row(*task) for task in grid])
    recorder = _patch_pool_and_recorder(monkeypatch)

    result_rows = mod.cmd_full(
        base_config={},
        workers=1,
        horizon_hours=mod.HORIZON_HOURS,
        smoke_size=5,
        results_dir=tmp_path,
        grid=grid,
        total_runs=len(grid),
        k_values=(1.0,),
        modes=("proportional",),
        output_path=None,
    )

    assert recorder == []
    assert len(result_rows) == len(grid)


# ---------------------------------------------------------------------------
# D10 backfill: a checkpoint row written before capacity_surcharge_mode
# existed (column absent from the CSV entirely) still keys as "proportional"
# and is skipped. Individually passable by an "always empty" degenerate,
# same caveat as the all-done test above.
# ---------------------------------------------------------------------------


def test_cmd_full_pre_d10_row_missing_mode_column_backfills_and_is_skipped(tmp_path, monkeypatch):
    task = (1.0, 3, "capacity_both", 111, "proportional")
    grid = (task,)

    row = _make_row(*task)
    del row["capacity_surcharge_mode"]  # simulate a pre-D10 checkpoint file
    paths = mod._result_paths(tmp_path)
    frame = pd.DataFrame([row])  # deliberately not reindexed: column genuinely absent
    frame.to_csv(paths["checkpoint"], index=False)
    recorder = _patch_pool_and_recorder(monkeypatch)

    result_rows = mod.cmd_full(
        base_config={},
        workers=1,
        horizon_hours=mod.HORIZON_HOURS,
        smoke_size=5,
        results_dir=tmp_path,
        grid=grid,
        total_runs=len(grid),
        k_values=(1.0,),
        modes=("proportional",),
        output_path=None,
    )

    assert recorder == []
    assert len(result_rows) == 1


# ---------------------------------------------------------------------------
# What the tests above do and do not individually prove (see
# docs/verification/reproducibility.md section 4 for the fuller version):
# test_cmd_full_cold_start_with_no_checkpoint_runs_every_task would also pass
# under a constant "remaining = full grid" implementation; test_cmd_full_
# all_done_takes_the_early_exit_with_zero_runs and test_cmd_full_pre_d10_
# row_missing_mode_column_backfills_and_is_skipped would each also pass
# under a constant "remaining = []" implementation, since each of those
# three tests' correct answer happens to coincide with one of those two
# constants. test_cmd_full_pair_present_is_skipped_pair_absent_is_not and
# test_cmd_full_partial_checkpoint_resumes_at_the_right_place each contain
# both a checkpointed and a non-checkpointed task in the same call and
# check both directions, so each fails under both constants on its own;
# the suite's protection against either degenerate does not depend on the
# three non-mixed tests.
# ---------------------------------------------------------------------------
