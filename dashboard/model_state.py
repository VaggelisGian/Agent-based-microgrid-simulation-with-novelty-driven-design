"""Session state for the live thesis-defense dashboard.

Owns exactly one real MicrogridModel instance at a time. Every number shown
in the UI is read from that instance's own public state after a real
model.step() call: nothing here recomputes broker prices, agent decisions,
or the capacity mechanism's arithmetic.

Three sources feed the display history, none of which alter model behaviour
or the RNG stream:

1. Direct reads of already-public attributes (model.feeder_net_import_history,
   model.broker_surcharges, agent.last_broker_id, agent.last_net_import_kwh,
   broker.cumulative_energy_served_kwh, ...).
2. The real metrics module (microgrid_sim.environment.metrics), called via
   model.compute_metrics(), the same method scripts/run_baseline.py calls.
3. A transparent wrapper around CapacityMechanism.step that records its
   already-computed return value (threshold, excess, charge, allocations,
   surcharges) for the rolling-threshold chart overlay. The wrapper calls the
   original method unchanged and returns its result unchanged; it draws no
   random numbers and does not touch model.random, so it cannot perturb the
   RNG stream consumed elsewhere in the model.

Broker-count and ablation config construction reuses
scripts/run_sensitivity_sweep.py's build_run_config / _build_broker_list,
the same helper scripts/run_monopoly_arm.py already reuses for its bc=1 arm,
so the dashboard's four ablations and broker-count arms are built exactly the
way the thesis sweep builds them, not a second, divergent implementation.
"""

from __future__ import annotations

import sys
import time
import warnings
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _path in (_SRC_DIR, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Matches scripts/run_baseline.py and scripts/run_sensitivity_sweep.py: Mesa
# 3.5.1 still supports seed= (the API this whole codebase is built on) but
# nudges toward rng=; functionally identical per Mesa's own message.
warnings.filterwarnings(
    "ignore", message="The use of the `seed` keyword argument is deprecated", category=FutureWarning
)

from microgrid_sim.config.loader import load_config  # noqa: E402
from microgrid_sim.environment.model import MicrogridModel  # noqa: E402
from microgrid_sim.environment import metrics as metrics_module  # noqa: E402

import run_sensitivity_sweep as sweep_lib  # noqa: E402  (config construction reuse)

DEMO_SEED = 20260704  # config/default.yaml's own canonical seed, unchanged
HORIZON_HOURS = sweep_lib.HORIZON_HOURS  # 8760: the mandated one-year horizon (D8)
WINDOW_HOURS = sweep_lib.WINDOW_HOURS  # 168: the mandated rolling-threshold window (D8)

ABLATIONS = sweep_lib.ABLATIONS
BROKER_COUNTS = (1, 2, 3, 5)  # every broker_count build_run_config supports (D8 + the bc=1 monopoly arm)

DEFAULT_K = 1.0
DEFAULT_ABLATION = "capacity_both"
DEFAULT_BROKER_COUNT = 3

# Rolling display window for the live charts (memory/render bound only; the
# model's own history and every metric are still computed over the FULL run,
# unaffected by this trim).
DISPLAY_WINDOW_HOURS = 500

_BASE_CONFIG = load_config(str(_REPO_ROOT / "config" / "default.yaml"))

ABLATION_LABELS = {
    "capacity_disabled": "Disabled (plain baseline)",
    "capacity_pnl_only": "P&L only (ledger debit, no price signal)",
    "capacity_pricing_only": "Pricing only (surcharge, no ledger debit)",
    "capacity_both": "Both channels (full mechanism)",
}

BROKER_COLORS = {
    "regulated_utility": "#4C72B0",
    "volatile_low_cost": "#DD8452",
    "premium_green": "#55A868",
    "volatile_low_cost_dup2": "#C44E52",
    "premium_green_dup2": "#8172B2",
}
_FALLBACK_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#64B5CD"]


def broker_color(broker_id: str, index: int) -> str:
    return BROKER_COLORS.get(broker_id, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)])


def _instrument_capacity(model: MicrogridModel) -> deque:
    """Wrap model._capacity.step (if the mechanism is enabled) so every
    CapacityStepResult it computes is also recorded here, unmodified, for the
    threshold/scarcity chart overlay. Pure instrumentation: the wrapper calls
    the original bound method and returns its result unchanged, so nothing
    about what the mechanism computes, or the RNG stream, is altered."""
    capacity = getattr(model, "_capacity", None)
    history: deque = deque(maxlen=DISPLAY_WINDOW_HOURS)
    if capacity is not None:
        original_step = capacity.step

        def _wrapped(feeder_net_import_kwh, broker_contributions_kwh, _original=original_step, _history=history):
            result = _original(feeder_net_import_kwh, broker_contributions_kwh)
            _history.append(result)
            return result

        capacity.step = _wrapped
    return history


def build_demo_config(k: float, broker_count: int, ablation: str, horizon_hours: int = HORIZON_HOURS) -> dict:
    """The exact config the sweep would build for this (k, broker_count,
    ablation) cell, at the project's canonical seed. Reuses
    run_sensitivity_sweep.build_run_config verbatim (see module docstring)."""
    return sweep_lib.build_run_config(_BASE_CONFIG, k, broker_count, ablation, DEMO_SEED, horizon_hours=horizon_hours)


class DemoSession:
    """One live MicrogridModel plus the display-history buffers derived from
    reading its public state. reset() is the only place a new model is ever
    constructed; every buffer is rebuilt from scratch there so no state from
    a previous (k, ablation, broker_count) choice can leak into the next."""

    def __init__(self):
        self.model: MicrogridModel | None = None
        self.k = DEFAULT_K
        self.ablation = DEFAULT_ABLATION
        self.broker_count = DEFAULT_BROKER_COUNT
        self.running = False

        self.hours: deque = deque()
        self.feeder: deque = deque()
        self.threshold: deque = deque()
        self.scarcity: deque = deque()
        self.broker_share_history: dict[str, deque] = {}
        self.surcharge_current: dict[str, float] = {}
        self.deferred_total_kwh = 0.0
        self.deferred_last_step_kwh = 0.0
        self.metrics = None
        self._capacity_history: deque = deque()
        self._last_batch_wall_sec: float | None = None
        self._last_batch_steps = 0

        self.reset(self.k, self.ablation, self.broker_count)

    # -- lifecycle -----------------------------------------------------

    def reset(self, k: float, ablation: str, broker_count: int) -> None:
        """Build a brand new MicrogridModel for (k, ablation, broker_count)
        and discard every buffer from whatever ran before. This is the only
        method that constructs a model, so "clean reset" is structural, not
        a convention callers have to remember to honour."""
        config = build_demo_config(k, broker_count, ablation)
        model = MicrogridModel(config)

        self.model = model
        self.k = k
        self.ablation = ablation
        self.broker_count = broker_count
        self.running = False

        self.hours = deque(maxlen=DISPLAY_WINDOW_HOURS)
        self.feeder = deque(maxlen=DISPLAY_WINDOW_HOURS)
        self.threshold = deque(maxlen=DISPLAY_WINDOW_HOURS)
        self.scarcity = deque(maxlen=DISPLAY_WINDOW_HOURS)
        self.broker_share_history = {broker_id: deque(maxlen=DISPLAY_WINDOW_HOURS) for broker_id in model.brokers}
        self.surcharge_current = {broker_id: 0.0 for broker_id in model.brokers}
        self.deferred_total_kwh = 0.0
        self.deferred_last_step_kwh = 0.0
        self._capacity_history = _instrument_capacity(model)
        self._last_batch_wall_sec = None
        self._last_batch_steps = 0

        self.metrics = model.compute_metrics()

    @property
    def finished(self) -> bool:
        return self.model.current_hour >= self.model.horizon_hours

    @property
    def broker_names(self) -> dict[str, str]:
        return {broker_id: broker.name for broker_id, broker in self.model.brokers.items()}

    # -- stepping --------------------------------------------------------

    def _step_once(self) -> None:
        model = self.model
        prev_deferred = metrics_module.compute_deferral_audit(model)

        model.step()  # the one and only call into the real model per hour

        hour = model.current_hour - 1
        # Order of operations matters here and must match the model's own.
        # environment/model.py sums the SIGNED per-agent net import per broker,
        # and environment/capacity.py then clips each broker TOTAL at zero
        # (positive_contrib). Clipping per agent first and summing afterwards is
        # a different quantity whenever one broker serves both importing and
        # exporting customers in the same hour, which is every sunny hour: a
        # prosumer exporting 2 kWh no longer cancels a neighbour importing 2 kWh.
        # Measured on the shipped config at k=1.0, bc=3 over 2000 hours, the two
        # orderings disagreed in 17.8 percent of hours, by up to 38 percentage
        # points of share. The model does not compute this quantity at all when
        # the mechanism is disabled, so the dashboard cannot simply read it; it
        # has to apply the same formula in the same order.
        signed_contributions: dict[str, float] = {broker_id: 0.0 for broker_id in model.brokers}
        for agent in model.agents:
            signed_contributions[agent.last_broker_id] += agent.last_net_import_kwh
        contributions = {broker_id: max(0.0, value) for broker_id, value in signed_contributions.items()}
        total = sum(contributions.values())

        self.hours.append(hour)
        self.feeder.append(model.feeder_net_import_history[-1])
        for broker_id in model.brokers:
            share = (contributions.get(broker_id, 0.0) / total) if total > 0.0 else 0.0
            self.broker_share_history[broker_id].append(share)

        if self._capacity_history:
            result = self._capacity_history[-1]
            self.threshold.append(result.threshold_eur)
            self.scarcity.append(result.excess_kwh > 0.0)
        else:
            self.threshold.append(None)
            self.scarcity.append(False)

        self.surcharge_current = dict(model.broker_surcharges)
        new_deferred = metrics_module.compute_deferral_audit(model)
        self.deferred_last_step_kwh = new_deferred - prev_deferred
        self.deferred_total_kwh = new_deferred

    def step_batch(self, n: int) -> int:
        """Advance up to n hours (fewer if the horizon ends first). Returns
        how many hours were actually stepped, and records real wall-clock
        time for the live steps/sec readout."""
        start = time.perf_counter()
        performed = 0
        for _ in range(n):
            if self.finished:
                break
            self._step_once()
            performed += 1
        self._last_batch_wall_sec = time.perf_counter() - start
        self._last_batch_steps = performed
        self.metrics = self.model.compute_metrics()
        return performed

    def steps_per_second(self) -> float | None:
        if not self._last_batch_wall_sec or self._last_batch_steps == 0:
            return None
        return self._last_batch_steps / self._last_batch_wall_sec
