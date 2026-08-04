# Microgrid broker simulation

Agent-based simulation (Mesa 3.5.1) of a neighborhood microgrid in which three fixed-rule broker
agents compete for a mixed population of consumers and prosumers. A shared-feeder scarcity charge,
allocated by each broker's contribution to the peak, feeds back into broker ledgers and next-step
quotes, and the model measures whether that competition can coexist with a more stable feeder.

No reinforcement learning and no grid physics, on purpose. The scarcity charge is a billing
signal, not a power-flow quantity, and its passthrough coefficient is not a EUR/kWh price.

## Install

Needs Python 3.12+. From the repo root:

    python -m venv .venv
    .venv\Scripts\activate          # Windows
    source .venv/bin/activate       # Linux/macOS
    pip install -r requirements.txt

All versions are pinned to what the results were produced with. Note that networkx is listed even
though this project never imports it directly: mesa needs it and does not declare it, so a clean
install fails at `import mesa` without it.

## Tests

    python -m pytest -n 16 -q

About a minute on 16 cores once bytecode is warm; expect a few minutes on the first run in a
fresh environment. Three tests skip by default; they re-run the 8760-hour feeder statistics and
take several minutes, and run with `RUN_SLOW_TESTS=1` set.

## Running the model

    python scripts/run_baseline.py

runs the deterministic 8760-hour single-broker monopoly baseline
(`config/scenarios/monopoly_baseline.yaml`) and prints the four metrics (average cost per agent,
load spread across brokers, feeder stability, prosumer self-sufficiency). The competitive
three-broker scenario and the fixed seed live in `config/default.yaml`; the sensitivity and
structural sweeps below run it across the full grid, and the two monopoly arms run single-broker
controls against it.

## Results

Sweep outputs, summary CSVs and figures live under `results/`. The committed artifacts there are
the pinned outputs the analysis is based on; the test suite asserts the claim-bearing numbers in
them against golden copies under `tests/golden/`, so an analysis regeneration that drifts fails
the tests rather than silently replacing a number.

The sweep entry points (`scripts/run_sensitivity_sweep.py`, `scripts/run_structural_sweep.py`,
`scripts/run_monopoly_arm.py`, `scripts/run_dose_matched_monopoly.py`) are multi-hour runs. They
checkpoint as they go and are resumable: re-running skips every (config, seed) cell already
present in the checkpoint, so an interrupted sweep continues where it stopped rather than starting
over. `scripts/analyze_sweep.py` builds the statistics and figures from the parquet outputs.
