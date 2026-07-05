# REVIEW_ME - reading order for the author

This project is an agent-based microgrid simulation (Mesa) with a novelty-driven capacity
mechanism, built for the diploma thesis "Autonomous Agents (Agentic AI) for Dynamic Load
Management and Energy Exchange in Microgrids." Read the items below in order; each line says what
to look for.

Note on where the writeup lives: the docs/ tree (PROJECT_CONTEXT, PROGRESS, DECISIONS, HARDWARE,
HANDOFF, and the whole docs/thesis/ package) is LOCAL-ONLY by author choice. docs/ is gitignored,
so the thesis writeup is on your disk, not in the GitHub repo. The repo tracks only src/, tests/,
config/, scripts/, results/, and root files like this one. The paths below under docs/ are on your
local disk.

1. docs/DECISIONS.md - the scenario and design choices (D1 to D8) with the underexplored-option
   novelty check for each, the Data provenance note, the Phase 2 fix observations, and the Phase 5
   observation on the bc=5 duplicate-broker jitter fix. This is the intellectual core; confirm you
   agree with each choice and can defend why it is underexplored.

2. results/summary_stats.csv - the numbers. Per (k, broker_count, ablation) mean, std, and t-based
   95 percent CI for the four metrics, plus paired effect sizes and p-values versus the disabled
   baseline. results/effect_sizes.csv splits the metric-3 effect from the metric 1, 2, 4 effects
   and carries the pnl_physical_leak sanity flag. Check that the headline numbers match the prose.

3. docs/thesis/04_capacity_mechanism.md - the contribution. The shared-feeder scarcity charge, the
   two independently ablatable feedback channels (P&L accounting versus pricing signal), and the
   price-elastic demand-deferral physical channel. Confirm capacity_passthrough is described as a
   signal-strength coefficient, never a EUR/kWh price.

4. docs/thesis/06_results.md - what was found. Clean channel isolation (pnl_only equals disabled on
   the physical metrics to about 1e-16); metric 3 improves within a bounded k envelope; the
   coefficient of variation improves at every k while the peak-to-average ratio reverses at high k
   with a broker-count-dependent onset; a modest, k-shrinking cost penalty; metrics 2 and 4 near
   invariant; broker-surcharge heterogeneity; switching behaviour.

5. docs/thesis/07_discussion.md - honest limitations. Tariff signal not a physical grid constraint;
   small-N brokers; bounded demand flexibility and the peak-to-average reversal; the absent bc=1
   monopoly arm; bc=5 near-substitute collapse; representative synthetic demand. Confirm nothing is
   overclaimed.

6. docs/thesis/qa_prep.md - defense answers. Twenty-one anticipated questions grouped Mechanism,
   Novelty, Results, Methodology, and Threats to validity, each with an out-loud answer and a
   pointer to the file or CSV to show. Rehearse the hardest ones (the k=2.0 reversal, why
   passthrough is not a price, what the P&L channel does, the bc=5 collapse, and whether the result
   is about broker competition or the mechanism).

7. docs/thesis/defense_slides.md - the presentation. About 15 slides with speaker notes. Check the
   framing matches the chapters and does not oversell.

8. The Phase 8 audit's weakest spots. The final defense-readiness audit scored the package 8 out of
   10. Its top weak spots are the gap between the fixed abstract's "competition improves outcomes"
   and the body's honest "a capacity-charge pricing channel improves stability within a bounded
   envelope," the fact that only one of the four metrics improves and at a cost, and the
   single-annual-maximum nature of the peak-to-average statistic. The final fix pass addressed these
   in the chapters, slides, and Q&A; the compact Phase 8 log at the end of docs/PROGRESS.md records
   what changed. Read the reframed sections 6, 7, and 8 to confirm you are comfortable defending the
   narrowed claim.

## Reproducing the pipeline

- Tests: python -m pytest -n 16 -q (117 passing).
- Sensitivity sweep: python scripts/run_sensitivity_sweep.py --mode full (1440 runs, resumable,
  writes results/sweep_raw.parquet).
- Analysis and figures: python scripts/analyze_sweep.py (reads the parquet, writes
  results/summary_stats.csv, results/effect_sizes.csv, and results/plots/).
