"""Sensitivity of the multiple-comparison conclusion to how the correction family is drawn.

Run from the repository root:

    python scripts/family_sensitivity.py

WHAT THIS READS (read-only, never modified):
  results/summary_stats_corrected.csv   the main D8 sweep's per-comparison rows, plus the
                                        supplementary broker_count=1 monopoly arm's rows
  results/structural_sensitivity.csv    the D9 structural-coefficient sweep's per-test rows
  results/monopoly_comparison.csv       the monopoly arm's per-k paired p-values, used only as
                                        an independent cross-check of the monopoly p-values
                                        already carried in summary_stats_corrected.csv

WHAT THIS WRITES:
  results/family_sensitivity.csv        and nothing else. No other file is created, modified,
                                        or deleted anywhere in the repository.

WHAT THIS DOES NOT DO:
  It re-runs no simulation. It reads no parquet. It recomputes no t-test. It takes the
  already-computed uncorrected p-values as given and only re-applies Holm-Bonferroni and
  Benjamini-Hochberg FDR corrections at alpha=0.05 under seven different definitions of the
  correction family, so that the robustness of the headline conclusion to the family choice
  is recorded in a shipped artifact rather than asserted.

The seven family definitions:
  F1 main_sweep_primary_120          the shipped primary family: the main sweep's real
                                     (non-degenerate) tests. Included as the reference.
  F2 main_sweep_with_degenerates_180 F1 plus the 60 capacity_pnl_only contrasts that are
                                     deterministic identities (p=1.0 by convention).
  F3 per_metric_24                   each metric corrected inside its own family.
  F4 pooled_with_monopoly            F1 pooled with the supplementary broker_count=1 arm.
  F5 everything_pooled               the D8 main sweep's real tests, the monopoly arm and the
                                     D9 structural sweep, in one family.
  F6 per_cell_k_by_broker_count      family = the tests sharing a single (k, broker_count) cell.
  F7 everything_pooled_with_degenerates
                                     the union of F2 and F5: F5 plus the 60 deterministic
                                     identities. The widest family this script computes.

F5 and F7 do NOT include the Phase 12 OPSD paired tests of Section 6.12, which are real paired
tests but carry no p-value column in results/demand_source_comparison.csv, so there is nothing
here to pool. "Everything" therefore means the D8 main sweep, the broker_count=1 monopoly arm
and the D9 structural sweep, not literally every paired test in the thesis.

Determinism: the script sorts every table explicitly, uses stable sorts, and writes with LF
line endings, so two consecutive runs produce a byte-identical results/family_sensitivity.csv.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

SUMMARY_CORRECTED_CSV = RESULTS_DIR / "summary_stats_corrected.csv"
STRUCTURAL_CSV = RESULTS_DIR / "structural_sensitivity.csv"
MONOPOLY_CSV = RESULTS_DIR / "monopoly_comparison.csv"
OUTPUT_CSV = RESULTS_DIR / "family_sensitivity.csv"

# The three metrics that make up the headline claim. Load-concentration HHI and prosumer
# self-sufficiency are reported but are not part of the 72.
HEADLINE_METRICS = (
    "feeder_coefficient_of_variation",
    "feeder_peak_to_average_ratio",
    "avg_cost_per_agent_eur",
)
HEADLINE_ABLATIONS = ("capacity_pricing_only", "capacity_both")

OUTPUT_COLUMNS = [
    "family_definition",
    "family_key",
    "family_size",
    "n_families_in_definition",
    "scope",
    "source",
    "k",
    "broker_count",
    "ablation",
    "cell_is_stamped_not_native",
    "metric",
    "test_id",
    "is_headline_72",
    "degenerate",
    "p_uncorrected",
    "p_holm",
    "p_bh",
    "survives_holm_alpha05",
    "survives_bh_alpha05",
    "alpha",
    "n_headline_tests",
    "n_headline_surviving_holm",
    "n_headline_surviving_bh",
    "worst_headline_p_uncorrected",
    "worst_headline_p_holm",
    "worst_headline_p_bh",
    "bonferroni_alpha_over_family_size",
    "notes",
]


# ---------------------------------------------------------------------------
# Correction procedures, implemented here and cross-checked against statsmodels
# ---------------------------------------------------------------------------


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values.

    Adjusted p for the i-th smallest raw p (0-based) is the running maximum of
    (m - i) * p_(i), then clipped at 1. p exactly 0.0 and denormal p are handled
    without special-casing: 0 * m is 0, and the running maximum stays monotone.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    multipliers = (m - np.arange(m)).astype(float)
    adjusted_sorted = np.maximum.accumulate(multipliers * p_sorted)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted_sorted
    return out


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p-values (step-up).

    Adjusted p for the i-th smallest raw p (0-based) is the running minimum, taken from the
    largest p downwards, of (m / (i + 1)) * p_(i), then clipped at 1.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    scaled = (m / ranks) * p_sorted
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted_sorted
    return out


def cross_check_against_statsmodels(p_values: np.ndarray) -> tuple[float, float] | None:
    """Return (max abs Holm disagreement, max abs BH disagreement) versus statsmodels."""
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return None
    _, sm_holm, _, _ = multipletests(p_values, alpha=ALPHA, method="holm")
    _, sm_bh, _, _ = multipletests(p_values, alpha=ALPHA, method="fdr_bh")
    return (
        float(np.max(np.abs(sm_holm - holm_adjust(p_values)))),
        float(np.max(np.abs(sm_bh - bh_adjust(p_values)))),
    )


# ---------------------------------------------------------------------------
# Building the pool of tests
# ---------------------------------------------------------------------------


def load_test_pool() -> pd.DataFrame:
    """Assemble every paired test in the thesis into one long table.

    Columns: source, scope, k, broker_count, ablation, metric, test_id, p_uncorrected,
    degenerate, is_headline_72.
    """
    # float_precision="round_trip" everywhere: pandas' default float parser is not
    # round-trip exact, and without this the p-values echoed into the artifact differ from
    # the source decimals by up to ~1e-13 relative, which would be an artifact of parsing
    # rather than of the correction being studied.
    summary = pd.read_csv(SUMMARY_CORRECTED_CSV, float_precision="round_trip")
    structural = pd.read_csv(STRUCTURAL_CSV, float_precision="round_trip")

    rows = []

    # --- main sweep, real tests and deterministic identities -------------------------
    main = summary[summary["scope"] == "main_sweep"].copy()
    for _, r in main.iterrows():
        degenerate = bool(r["degenerate"])
        rows.append(
            {
                "source": "main_sweep_degenerate" if degenerate else "main_sweep_real",
                "scope": "main_sweep",
                "k": float(r["k"]),
                "broker_count": int(r["broker_count"]),
                "ablation": str(r["ablation"]),
                "cell_is_stamped_not_native": False,
                "metric": str(r["metric"]),
                "test_id": "main|k={:g}|bc={:d}|{}|{}".format(
                    float(r["k"]), int(r["broker_count"]), r["ablation"], r["metric"]
                ),
                "p_uncorrected": float(r["p_uncorrected"]),
                "degenerate": degenerate,
                "shipped_p_holm": float(r["p_holm"]),
            }
        )

    # --- supplementary broker_count=1 monopoly arm -----------------------------------
    mono = summary[summary["scope"] == "monopoly_supplement"].copy()
    for _, r in mono.iterrows():
        rows.append(
            {
                "source": "monopoly_supplement",
                "scope": "monopoly_supplement",
                "k": float(r["k"]),
                "broker_count": int(r["broker_count"]),
                "ablation": str(r["ablation"]),
                "cell_is_stamped_not_native": False,
                "metric": str(r["metric"]),
                "test_id": "monopoly|k={:g}|bc={:d}|{}|{}".format(
                    float(r["k"]), int(r["broker_count"]), r["ablation"], r["metric"]
                ),
                "p_uncorrected": float(r["p_uncorrected"]),
                "degenerate": bool(r["degenerate"]),
                "shipped_p_holm": float(r["p_holm"]),
            }
        )

    # --- D9 structural sweep ---------------------------------------------------------
    # Rows with no correction_family are the per-coefficient verdict summaries, not tests.
    struct = structural[structural["correction_family"].notna()].copy()
    for _, r in struct.iterrows():
        if r["scope"] == "cell_full_factorial":
            ident = "struct_cell|df={}|rr={}|pcf={}|{}".format(
                r["deferrable_fraction"],
                r["response_reference_eur_per_kwh"],
                r["payback_cap_fraction"],
                r["metric"],
            )
        else:
            ident = "struct_marginal|{}={:g}|{}".format(
                r["coefficient"], float(r["coefficient_level"]), r["metric"]
            )
        rows.append(
            {
                "source": "structural_" + str(r["scope"]),
                "scope": "structural_sweep",
                # The D9 sweep is run at the headline cell: k=1.0, broker_count=3,
                # capacity_both vs capacity_disabled. The k / broker_count / ablation values
                # below are therefore a STAMP describing where D9 was run, not a native
                # coordinate of the row, and cell_is_stamped_not_native records that so a
                # reader who groups the artifact by (k, broker_count, ablation) does not read
                # these 118 rows as extra tests sitting at the headline cell. Family
                # assignment never uses the stamp: no definition keyed on (k, broker_count)
                # includes the D9 rows.
                "k": 1.0,
                "broker_count": 3,
                "ablation": "capacity_both",
                "cell_is_stamped_not_native": True,
                "metric": str(r["metric"]),
                "test_id": ident,
                "p_uncorrected": float(r["p_uncorrected"]),
                "degenerate": bool(r["degenerate"]),
                "shipped_p_holm": float(r["p_holm"]),
            }
        )

    pool = pd.DataFrame(rows)

    # The 72 headline comparisons live in the main sweep's real tests only.
    pool["is_headline_72"] = (
        (pool["source"] == "main_sweep_real")
        & pool["metric"].isin(HEADLINE_METRICS)
        & pool["ablation"].isin(HEADLINE_ABLATIONS)
    )

    if pool["test_id"].duplicated().any():
        dupes = pool.loc[pool["test_id"].duplicated(keep=False), "test_id"].tolist()
        raise RuntimeError("non-unique test_id values: {}".format(sorted(set(dupes))))

    pool = pool.sort_values(
        ["source", "metric", "ablation", "k", "broker_count", "test_id"], kind="stable"
    ).reset_index(drop=True)
    return pool


def family_assignments(pool: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {family_definition: sub-pool with a family_key column}."""
    is_main_real = pool["source"] == "main_sweep_real"
    is_main_degen = pool["source"] == "main_sweep_degenerate"
    is_mono = pool["source"] == "monopoly_supplement"
    is_struct = pool["source"].str.startswith("structural_")

    definitions: dict[str, pd.DataFrame] = {}

    f1 = pool[is_main_real].copy()
    f1["family_key"] = "main_sweep_primary_120"
    definitions["F1_main_sweep_primary_120"] = f1

    f2 = pool[is_main_real | is_main_degen].copy()
    f2["family_key"] = "main_sweep_with_degenerates_180"
    definitions["F2_main_sweep_with_degenerates_180"] = f2

    f3 = pool[is_main_real].copy()
    f3["family_key"] = "metric=" + f3["metric"]
    definitions["F3_per_metric_24"] = f3

    f4 = pool[is_main_real | is_mono].copy()
    f4["family_key"] = "pooled_with_monopoly"
    definitions["F4_pooled_with_monopoly"] = f4

    f5 = pool[is_main_real | is_mono | is_struct].copy()
    f5["family_key"] = "everything_pooled"
    definitions["F5_everything_pooled"] = f5

    f6 = pool[is_main_real].copy()
    f6["family_key"] = (
        "k=" + f6["k"].map(lambda v: "{:g}".format(v)) + "|bc=" + f6["broker_count"].astype(str)
    )
    definitions["F6_per_cell_k_by_broker_count"] = f6

    # F7 crosses the two axes the other definitions vary separately: "fold the deterministic
    # identities back in" (F2) and "pool the monopoly arm and D9 in" (F5). It is the union of
    # the two and the widest family here.
    f7 = pool[is_main_real | is_main_degen | is_mono | is_struct].copy()
    f7["family_key"] = "everything_pooled_with_degenerates"
    definitions["F7_everything_pooled_with_degenerates"] = f7

    return definitions


FAMILY_NOTES = {
    "F1_main_sweep_primary_120": (
        "shipped primary family: the main D8 sweep's 120 real (non-degenerate) paired tests; "
        "reference row set"
    ),
    "F2_main_sweep_with_degenerates_180": (
        "F1 plus the 60 capacity_pnl_only deterministic identities (p=1.0 by convention) "
        "folded back into the correction budget"
    ),
    "F3_per_metric_24": (
        "each metric corrected inside its own family of 4 k x 3 broker_count x 2 ablations"
    ),
    "F4_pooled_with_monopoly": (
        "F1 pooled with the 8 real tests of the supplementary broker_count=1 monopoly arm"
    ),
    "F5_everything_pooled": (
        "the D8 main sweep's real tests + the monopoly arm + the D9 structural sweep in one "
        "family (D9 contributes both its cell_full_factorial and marginal_per_coefficient "
        "rows, which overlap in the underlying runs, so those runs are double counted on "
        "purpose); it excludes the Phase 12 OPSD paired tests, which carry no p-value column "
        "in results/demand_source_comparison.csv; it is the largest family of real tests but "
        "not the harshest on the headline set, because Holm is step-down and the added D9 "
        "tests are themselves highly significant, so they raise the family size and the "
        "headline tests' rank together"
    ),
    "F6_per_cell_k_by_broker_count": (
        "family = the 5 metrics x 2 ablations sharing a single (k, broker_count) cell"
    ),
    "F7_everything_pooled_with_degenerates": (
        "the union of F2 and F5 and the widest family computed here: the D8 main sweep's real "
        "tests + the 60 deterministic capacity_pnl_only identities + the monopoly arm + the "
        "D9 structural sweep; it crosses the two axes F2 and F5 vary separately and is the "
        "harshest definition on the headline set"
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    pool = load_test_pool()

    n_headline = int(pool["is_headline_72"].sum())
    print("test pool: {} tests total".format(len(pool)))
    print("  by source:")
    for src, n in pool.groupby("source", sort=True).size().items():
        print("    {:<34} {}".format(src, n))
    print("headline tests found: {} (expected 72)".format(n_headline))
    if n_headline != 72:
        # Every number the thesis quotes from this artifact is "of the 72". Shipping a CSV
        # whose summary rows are computed over some other count would silently contradict
        # the prose, so this is fatal rather than a warning.
        raise SystemExit(
            "headline count is {}, expected 72. HEADLINE_METRICS x HEADLINE_ABLATIONS no "
            "longer selects the 72 comparisons the thesis reports; either the inputs under "
            "results/ changed or those two constants did. Refusing to write {}.".format(
                n_headline, OUTPUT_CSV
            )
        )

    definitions = family_assignments(pool)

    out_rows: list[dict] = []
    max_holm_gap = 0.0
    max_bh_gap = 0.0
    cross_check_done = False

    for family_definition in sorted(definitions):
        sub = definitions[family_definition].copy()
        n_families = sub["family_key"].nunique()
        corrected_parts = []
        for family_key in sorted(sub["family_key"].unique()):
            block = sub[sub["family_key"] == family_key].copy()
            block = block.sort_values("test_id", kind="stable").reset_index(drop=True)
            p_raw = block["p_uncorrected"].to_numpy(dtype=float)
            block["family_size"] = len(block)
            block["p_holm"] = holm_adjust(p_raw)
            block["p_bh"] = bh_adjust(p_raw)
            gaps = cross_check_against_statsmodels(p_raw)
            if gaps is not None:
                max_holm_gap = max(max_holm_gap, gaps[0])
                max_bh_gap = max(max_bh_gap, gaps[1])
                cross_check_done = True
            corrected_parts.append(block)
        sub = pd.concat(corrected_parts, ignore_index=True)
        # Survival convention matches build_corrected_summary() in scripts/analyze_sweep.py
        # exactly: non-strict "<= ALPHA", and a degenerate test never counts as surviving.
        # The two artifacts are read side by side, so they must agree on the threshold on
        # purpose, not by the accident that no p currently sits at 0.05 and the 60
        # deterministic identities happen to carry p_uncorrected = 1.0.
        not_degenerate = ~sub["degenerate"].astype(bool)
        sub["survives_holm_alpha05"] = not_degenerate & (sub["p_holm"] <= ALPHA)
        sub["survives_bh_alpha05"] = not_degenerate & (sub["p_bh"] <= ALPHA)
        sub["family_definition"] = family_definition
        sub["n_families_in_definition"] = n_families
        sub["alpha"] = ALPHA
        sub["notes"] = FAMILY_NOTES[family_definition]
        sub["bonferroni_alpha_over_family_size"] = ALPHA / sub["family_size"]

        sub = sub.sort_values(
            ["family_key", "metric", "ablation", "k", "broker_count", "test_id"], kind="stable"
        ).reset_index(drop=True)
        out_rows.extend(sub.to_dict("records"))

        # --- one summary row per family definition ----------------------------------
        head = sub[sub["is_headline_72"]]
        largest_family_size = int(sub["family_size"].max())
        summary_row = {
            "family_definition": family_definition,
            "family_key": "ALL",
            "family_size": largest_family_size,
            "n_families_in_definition": n_families,
            "scope": "family_summary",
            "source": "family_summary",
            "test_id": "family_summary|{}".format(family_definition),
            "is_headline_72": "",
            "degenerate": "",
            "cell_is_stamped_not_native": "",
            "alpha": ALPHA,
            "n_headline_tests": len(head),
            "n_headline_surviving_holm": int(head["survives_holm_alpha05"].sum()),
            "n_headline_surviving_bh": int(head["survives_bh_alpha05"].sum()),
            "worst_headline_p_uncorrected": float(head["p_uncorrected"].max()),
            "worst_headline_p_holm": float(head["p_holm"].max()),
            "worst_headline_p_bh": float(head["p_bh"].max()),
            "bonferroni_alpha_over_family_size": ALPHA / largest_family_size,
            "notes": (
                "summary over the {} headline comparisons; family_size is the largest "
                "single family in this definition; {}".format(
                    len(head), FAMILY_NOTES[family_definition]
                )
            ),
        }
        out_rows.append(summary_row)

        print(
            "{:<36} n_families={:<3} max_family_size={:<4} "
            "headline Holm {}/{}  BH {}/{}  worst_p_holm={:.6g}".format(
                family_definition,
                n_families,
                largest_family_size,
                summary_row["n_headline_surviving_holm"],
                summary_row["n_headline_tests"],
                summary_row["n_headline_surviving_bh"],
                summary_row["n_headline_tests"],
                summary_row["worst_headline_p_holm"],
            )
        )

    out = pd.DataFrame(out_rows)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[OUTPUT_COLUMNS]
    # summary rows sort last inside each family definition
    out["_is_summary"] = (out["scope"] == "family_summary").astype(int)
    out = out.sort_values(
        ["family_definition", "_is_summary", "family_key", "test_id"], kind="stable"
    ).drop(columns=["_is_summary"])
    out.to_csv(OUTPUT_CSV, index=False, lineterminator="\n")

    # --- checks reported to stdout, not written to any file --------------------------
    print()
    if cross_check_done:
        print(
            "statsmodels cross-check, max abs disagreement: Holm {:.3e}, BH {:.3e}".format(
                max_holm_gap, max_bh_gap
            )
        )
    else:
        print("statsmodels not available, cross-check skipped")

    f1 = definitions["F1_main_sweep_primary_120"].copy()
    f1_holm = holm_adjust(f1["p_uncorrected"].to_numpy(dtype=float))
    f1_gap = float(np.max(np.abs(f1_holm - f1["shipped_p_holm"].to_numpy(dtype=float))))
    print("F1 vs shipped p_holm, max abs disagreement: {:.6e}".format(f1_gap))

    head_pool = pool[pool["is_headline_72"]]
    worst_p = float(head_pool["p_uncorrected"].max())
    worst_tests = head_pool[head_pool["p_uncorrected"] == worst_p]
    print(
        "worst uncorrected p among the {} headline tests: {:.6e}, held by {} test(s):".format(
            n_headline, worst_p, len(worst_tests)
        )
    )
    for _, w in worst_tests.iterrows():
        print(
            "    {} k={:g} bc={} {}".format(
                w["metric"], w["k"], w["broker_count"], w["ablation"]
            )
        )
    largest_n = int(out["family_size"].max())
    print(
        "largest family size computed: {}, Bonferroni alpha/N = {:.6e}".format(
            largest_n, ALPHA / largest_n
        )
    )

    # cross-check the monopoly p-values against the monopoly comparison file
    mono_cmp = pd.read_csv(MONOPOLY_CSV, float_precision="round_trip")
    mono_cmp = mono_cmp[mono_cmp["broker_count"] == 1]
    mono_pool = pool[pool["source"] == "monopoly_supplement"]
    gaps = []
    for _, r in mono_cmp.iterrows():
        for metric, col in (
            ("feeder_peak_to_average_ratio", "feeder_peak_to_average_ratio_paired_p"),
            ("feeder_coefficient_of_variation", "feeder_coefficient_of_variation_paired_p"),
        ):
            match = mono_pool[
                (mono_pool["metric"] == metric) & (np.isclose(mono_pool["k"], r["k"]))
            ]
            if len(match) == 1:
                gaps.append(abs(float(match.iloc[0]["p_uncorrected"]) - float(r[col])))
    print(
        "monopoly p cross-check vs monopoly_comparison.csv: {} pairs, max abs gap {:.3e}".format(
            len(gaps), max(gaps) if gaps else float("nan")
        )
    )

    print()
    print("wrote {}".format(OUTPUT_CSV))
    print("sha256 {}".format(sha256_of(OUTPUT_CSV)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
