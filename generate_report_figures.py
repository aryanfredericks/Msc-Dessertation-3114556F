"""
generate_report_figures.py

Scans a directory tree for full_metrics.json files (as written by scorer.py
for every tier in this project) and generates a set of comparison figures
suitable for dropping directly into the dissertation report.

Handles tiers with missing per_type_strict entries, zero-support types
(e.g. an out-of-distribution run like BC5CDR where several BioRED-only
types have no gold at all), and any number of tiers - it discovers
whatever full_metrics.json files exist rather than hardcoding tier names.

Usage:
    python generate_report_figures.py --root outputs --out report_figures
    python generate_report_figures.py --root outputs --out report_figures --exclude tier4_agent_bc5cdr

Output (in --out):
    01_results_summary.png       - strict/relaxed/macro F1 per tier, grouped bars
    02_per_type_f1_heatmap.png   - tier x entity-type F1 heatmap (zero-support cells left blank)
    03_precision_recall.png      - precision vs recall scatter, one point per tier, iso-F1 reference lines
    04_error_taxonomy.png        - error category breakdown per tier, stacked bars
    05_tier4_vs_tier4_extended.png  - only generated if both tier4_agent and
                                      tier5_agent (or equivalents) are found
    06_ablation_<tier>.png       - one per base tier that has ablation variants
                                    detected (folder name differs from the JSON's
                                    internal "name" field, e.g. ablation_no_overseer/)
    07_strict_sanity_summary.png - P/R/F1 from each run's inline sanity check
                                    (test_strict_metrics.json)
    08_strict_tp_fp_fn.png       - TP/FP/FN counts from the same sanity-check files
    09_sanity_vs_scorer_consistency.png - cross-checks test_strict_metrics.json's
                                    F1 against scorer.py's full_metrics.json F1 per
                                    tier; flags any tier where they disagree by >0.5 F1
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Friendlier display names for known tier folder/name conventions. Anything
# not in this map falls back to a title-cased version of its raw name.
DISPLAY_NAMES = {
    "tier1_pubmedbert": "Tier 1: PubMedBERT",
    "tier2_gliner": "Tier 2: GLiNER",
    "tier3_llm_0shot": "Tier 3: LLM (0-shot)",
    "tier3_llm_3shot": "Tier 3: LLM (3-shot)",
    "tier4_agent": "Tier 4: Multi-Agent",
    "tier5_agent": "Tier 4 Extended: LLM-Orchestrated",
    "tier4_extended": "Tier 4 Extended: LLM-Orchestrated",
}

# preferred left-to-right / top-to-bottom ordering for known tiers; anything
# unrecognised is appended after these, alphabetically
PREFERRED_ORDER = [
    "tier1_pubmedbert", "tier2_gliner", "tier3_llm_0shot", "tier3_llm_3shot",
    "tier4_agent", "tier5_agent", "tier4_extended",
]

ENTITY_TYPES = [
    "GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "ChemicalEntity",
    "SequenceVariant", "CellLine", "OrganismTaxon",
]

ERROR_CATEGORIES = ["boundary_only", "type_and_boundary", "type_only", "spurious_fp", "missed_fn"]
ERROR_LABELS = {
    "boundary_only": "boundary only",
    "type_and_boundary": "type + boundary",
    "type_only": "type only",
    "spurious_fp": "spurious (pure FP)",
    "missed_fn": "missed (pure FN)",
}


# Friendlier labels for known ablation folder-name suffixes. Falls back to a
# cleaned-up version of the folder name for anything not listed here.
ABLATION_LABELS = {
    "ablation_no_overseer": "no overseer / requery",
    "ablation_no_pattern_branch": "no pattern branch",
    "ablation_no_rare_branch": "no rare branch",
    "ablation_no_kb_gate": "no KB confidence gate",
    "ablation_no_confidence_gate": "no confidence gate",
}


def ablation_label(folder_name: str) -> str:
    if folder_name in ABLATION_LABELS:
        return ABLATION_LABELS[folder_name]
    cleaned = folder_name.replace("ablation_", "").replace("_", " ")
    return cleaned


def display_name(unique_id: str) -> str:
    if "__" in unique_id:
        base, suffix = unique_id.split("__", 1)
        base_label = DISPLAY_NAMES.get(base, base.replace("_", " ").title())
        return f"{base_label} (\u2212 {ablation_label(suffix)})"
    return DISPLAY_NAMES.get(unique_id, unique_id.replace("_", " ").title())


def sort_key(unique_id: str):
    base = unique_id.split("__")[0]
    is_variant = "__" in unique_id
    if base in PREFERRED_ORDER:
        return (0, PREFERRED_ORDER.index(base), 1 if is_variant else 0, unique_id)
    return (1, base, 1 if is_variant else 0, unique_id)


def load_all_metrics(root: Path, exclude: set[str]) -> list[dict]:
    results = []
    for path in sorted(root.rglob("full_metrics.json")):
        with open(path, "r") as f:
            data = json.load(f)

        json_name = data.get("name", path.parent.name)
        folder_name = path.parent.name

        # if the immediate containing folder name differs from the JSON's
        # own "name" field, this is a variant nested under a tier's output
        # directory (e.g. outputs/tier4_agent/ablation_no_overseer/) whose
        # scorer.py --name argument wasn't changed per-variant. Use the
        # folder name to disambiguate rather than silently collapsing every
        # variant into one label.
        if folder_name != json_name:
            unique_id = f"{json_name}__{folder_name}"
        else:
            unique_id = json_name

        base_name = unique_id.split("__")[0]
        if base_name in exclude or json_name in exclude or folder_name in exclude or unique_id in exclude:
            print(f"[skip] {path} (excluded)")
            continue

        data["_raw_name"] = unique_id
        data["_base_name"] = base_name
        data["_is_variant"] = "__" in unique_id
        data["_path"] = str(path)
        results.append(data)
        print(f"[load] {path} -> id='{unique_id}'")
    results.sort(key=lambda d: sort_key(d["_raw_name"]))
    return results


def fig01_results_summary(all_metrics: list[dict], out_dir: Path):
    names = [display_name(d["_raw_name"]) for d in all_metrics]
    strict_f1 = [d["strict"]["f1"] * 100 for d in all_metrics]
    relaxed_f1 = [d["relaxed"]["f1"] * 100 for d in all_metrics]
    macro_f1 = [d.get("macro_f1_strict", 0) * 100 for d in all_metrics]

    y = np.arange(len(names))
    height = 0.25

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(names) + 1.5)))
    ax.barh(y + height, strict_f1, height, label="Strict F1", color="#4C72B0")
    ax.barh(y, relaxed_f1, height, label="Relaxed F1", color="#55A868")
    ax.barh(y - height, macro_f1, height, label="Macro F1", color="#C44E52")

    for i, (s, r, m) in enumerate(zip(strict_f1, relaxed_f1, macro_f1)):
        ax.text(s + 1, y[i] + height, f"{s:.1f}", va="center", fontsize=8)
        ax.text(r + 1, y[i], f"{r:.1f}", va="center", fontsize=8)
        ax.text(m + 1, y[i] - height, f"{m:.1f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 110)
    ax.set_title("Results summary across tiers")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    path = out_dir / "01_results_summary.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def fig02_per_type_heatmap(all_metrics: list[dict], out_dir: Path):
    names = [display_name(d["_raw_name"]) for d in all_metrics]
    matrix = np.full((len(all_metrics), len(ENTITY_TYPES)), np.nan)

    for i, d in enumerate(all_metrics):
        per_type = d.get("per_type_strict", {})
        for j, etype in enumerate(ENTITY_TYPES):
            info = per_type.get(etype)
            if info is None or info.get("support", 0) == 0:
                continue  # leave NaN - zero/no support, not a meaningful score
            matrix[i, j] = info["f1"] * 100

    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(names) + 1.5)))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(len(names)):
        for j in range(len(ENTITY_TYPES)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999")
            else:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                         color="white" if val < 50 or val > 85 else "black")

    ax.set_xticks(np.arange(len(ENTITY_TYPES)))
    ax.set_xticklabels(ENTITY_TYPES, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Per-type strict F1 across tiers\n(— = zero support in this evaluation)")
    fig.colorbar(im, ax=ax, label="Strict F1", shrink=0.8)

    plt.tight_layout()
    path = out_dir / "02_per_type_f1_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def fig03_precision_recall(all_metrics: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 7))

    # iso-F1 reference curves
    recall_range = np.linspace(0.01, 1, 200)
    for f1_target in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        with np.errstate(divide="ignore", invalid="ignore"):
            precision_curve = (f1_target * recall_range) / (2 * recall_range - f1_target)
        valid = (precision_curve > 0) & (precision_curve <= 1)
        ax.plot(recall_range[valid] * 100, precision_curve[valid] * 100,
                 color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
        # label near the curve's right end within plot bounds
        idxs = np.where(valid)[0]
        if len(idxs):
            xi = idxs[-1]
            ax.text(recall_range[xi] * 100, precision_curve[xi] * 100, f"F1={f1_target}",
                     fontsize=7, color="gray", alpha=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))
    for d, color in zip(all_metrics, colors):
        p = d["strict"]["precision"] * 100
        r = d["strict"]["recall"] * 100
        name = display_name(d["_raw_name"])
        ax.scatter(r, p, s=120, color=color, edgecolor="black", zorder=5, label=name)
        ax.annotate(name, (r, p), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Precision vs recall (strict), all tiers\ndashed lines = iso-F1 reference")

    plt.tight_layout()
    path = out_dir / "03_precision_recall.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def fig04_error_taxonomy(all_metrics: list[dict], out_dir: Path):
    names = [display_name(d["_raw_name"]) for d in all_metrics]
    colors = {"boundary_only": "#4C72B0", "type_and_boundary": "#8172B2",
              "type_only": "#CCB974", "spurious_fp": "#C44E52", "missed_fn": "#937860"}

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(names) + 1.5)))
    y = np.arange(len(names))
    left = np.zeros(len(names))

    for cat in ERROR_CATEGORIES:
        values = np.array([d.get("error_taxonomy", {}).get(cat, 0) for d in all_metrics])
        ax.barh(y, values, left=left, color=colors[cat], label=ERROR_LABELS[cat])
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Error count")
    ax.set_title("Error taxonomy breakdown per tier\n(excludes exact matches)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    path = out_dir / "04_error_taxonomy.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def fig05_tier4_vs_extended(all_metrics: list[dict], out_dir: Path):
    by_name = {d["_raw_name"]: d for d in all_metrics}
    tier4 = by_name.get("tier4_agent")
    extended = by_name.get("tier5_agent") or by_name.get("tier4_extended")

    if tier4 is None or extended is None:
        print("[skip] 05_tier4_vs_tier4_extended.png - need both tier4_agent and "
              "tier5_agent/tier4_extended full_metrics.json present")
        return

    types_present = [t for t in ENTITY_TYPES
                      if tier4.get("per_type_strict", {}).get(t, {}).get("support", 0) > 0]

    t4_f1 = [tier4["per_type_strict"][t]["f1"] * 100 for t in types_present]
    t5_f1 = [extended["per_type_strict"][t]["f1"] * 100 for t in types_present]

    deltas = [b - a for a, b in zip(t4_f1, t5_f1)]
    order = np.argsort(deltas)[::-1]
    types_sorted = [types_present[i] for i in order]
    t4_sorted = [t4_f1[i] for i in order]
    t5_sorted = [t5_f1[i] for i in order]

    y = np.arange(len(types_sorted))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(4, 0.7 * len(types_sorted) + 1.5)))
    ax.barh(y + height/2, t4_sorted, height, label=display_name("tier4_agent"), color="#C44E52")
    ax.barh(y - height/2, t5_sorted, height, label=display_name(extended["_raw_name"]), color="#4C72B0")

    for i, (v4, v5) in enumerate(zip(t4_sorted, t5_sorted)):
        ax.text(v4 + 1, y[i] + height/2, f"{v4:.1f}", va="center", fontsize=9)
        ax.text(v5 + 1, y[i] - height/2, f"{v5:.1f}", va="center", fontsize=9)
        delta = v5 - v4
        sign = "+" if delta >= 0 else ""
        color = "#2ca02c" if delta > 2 else ("#888" if abs(delta) <= 2 else "#C44E52")
        ax.text(max(v4, v5) + 8, y[i], f"{sign}{delta:.1f}", va="center",
                 fontsize=9, fontweight="bold", color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(types_sorted)
    ax.set_xlabel("Strict F1")
    ax.set_xlim(0, 115)
    ax.set_title("Tier 4 vs Tier 4 Extended — per-type F1")
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = out_dir / "05_tier4_vs_tier4_extended.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def fig06_ablation_breakdown(all_metrics: list[dict], out_dir: Path):
    variants_by_base: dict[str, list[dict]] = {}
    full_by_base: dict[str, dict] = {}

    for d in all_metrics:
        if d["_is_variant"]:
            variants_by_base.setdefault(d["_base_name"], []).append(d)
        else:
            full_by_base[d["_base_name"]] = d

    bases_with_ablations = [b for b in variants_by_base if b in full_by_base]
    if not bases_with_ablations:
        print("[skip] 06_ablation_breakdown.png - no ablation variants found "
              "(a variant is any full_metrics.json whose containing folder name "
              "differs from its internal 'name' field)")
        return

    for base in bases_with_ablations:
        full_run = full_by_base[base]
        variants = sorted(variants_by_base[base], key=lambda d: d["strict"]["f1"])

        labels = [display_name(full_run["_raw_name"]) + " (full system)"] + \
                 [display_name(v["_raw_name"]) for v in variants]
        f1_values = [full_run["strict"]["f1"] * 100] + [v["strict"]["f1"] * 100 for v in variants]
        full_f1 = f1_values[0]

        y = np.arange(len(labels))
        colors = ["#4C72B0"] + ["#C44E52"] * len(variants)

        fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(labels) + 1.5)))
        ax.barh(y, f1_values, color=colors)
        for i, v in enumerate(f1_values):
            delta = v - full_f1
            label = f"{v:.1f}" if i == 0 else f"{v:.1f}  ({delta:+.1f})"
            ax.text(v + 1, y[i], label, va="center", fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Strict F1")
        ax.set_xlim(0, max(f1_values) + 15)
        ax.set_title(f"Ablation breakdown — {display_name(base)}")
        ax.invert_yaxis()

        plt.tight_layout()
        path = out_dir / f"06_ablation_{base}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {path}")


def load_test_strict(root: Path, exclude: set[str]) -> list[dict]:
    """Loads test_strict_metrics.json files. Unlike full_metrics.json these
    have no internal 'name' field, so tiers/variants are identified purely
    from their position in the folder tree relative to --root."""
    results = []
    for path in sorted(root.rglob("test_strict_metrics.json")):
        with open(path, "r") as f:
            data = json.load(f)

        rel_parts = path.parent.relative_to(root).parts
        if not rel_parts:
            continue
        base_name = rel_parts[0]
        is_variant = len(rel_parts) > 1
        unique_id = f"{base_name}__{rel_parts[-1]}" if is_variant else base_name

        if base_name in exclude or unique_id in exclude:
            print(f"[skip] {path} (excluded)")
            continue

        data["_raw_name"] = unique_id
        data["_base_name"] = base_name
        data["_is_variant"] = is_variant
        data["_path"] = str(path)
        results.append(data)
        print(f"[load] {path} -> id='{unique_id}'")
    results.sort(key=lambda d: sort_key(d["_raw_name"]))
    return results


def fig07_strict_sanity_summary(strict_results: list[dict], out_dir: Path):
    names = [display_name(d["_raw_name"]) for d in strict_results]
    p = [d["precision"] * 100 for d in strict_results]
    r = [d["recall"] * 100 for d in strict_results]
    f1 = [d["f1"] * 100 for d in strict_results]

    y = np.arange(len(names))
    height = 0.25

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(names) + 1.5)))
    ax.barh(y + height, p, height, label="Precision", color="#4C72B0")
    ax.barh(y, r, height, label="Recall", color="#55A868")
    ax.barh(y - height, f1, height, label="F1", color="#C44E52")

    for i, (pv, rv, fv) in enumerate(zip(p, r, f1)):
        ax.text(pv + 1, y[i] + height, f"{pv:.1f}", va="center", fontsize=8)
        ax.text(rv + 1, y[i], f"{rv:.1f}", va="center", fontsize=8)
        ax.text(fv + 1, y[i] - height, f"{fv:.1f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0, 110)
    ax.set_title("Strict sanity-check results across tiers\n(entity-level P/R/F1, from each run's inline sanity check)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    path = out_dir / "07_strict_sanity_summary.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def fig08_strict_tp_fp_fn(strict_results: list[dict], out_dir: Path):
    names = [display_name(d["_raw_name"]) for d in strict_results]
    tp = np.array([d["tp"] for d in strict_results])
    fp = np.array([d["fp"] for d in strict_results])
    fn = np.array([d["fn"] for d in strict_results])

    y = np.arange(len(names))
    height = 0.25

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(names) + 1.5)))
    ax.barh(y + height, tp, height, label="True Positives", color="#55A868")
    ax.barh(y, fp, height, label="False Positives", color="#C44E52")
    ax.barh(y - height, fn, height, label="False Negatives", color="#DD8452")

    for i in range(len(names)):
        ax.text(tp[i] + max(tp) * 0.01, y[i] + height, str(tp[i]), va="center", fontsize=8)
        ax.text(fp[i] + max(tp) * 0.01, y[i], str(fp[i]), va="center", fontsize=8)
        ax.text(fn[i] + max(tp) * 0.01, y[i] - height, str(fn[i]), va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Count")
    ax.set_title("Strict TP / FP / FN counts across tiers")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    path = out_dir / "08_strict_tp_fp_fn.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def fig09_sanity_vs_scorer_consistency(strict_results: list[dict], full_metrics: list[dict], out_dir: Path):
    """Cross-checks the inline sanity-check F1 (test_strict_metrics.json,
    computed in run_agent.py/run_tier5.py itself) against the official
    scorer.py strict F1 (full_metrics.json). These should closely agree -
    a divergence would indicate the two code paths are scoring against
    different predictions/gold, which is worth catching before reporting
    numbers in the dissertation."""
    full_by_id = {d["_raw_name"]: d for d in full_metrics}

    rows = [(d, full_by_id[d["_raw_name"]]) for d in strict_results if d["_raw_name"] in full_by_id]
    if not rows:
        print("[skip] 09_sanity_vs_scorer_consistency.png - no matching tier IDs "
              "found between test_strict_metrics.json and full_metrics.json")
        return

    names = [display_name(d["_raw_name"]) for d, _ in rows]
    sanity_f1 = [d["f1"] * 100 for d, _ in rows]
    scorer_f1 = [f["strict"]["f1"] * 100 for _, f in rows]

    y = np.arange(len(names))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(names) + 1.5)))
    ax.barh(y + height/2, sanity_f1, height, label="Inline sanity check (test_strict_metrics.json)", color="#8172B2")
    ax.barh(y - height/2, scorer_f1, height, label="scorer.py (full_metrics.json)", color="#4C72B0")

    for i, (s, sc) in enumerate(zip(sanity_f1, scorer_f1)):
        diff = abs(s - sc)
        flag = "  \u26a0 MISMATCH" if diff > 0.5 else ""
        ax.text(max(s, sc) + 1, y[i], f"\u0394={diff:.2f}{flag}", va="center", fontsize=8,
                 color="red" if diff > 0.5 else "gray")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Strict F1")
    ax.set_xlim(0, 110)
    ax.set_title("Consistency check: inline sanity F1 vs scorer.py F1\n(should closely agree per tier)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    path = out_dir / "09_sanity_vs_scorer_consistency.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs", help="Directory to search for full_metrics.json")
    ap.add_argument("--out", default="report_figures", help="Output directory for generated images")
    ap.add_argument("--exclude", nargs="*", default=[], help="Tier/base names to exclude (matches JSON 'name', folder name, or full unique id)")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = load_all_metrics(root, set(args.exclude))
    strict_results = load_test_strict(root, set(args.exclude))

    if not all_metrics and not strict_results:
        print(f"[error] no full_metrics.json or test_strict_metrics.json files found under {root}")
        return

    tier_level = [d for d in all_metrics if not d["_is_variant"]]
    n_variants = len(all_metrics) - len(tier_level)
    strict_tier_level = [d for d in strict_results if not d["_is_variant"]]

    print(f"\n[info] found {len(tier_level)} tier-level full_metrics result(s): "
          f"{', '.join(d['_raw_name'] for d in tier_level)}")
    if n_variants:
        print(f"[info] found {n_variants} full_metrics ablation variant(s): "
              f"{', '.join(d['_raw_name'] for d in all_metrics if d['_is_variant'])}")
    print(f"[info] found {len(strict_tier_level)} tier-level test_strict_metrics result(s): "
          f"{', '.join(d['_raw_name'] for d in strict_tier_level)}")
    print()

    if all_metrics:
        fig01_results_summary(tier_level, out_dir)
        fig02_per_type_heatmap(tier_level, out_dir)
        fig03_precision_recall(tier_level, out_dir)
        fig04_error_taxonomy(tier_level, out_dir)
        fig05_tier4_vs_extended(all_metrics, out_dir)
        fig06_ablation_breakdown(all_metrics, out_dir)

    if strict_results:
        fig07_strict_sanity_summary(strict_tier_level, out_dir)
        fig08_strict_tp_fp_fn(strict_tier_level, out_dir)

    if strict_results and all_metrics:
        fig09_sanity_vs_scorer_consistency(strict_results, all_metrics, out_dir)

    print(f"\n[done] figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()