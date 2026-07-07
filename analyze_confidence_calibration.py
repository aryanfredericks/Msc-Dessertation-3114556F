"""
analyze_confidence_calibration.py

Answers: "would raising the common-branch confidence threshold from 0.5 to
0.75 (or anywhere else) actually help?" - empirically, using real run data.

Joins three files from a single run:
  - entity_log.jsonl        (doc_id, span_text, entity_type, source, confidence, requeried)
    written by AgentWorkflow.save_stats() - confidence BEFORE offset localization
  - test_predictions.json   ({doc_id: [{start, end, type, text}, ...]})
    final localized predictions
  - gold_test.json          ({doc_id: [{start, end, type, text}, ...]})
    gold labels, same format

Join key: (doc_id, span_text). Span extraction dedupes surface strings per
document, so a given (doc_id, text) should map to exactly one entity_log
row; if extraction produced accidental duplicates, they'll have identical
confidence anyway since the underlying model call is deterministic, so
picking the first match is safe.

Outputs:
  - confidence_calibration.csv   (prediction-level: doc_id, text, type, branch, confidence, correct)
  - confidence_calibration_summary.json  (mean/median confidence, TP vs FP, per type/branch)
  - threshold_sweep.csv          (for a range of thresholds: how many currently-accepted
                                   TP/FP predictions would newly fall below it)
  - confidence_calibration_chart.png  (histograms: correct vs incorrect confidence, per type)

Usage:
    python analyze_confidence_calibration.py \
        --entity_log outputs/tier4_agent_bc5cdr/branch_analysis/entity_log_<ts>.jsonl \
        --predictions outputs/tier4_agent_bc5cdr/test_predictions.json \
        --gold outputs/tier4_agent_bc5cdr/gold_test.json \
        --out_dir outputs/tier4_agent_bc5cdr/calibration_analysis
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict


def load_entity_log(path: str) -> dict:
    """Returns {(doc_id, span_text): {entity_type, source, confidence, requeried}}"""
    lookup = {}
    collisions = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["doc_id"], row["span_text"])
            if key in lookup:
                collisions += 1
                continue  # keep first; confidence should be identical anyway
            lookup[key] = {
                "entity_type": row["entity_type"],
                "source": row["source"],
                "confidence": row["confidence"],
                "requeried": row["requeried"],
            }
    if collisions:
        print(f"[join] note: {collisions} duplicate (doc_id, span_text) keys in entity_log "
              f"(kept first occurrence each - expected to be harmless)")
    return lookup


def load_canonical(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_gold_index(gold: dict) -> set:
    """Returns set of (doc_id, start, end, type) for exact-match checking."""
    index = set()
    for doc_id, entities in gold.items():
        for e in entities:
            index.add((doc_id, e["start"], e["end"], e["type"]))
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_log", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--out_dir", default="calibration_analysis")
    args = parser.parse_args()

    print("Loading files...")
    conf_lookup = load_entity_log(args.entity_log)
    predictions = load_canonical(args.predictions)
    gold = load_canonical(args.gold)
    gold_index = build_gold_index(gold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    unmatched = 0
    total_preds = 0

    for doc_id, entities in predictions.items():
        for e in entities:
            total_preds += 1
            key = (doc_id, e["text"])
            info = conf_lookup.get(key)
            if info is None:
                unmatched += 1
                continue

            is_correct = (doc_id, e["start"], e["end"], e["type"]) in gold_index

            rows.append({
                "doc_id": doc_id,
                "text": e["text"],
                "type": e["type"],
                "branch": info["source"],
                "confidence": info["confidence"],
                "requeried": info["requeried"],
                "correct": is_correct,
            })

    print(f"[join] {total_preds} predictions total, {unmatched} unmatched "
          f"({100*unmatched/total_preds:.1f}%), {len(rows)} joined rows")

    # --- write prediction-level CSV ---
    csv_path = out_dir / "confidence_calibration.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "text", "type", "branch",
                                                 "confidence", "requeried", "correct"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[write] {csv_path}")

    # --- summary stats: mean/median confidence, TP vs FP, per type + branch ---
    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    def median(xs):
        if not xs:
            return None
        s = sorted(xs)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid-1] + s[mid]) / 2

    summary = {}
    by_type_branch = defaultdict(lambda: {"correct": [], "incorrect": []})
    for r in rows:
        bucket = "correct" if r["correct"] else "incorrect"
        by_type_branch[(r["type"], r["branch"])][bucket].append(r["confidence"])

    for (etype, branch), buckets in by_type_branch.items():
        summary[f"{etype} | {branch}"] = {
            "n_correct": len(buckets["correct"]),
            "n_incorrect": len(buckets["incorrect"]),
            "mean_confidence_correct": mean(buckets["correct"]),
            "mean_confidence_incorrect": mean(buckets["incorrect"]),
            "median_confidence_correct": median(buckets["correct"]),
            "median_confidence_incorrect": median(buckets["incorrect"]),
        }

    summary_path = out_dir / "confidence_calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {summary_path}")

    print("\n=== Confidence calibration summary ===")
    for key, s in summary.items():
        print(f"\n  {key}  (n_correct={s['n_correct']}, n_incorrect={s['n_incorrect']})")
        print(f"    mean conf   correct={s['mean_confidence_correct']}  incorrect={s['mean_confidence_incorrect']}")
        print(f"    median conf correct={s['median_confidence_correct']}  incorrect={s['median_confidence_incorrect']}")

    # --- threshold sweep: for common-branch predictions only (the ones actually gated) ---
    common_rows = [r for r in rows if r["branch"] == "common"]
    thresholds = [round(0.05 * i, 2) for i in range(10, 20)]  # 0.50 .. 0.95

    sweep_path = out_dir / "threshold_sweep.csv"
    with open(sweep_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "tp_now_flagged", "fp_now_flagged",
                          "tp_retained", "fp_retained",
                          "pct_fp_caught", "pct_tp_lost"])
        total_tp = sum(1 for r in common_rows if r["correct"])
        total_fp = sum(1 for r in common_rows if not r["correct"])
        for t in thresholds:
            tp_flagged = sum(1 for r in common_rows if r["correct"] and r["confidence"] < t)
            fp_flagged = sum(1 for r in common_rows if not r["correct"] and r["confidence"] < t)
            tp_retained = total_tp - tp_flagged
            fp_retained = total_fp - fp_flagged
            pct_fp_caught = 100 * fp_flagged / total_fp if total_fp else 0.0
            pct_tp_lost = 100 * tp_flagged / total_tp if total_tp else 0.0
            writer.writerow([t, tp_flagged, fp_flagged, tp_retained, fp_retained,
                              round(pct_fp_caught, 2), round(pct_tp_lost, 2)])
    print(f"\n[write] {sweep_path}")
    print(f"[sweep] common branch: {total_tp} correct, {total_fp} incorrect (baseline, threshold=0.5 status quo)")

    print("\n=== Threshold sweep preview ===")
    print(f"{'thresh':>7} {'tp_flagged':>11} {'fp_flagged':>11} {'%fp_caught':>11} {'%tp_lost':>9}")
    for t in thresholds:
        tp_flagged = sum(1 for r in common_rows if r["correct"] and r["confidence"] < t)
        fp_flagged = sum(1 for r in common_rows if not r["correct"] and r["confidence"] < t)
        pct_fp_caught = 100 * fp_flagged / total_fp if total_fp else 0.0
        pct_tp_lost = 100 * tp_flagged / total_tp if total_tp else 0.0
        print(f"{t:>7} {tp_flagged:>11} {fp_flagged:>11} {pct_fp_caught:>10.1f}% {pct_tp_lost:>8.1f}%")

    # --- chart: confidence histograms, correct vs incorrect, per entity type ---
    try:
        import matplotlib.pyplot as plt

        types_present = sorted(set(r["type"] for r in common_rows))
        fig, axes = plt.subplots(1, len(types_present), figsize=(6 * len(types_present), 4.5))
        if len(types_present) == 1:
            axes = [axes]

        for ax, etype in zip(axes, types_present):
            correct_conf = [r["confidence"] for r in common_rows if r["type"] == etype and r["correct"]]
            incorrect_conf = [r["confidence"] for r in common_rows if r["type"] == etype and not r["correct"]]
            bins = [i / 20 for i in range(21)]
            ax.hist(correct_conf, bins=bins, alpha=0.6, label=f"correct (n={len(correct_conf)})", color="#4C72B0")
            ax.hist(incorrect_conf, bins=bins, alpha=0.6, label=f"incorrect (n={len(incorrect_conf)})", color="#C44E52")
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="current threshold (0.5)")
            ax.axvline(0.75, color="black", linestyle=":", linewidth=1, label="proposed threshold (0.75)")
            ax.set_title(f"{etype} (common branch)")
            ax.set_xlabel("confidence")
            ax.set_ylabel("count")
            ax.legend(fontsize=7)

        plt.tight_layout()
        chart_path = out_dir / "confidence_calibration_chart.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        print(f"[write] {chart_path}")
    except ImportError:
        print("[chart] matplotlib not available, skipping chart generation")


if __name__ == "__main__":
    main()