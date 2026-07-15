import json
import sys
import csv
from pathlib import Path
from collections import defaultdict


def load_entity_log(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_distribution(rows: list[dict]) -> dict:
    branch_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overall_totals: dict[str, int] = defaultdict(int)

    for row in rows:
        entity_type = row["entity_type"]
        source = row["source"]
        branch_stats[entity_type][source] += 1
        overall_totals[source] += 1

    grand_total = sum(overall_totals.values())

    summary = {
        "grand_total_entities": grand_total,
        "overall_branch_totals": dict(overall_totals),
        "overall_branch_percentages": {
            branch: round(100 * count / grand_total, 2) if grand_total else 0.0
            for branch, count in overall_totals.items()
        },
        "by_entity_type": {
            entity_type: {
                "branch_counts": dict(branch_counts),
                "total": sum(branch_counts.values()),
                "branch_percentages": {
                    branch: round(100 * count / sum(branch_counts.values()), 2)
                    for branch, count in branch_counts.items()
                } if sum(branch_counts.values()) else {}
            }
            for entity_type, branch_counts in branch_stats.items()
        }
    }
    return summary


def write_outputs(summary: dict, out_dir: str = "output/branch_analysis") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "branch_distribution.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stats] summary -> {json_path}")

    csv_path = out / "branch_distribution.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_type", "branch", "count", "pct_within_entity_type"])
        for entity_type, data in summary["by_entity_type"].items():
            for branch, count in data["branch_counts"].items():
                pct = data["branch_percentages"][branch]
                writer.writerow([entity_type, branch, count, pct])
        for branch, count in summary["overall_branch_totals"].items():
            pct = summary["overall_branch_percentages"][branch]
            writer.writerow(["ALL", branch, count, pct])
    print(f"[stats] CSV -> {csv_path}")

    # console preview
    print("\n=== Overall branch resolution distribution ===")
    for branch, pct in summary["overall_branch_percentages"].items():
        count = summary["overall_branch_totals"][branch]
        print(f"  {branch:15s} {count:6d}  ({pct}%)")

    print("\n=== Per entity type ===")
    for entity_type, data in summary["by_entity_type"].items():
        print(f"\n  {entity_type} (n={data['total']})")
        for branch, pct in data["branch_percentages"].items():
            count = data["branch_counts"][branch]
            print(f"    {branch:15s} {count:6d}  ({pct}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_branch_stats.py <path_to_entity_log.jsonl>")
        sys.exit(1)

    log_path = sys.argv[1]
    rows = load_entity_log(log_path)
    print(f"Loaded {len(rows)} resolved entities from {log_path}")

    summary = compute_distribution(rows)
    write_outputs(summary)