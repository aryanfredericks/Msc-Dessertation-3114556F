"""
extract_rare_entities.py

Pulls every gold OrganismTaxon and CellLine entity out of the BioRED test
set, for inspecting whether resolve_rare_entity()'s sequential
Cellosaurus-then-NCBI-Taxonomy ordering is misclassifying spans that should
go to the other type.

Writes two files:
  - a JSON file, doc-grouped, same shape as your canonical gold format
    (useful for joining back against predictions/entity_log later)
  - a flat .txt file of unique surface strings per type, one per line
    (useful for feeding straight into a curl/API-testing loop)

Usage:
    PYTHONPATH=. uv run extract_rare_entities.py \
        --test_json ./dataset/test/Test.BioC.JSON \
        --out outputs/rare_entity_analysis/organism_cellline_gold.json \
        --txt_out outputs/rare_entity_analysis/organism_cellline_spans.txt
"""

import json
import argparse
from pathlib import Path

from biored import load_biored_documents, gold_entities_by_doc

TARGET_TYPES = {"OrganismTaxon", "CellLine"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", default="./dataset/test/Test.BioC.JSON")
    ap.add_argument("--out", default="outputs/rare_entity_analysis/organism_cellline_gold.json")
    ap.add_argument("--txt_out", default="outputs/rare_entity_analysis/organism_cellline_spans.txt")
    args = ap.parse_args()

    docs, _, _ = load_biored_documents(args.test_json)
    gold_all = gold_entities_by_doc(docs)

    results: dict[str, list[dict]] = {}
    unique_spans_by_type: dict[str, set[str]] = {"OrganismTaxon": set(), "CellLine": set()}
    total_mentions_by_type: dict[str, int] = {"OrganismTaxon": 0, "CellLine": 0}

    for doc_id, entities in gold_all.items():
        matches = [e for e in entities if e["type"] in TARGET_TYPES]
        if matches:
            results[doc_id] = matches
        for e in matches:
            unique_spans_by_type[e["type"]].add(e["text"])
            total_mentions_by_type[e["type"]] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    total = sum(total_mentions_by_type.values())
    print(f"[extract] {total} entities across {len(results)} documents")
    for etype in ("OrganismTaxon", "CellLine"):
        print(f"[extract]   {etype}: {len(unique_spans_by_type[etype])} unique spans, "
              f"{total_mentions_by_type[etype]} total mentions")
    print(f"[extract] wrote {out_path}")

    if args.txt_out:
        txt_path = Path(args.txt_out)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w") as f:
            for etype in ("OrganismTaxon", "CellLine"):
                f.write(f"# {etype} ({len(unique_spans_by_type[etype])} unique spans)\n")
                for span in sorted(unique_spans_by_type[etype], key=str.lower):
                    f.write(span + "\n")
                f.write("\n")
        print(f"[extract] wrote {txt_path}")


if __name__ == "__main__":
    main()