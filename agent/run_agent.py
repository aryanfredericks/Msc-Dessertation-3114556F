import json
import os
from biored import load_biored_documents, gold_entities_by_doc, write_canonical, strict_prf
from workflow import AgentWorkflow

def reconstruct_doc_text(doc):
    end = max(p.offset + len(p.text) for p in doc.passages) if doc.passages else 0
    buf = [" "] * end
    for p in doc.passages:
        for i, ch in enumerate(p.text):
            buf[p.offset + i] = ch
    return "".join(buf)

def build_initial_state(doc, doc_text):
    return {
        "doc_id": doc.doc_id,
        "doc_text": doc_text,
        "doc" : doc,
        "candidate_spans": [],
        "common_votes": [],
        "rare_votes": [],
        "pattern_votes": [],
        "decisions": [],
        "requery_spans": [],
        "final_entities": [],
        "requery_count": 0,
        "skipped_spans": [],
        "branch_sources": {},
        "errors": []
    }

def entity_type_to_canonical(entity_type) -> str:
    """Map EntityTypes enum to canonical BioRED string."""
    mapping = {
        "GENE": "GeneOrGeneProduct",
        "DISEASE": "DiseaseOrPhenotypicFeature",
        "CHEMICAL": "ChemicalEntity",
        "SEQUENCE": "SequenceVariant",
        "CELL": "CellLine",
        "ORGANISM": "OrganismTaxon",
    }
    # handle both enum and string
    key = entity_type.name if hasattr(entity_type, "name") else str(entity_type)
    return mapping.get(key, key)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", default="./dataset/test/Test.BioC.JSON")
    ap.add_argument("--output_dir", default="outputs/tier4_agent")
    ap.add_argument("--limit", type=int, default=0,
                    help="DEBUG: limit to N docs")
    args = ap.parse_args()

    docs, _, _ = load_biored_documents(args.test_json)
    os.makedirs(args.output_dir, exist_ok=True)
    workflow = AgentWorkflow()

    if args.limit > 0:
        docs = docs[:args.limit]
        print(f"[limit] DEBUG: running {len(docs)} docs only")

    preds_by_doc = {}
    gold_all = gold_entities_by_doc(docs)

    for idx, doc in enumerate(docs, 1):
        print("##############################################")
        doc_text = reconstruct_doc_text(doc)
        state = build_initial_state(doc, doc_text)

        try:
            result = workflow.graph.invoke(state)
        except Exception as e:
            print(f"[{doc.doc_id}] graph failed: {e}")
            preds_by_doc[doc.doc_id] = []
            continue

        items = []
        seen = set()
        for d in result["final_entities"]:
            if d.start == -1 or d.end == -1:
                continue 
            canon_type = entity_type_to_canonical(d.entity_type)
            key = (d.start, d.end, canon_type)
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "start": d.start,
                "end": d.end,
                "type": canon_type,
                "text": doc_text[d.start:d.end]
            })
        preds_by_doc[doc.doc_id] = items

        sources = result.get("branch_sources", {})
        print(f"  [{idx}/{len(docs)}] {doc.doc_id}: "
              f"{len(items)} entities | "
              f"pattern={sources.get('pattern',0)} "
              f"rare={sources.get('rare',0)} "
              f"common={sources.get('common',0)} "
              f"requery={sources.get('requery_needed',0)} "
              f"dropped={sources.get('dropped',0)}")

        if result.get("errors"):
            print(f"    errors: {result['errors']}")

    write_canonical(
        os.path.join(args.output_dir, "test_predictions.json"), preds_by_doc
    )
    write_canonical(
        os.path.join(args.output_dir, "gold_test.json"), gold_all
    )

    score = strict_prf(preds_by_doc, gold_all)
    with open(os.path.join(args.output_dir, "test_strict_metrics.json"), "w") as f:
        json.dump(score, f, indent=2)
    print("\n[strict entity-level sanity]", json.dumps(score, indent=2))
    print(f"\nNow run scorer.py for the full strict/relaxed/per-type table:")
    print(f"PYTHONPATH=. python scorer.py "
          f"--pred {args.output_dir}/test_predictions.json "
          f"--gold {args.output_dir}/gold_test.json "
          f"--name tier4_agent "
          f"--out {args.output_dir}/full_metrics.json")

if __name__ == "__main__":
    main()