import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from biored import load_biored_documents, gold_entities_by_doc
from agent.utils.bert_span_extractor import extract_spans_with_bert
from agent.config import Configs


def reconstruct_doc_text(doc) -> str:
    """Same convention used throughout the project - see agent/run_agent.py."""
    end = max(p.offset + len(p.text) for p in doc.passages) if doc.passages else 0
    buf = [" "] * end
    for p in doc.passages:
        for i, ch in enumerate(p.text):
            buf[p.offset + i] = ch
    return "".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", default="./dataset/test/Test.BioC.JSON")
    ap.add_argument("--type", default="OrganismTaxon",
                     help="Entity type to check (OrganismTaxon, CellLine, SequenceVariant, etc.)")
    ap.add_argument("--max_examples", type=int, default=15,
                     help="How many example missed spans to print")
    args = ap.parse_args()

    configs = Configs()
    print(f"Loading PubMedBERT from {configs.pubmed_model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(configs.pubmed_model_dir)
    model = AutoModelForTokenClassification.from_pretrained(configs.pubmed_model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Loading {args.test_json} ...")
    docs, _, _ = load_biored_documents(args.test_json)
    gold_all = gold_entities_by_doc(docs)

    found = 0
    missed = 0
    missed_examples = []
    found_examples = []

    for idx, doc in enumerate(docs, 1):
        doc_text = reconstruct_doc_text(doc)
        try:
            candidate_spans = extract_spans_with_bert(
                doc=doc, doc_text=doc_text,
                tokenizer=tokenizer, model=model, device=device,
            )
        except Exception as e:
            print(f"  [{doc.doc_id}] span extraction failed: {e}")
            continue

        candidates_lower = set(s.lower() for s in candidate_spans)

        for e in gold_all.get(doc.doc_id, []):
            if e["type"] != args.type:
                continue
            if e["text"].lower() in candidates_lower:
                found += 1
                if len(found_examples) < args.max_examples:
                    found_examples.append(e["text"])
            else:
                missed += 1
                if len(missed_examples) < args.max_examples:
                    missed_examples.append((doc.doc_id, e["text"]))

        if idx % 20 == 0:
            print(f"  ...processed {idx}/{len(docs)} docs")

    total = found + missed
    print(f"\n=== Extraction ceiling for {args.type} ===")
    print(f"Total gold mentions: {total}")
    print(f"  Extracted as a candidate span: {found}")
    print(f"  NEVER extracted as a candidate: {missed}")
    if total:
        print(f"  Extraction recall ceiling: {100 * found / total:.1f}%")
        print(f"\n  (This is the maximum recall ANY downstream branch/agent could possibly")
        print(f"   achieve for {args.type}, since a span that's never extracted can't be typed.)")

    if missed_examples:
        print(f"\nSample missed spans (never extracted as candidates), up to {args.max_examples}:")
        for doc_id, text in missed_examples:
            print(f"    doc={doc_id}  '{text}'")

    if found_examples:
        print(f"\nSample spans that WERE extracted, up to {args.max_examples}:")
        for text in found_examples:
            print(f"    '{text}'")


if __name__ == "__main__":
    main()