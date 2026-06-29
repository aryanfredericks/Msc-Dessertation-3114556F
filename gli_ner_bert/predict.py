import argparse
import json
import os

import torch
from gliner import GLiNER

from biored import (
    gold_entities_by_doc,
    load_biored_documents,
    strict_prf,
    write_canonical,
)

# GLiNER reads label names as natural language, so the BioRED type strings are
# given human phrasings. The value is the canonical BioRED type used everywhere
# else. EDIT THESE PHRASINGS - they change the score; tune on dev.
LABEL_PHRASES = {
    "gene or gene product": "GeneOrGeneProduct",
    "disease or phenotypic feature": "DiseaseOrPhenotypicFeature",
    "chemical": "ChemicalEntity",
    "sequence variant": "SequenceVariant",
    "organism or species": "OrganismTaxon",
    "cell line": "CellLine",
}


def chunk_by_words(text, max_words=300, overlap_words=50):
    """Split text into overlapping word windows, never cutting a word.

    Returns list of (chunk_text, char_offset) where char_offset is the start of
    the chunk within `text`. GLiNER prefers short inputs, so this keeps recall up
    on long abstracts while preserving exact offsets.
    """
    n = len(text)
    words, i = [], 0
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        words.append((start, i))  # (char_start, char_end) of one word
    if not words:
        return []

    chunks, step = [], max(1, max_words - overlap_words)
    for s in range(0, len(words), step):
        win = words[s : s + max_words]
        if not win:
            break
        c_start, c_end = win[0][0], win[-1][1]
        chunks.append((text[c_start:c_end], c_start))
        if s + max_words >= len(words):
            break
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--model_name", default="Ihor/gliner-biomed-large-v1.0")
    ap.add_argument("--output_dir", default="outputs/tier2_gliner")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_words", type=int, default=300)
    ap.add_argument("--overlap_words", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(args.model_name).to(device).eval()
    phrases = list(LABEL_PHRASES.keys())

    docs, _, _ = load_biored_documents(args.test_json)
    preds_by_doc = {d.doc_id: [] for d in docs}
    seen = {d.doc_id: set() for d in docs}

    for d in docs:
        for p in d.passages:
            for chunk_text, c_off in chunk_by_words(
                p.text, args.max_words, args.overlap_words
            ):
                ents = model.predict_entities(
                    chunk_text, phrases, threshold=args.threshold, flat_ner=True
                )
                for e in ents:
                    canon = LABEL_PHRASES.get(e["label"])
                    if canon is None:
                        continue
                    abs_start = p.offset + c_off + e["start"]
                    abs_end = p.offset + c_off + e["end"]
                    key = (abs_start, abs_end, canon)
                    if key in seen[d.doc_id]:  # dedupe across overlapping windows
                        continue
                    seen[d.doc_id].add(key)
                    surface = e.get("text") or chunk_text[e["start"] : e["end"]]
                    preds_by_doc[d.doc_id].append(
                        {"start": abs_start, "end": abs_end, "type": canon, "text": surface}
                    )

    gold_by_doc = gold_entities_by_doc(docs)
    os.makedirs(args.output_dir, exist_ok=True)
    write_canonical(
        os.path.join(args.output_dir, "test_predictions.json"), preds_by_doc
    )
    write_canonical(os.path.join(args.output_dir, "gold_test.json"), gold_by_doc)

    score = strict_prf(preds_by_doc, gold_by_doc)
    with open(os.path.join(args.output_dir, "test_strict_metrics.json"), "w") as f:
        json.dump(score, f, indent=2)
    print("[test strict entity-level]", json.dumps(score, indent=2))
    print("\nNow run the shared scorer for the full strict/relaxed/per-type table.")


if __name__ == "__main__":
    main()