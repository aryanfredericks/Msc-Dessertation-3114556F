"""
convert_bc5cdr.py

Parses BC5CDR PubTator-format files (downloaded by download_bc5cdr.py) into
the same document format used for BioRED elsewhere in this project:

    { "<doc_id>": [{"start": 123, "end": 130, "type": "ChemicalEntity", "text": "aspirin"}, ...] }

Also writes a companion doc_text json mapping doc_id -> full document text
(title + " " + abstract), since PubTator annotation offsets are indexed
into that exact concatenation (verified against the raw corpus - single
space joiner, not newline). Your existing find_occurrences / offset
localisation logic depends on having this exact string to index into.

PubTator format reference:
    <PMID>|t|<title>
    <PMID>|a|<abstract>
    <PMID><TAB>start<TAB>end<TAB>mention_text<TAB>type<TAB>id[|id...][<TAB>individual_mentions]
    <PMID><TAB>CID<TAB>chem_id<TAB>disease_id      (relation lines - ignored here, NER only)
    <blank line separates documents>

Usage:
    python convert_bc5cdr.py --split test
    python convert_bc5cdr.py --split train --raw_dir ./datasets/bc5cdr/raw --out_dir ./datasets/bc5cdr/processed
"""

import json
import argparse
from pathlib import Path

# PubTator label -> BioRED label (only Chemical/Disease exist in BC5CDR;
# this is exactly the subset common_relation_agent is scoped to, minus Gene)
TYPE_MAP = {
    "Chemical": "ChemicalEntity",
    "Disease": "DiseaseOrPhenotypicFeature",
}

SPLIT_FILENAMES = {
    "train": "CDR_TrainingSet.PubTator.txt",
    "validation": "CDR_DevelopmentSet.PubTator.txt",
    "test": "CDR_TestSet.PubTator.txt",
}


def parse_pubtator_file(path: Path):
    """Yields (doc_id, doc_text, annotations) per document."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # documents are separated by blank lines
    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip("\n").split("\n")
        if not lines or not lines[0].strip():
            continue

        title = None
        abstract = None
        doc_id = None
        annotations = []

        for line in lines:
            if "|t|" in line:
                doc_id, _, title = line.partition("|t|")
                title = line.split("|t|", 1)[1]
                doc_id = line.split("|t|", 1)[0]
            elif "|a|" in line:
                doc_id2, _, abstract = line.partition("|a|")
                abstract = line.split("|a|", 1)[1]
            else:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                if parts[1] == "CID":
                    # relation annotation - not needed for NER, skip
                    continue
                if len(parts) < 6:
                    continue
                _pmid, start, end, text, etype, *_rest = parts
                mapped_type = TYPE_MAP.get(etype)
                if mapped_type is None:
                    continue
                annotations.append({
                    "start": int(start),
                    "end": int(end),
                    "type": mapped_type,
                    "text": text,
                })

        if doc_id is None or title is None or abstract is None:
            continue

        doc_text = title + " " + abstract
        yield doc_id, doc_text, annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--raw_dir", default="./datasets/bc5cdr/raw")
    parser.add_argument("--out_dir", default="./datasets/bc5cdr/processed")
    args = parser.parse_args()

    corpus_dir = Path(args.raw_dir) / "CDR_Data" / "CDR.Corpus.v010516"
    pubtator_path = corpus_dir / SPLIT_FILENAMES[args.split]

    if not pubtator_path.exists():
        raise FileNotFoundError(
            f"Could not find {pubtator_path}. Did you run download_bc5cdr.py first?"
        )

    print(f"Parsing {pubtator_path} ...")

    doc_annotations = {}
    doc_texts = {}

    for doc_id, doc_text, annotations in parse_pubtator_file(pubtator_path):
        doc_annotations[doc_id] = annotations
        doc_texts[doc_id] = doc_text

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = out_dir / f"bc5cdr_{args.split}_annotations.json"
    text_path = out_dir / f"bc5cdr_{args.split}_doc_text.json"

    with open(ann_path, "w") as f:
        json.dump(doc_annotations, f, indent=2)
    with open(text_path, "w") as f:
        json.dump(doc_texts, f, indent=2)

    total_entities = sum(len(v) for v in doc_annotations.values())
    type_counts = {}
    for anns in doc_annotations.values():
        for a in anns:
            type_counts[a["type"]] = type_counts.get(a["type"], 0) + 1

    print(f"\n[convert] wrote {ann_path}")
    print(f"[convert] wrote {text_path}")
    print(f"[convert] {len(doc_annotations)} documents, {total_entities} entities")
    for t, c in type_counts.items():
        print(f"    {t}: {c}")

    # --- sanity check: verify offsets actually match the text ---
    print("\n[sanity check] verifying span offsets against doc_text for 5 docs...")
    checked = 0
    mismatches = 0
    for doc_id, anns in list(doc_annotations.items())[:5]:
        text = doc_texts[doc_id]
        for a in anns:
            checked += 1
            actual = text[a["start"]:a["end"]]
            if actual != a["text"]:
                mismatches += 1
                print(f"  MISMATCH doc={doc_id} expected='{a['text']}' got='{actual}'")
    print(f"[sanity check] {checked} spans checked, {mismatches} mismatches")


if __name__ == "__main__":
    main()