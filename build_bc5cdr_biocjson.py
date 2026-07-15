
import json
import argparse
from pathlib import Path

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
    """Yields (doc_id, title, abstract, annotations) per document.
    annotations: list of (start, end, text, mapped_type)  [global offsets]
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip("\n").split("\n")
        if not lines or not lines[0].strip():
            continue

        title = None
        abstract = None
        doc_id = None
        raw_annotations = []

        for line in lines:
            if "|t|" in line:
                doc_id = line.split("|t|", 1)[0]
                title = line.split("|t|", 1)[1]
            elif "|a|" in line:
                abstract = line.split("|a|", 1)[1]
            else:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                if parts[1] == "CID":
                    continue
                if len(parts) < 6:
                    continue
                _pmid, start, end, text, etype, *_rest = parts
                mapped_type = TYPE_MAP.get(etype)
                if mapped_type is None:
                    continue
                raw_annotations.append((int(start), int(end), text, mapped_type))

        if doc_id is None or title is None or abstract is None:
            continue

        yield doc_id, title, abstract, raw_annotations


def build_biocjson(pubtator_path: Path) -> dict:
    documents = []

    for doc_id, title, abstract, raw_annotations in parse_pubtator_file(pubtator_path):
        title_offset = 0
        abstract_offset = len(title) + 1  # single-space joiner, verified earlier

        title_annotations = []
        abstract_annotations = []
        ann_id = 0

        for start, end, text, mapped_type in raw_annotations:
            entry = {
                "id": str(ann_id),
                "infons": {
                    "identifier": "-1",
                    "type": mapped_type
                },
                "text": text,
                "locations": [
                    {"offset": start, "length": end - start}
                ]
            }
            ann_id += 1

            if start < abstract_offset:
                title_annotations.append(entry)
            else:
                abstract_annotations.append(entry)

        doc = {
            "id": doc_id,
            "passages": [
                {
                    "offset": title_offset,
                    "text": title,
                    "annotations": title_annotations
                },
                {
                    "offset": abstract_offset,
                    "text": abstract,
                    "annotations": abstract_annotations
                }
            ],
            "relations": []
        }
        documents.append(doc)

    return {
        "source": "BC5CDR",
        "date": "converted",
        "key": "BioC.key",
        "documents": documents
    }


def reconstruct_doc_text(doc: dict) -> str:
    """Mirrors agent/run_agent.py's reconstruct_doc_text for verification."""
    end = max(p["offset"] + len(p["text"]) for p in doc["passages"]) if doc["passages"] else 0
    buf = [" "] * end
    for p in doc["passages"]:
        for i, ch in enumerate(p["text"]):
            buf[p["offset"] + i] = ch
    return "".join(buf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--raw_dir", default="./dataset/bc5cdr/raw")
    parser.add_argument("--out", default="./dataset/bc5cdr/test/BC5CDR_Test.BioC.JSON")
    args = parser.parse_args()

    corpus_dir = Path(args.raw_dir) / "CDR_Data" / "CDR.Corpus.v010516"
    pubtator_path = corpus_dir / SPLIT_FILENAMES[args.split]

    if not pubtator_path.exists():
        raise FileNotFoundError(
            f"Could not find {pubtator_path}. Did you run download_bc5cdr.py first?"
        )

    print(f"Parsing {pubtator_path} ...")
    biocjson = build_biocjson(pubtator_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(biocjson, f, indent=2)

    n_docs = len(biocjson["documents"])
    n_entities = sum(
        len(p["annotations"]) for d in biocjson["documents"] for p in d["passages"]
    )
    print(f"[build] wrote {out_path}")
    print(f"[build] {n_docs} documents, {n_entities} entities")

    print("\n[verify] checking all annotation offsets against reconstructed doc_text...")
    checked = 0
    mismatches = 0
    for doc in biocjson["documents"]:
        doc_text = reconstruct_doc_text(doc)
        for p in doc["passages"]:
            for a in p["annotations"]:
                loc = a["locations"][0]
                start, length = loc["offset"], loc["length"]
                actual = doc_text[start:start + length]
                checked += 1
                if actual != a["text"]:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"  MISMATCH doc={doc['id']} expected='{a['text']}' got='{actual}'")
    print(f"[verify] {checked} annotations checked across {n_docs} docs, {mismatches} mismatches")


if __name__ == "__main__":
    main()