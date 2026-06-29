import json
from collections import Counter
from dataclasses import dataclass


CANONICAL_TYPES = [
    "GeneOrGeneProduct",
    "DiseaseOrPhenotypicFeature",
    "ChemicalEntity",
    "SequenceVariant",
    "OrganismTaxon",
    "CellLine",
]


def build_label_maps(types=CANONICAL_TYPES):
    """Return (label_list, label2id, id2label) for a BIO scheme over `types`."""
    labels = ["O"]
    for t in types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


@dataclass
class Entity:
    start: int
    end: int
    type: str
    text: str


@dataclass
class Passage:
    offset: int
    text: str
    entities: list


@dataclass
class Document:
    doc_id: str
    passages: list 


def load_biored_documents(json_path, type_whitelist=frozenset(CANONICAL_TYPES)):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    docs, skipped_types, mismatches = [], Counter(), 0
    for d in data["documents"]:
        passages = []
        for p in d["passages"]:
            p_off = int(p["offset"])
            text = p.get("text", "")
            ents = []
            for ann in p.get("annotations", []):
                etype = ann.get("infons", {}).get("type")
                if etype not in type_whitelist:
                    skipped_types[etype] += 1
                    continue
                locations = ann.get("locations", [])
                for loc in locations:
                    s = int(loc["offset"])
                    e = s + int(loc["length"])
                    rel_s, rel_e = s - p_off, e - p_off
                    surface = (
                        text[rel_s:rel_e]
                        if 0 <= rel_s <= rel_e <= len(text)
                        else ann.get("text", "")
                    )
                    if len(locations) == 1 and surface != ann.get("text", surface):
                        mismatches += 1
                    ents.append(Entity(s, e, etype, surface))
            passages.append(Passage(p_off, text, ents))
        docs.append(Document(str(d["id"]), passages))
    return docs, skipped_types, mismatches


def gold_entities_by_doc(docs):
    """Canonical gold dict: {doc_id: [ {start,end,type,text}, ... ]}."""
    gold = {}
    for doc in docs:
        items = []
        for p in doc.passages:
            for e in p.entities:
                items.append(
                    {"start": e.start, "end": e.end, "type": e.type, "text": e.text}
                )
        gold[doc.doc_id] = items
    return gold


def _filter_overlaps(entities_rel):
    """Greedily keep longest-first non-overlapping spans so BIO is representable.

    entities_rel : list[(start, end, type)] passage-relative.
    Returns (kept_sorted_by_start, dropped_count).
    """
    kept, occupied, dropped = [], [], 0
    for s, e, t in sorted(entities_rel, key=lambda x: (-(x[1] - x[0]), x[0])):
        if any(s < oe and os < e for os, oe in occupied):  # [s,e) overlaps [os,oe)
            dropped += 1
            continue
        occupied.append((s, e))
        kept.append((s, e, t))
    kept.sort(key=lambda x: x[0])
    return kept, dropped


def align_labels(offset_mapping, sequence_ids, entities_rel, label2id):
    """Produce BIO label ids for one tokenized chunk.

    entities_rel must be non-overlapping and sorted by start (passage-relative).
    Special tokens / padding receive -100 so they are ignored by the loss.
    """
    labels, prev_idx = [], None
    for (ts, te), seq in zip(offset_mapping, sequence_ids):
        if seq is None or (ts == 0 and te == 0):  # special token
            labels.append(-100)
            prev_idx = None
            continue
        match = None
        for i, (s, e, t) in enumerate(entities_rel):
            if s <= ts < e:  # token starts inside this entity
                match = (i, t)
                break
        if match is None:
            labels.append(label2id["O"])
            prev_idx = None
        else:
            i, t = match
            tag = f"B-{t}" if i != prev_idx else f"I-{t}"
            labels.append(label2id[tag])
            prev_idx = i
    return labels


def encode_for_training(docs, tokenizer, label2id, max_length=512, stride=128):
    """Tokenize per passage with a sliding window and align BIO labels.

    Returns (datasets.Dataset, total_overlaps_dropped). The Dataset has only
    input_ids / attention_mask / labels so it feeds Trainer directly.
    """
    input_ids, attention_mask, all_labels = [], [], []
    total_dropped = 0
    for doc in docs:
        for p in doc.passages:
            if not p.text:
                continue
            ents_rel = [
                (e.start - p.offset, e.end - p.offset, e.type) for e in p.entities
            ]
            ents_rel = [(s, e, t) for s, e, t in ents_rel if 0 <= s < e <= len(p.text)]
            ents_rel, dropped = _filter_overlaps(ents_rel)
            total_dropped += dropped

            enc = tokenizer(
                p.text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )
            for ci in range(len(enc["input_ids"])):
                labels = align_labels(
                    enc["offset_mapping"][ci], enc.sequence_ids(ci), ents_rel, label2id
                )
                input_ids.append(enc["input_ids"][ci])
                attention_mask.append(enc["attention_mask"][ci])
                all_labels.append(labels)

    from datasets import Dataset

    ds = Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": all_labels}
    )
    return ds, total_dropped


def encode_for_inference(docs, tokenizer, max_length=512, stride=128):
    """Tokenize per passage, keeping everything needed to decode back to offsets."""
    chunks = []
    for doc in docs:
        for p in doc.passages:
            if not p.text:
                continue
            enc = tokenizer(
                p.text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )
            for ci in range(len(enc["input_ids"])):
                chunks.append(
                    {
                        "doc_id": doc.doc_id,
                        "passage_offset": p.offset,
                        "input_ids": enc["input_ids"][ci],
                        "attention_mask": enc["attention_mask"][ci],
                        "offset_mapping": enc["offset_mapping"][ci],
                        "sequence_ids": enc.sequence_ids(ci),
                    }
                )
    return chunks


def decode_chunk(pred_label_strs, offset_mapping, sequence_ids, passage_offset):
    """Turn one chunk's predicted BIO tags into document-absolute entity spans."""
    spans, cur = [], None  # cur = [type, start_char, end_char]
    for lab, (ts, te), seq in zip(pred_label_strs, offset_mapping, sequence_ids):
        if seq is None or (ts == 0 and te == 0):
            continue
        if lab == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        bio, _, t = lab.partition("-")
        if bio == "B" or cur is None or cur[0] != t:
            if cur:
                spans.append(cur)
            cur = [t, ts, te]
        else:  # I- continuing the same type
            cur[2] = te
    if cur:
        spans.append(cur)
    return [
        {"start": s + passage_offset, "end": e + passage_offset, "type": t}
        for t, s, e in spans
    ]


def write_canonical(path, entities_by_doc):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entities_by_doc, f, ensure_ascii=False, indent=2)


def strict_prf(pred_by_doc, gold_by_doc):
    """Strict entity-level micro precision/recall/F1 over (doc,start,end,type)."""
    def to_set(d):
        return {
            (doc_id, e["start"], e["end"], e["type"])
            for doc_id, ents in d.items()
            for e in ents
        }

    P, G = to_set(pred_by_doc), to_set(gold_by_doc)
    tp, fp, fn = len(P & G), len(P - G), len(G - P)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }