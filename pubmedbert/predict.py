"""Tier 1 - run the fine-tuned PubMedBERT on the BioRED test set and save outputs.

Run:
  python predict.py \
    --model_dir outputs/tier1_pubmedbert/model \
    --test_json /path/to/data/test/Test.BioC.JSON \
    --output_dir outputs/tier1_pubmedbert

Outputs (canonical format, reused by every later tier's scorer):
  outputs/tier1_pubmedbert/test_predictions.json
  outputs/tier1_pubmedbert/gold_test.json
  outputs/tier1_pubmedbert/test_strict_metrics.json   (quick sanity F1)

The strict number here is a sanity check. The full strict / relaxed / per-type /
cost table comes from the shared cross-tier scorer built in a later step.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from biored import (
    decode_chunk,
    encode_for_inference,
    gold_entities_by_doc,
    load_biored_documents,
    strict_prf,
    write_canonical,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--output_dir", default="outputs/tier1_pubmedbert")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    docs, _, _ = load_biored_documents(args.test_json)
    chunks = encode_for_inference(docs, tokenizer, args.max_length, args.stride)

    preds_by_doc = {d.doc_id: [] for d in docs}
    seen = {d.doc_id: set() for d in docs}

    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i : i + args.batch_size]
        maxlen = max(len(c["input_ids"]) for c in batch)
        input_ids = torch.zeros(len(batch), maxlen, dtype=torch.long)
        attn = torch.zeros(len(batch), maxlen, dtype=torch.long)
        for j, c in enumerate(batch):
            L = len(c["input_ids"])
            input_ids[j, :L] = torch.tensor(c["input_ids"], dtype=torch.long)
            attn[j, :L] = torch.tensor(c["attention_mask"], dtype=torch.long)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids.to(device), attention_mask=attn.to(device)
            ).logits
        pred_ids = logits.argmax(-1).cpu().numpy()

        for j, c in enumerate(batch):
            L = len(c["input_ids"])
            lab_strs = [id2label[int(x)] for x in pred_ids[j, :L]]
            ents = decode_chunk(
                lab_strs, c["offset_mapping"], c["sequence_ids"], c["passage_offset"]
            )
            for e in ents:
                key = (e["start"], e["end"], e["type"])
                if key in seen[c["doc_id"]]:  # dedupe across sliding-window overlap
                    continue
                seen[c["doc_id"]].add(key)
                preds_by_doc[c["doc_id"]].append(
                    {"start": e["start"], "end": e["end"], "type": e["type"], "text": ""}
                )

    # Fill surface text from the source passages.
    for d in docs:
        for e in preds_by_doc[d.doc_id]:
            for p in d.passages:
                rs, re = e["start"] - p.offset, e["end"] - p.offset
                if 0 <= rs < re <= len(p.text):
                    e["text"] = p.text[rs:re]
                    break

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


if __name__ == "__main__":
    main()