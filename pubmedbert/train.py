"""Tier 1 - fine-tune PubMedBERT for BioRED NER (token classification).

Run:
  python train.py \
    --train_json /path/to/data/train/Train.BioC.JSON \
    --dev_json   /path/to/data/dev/Dev.BioC.JSON \
    --output_dir outputs/tier1_pubmedbert

Outputs:
  outputs/tier1_pubmedbert/model/            fine-tuned model + tokenizer
  outputs/tier1_pubmedbert/dev_metrics.json  seqeval dev metrics (strict, BIO)
"""
import argparse
import json
import os

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from biored import build_label_maps, encode_for_training, load_biored_documents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--dev_json", required=True)
    ap.add_argument(
        "--model_name",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    ap.add_argument("--output_dir", default="outputs/tier1_pubmedbert")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    label_list, label2id, id2label = build_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_docs, skipped_tr, mm_tr = load_biored_documents(args.train_json)
    dev_docs, _, _ = load_biored_documents(args.dev_json)
    print(
        f"[data] train docs={len(train_docs)} dev docs={len(dev_docs)} "
        f"skipped_types={dict(skipped_tr)} offset_mismatches={mm_tr}"
    )

    train_ds, dropped_tr = encode_for_training(
        train_docs, tokenizer, label2id, args.max_length, args.stride
    )
    dev_ds, dropped_dev = encode_for_training(
        dev_docs, tokenizer, label2id, args.max_length, args.stride
    )
    print(
        f"[encode] train chunks={len(train_ds)} (overlaps dropped={dropped_tr}); "
        f"dev chunks={len(dev_ds)} (overlaps dropped={dropped_dev})"
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    collator = DataCollatorForTokenClassification(tokenizer)

    import evaluate

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_preds, true_labs = [], []
        for pr, gl in zip(preds, gold):
            tp_, tl_ = [], []
            for pi, gi in zip(pr, gl):
                if gi == -100:
                    continue
                tp_.append(id2label[int(pi)])
                tl_.append(id2label[int(gi)])
            true_preds.append(tp_)
            true_labs.append(tl_)
        r = seqeval.compute(
            predictions=true_preds, references=true_labs, zero_division=0
        )
        return {
            "precision": r["overall_precision"],
            "recall": r["overall_recall"],
            "f1": r["overall_f1"],
            "accuracy": r["overall_accuracy"],
        }

    targs = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        weight_decay=0.01,
        warmup_steps=0.1,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    dev_metrics = trainer.evaluate()
    print("[dev]", dev_metrics)

    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(args.output_dir, "dev_metrics.json"), "w") as f:
        json.dump(dev_metrics, f, indent=2)
    print(f"[done] model saved to {model_dir}")


if __name__ == "__main__":
    main()