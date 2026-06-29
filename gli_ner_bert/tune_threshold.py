import argparse

import torch
from gliner import GLiNER

import scorer
from biored import gold_entities_by_doc, load_biored_documents
from predict import LABEL_PHRASES, chunk_by_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--model_name", default="Ihor/gliner-biomed-large-v1.0")
    ap.add_argument("--floor", type=float, default=0.05, help="catch-all threshold")
    ap.add_argument("--max_words", type=int, default=300)
    ap.add_argument("--overlap_words", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(args.model_name).to(device).eval()
    phrases = list(LABEL_PHRASES.keys())

    docs, _, _ = load_biored_documents(args.dev_json)
    gold_by_doc = gold_entities_by_doc(docs)

    # Predict once at the floor; keep (start, end, type, score), deduped per doc
    # keeping the max score for identical spans.
    cand = {d.doc_id: {} for d in docs}
    for d in docs:
        for p in d.passages:
            for chunk_text, c_off in chunk_by_words(
                p.text, args.max_words, args.overlap_words
            ):
                for e in model.predict_entities(
                    chunk_text, phrases, threshold=args.floor, flat_ner=True
                ):
                    canon = LABEL_PHRASES.get(e["label"])
                    if canon is None:
                        continue
                    key = (p.offset + c_off + e["start"], p.offset + c_off + e["end"], canon)
                    cand[d.doc_id][key] = max(cand[d.doc_id].get(key, 0.0), e["score"])

    print(f"{'thr':>5}{'strictP':>9}{'strictR':>9}{'strictF1':>10}{'relaxF1':>9}")
    best = (None, -1.0)
    for thr in [round(0.30 + 0.05 * k, 2) for k in range(9)]:  # 0.30 .. 0.70
        preds = {
            doc_id: [
                {"start": s, "end": e, "type": t}
                for (s, e, t), sc in spans.items()
                if sc >= thr
            ]
            for doc_id, spans in cand.items()
        }
        r = scorer.score(preds, gold_by_doc)
        s, rel = r["strict"], r["relaxed"]
        print(
            f"{thr:>5}{s['precision']*100:>9.1f}{s['recall']*100:>9.1f}"
            f"{s['f1']*100:>10.1f}{rel['f1']*100:>9.1f}"
        )
        if s["f1"] > best[1]:
            best = (thr, s["f1"])

    print(f"\nBest strict F1 on dev: threshold={best[0]}  (F1 {best[1]*100:.1f})")
    print("Use that with: python predict.py --threshold", best[0])


if __name__ == "__main__":
    main()