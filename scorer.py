import argparse
import json
from collections import Counter, defaultdict


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ents(doc_list):
    """JSON entity dicts -> list of (start, end, type)."""
    return [(int(e["start"]), int(e["end"]), e["type"]) for e in doc_list]


def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _greedy_same_type_overlap(gold, pred):
    """One-to-one greedy match on (same type AND char overlap). Returns matched count."""
    cands = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if g[2] == p[2]:
                ov = _overlap(g, p)
                if ov > 0:
                    cands.append((ov, gi, pi))
    cands.sort(reverse=True)
    used_g, used_p, matched = set(), set(), 0
    for _, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched += 1
    return matched


def _taxonomy_doc(gold, pred, tax, confusion):
    """Categorise every gold/pred via priority passes; mutate `tax` and `confusion`."""
    g_used, p_used = set(), set()

    # Pass 1: exact (start, end, type)
    pred_index = defaultdict(list)
    for pi, p in enumerate(pred):
        pred_index[p].append(pi)
    for gi, g in enumerate(gold):
        for pi in pred_index.get(g, []):
            if pi not in p_used:
                g_used.add(gi)
                p_used.add(pi)
                tax["exact"] += 1
                break

    # Pass 2: same (start, end), different type -> pure type error
    span_to_pred = defaultdict(list)
    for pi, p in enumerate(pred):
        if pi not in p_used:
            span_to_pred[(p[0], p[1])].append(pi)
    for gi, g in enumerate(gold):
        if gi in g_used:
            continue
        for pi in span_to_pred.get((g[0], g[1]), []):
            if pi not in p_used:
                g_used.add(gi)
                p_used.add(pi)
                tax["type_only"] += 1
                confusion[f"{g[2]} -> {pred[pi][2]}"] += 1
                break

    # Pass 3: same type, overlapping (not exact) -> boundary error
    cands = []
    for gi, g in enumerate(gold):
        if gi in g_used:
            continue
        for pi, p in enumerate(pred):
            if pi in p_used or g[2] != p[2]:
                continue
            ov = _overlap(g, p)
            if ov > 0:
                cands.append((ov, gi, pi))
    cands.sort(reverse=True)
    for _, gi, pi in cands:
        if gi in g_used or pi in p_used:
            continue
        g_used.add(gi)
        p_used.add(pi)
        tax["boundary_only"] += 1

    # Pass 4: overlapping, different type -> both wrong
    cands = []
    for gi, g in enumerate(gold):
        if gi in g_used:
            continue
        for pi, p in enumerate(pred):
            if pi in p_used:
                continue
            ov = _overlap(g, p)
            if ov > 0:
                cands.append((ov, gi, pi))
    cands.sort(reverse=True)
    for _, gi, pi in cands:
        if gi in g_used or pi in p_used:
            continue
        g_used.add(gi)
        p_used.add(pi)
        tax["type_and_boundary"] += 1

    # Leftovers
    tax["spurious_fp"] += sum(1 for pi in range(len(pred)) if pi not in p_used)
    tax["missed_fn"] += sum(1 for gi in range(len(gold)) if gi not in g_used)


def score(pred_by_doc, gold_by_doc):
    all_types = set()
    strict = Counter()                       # overall strict tp/fp/fn
    per_type = defaultdict(Counter)          # type -> tp/fp/fn (+ support)
    relaxed_tp = relaxed_pred = relaxed_gold = 0
    tax = Counter()
    confusion = Counter()

    doc_ids = set(pred_by_doc) | set(gold_by_doc)
    for doc_id in doc_ids:
        gold = _ents(gold_by_doc.get(doc_id, []))
        pred = _ents(pred_by_doc.get(doc_id, []))
        for _, _, t in gold + pred:
            all_types.add(t)

        gset, pset = set(gold), set(pred)

        # strict overall
        tp = len(gset & pset)
        strict["tp"] += tp
        strict["fp"] += len(pset - gset)
        strict["fn"] += len(gset - pset)

        # strict per type
        for t in all_types:
            g_t = {x for x in gset if x[2] == t}
            p_t = {x for x in pset if x[2] == t}
            per_type[t]["tp"] += len(g_t & p_t)
            per_type[t]["fp"] += len(p_t - g_t)
            per_type[t]["fn"] += len(g_t - p_t)
            per_type[t]["support"] += len(g_t)

        # relaxed (same type + overlap)
        m = _greedy_same_type_overlap(gold, pred)
        relaxed_tp += m
        relaxed_pred += len(pred)
        relaxed_gold += len(gold)

        _taxonomy_doc(gold, pred, tax, confusion)

    strict_metrics = _prf(strict["tp"], strict["fp"], strict["fn"])
    relaxed_metrics = _prf(
        relaxed_tp, relaxed_pred - relaxed_tp, relaxed_gold - relaxed_tp
    )

    per_type_out, f1s = {}, []
    for t in sorted(all_types):
        c = per_type[t]
        m = _prf(c["tp"], c["fp"], c["fn"])
        m["support"] = c["support"]
        per_type_out[t] = m
        f1s.append(m["f1"])
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "n_docs": len(doc_ids),
        "strict": strict_metrics,
        "relaxed": relaxed_metrics,
        "macro_f1_strict": macro_f1,
        "per_type_strict": per_type_out,
        "error_taxonomy": dict(tax),
        "type_confusion": dict(confusion.most_common()),
    }


def _print_report(name, r):
    def line(label, m):
        return (f"  {label:9s}  P {m['precision']*100:5.1f}  R {m['recall']*100:5.1f}"
                f"  F1 {m['f1']*100:5.1f}   (tp {m['tp']} fp {m['fp']} fn {m['fn']})")

    print(f"\n=== {name}  ({r['n_docs']} docs) ===")
    print("Overall micro:")
    print(line("strict", r["strict"]))
    print(line("relaxed", r["relaxed"]))
    print(f"  macro-F1 (strict, unweighted over types): {r['macro_f1_strict']*100:5.1f}")

    print("\nPer-type (strict):")
    print(f"  {'type':<28}{'support':>8}{'P':>7}{'R':>7}{'F1':>7}")
    for t, m in r["per_type_strict"].items():
        print(f"  {t:<28}{m['support']:>8}{m['precision']*100:>7.1f}"
              f"{m['recall']*100:>7.1f}{m['f1']*100:>7.1f}")

    print("\nError taxonomy (every pred/gold bucketed once):")
    labels = {
        "exact": "exact match (strict TP)",
        "type_only": "right span, wrong type",
        "boundary_only": "right type, wrong span",
        "type_and_boundary": "overlap, wrong type+span",
        "spurious_fp": "spurious prediction (pure FP)",
        "missed_fn": "missed gold (pure FN)",
    }
    for k, lbl in labels.items():
        print(f"  {lbl:<32}{r['error_taxonomy'].get(k, 0):>6}")

    if r["type_confusion"]:
        print("\nType confusion (boundary-correct, type wrong):")
        for pair, n in list(r["type_confusion"].items())[:10]:
            print(f"  {pair:<48}{n:>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--name", default="system")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = score(_load(args.pred), _load(args.gold))
    _print_report(args.name, r)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"name": args.name, **r}, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()