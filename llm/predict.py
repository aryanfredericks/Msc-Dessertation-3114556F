"""Tier 3 - single-call LLM NER on BioRED (the orchestration-free control).

One Groq chat completion per document (JSON mode, temperature 0). The LLM returns
surface strings + types; this script maps each surface string back to character
offsets deterministically (LLMs cannot count characters reliably). Same canonical
output + scorer as Tiers 1-2.

Built for tight free-tier limits (e.g. llama-3.1-8b-instant: 6K tokens/min,
500K/day). Paces requests ADAPTIVELY from real per-response token usage to stay
under the per-minute token budget, backs off on rate-limit errors, and
checkpoints each document to JSONL so an interrupted run RESUMES.

Setup:
  pip install -r requirements.txt
  export GROQ_API_KEY=...

Run zero-shot (cheap; do this for the mean+/-std headline):
  PYTHONPATH=. python tier3_llm/predict.py --test_json ./dataset/test/Test.BioC.JSON \
    --shots 0 --output_dir outputs/tier3_llm_0shot

Run few-shot (heavier; one full run ~1h under 6K tpm):
  PYTHONPATH=. python tier3_llm/predict.py --test_json ./dataset/test/Test.BioC.JSON \
    --example_json ./dataset/train/Train.BioC.JSON --shots 3 \
    --output_dir outputs/tier3_llm_3shot

If a run stops, RERUN the same command - it resumes from the checkpoint.
Delete <output_dir>/checkpoint.jsonl to start fresh.
"""
import argparse
import json
import os
import time
from collections import deque

from biored import (
    CANONICAL_TYPES,
    gold_entities_by_doc,
    load_biored_documents,
    strict_prf,
    write_canonical,
)
from prompt import build_fewshot_messages, build_system_prompt, reconstruct_doc_text


# ---------------------------------------------------------------- offset mapping
def find_occurrences(doc_text, surface):
    """All token-boundary-aligned exact matches of `surface` in `doc_text`."""
    out, L, start = [], len(surface), 0
    if not surface:
        return out
    while True:
        i = doc_text.find(surface, start)
        if i == -1:
            break
        j = i + L
        left_ok = i == 0 or not doc_text[i - 1].isalnum()
        right_ok = j == len(doc_text) or not doc_text[j].isalnum()
        if left_ok and right_ok:
            out.append((i, j))
        start = i + 1
    return out


def parse_entities(content):
    """Parse the JSON response into (text, type) pairs, tolerating ```json fences."""
    txt = content.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[txt.find("{") : txt.rfind("}") + 1]
    data = json.loads(txt)
    pairs = []
    for e in data.get("entities", []):
        if isinstance(e, dict) and "text" in e and "type" in e:
            pairs.append((str(e["text"]), str(e["type"])))
    return pairs


def map_pairs_to_items(pairs, doc_text):
    seen, items = set(), []
    for surface, etype in pairs:
        if etype not in CANONICAL_TYPES or not surface:
            continue
        for a, b in find_occurrences(doc_text, surface):
            key = (a, b, etype)
            if key in seen:
                continue
            seen.add(key)
            items.append({"start": a, "end": b, "type": etype, "text": doc_text[a:b]})
    return items


# ----------------------------------------------------------- adaptive rate budget
class RateBudget:
    """Rolling 60s window over (requests, tokens) to respect per-minute limits."""

    def __init__(self, tpm, rpm, safety=0.90):
        self.tpm = tpm * safety
        self.rpm = rpm * safety
        self.events = deque()  # (timestamp, tokens)

    def _prune(self, now):
        while self.events and now - self.events[0][0] >= 60.0:
            self.events.popleft()

    def wait(self, est_tokens):
        """Block until sending an est_tokens call keeps us under both limits."""
        while True:
            now = time.time()
            self._prune(now)
            tok = sum(t for _, t in self.events)
            if tok + est_tokens <= self.tpm and len(self.events) + 1 <= self.rpm:
                return
            sleep_for = 60.0 - (now - self.events[0][0]) + 0.2 if self.events else 1.0
            time.sleep(max(0.5, sleep_for))

    def record(self, tokens):
        self.events.append((time.time(), tokens))


def estimate_tokens(messages, max_output):
    chars = sum(len(m["content"]) for m in messages)
    return chars // 4 + max_output  # ~4 chars/token + output ceiling (conservative)


def is_rate_limit(ex):
    name = ex.__class__.__name__.lower()
    msg = str(ex).lower()
    return ("ratelimit" in name or "rate limit" in msg or "429" in msg
            or "too many requests" in msg)


def is_content_400(ex):
    msg = str(ex).lower()
    return "400" in msg and (
        "json" in msg or "failed_generation" in msg or "adjust your prompt" in msg
    )


def call_llm(client, model, messages, temperature, seed, max_output, max_retries, cooldown):
    """Return (pairs, total_tokens, status).

    status is one of:
      "ok"   - parsed successfully (pairs may be empty if the doc has no entities)
      "skip" - this document failed JSON generation; record empty and move on
      "stop" - sustained rate limit; stop the run so it can resume later

    On a content-400 (8B failed to emit valid JSON), drops strict JSON mode and
    retries once; if it still fails, the document is skipped rather than halting.
    """
    last, use_json, last_was_rate = "", True, False
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_output,
            )
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            pairs = parse_entities(resp.choices[0].message.content)
            total = getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
            return pairs, total, "ok"
        except Exception as ex:  # noqa: BLE001
            last = str(ex)
            if is_rate_limit(ex):
                last_was_rate = True
                print(f"    [rate-limit] attempt {attempt + 1}/{max_retries}; "
                      f"waiting {cooldown}s")
                time.sleep(cooldown)
                continue
            last_was_rate = False
            if is_content_400(ex):
                if use_json:  # one fallback: retry without strict JSON mode
                    use_json = False
                    print("    [json-400] retrying once without strict JSON mode")
                    continue
                print(f"    [skip doc] JSON generation failed twice: {last[:80]}")
                return [], 0, "skip"
            wait = 2 * (attempt + 1)
            print(f"    [error] attempt {attempt + 1}/{max_retries}; waiting {wait}s "
                  f"({last[:80]})")
            time.sleep(wait)
    # retries exhausted: stop only if the last error was a rate limit, else skip
    if last_was_rate:
        print(f"    [give up] sustained rate limit: {last[:100]}")
        return None, 0, "stop"
    print(f"    [skip doc] persistent error: {last[:100]}")
    return [], 0, "skip"


# ------------------------------------------------------------------- checkpoints
def load_checkpoint(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["doc_id"]] = {"items": rec["items"], "raw": rec["raw"]}
    return done


def append_checkpoint(path, doc_id, items, raw):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"doc_id": doc_id, "items": items, "raw": raw},
                           ensure_ascii=False) + "\n")


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--example_json", default="./dataset/train/Train.BioC.JSON")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="outputs/tier3_llm")
    # rate limits (defaults match llama-3.1-8b-instant free tier)
    ap.add_argument("--tpm", type=int, default=6000, help="tokens per minute limit")
    ap.add_argument("--rpm", type=int, default=30, help="requests per minute limit")
    ap.add_argument("--max_output", type=int, default=1024,
                    help="cap on output tokens (also used for conservative pacing)")
    ap.add_argument("--cooldown", type=float, default=30.0,
                    help="seconds to wait after a rate-limit error")
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="DEBUG ONLY: process at most N new docs (NOT comparable "
                         "across tiers - report only full-100 runs)")
    args = ap.parse_args()

    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")

    client = Groq(api_key=api_key)

    system_msg = {"role": "system", "content": build_system_prompt()}
    fewshot = (
        build_fewshot_messages(args.example_json, args.shots) if args.shots > 0 else []
    )
    print(f"[setup] model={args.model} shots={args.shots} temp={args.temperature} "
          f"budget={args.tpm} tpm / {args.rpm} rpm (90% safety)")

    docs, _, _ = load_biored_documents(args.test_json)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "checkpoint.jsonl")

    done = load_checkpoint(ckpt_path)
    if done:
        print(f"[resume] {len(done)} docs already in checkpoint; skipping those")

    to_do = [d for d in docs if d.doc_id not in done]
    if args.limit > 0:
        to_do = to_do[: args.limit]
        print(f"[limit] DEBUG: processing only {len(to_do)} new docs this run")

    budget = RateBudget(args.tpm, args.rpm)
    stopped, skipped = False, []
    for n, d in enumerate(to_do, 1):
        doc_text = reconstruct_doc_text(d)
        messages = [system_msg] + fewshot + [{"role": "user", "content": doc_text}]

        budget.wait(estimate_tokens(messages, args.max_output)) 
        pairs, used, status = call_llm(
            client, args.model, messages, args.temperature, args.seed,
            args.max_output, args.max_retries, args.cooldown
        )
        if pairs is None:
            print("llm returned a null (text, type) pair")
            continue
        if status == "stop":
            print("[stop] sustained rate limit. Rerun the same command later to "
                  "resume from checkpoint.")
            stopped = True
            break
        budget.record(used or estimate_tokens(messages, args.max_output))

        items = map_pairs_to_items(pairs, doc_text)
        raw = [{"text": t, "type": ty} for t, ty in pairs]
        append_checkpoint(ckpt_path, d.doc_id, items, raw)
        done[d.doc_id] = {"items": items, "raw": raw}
        if status == "skip":
            skipped.append(d.doc_id)
            print(f"  [{n}/{len(to_do)}] {d.doc_id}: SKIPPED (no parseable output) "
                  "-> 0 entities")
        else:
            print(f"  [{n}/{len(to_do)}] {d.doc_id}: {len(items)} entities, {used} tok")

    # Finalize over whatever is done (consistent pred+gold subset).
    done_ids = [d.doc_id for d in docs if d.doc_id in done]
    preds_by_doc = {doc_id: done[doc_id]["items"] for doc_id in done_ids}
    raw_log = [{"doc_id": doc_id, "entities": done[doc_id]["raw"]} for doc_id in done_ids]

    gold_all = gold_entities_by_doc(docs)
    gold_by_doc = {doc_id: gold_all[doc_id] for doc_id in done_ids}

    write_canonical(os.path.join(args.output_dir, "test_predictions.json"), preds_by_doc)
    write_canonical(os.path.join(args.output_dir, "gold_test.json"), gold_by_doc)
    with open(os.path.join(args.output_dir, "raw_responses.json"), "w", encoding="utf-8") as f:
        json.dump(raw_log, f, indent=2, ensure_ascii=False)

    score = strict_prf(preds_by_doc, gold_by_doc)
    with open(os.path.join(args.output_dir, "test_strict_metrics.json"), "w") as f:
        json.dump(score, f, indent=2)

    complete = len(done_ids) == len(docs)
    print(f"\n[done] {len(done_ids)}/{len(docs)} docs scored "
          f"({'complete' if complete else 'INCOMPLETE - rerun to finish'})")
    if skipped:
        with open(os.path.join(args.output_dir, "skipped.json"), "w") as f:
            json.dump(skipped, f, indent=2)
        print(f"[skipped] {len(skipped)} docs produced no parseable output "
              f"(counted as 0 predictions): {skipped}")
    print("[test strict entity-level]", json.dumps(score, indent=2))
    if not complete:
        print("\nPARTIAL result. Rerun the same command to process the remaining docs.")
    else:
        print("\nNow run scorer.py for the full strict/relaxed/per-type table.")


if __name__ == "__main__":
    main()