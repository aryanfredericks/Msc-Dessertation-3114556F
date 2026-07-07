"""
agent/run_tier5.py

Tier 5: replaces Tier 4's deterministic confidence-gated combiner with a
single Groq LLM agent per document that decides, per candidate span, which
of the three existing branches (common / pattern / rare) to invoke as
tools, then emits a final type assignment. Span extraction (PubMedBERT)
is UNCHANGED from Tier 4 - only the arbitration/routing step is replaced,
so any F1 delta vs Tier 4 is attributable to routing strategy alone.

NOT YET RUN/VERIFIED - no access to GROQ_API_KEY, the PubMedBERT
checkpoint, or your config.py/biored.py/utils modules in this environment.
Treat as a first draft to debug against your actual stack, not a
guaranteed-working script. See the checklist at the bottom of this file
docstring for the most likely breakage points.

Usage:
    PYTHONPATH=. uv run agent/run_tier5.py \
        --test_json ./dataset/test/Test.BioC.JSON \
        --output_dir outputs/tier5_agent \
        --limit 5          # ALWAYS smoke-test with --limit first

Checklist before a full run:
  1. Add a tool-calling-capable model to config.py, e.g.:
         self.tier5_agent_model = "llama-3.3-70b-versatile"
     Your existing configs.reasoning_model (qwen3, used for the Tier 4
     overseer) may not support function calling reliably - reasoning
     models and tool-calling don't always mix well on Groq. Verify with
     a 1-span smoke test before trusting it at scale.
  2. Rate limits: this makes 1 call/turn per document, with up to
     max_turns turns each involving several tool calls. On a 500-doc
     test set that could be several hundred to 1000+ LLM calls. Check
     your Groq plan's rpm/tpm for whichever model you pick and adjust
     TIER5_RATE_BUDGET below accordingly, or you WILL hit 429s.
  3. JSON compliance: smaller/faster Groq models are less reliable at
     strict JSON-only output than large ones. _parse_final has the same
     <think> / markdown-fence stripping as your Tier 4 overseer, but if
     you see frequent parse failures in the logs, tighten the prompt or
     add a retry-with-correction turn.
  4. reconstruct_doc_text is duplicated from agent/run_agent.py here to
     keep this file standalone - keep them in sync if you change one.
"""

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import csv

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from config import Configs
from biored import load_biored_documents, gold_entities_by_doc, write_canonical, strict_prf
from utils.pattern_matching import match_sequence_variant
from utils.rare_agent_utils import resolve_rare_entity
from utils.common_agent_utils import predict_span_type
from utils.bert_span_extractor import extract_spans_with_bert
from utils.find_occurences import find_occurrences
from utils.rate_budget import RateBudget

from dotenv import load_dotenv
load_dotenv()

configs = Configs()

VALID_TYPES = [
    "GeneOrGeneProduct",
    "DiseaseOrPhenotypicFeature",
    "ChemicalEntity",
    "SequenceVariant",
    "CellLine",
    "OrganismTaxon",
]

MAX_OCCURRENCES_PER_SPAN = 5
MAX_AGENT_TURNS = 6

# adjust to your actual Groq plan limits for whichever model you pick
TIER5_RATE_BUDGET = {"tpm": 6000, "rpm": 60}

SYSTEM_PROMPT_TOOL_PHASE = """You are a biomedical named-entity typing agent. You will be given a \
passage and a list of candidate spans already extracted from it. Your job right now is ONLY to \
gather evidence using tools - do NOT attempt to give a final answer yet, and do NOT output any \
JSON in this phase.

You have three tools. Use whichever is appropriate for each span's likely type - you do not need \
to call every tool for every span, only the one(s) relevant to what the span looks like. You may \
call more than one tool for a span if you are unsure or a result seems wrong given context.

- common_classifier(span_text): a fine-tuned biomedical NER model. Best for \
GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity.
- pattern_matcher(span_text): regex-based detector for SequenceVariant \
(HGVS-style mutation notation, e.g. "V1763M", "G-->A substitution at codon 1763").
- rare_lookup(span_text): external knowledge-base lookup (Cellosaurus for CellLine, \
NCBI Taxonomy for OrganismTaxon).

Call tools for every candidate span before you are done. Once you have called tools for all \
spans and have no more evidence to gather, respond with the single word: DONE
Do not produce any JSON or final entity list yet - that will be requested separately.
"""

FINAL_ANSWER_PROMPT = """You have now gathered tool evidence for the candidate spans above. \
Using that evidence and the passage context, decide the final entity type for each candidate \
span.

Respond with ONLY a JSON object, no markdown fences, no explanation, no extra text, in exactly \
this format:
{"entities": {"<span_text>": "<entity_type_or_null>", ...}}

Use exactly one of these type strings, or null if the span is not a real entity of any listed \
type: GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity, SequenceVariant, CellLine, \
OrganismTaxon
"""

def build_tier5_prompt(doc_text: str, candidate_spans: list[str]) -> str:
    spans_block = "\n".join(f'{i}. "{s}"' for i, s in enumerate(candidate_spans))
    types_block = ", ".join(VALID_TYPES)
    return f"""Passage:
{doc_text}

Candidate spans to classify:
{spans_block}

Valid entity types: {types_block}
"""


def reconstruct_doc_text(doc) -> str:
    """Duplicated from agent/run_agent.py - keep in sync if that changes."""
    end = max(p.offset + len(p.text) for p in doc.passages) if doc.passages else 0
    buf = [" "] * end
    for p in doc.passages:
        for i, ch in enumerate(p.text):
            buf[p.offset + i] = ch
    return "".join(buf)


class Tier5Orchestrator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "")

        agent_model_name = getattr(configs, "tier5_agent_model", None)
        if agent_model_name is None:
            raise AttributeError(
                "configs.tier5_agent_model is not set. Add a tool-calling-capable "
                "Groq model name to config.py, e.g. self.tier5_agent_model = "
                "'llama-3.3-70b-versatile'"
            )

        self.agent_model = ChatGroq(model=agent_model_name, api_key=api_key, max_tokens=4096)
        self.rate_budget = RateBudget(**TIER5_RATE_BUDGET)

        model_dir = configs.pubmed_model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.ner_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ner_model = self.ner_model.to(self.device)

        # stats logging - same shape as Tier 4's, so the same
        # analyze_branch_stats.py / calibration scripts work unchanged.
        # "branch" here means which tool the agent invoked, analogous to
        # AgentSource in Tier 4.
        self.branch_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.entity_log: list[dict] = []
        self.doc_count = 0

    def _make_tools(self, doc_text: str, tool_call_log: dict):
        """tool_call_log accumulates {span_text: [tool_names_called]} for this doc,
        used afterward to tag each final entity with which tool(s) resolved it."""

        @tool
        def common_classifier(span_text: str) -> str:
            """Classify a span using the fine-tuned PubMedBERT common-entity
            classifier. Best for GeneOrGeneProduct, DiseaseOrPhenotypicFeature,
            ChemicalEntity. Returns JSON with entity_type and confidence."""
            entity_type, conf = predict_span_type(
                span_text=span_text,
                passage_text=doc_text,
                passage_offset=0,
                doc_text=doc_text,
                tokenizer=self.tokenizer,
                model=self.ner_model,
                device=self.device,
            )
            tool_call_log.setdefault(span_text, []).append("common")
            return json.dumps({"entity_type": entity_type, "confidence": conf})

        @tool
        def pattern_matcher(span_text: str) -> str:
            """Check if a span matches a SequenceVariant regex pattern (HGVS-style
            mutation nomenclature). Returns JSON with matched (bool) and reasoning."""
            result = match_sequence_variant(span_text)
            tool_call_log.setdefault(span_text, []).append("pattern")
            return json.dumps({"matched": bool(result), "reasoning": result or "no_match"})

        @tool
        def rare_lookup(span_text: str) -> str:
            """Look up a span against external knowledge bases (Cellosaurus for
            CellLine, NCBI Taxonomy for OrganismTaxon). Returns JSON with
            entity_type, confidence, reasoning."""
            entity_type, conf, reasoning = resolve_rare_entity(span_text)
            tool_call_log.setdefault(span_text, []).append("rare")
            return json.dumps({"entity_type": entity_type, "confidence": conf, "reasoning": reasoning})

        return [common_classifier, pattern_matcher, rare_lookup]

    def _parse_final(self, text: str) -> dict:
        txt = text.strip()
        if "<think>" in txt:
            end_think = txt.rfind("</think>")
            txt = txt[end_think + len("</think>"):].strip() if end_think != -1 else txt[txt.find("{"):]
        if txt.startswith("```"):
            txt = txt[txt.find("{"):txt.rfind("}") + 1]
        try:
            data = json.loads(txt)
        except json.JSONDecodeError as e:
            print(f"  [tier5] JSON parse failed: {e}. raw length={len(txt)} chars. "
                  f"Attempting truncation repair...")
            repaired = self._repair_truncated_json(txt)
            if repaired is not None:
                print(f"  [tier5] repair succeeded, recovered {len(repaired.get('entities', repaired))} entities")
                return repaired.get("entities", repaired if isinstance(repaired, dict) else {})
            print(f"  [tier5] repair failed. raw (first 200 chars): {txt[:200]}")
            return {}
        return data.get("entities", data if isinstance(data, dict) else {})

    @staticmethod
    def _repair_truncated_json(txt: str) -> dict | None:
        """Best-effort recovery when the response got cut off mid-string (token
        limit hit). Finds the last fully-closed "key": "value" pair and
        truncates there, discarding only the incomplete tail entry rather than
        losing the whole document's predictions."""
        # find the last comma that occurs after a complete "..."  pair
        last_good_comma = -1
        i = 0
        depth_ok = txt.find('{"entities":')
        if depth_ok == -1:
            return None
        cursor = txt.find("{", depth_ok + len('"entities":'))
        if cursor == -1:
            return None

        # scan for complete "key": "value" pairs separated by commas
        pos = cursor + 1
        pairs_end = pos
        while True:
            # match "key"
            m_key_start = txt.find('"', pos)
            if m_key_start == -1:
                break
            m_key_end = txt.find('"', m_key_start + 1)
            if m_key_end == -1:
                break
            # expect : "value"
            colon = txt.find(":", m_key_end)
            if colon == -1:
                break
            val_start = txt.find('"', colon)
            null_start = txt.find("null", colon)
            if null_start != -1 and (val_start == -1 or null_start < val_start):
                val_end = null_start + len("null")
                pairs_end = val_end
                pos = val_end
            else:
                if val_start == -1:
                    break
                val_end = txt.find('"', val_start + 1)
                if val_end == -1:
                    break  # this value is truncated - stop here, don't include it
                pairs_end = val_end + 1
                pos = val_end + 1

            comma = txt.find(",", pos)
            next_quote = txt.find('"', pos)
            if comma == -1 or (next_quote != -1 and comma > next_quote and txt[pos:next_quote].strip() not in (",", "")):
                # no more complete pairs follow cleanly
                nxt = txt[pos:pos+3]
                if nxt.strip().startswith(","):
                    pos = txt.find(",", pos) + 1
                    continue
                break
            pos = comma + 1

        salvaged = txt[cursor:pairs_end] + "}"
        wrapped = '{"entities": ' + salvaged + "}"
        try:
            return json.loads(wrapped)
        except json.JSONDecodeError:
            return None

    def process_document(self, doc_id: str, doc_text: str, candidate_spans: list[str]):
        if not candidate_spans:
            return {}, {}

        tool_call_log: dict[str, list[str]] = {}
        tools = self._make_tools(doc_text, tool_call_log)
        tool_by_name = {t.name: t for t in tools}

        # force at least one tool call on the first turn - without this, models
        # frequently skip tools entirely and answer directly from parametric
        # knowledge, which silently degrades this into Tier 3's zero-shot
        # classification. The tool-gathering phase is kept SEPARATE from the
        # final-answer phase (below) because forcing tool_choice while also
        # allowing a final-JSON response in the same turn causes some models
        # to try to stuff their final answer into a malformed tool call
        # (Groq: "tool_use_failed" - confirmed empirically on this project).
        model_forced = self.agent_model.bind_tools(tools, tool_choice="required")
        model_auto = self.agent_model.bind_tools(tools, tool_choice="auto")

        user_prompt = build_tier5_prompt(doc_text, candidate_spans)
        messages = [SystemMessage(content=SYSTEM_PROMPT_TOOL_PHASE), HumanMessage(content=user_prompt)]

        # --- phase 1: tool-gathering only, no final answer allowed here ---
        for turn in range(MAX_AGENT_TURNS):
            est_tokens = len(user_prompt) // 4 + 500
            self.rate_budget.wait(est_tokens)

            model_with_tools = model_forced if turn == 0 else model_auto
            response = model_with_tools.invoke(messages)
            self.rate_budget.record(est_tokens)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None)
            print(f"    [tier5 debug] gather-phase turn {turn}: "
                  f"{len(tool_calls) if tool_calls else 0} tool_calls "
                  f"| content preview: {str(response.content)[:60]!r}")

            if not tool_calls:
                # model signaled DONE (or gave up) - stop gathering evidence
                break

            for tc in tool_calls:
                tool_fn = tool_by_name.get(tc["name"])
                if tool_fn is None:
                    result = json.dumps({"error": f"unknown tool {tc['name']}"})
                else:
                    try:
                        result = tool_fn.invoke(tc["args"])
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # --- phase 2: final answer, NO tools bound at all - guarantees the
        # model can only respond with plain text, never attempt a tool call ---
        messages.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
        est_tokens = len(user_prompt) // 4 + 500
        self.rate_budget.wait(est_tokens)
        final_response = self.agent_model.invoke(messages)  # unbound - no tools param
        self.rate_budget.record(est_tokens)
        final_json = self._parse_final(final_response.content)

        return final_json, tool_call_log

    def _record_stats(self, doc_id, items: list[dict], tool_used: dict):
        self.doc_count += 1
        for item in items:
            entity_type = item["type"]
            source = tool_used.get(item["text"], "unknown")
            self.branch_stats[entity_type][source] += 1
            self.entity_log.append({
                "doc_id": doc_id,
                "span_text": item["text"],
                "entity_type": entity_type,
                "source": source,
            })

    def save_stats(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        overall_totals: dict[str, int] = defaultdict(int)
        for entity_type, branch_counts in self.branch_stats.items():
            for branch, count in branch_counts.items():
                overall_totals[branch] += count
        grand_total = sum(overall_totals.values())

        summary = {
            "doc_count": self.doc_count,
            "grand_total_entities": grand_total,
            "overall_branch_totals": dict(overall_totals),
            "overall_branch_percentages": {
                b: round(100 * c / grand_total, 2) if grand_total else 0.0
                for b, c in overall_totals.items()
            },
            "by_entity_type": {
                et: {
                    "branch_counts": dict(bc),
                    "total": sum(bc.values()),
                    "branch_percentages": {
                        b: round(100 * c / sum(bc.values()), 2) for b, c in bc.items()
                    } if sum(bc.values()) else {}
                }
                for et, bc in self.branch_stats.items()
            }
        }

        with open(out / f"branch_distribution_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2)

        with open(out / f"branch_distribution_{timestamp}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["entity_type", "branch", "count", "pct_within_entity_type"])
            for et, bc in self.branch_stats.items():
                et_total = sum(bc.values())
                for b, c in bc.items():
                    writer.writerow([et, b, c, round(100 * c / et_total, 2) if et_total else 0.0])
            for b, c in overall_totals.items():
                writer.writerow(["ALL", b, c, round(100 * c / grand_total, 2) if grand_total else 0.0])

        with open(out / f"entity_log_{timestamp}.jsonl", "w") as f:
            for row in self.entity_log:
                f.write(json.dumps(row) + "\n")

        print(f"[stats] written to {out} (timestamp={timestamp})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", default="./dataset/test/Test.BioC.JSON")
    ap.add_argument("--output_dir", default="outputs/tier5_agent")
    ap.add_argument("--limit", type=int, default=0, help="DEBUG: limit to N docs")
    args = ap.parse_args()

    docs, _, _ = load_biored_documents(args.test_json)
    os.makedirs(args.output_dir, exist_ok=True)
    orchestrator = Tier5Orchestrator()

    if args.limit > 0:
        docs = docs[:args.limit]
        print(f"[limit] DEBUG: running {len(docs)} docs only")

    preds_by_doc = {}
    gold_all = gold_entities_by_doc(docs)

    for idx, doc in enumerate(docs, 1):
        print("##############################################")
        doc_text = reconstruct_doc_text(doc)

        try:
            candidate_spans = extract_spans_with_bert(
                doc=doc, doc_text=doc_text,
                tokenizer=orchestrator.tokenizer, model=orchestrator.ner_model,
                device=orchestrator.device,
            )
        except Exception as e:
            print(f"[{doc.doc_id}] span extraction failed: {e}")
            preds_by_doc[doc.doc_id] = []
            continue

        try:
            final_json, tool_call_log = orchestrator.process_document(doc.doc_id, doc_text, candidate_spans)
        except Exception as e:
            print(f"[{doc.doc_id}] agent failed: {e}")
            preds_by_doc[doc.doc_id] = []
            continue

        # take the LAST tool called for a span as its "resolving" tool
        # (mirrors Tier 4 branch_sources semantics: which branch's opinion won)
        tool_used = {span: calls[-1] for span, calls in tool_call_log.items()}

        items = []
        seen = set()
        for span_text, etype in final_json.items():
            if not etype or str(etype).lower() == "null":
                continue
            if etype not in VALID_TYPES:
                print(f"    [warn] invalid type '{etype}' for span '{span_text}' - dropping")
                continue
            occurrences = find_occurrences(doc_text, span_text)
            if not occurrences:
                continue
            for start, end in occurrences[:MAX_OCCURRENCES_PER_SPAN]:
                key = (start, end, etype)
                if key in seen:
                    continue
                seen.add(key)
                items.append({"start": start, "end": end, "type": etype, "text": span_text})

        preds_by_doc[doc.doc_id] = items
        orchestrator._record_stats(doc.doc_id, items, tool_used)

        print(f"  [{idx}/{len(docs)}] {doc.doc_id}: {len(items)} entities "
              f"from {len(candidate_spans)} candidate spans")

    write_canonical(os.path.join(args.output_dir, "test_predictions.json"), preds_by_doc)
    write_canonical(os.path.join(args.output_dir, "gold_test.json"), gold_all)
    orchestrator.save_stats(output_dir=os.path.join(args.output_dir, "branch_analysis"))

    score = strict_prf(preds_by_doc, gold_all)
    with open(os.path.join(args.output_dir, "test_strict_metrics.json"), "w") as f:
        json.dump(score, f, indent=2)
    print("\n[strict entity-level sanity]", json.dumps(score, indent=2))
    print(f"\nNow run scorer.py for the full strict/relaxed/per-type table:")
    print(f"PYTHONPATH=. python scorer.py "
          f"--pred {args.output_dir}/test_predictions.json "
          f"--gold {args.output_dir}/gold_test.json "
          f"--name tier5_agent "
          f"--out {args.output_dir}/full_metrics.json")


if __name__ == "__main__":
    main()