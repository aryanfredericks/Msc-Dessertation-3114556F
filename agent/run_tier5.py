# """
# agent/run_tier5.py

# Tier 5: replaces Tier 4's deterministic confidence-gated combiner with a
# single Groq LLM agent per document that decides, per candidate span, which
# of the three existing branches (common / pattern / rare) to invoke as
# tools, then emits a final type assignment. Span extraction (PubMedBERT)
# is UNCHANGED from Tier 4 - only the arbitration/routing step is replaced,
# so any F1 delta vs Tier 4 is attributable to routing strategy alone.

# Safety features for running the full 100-doc BioRED test set against a
# rate-limited free-tier model (llama-3.3-70b-versatile @ 30 rpm / 1K rpd /
# 12K tpm / 100K tpd on Groq):

#   - Checkpointing: predictions are written to
#     <output_dir>/checkpoint_predictions.jsonl incrementally, one line per
#     document, immediately after each doc finishes. If the run dies
#     partway through (e.g. the daily token cap is a HARD stop, unlike the
#     per-minute caps which just throttle), simply rerun the exact same
#     command - already-completed docs are skipped automatically.
#   - Accurate token estimation: RateBudget.wait()/record() now measure the
#     actual accumulated conversation (system prompt + passage + every
#     prior tool call/result so far), not just the original static prompt.
#     Token usage grows every turn within a document as history
#     accumulates: the old fixed-size estimate silently under-charged
#     later turns in long documents, risking a real 429 that the local
#     throttle didn't see coming.
#   - MAX_AGENT_TURNS lowered from 6 to 4: bounds worst-case per-document
#     token spend on dense documents (many candidate spans) that don't
#     reach a clean "DONE" quickly.

# Usage:
#     PYTHONPATH=. uv run agent/run_tier5.py \
#         --test_json ./dataset/test/Test.BioC.JSON \
#         --output_dir outputs/tier5_agent \
#         --limit 5          # ALWAYS smoke-test with --limit first

#     # if a full run gets interrupted (rate limit, crash, etc.) just rerun
#     # the identical command - it resumes from checkpoint_predictions.jsonl
#     PYTHONPATH=. uv run agent/run_tier5.py \
#         --test_json ./dataset/test/Test.BioC.JSON \
#         --output_dir outputs/tier5_agent
# """

# import json
# import os
# import argparse
# from collections import defaultdict
# from pathlib import Path
# from datetime import datetime
# import csv

# import torch
# from transformers import AutoModelForTokenClassification, AutoTokenizer
# from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# from langchain_core.tools import tool
# from langchain_groq import ChatGroq

# from config import Configs
# from biored import load_biored_documents, gold_entities_by_doc, write_canonical, strict_prf
# from utils.pattern_matching import match_sequence_variant
# from utils.rare_agent_utils import resolve_rare_entity
# from utils.common_agent_utils import predict_span_type
# from utils.bert_span_extractor import extract_spans_with_bert
# from utils.find_occurences import find_occurrences
# from utils.rate_budget import RateBudget

# from dotenv import load_dotenv
# load_dotenv()

# configs = Configs()

# VALID_TYPES = [
#     "GeneOrGeneProduct",
#     "DiseaseOrPhenotypicFeature",
#     "ChemicalEntity",
#     "SequenceVariant",
#     "CellLine",
#     "OrganismTaxon",
# ]

# MAX_OCCURRENCES_PER_SPAN = 5
# MAX_AGENT_TURNS = 4  # lowered from 6 - bounds worst-case token growth per doc

# # adjust to your actual Groq plan limits for whichever model you pick
# TIER5_RATE_BUDGET = {"tpm": 8000, "rpm": 30}

# SYSTEM_PROMPT_TOOL_PHASE = """You are a biomedical named-entity typing agent. You will be given a \
# passage and a list of candidate spans already extracted from it. Your job right now is ONLY to \
# gather evidence using tools - do NOT attempt to give a final answer yet, and do NOT output any \
# JSON in this phase.

# You have three tools. Each tool accepts a LIST of spans, not a single span - group spans by \
# which tool is relevant and call each tool ONCE with all the relevant spans, rather than calling \
# a tool separately for each individual span. For example, if 10 spans look like genes/diseases/ \
# chemicals, call common_classifier ONCE with all 10 span texts in the list, not 10 separate calls.

# - common_classifier(span_texts): a fine-tuned biomedical NER model. Best for \
# GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity.
# - pattern_matcher(span_texts): regex-based detector for SequenceVariant \
# (HGVS-style mutation notation, e.g. "V1763M", "G-->A substitution at codon 1763").
# - rare_lookup(span_texts): external knowledge-base lookup (Cellosaurus for CellLine, \
# NCBI Taxonomy for OrganismTaxon).

# Every candidate span must be covered by at least one tool call before you are done - but batch \
# spans into as few tool calls as possible (ideally one call per tool, covering all relevant spans \
# in that call). Once every span has been covered, respond with the single word: DONE
# Do not produce any JSON or final entity list yet - that will be requested separately.
# """

# FINAL_ANSWER_PROMPT = """You have now gathered tool evidence for the candidate spans above. \
# Using that evidence and the passage context, decide the final entity type for each candidate \
# span.

# Respond with ONLY a JSON object, no markdown fences, no explanation, no extra text, in exactly \
# this format:
# {"entities": {"<span_text>": "<entity_type_or_null>", ...}}

# Use exactly one of these type strings, or null if the span is not a real entity of any listed \
# type: GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity, SequenceVariant, CellLine, \
# OrganismTaxon
# """

# def build_tier5_prompt(doc_text: str, candidate_spans: list[str]) -> str:
#     spans_block = "\n".join(f'{i}. "{s}"' for i, s in enumerate(candidate_spans))
#     types_block = ", ".join(VALID_TYPES)
#     return f"""Passage:
# {doc_text}

# Candidate spans to classify:
# {spans_block}

# Valid entity types: {types_block}
# """


# def reconstruct_doc_text(doc) -> str:
#     """Duplicated from agent/run_agent.py - keep in sync if that changes."""
#     end = max(p.offset + len(p.text) for p in doc.passages) if doc.passages else 0
#     buf = [" "] * end
#     for p in doc.passages:
#         for i, ch in enumerate(p.text):
#             buf[p.offset + i] = ch
#     return "".join(buf)


# def _estimate_tokens(messages) -> int:
#     """Rough token estimate over the ACTUAL accumulated conversation so far,
#     not just the static initial prompt. This grows every turn within a
#     document as tool calls/results accumulate - using a fixed estimate here
#     silently under-charges later turns in long documents and risks a real
#     429 that the local rate throttle never saw coming."""
#     total_chars = sum(len(str(getattr(m, "content", m))) for m in messages)
#     return total_chars // 4 + 500


# class Tier5Orchestrator:
#     def __init__(self):
#         api_key = os.getenv("GROQ_API_KEY", "")

#         agent_model_name = getattr(configs, "gpt_120b_oss_model", None)
#         if agent_model_name is None:
#             raise AttributeError(
#                 "configs.span_extraction_model is not set. Add a tool-calling-capable "
#                 "Groq model name to config.py, e.g. self.span_extraction_model = "
#                 "'llama-3.3-70b-versatile'"
#             )

#         self.agent_model = ChatGroq(model=agent_model_name, api_key=api_key, max_tokens=4096)
#         self.rate_budget = RateBudget(**TIER5_RATE_BUDGET)

#         model_dir = configs.pubmed_model_dir
#         self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
#         self.ner_model = AutoModelForTokenClassification.from_pretrained(model_dir)
#         self.ner_model.eval()
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.ner_model = self.ner_model.to(self.device)

#         # stats logging - same shape as Tier 4's, so the same
#         # analyze_branch_stats.py / calibration scripts work unchanged.
#         # "branch" here means which tool the agent invoked, analogous to
#         # AgentSource in Tier 4.
#         self.branch_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
#         self.entity_log: list[dict] = []
#         self.doc_count = 0

#     def _make_tools(self, doc_text: str, tool_call_log: dict):
#         """tool_call_log accumulates {span_text: [tool_names_called]} for this doc,
#         used afterward to tag each final entity with which tool(s) resolved it.

#         Tools accept a LIST of spans per call, not a single span. This matters
#         because some models (observed on gpt-oss-120b) call exactly one tool
#         per turn rather than batching several tool calls in one turn - with a
#         single-span signature that means one turn covers one span, requiring
#         as many turns as there are candidate spans (30-70+ on dense BioRED
#         docs), which blows through MAX_AGENT_TURNS and the per-minute rate
#         budget alike. Batched spans-per-call lets a model that prefers one
#         tool call per turn still cover many spans per turn."""

#         @tool
#         def common_classifier(span_texts: list[str]) -> str:
#             """Classify one or more spans using the fine-tuned PubMedBERT
#             common-entity classifier. Best for GeneOrGeneProduct,
#             DiseaseOrPhenotypicFeature, ChemicalEntity. Pass ALL spans you
#             want checked with this tool in a single call as a list, not one
#             call per span. Returns a JSON list of {span_text, entity_type,
#             confidence} objects, one per input span."""
#             results = []
#             for span_text in span_texts:
#                 entity_type, conf = predict_span_type(
#                     span_text=span_text,
#                     passage_text=doc_text,
#                     passage_offset=0,
#                     doc_text=doc_text,
#                     tokenizer=self.tokenizer,
#                     model=self.ner_model,
#                     device=self.device,
#                 )
#                 tool_call_log.setdefault(span_text, []).append("common")
#                 results.append({"span_text": span_text, "entity_type": entity_type, "confidence": conf})
#             return json.dumps(results)

#         @tool
#         def pattern_matcher(span_texts: list[str]) -> str:
#             """Check one or more spans against a SequenceVariant regex pattern
#             (HGVS-style mutation notation, e.g. "V1763M", "G-->A substitution
#             at codon 1763"). Pass ALL spans you want checked with this tool in
#             a single call as a list, not one call per span. Returns a JSON
#             list of {span_text, matched, reasoning} objects, one per input
#             span."""
#             results = []
#             for span_text in span_texts:
#                 result = match_sequence_variant(span_text)
#                 tool_call_log.setdefault(span_text, []).append("pattern")
#                 results.append({"span_text": span_text, "matched": bool(result), "reasoning": result or "no_match"})
#             return json.dumps(results)

#         @tool
#         def rare_lookup(span_texts: list[str]) -> str:
#             """Look up one or more spans against external knowledge bases
#             (Cellosaurus for CellLine, NCBI Taxonomy for OrganismTaxon). Pass
#             ALL spans you want checked with this tool in a single call as a
#             list, not one call per span. Returns a JSON list of {span_text,
#             entity_type, confidence, reasoning} objects, one per input span."""
#             results = []
#             for span_text in span_texts:
#                 entity_type, conf, reasoning = resolve_rare_entity(span_text)
#                 tool_call_log.setdefault(span_text, []).append("rare")
#                 results.append({"span_text": span_text, "entity_type": entity_type, "confidence": conf, "reasoning": reasoning})
#             return json.dumps(results)

#         return [common_classifier, pattern_matcher, rare_lookup]

#     def _parse_final(self, text: str) -> dict:
#         txt = text.strip()
#         if "<think>" in txt:
#             end_think = txt.rfind("</think>")
#             txt = txt[end_think + len("</think>"):].strip() if end_think != -1 else txt[txt.find("{"):]
#         if txt.startswith("```"):
#             txt = txt[txt.find("{"):txt.rfind("}") + 1]
#         try:
#             data = json.loads(txt)
#         except json.JSONDecodeError as e:
#             print(f"  [tier5] JSON parse failed: {e}. raw length={len(txt)} chars. "
#                   f"Attempting truncation repair...")
#             repaired = self._repair_truncated_json(txt)
#             if repaired is not None:
#                 print(f"  [tier5] repair succeeded, recovered {len(repaired.get('entities', repaired))} entities")
#                 return repaired.get("entities", repaired if isinstance(repaired, dict) else {})
#             print(f"  [tier5] repair failed. raw (first 200 chars): {txt[:200]}")
#             return {}
#         return data.get("entities", data if isinstance(data, dict) else {})

#     @staticmethod
#     def _repair_truncated_json(txt: str) -> dict | None:
#         """Best-effort recovery when the response got cut off mid-string (token
#         limit hit). Finds the last fully-closed "key": "value" pair and
#         truncates there, discarding only the incomplete tail entry rather than
#         losing the whole document's predictions."""
#         depth_ok = txt.find('{"entities":')
#         if depth_ok == -1:
#             return None
#         cursor = txt.find("{", depth_ok + len('"entities":'))
#         if cursor == -1:
#             return None

#         pos = cursor + 1
#         pairs_end = pos
#         while True:
#             m_key_start = txt.find('"', pos)
#             if m_key_start == -1:
#                 break
#             m_key_end = txt.find('"', m_key_start + 1)
#             if m_key_end == -1:
#                 break
#             colon = txt.find(":", m_key_end)
#             if colon == -1:
#                 break
#             val_start = txt.find('"', colon)
#             null_start = txt.find("null", colon)
#             if null_start != -1 and (val_start == -1 or null_start < val_start):
#                 val_end = null_start + len("null")
#                 pairs_end = val_end
#                 pos = val_end
#             else:
#                 if val_start == -1:
#                     break
#                 val_end = txt.find('"', val_start + 1)
#                 if val_end == -1:
#                     break
#                 pairs_end = val_end + 1
#                 pos = val_end + 1

#             comma = txt.find(",", pos)
#             next_quote = txt.find('"', pos)
#             if comma == -1 or (next_quote != -1 and comma > next_quote and txt[pos:next_quote].strip() not in (",", "")):
#                 nxt = txt[pos:pos+3]
#                 if nxt.strip().startswith(","):
#                     pos = txt.find(",", pos) + 1
#                     continue
#                 break
#             pos = comma + 1

#         salvaged = txt[cursor:pairs_end] + "}"
#         wrapped = '{"entities": ' + salvaged + "}"
#         try:
#             return json.loads(wrapped)
#         except json.JSONDecodeError:
#             return None

#     def process_document(self, doc_id: str, doc_text: str, candidate_spans: list[str]):
#         if not candidate_spans:
#             return {}, {}

#         tool_call_log: dict[str, list[str]] = {}
#         tools = self._make_tools(doc_text, tool_call_log)
#         tool_by_name = {t.name: t for t in tools}

#         # force at least one tool call on the first turn - without this, models
#         # frequently skip tools entirely and answer directly from parametric
#         # knowledge, which silently degrades this into Tier 3's zero-shot
#         # classification. The tool-gathering phase is kept SEPARATE from the
#         # final-answer phase (below) because forcing tool_choice while also
#         # allowing a final-JSON response in the same turn causes some models
#         # to try to stuff their final answer into a malformed tool call
#         # (confirmed empirically against Groq's llama models during development).
#         model_forced = self.agent_model.bind_tools(tools, tool_choice="required")
#         model_auto = self.agent_model.bind_tools(tools, tool_choice="auto")

#         user_prompt = build_tier5_prompt(doc_text, candidate_spans)
#         messages = [SystemMessage(content=SYSTEM_PROMPT_TOOL_PHASE), HumanMessage(content=user_prompt)]

#         # --- phase 1: tool-gathering only, no final answer allowed here ---
#         for turn in range(MAX_AGENT_TURNS):
#             est_tokens = _estimate_tokens(messages)
#             self.rate_budget.wait(est_tokens)

#             model_with_tools = model_forced if turn == 0 else model_auto
#             response = model_with_tools.invoke(messages)
#             self.rate_budget.record(est_tokens)
#             messages.append(response)

#             tool_calls = getattr(response, "tool_calls", None)
#             print(f"    [tier5 debug] gather-phase turn {turn}: "
#                   f"{len(tool_calls) if tool_calls else 0} tool_calls "
#                   f"| content preview: {str(response.content)[:60]!r}")

#             if not tool_calls:
#                 # model signaled DONE (or gave up) - stop gathering evidence
#                 break

#             for tc in tool_calls:
#                 tool_fn = tool_by_name.get(tc["name"])
#                 if tool_fn is None:
#                     result = json.dumps({"error": f"unknown tool {tc['name']}"})
#                 else:
#                     try:
#                         result = tool_fn.invoke(tc["args"])
#                     except Exception as e:
#                         result = json.dumps({"error": str(e)})
#                 messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

#         # --- phase 2: final answer. Bind the same tools but with
#         # tool_choice="none" - some models (observed on gpt-oss-120b) still
#         # attempt a tool call even when no tools are bound at all, because
#         # they're conditioned by the tool-calling turns earlier in the
#         # conversation. Explicitly forbidding tool use via tool_choice="none"
#         # is a real instruction the model must honor, unlike an implicit
#         # absence of tools which it can apparently still act around. ---
#         model_no_tools = self.agent_model.bind_tools(tools, tool_choice="none")
#         messages.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
#         est_tokens = _estimate_tokens(messages)
#         self.rate_budget.wait(est_tokens)
#         final_response = model_no_tools.invoke(messages)
#         self.rate_budget.record(est_tokens)
#         final_json = self._parse_final(final_response.content)

#         return final_json, tool_call_log

#     def _record_stats(self, doc_id, items: list[dict], tool_used: dict):
#         self.doc_count += 1
#         for item in items:
#             entity_type = item["type"]
#             source = tool_used.get(item["text"], "unknown")
#             self.branch_stats[entity_type][source] += 1
#             self.entity_log.append({
#                 "doc_id": doc_id,
#                 "span_text": item["text"],
#                 "entity_type": entity_type,
#                 "source": source,
#             })

#     def save_stats(self, output_dir: str) -> None:
#         out = Path(output_dir)
#         out.mkdir(parents=True, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#         overall_totals: dict[str, int] = defaultdict(int)
#         for entity_type, branch_counts in self.branch_stats.items():
#             for branch, count in branch_counts.items():
#                 overall_totals[branch] += count
#         grand_total = sum(overall_totals.values())

#         summary = {
#             "doc_count": self.doc_count,
#             "grand_total_entities": grand_total,
#             "overall_branch_totals": dict(overall_totals),
#             "overall_branch_percentages": {
#                 b: round(100 * c / grand_total, 2) if grand_total else 0.0
#                 for b, c in overall_totals.items()
#             },
#             "by_entity_type": {
#                 et: {
#                     "branch_counts": dict(bc),
#                     "total": sum(bc.values()),
#                     "branch_percentages": {
#                         b: round(100 * c / sum(bc.values()), 2) for b, c in bc.items()
#                     } if sum(bc.values()) else {}
#                 }
#                 for et, bc in self.branch_stats.items()
#             }
#         }

#         with open(out / f"branch_distribution_{timestamp}.json", "w") as f:
#             json.dump(summary, f, indent=2)

#         with open(out / f"branch_distribution_{timestamp}.csv", "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["entity_type", "branch", "count", "pct_within_entity_type"])
#             for et, bc in self.branch_stats.items():
#                 et_total = sum(bc.values())
#                 for b, c in bc.items():
#                     writer.writerow([et, b, c, round(100 * c / et_total, 2) if et_total else 0.0])
#             for b, c in overall_totals.items():
#                 writer.writerow(["ALL", b, c, round(100 * c / grand_total, 2) if grand_total else 0.0])

#         with open(out / f"entity_log_{timestamp}.jsonl", "w") as f:
#             for row in self.entity_log:
#                 f.write(json.dumps(row) + "\n")

#         print(f"[stats] written to {out} (timestamp={timestamp})")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--test_json", default="./dataset/test/Test.BioC.JSON")
#     ap.add_argument("--output_dir", default="outputs/tier5_agent")
#     ap.add_argument("--limit", type=int, default=0, help="DEBUG: limit to N docs")
#     args = ap.parse_args()

#     docs, _, _ = load_biored_documents(args.test_json)
#     os.makedirs(args.output_dir, exist_ok=True)
#     orchestrator = Tier5Orchestrator()

#     if args.limit > 0:
#         docs = docs[:args.limit]
#         print(f"[limit] DEBUG: running {len(docs)} docs only")

#     # --- checkpoint/resume: load any progress from a previous (possibly
#     # interrupted) run of this exact output_dir before starting ---
#     checkpoint_path = os.path.join(args.output_dir, "checkpoint_predictions.jsonl")
#     preds_by_doc: dict[str, list] = {}
#     processed_ids: set[str] = set()
#     if os.path.exists(checkpoint_path):
#         with open(checkpoint_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 row = json.loads(line)
#                 preds_by_doc[row["doc_id"]] = row["items"]
#                 processed_ids.add(row["doc_id"])
#         print(f"[checkpoint] resuming - {len(processed_ids)} docs already done, skipping them")

#     gold_all = gold_entities_by_doc(docs)

#     for idx, doc in enumerate(docs, 1):
#         if doc.doc_id in processed_ids:
#             continue

#         print("##############################################")
#         doc_text = reconstruct_doc_text(doc)

#         try:
#             candidate_spans = extract_spans_with_bert(
#                 doc=doc, doc_text=doc_text,
#                 tokenizer=orchestrator.tokenizer, model=orchestrator.ner_model,
#                 device=orchestrator.device,
#             )
#         except Exception as e:
#             print(f"[{doc.doc_id}] span extraction failed: {e}")
#             preds_by_doc[doc.doc_id] = []
#             with open(checkpoint_path, "a") as f:
#                 f.write(json.dumps({"doc_id": doc.doc_id, "items": []}) + "\n")
#             continue

#         try:
#             final_json, tool_call_log = orchestrator.process_document(doc.doc_id, doc_text, candidate_spans)
#         except Exception as e:
#             print(f"[{doc.doc_id}] agent failed: {e}")
#             preds_by_doc[doc.doc_id] = []
#             with open(checkpoint_path, "a") as f:
#                 f.write(json.dumps({"doc_id": doc.doc_id, "items": []}) + "\n")
#             continue

#         # take the LAST tool called for a span as its "resolving" tool
#         # (mirrors Tier 4 branch_sources semantics: which branch's opinion won)
#         tool_used = {span: calls[-1] for span, calls in tool_call_log.items()}

#         items = []
#         seen = set()
#         for span_text, etype in final_json.items():
#             if not etype or str(etype).lower() == "null":
#                 continue
#             if etype not in VALID_TYPES:
#                 print(f"    [warn] invalid type '{etype}' for span '{span_text}' - dropping")
#                 continue
#             occurrences = find_occurrences(doc_text, span_text)
#             if not occurrences:
#                 continue
#             for start, end in occurrences[:MAX_OCCURRENCES_PER_SPAN]:
#                 key = (start, end, etype)
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 items.append({"start": start, "end": end, "type": etype, "text": span_text})

#         preds_by_doc[doc.doc_id] = items
#         orchestrator._record_stats(doc.doc_id, items, tool_used)

#         # persist progress immediately - if the process dies right after this
#         # (rate limit, crash, etc.), this doc's work is not lost
#         with open(checkpoint_path, "a") as f:
#             f.write(json.dumps({"doc_id": doc.doc_id, "items": items}) + "\n")

#         print(f"  [{idx}/{len(docs)}] {doc.doc_id}: {len(items)} entities "
#               f"from {len(candidate_spans)} candidate spans")

#     write_canonical(os.path.join(args.output_dir, "test_predictions.json"), preds_by_doc)
#     write_canonical(os.path.join(args.output_dir, "gold_test.json"), gold_all)
#     orchestrator.save_stats(output_dir=os.path.join(args.output_dir, "branch_analysis"))

#     score = strict_prf(preds_by_doc, gold_all)
#     with open(os.path.join(args.output_dir, "test_strict_metrics.json"), "w") as f:
#         json.dump(score, f, indent=2)
#     print("\n[strict entity-level sanity]", json.dumps(score, indent=2))
#     print(f"\nNow run scorer.py for the full strict/relaxed/per-type table:")
#     print(f"PYTHONPATH=. python scorer.py "
#           f"--pred {args.output_dir}/test_predictions.json "
#           f"--gold {args.output_dir}/gold_test.json "
#           f"--name tier5_agent "
#           f"--out {args.output_dir}/full_metrics.json")


# if __name__ == "__main__":
#     main()



################################################ UNIVERSITY HPC CLUSTER CODE #########################################
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
from langchain_openai import ChatOpenAI

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
MAX_AGENT_TURNS = 6  # batched tool calls should cover all spans in ~3 turns (one per tool type); this is headroom

# HPC endpoint (University of Glasgow) - use the non-UofG-network URL if
# running outside the university network:
#   "http://api.terrier.org/v1"
IDA_LLM_BASE_URL = "http://api.llm.apps.os.dcs.gla.ac.uk/v1"
IDA_LLM_MODEL = "gpt-oss-120b"

# The whole reason to use the HPC endpoint instead of Groq for this model is
# to escape Groq's free-tier 200K-token/day cap (gpt-oss-120b's hidden
# reasoning tokens burn through that in ~1.5 documents - verified
# empirically). This budget is a much more generous placeholder since the
# HPC endpoint's actual limits are unknown - check with whoever administers
# the cluster if you hit real throttling, and tighten this accordingly.
TIER5_RATE_BUDGET = {"tpm": 60000, "rpm": 60}

SYSTEM_PROMPT_TOOL_PHASE = """You are a biomedical named-entity typing agent. You will be given a \
passage and a list of candidate spans already extracted from it. Your job right now is ONLY to \
gather evidence using tools - do NOT attempt to give a final answer yet, and do NOT output any \
JSON in this phase.

You have three tools. Each tool accepts a LIST of spans, not a single span - group spans by \
which tool is relevant and call each tool ONCE with all the relevant spans, rather than calling \
a tool separately for each individual span. For example, if 10 spans look like genes/diseases/ \
chemicals, call common_classifier ONCE with all 10 span texts in the list, not 10 separate calls.

- common_classifier(span_texts): a fine-tuned biomedical NER model. Best for \
GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity.
- pattern_matcher(span_texts): regex-based detector for SequenceVariant \
(HGVS-style mutation notation, e.g. "V1763M", "G-->A substitution at codon 1763").
- rare_lookup(span_texts): external knowledge-base lookup (Cellosaurus for CellLine, \
NCBI Taxonomy for OrganismTaxon).

Every candidate span must be covered by at least one tool call before you are done - but batch \
spans into as few tool calls as possible (ideally one call per tool, covering all relevant spans \
in that call). Once every span has been covered, respond with the single word: DONE
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


def _estimate_tokens(messages) -> int:
    """Rough token estimate over the ACTUAL accumulated conversation so far,
    not just the static initial prompt. This grows every turn within a
    document as tool calls/results accumulate - using a fixed estimate here
    silently under-charges later turns in long documents and risks a real
    429 that the local rate throttle never saw coming."""
    total_chars = sum(len(str(getattr(m, "content", m))) for m in messages)
    return total_chars // 4 + 500


class Tier5Orchestrator:
    def __init__(self):
        api_key = os.environ["IDA_LLM_API_KEY"]

        self.agent_model = ChatOpenAI(
            model=IDA_LLM_MODEL,
            base_url=IDA_LLM_BASE_URL,
            api_key=api_key,
            max_tokens=4096,
        )
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
        used afterward to tag each final entity with which tool(s) resolved it.

        Tools accept a LIST of spans per call, not a single span. This matters
        because some models (observed on gpt-oss-120b) call exactly one tool
        per turn rather than batching several tool calls in one turn - with a
        single-span signature that means one turn covers one span, requiring
        as many turns as there are candidate spans (30-70+ on dense BioRED
        docs), which blows through MAX_AGENT_TURNS and the per-minute rate
        budget alike. Batched spans-per-call lets a model that prefers one
        tool call per turn still cover many spans per turn."""

        @tool
        def common_classifier(span_texts: list[str]) -> str:
            """Classify one or more spans using the fine-tuned PubMedBERT
            common-entity classifier. Best for GeneOrGeneProduct,
            DiseaseOrPhenotypicFeature, ChemicalEntity. Pass ALL spans you
            want checked with this tool in a single call as a list, not one
            call per span. Returns a JSON list of {span_text, entity_type,
            confidence} objects, one per input span."""
            results = []
            for span_text in span_texts:
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
                results.append({"span_text": span_text, "entity_type": entity_type, "confidence": conf})
            return json.dumps(results)

        @tool
        def pattern_matcher(span_texts: list[str]) -> str:
            """Check one or more spans against a SequenceVariant regex pattern
            (HGVS-style mutation notation, e.g. "V1763M", "G-->A substitution
            at codon 1763"). Pass ALL spans you want checked with this tool in
            a single call as a list, not one call per span. Returns a JSON
            list of {span_text, matched, reasoning} objects, one per input
            span."""
            results = []
            for span_text in span_texts:
                result = match_sequence_variant(span_text)
                tool_call_log.setdefault(span_text, []).append("pattern")
                results.append({"span_text": span_text, "matched": bool(result), "reasoning": result or "no_match"})
            return json.dumps(results)

        @tool
        def rare_lookup(span_texts: list[str]) -> str:
            """Look up one or more spans against external knowledge bases
            (Cellosaurus for CellLine, NCBI Taxonomy for OrganismTaxon). Pass
            ALL spans you want checked with this tool in a single call as a
            list, not one call per span. Returns a JSON list of {span_text,
            entity_type, confidence, reasoning} objects, one per input span."""
            results = []
            for span_text in span_texts:
                entity_type, conf, reasoning = resolve_rare_entity(span_text)
                tool_call_log.setdefault(span_text, []).append("rare")
                results.append({"span_text": span_text, "entity_type": entity_type, "confidence": conf, "reasoning": reasoning})
            return json.dumps(results)

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
        depth_ok = txt.find('{"entities":')
        if depth_ok == -1:
            return None
        cursor = txt.find("{", depth_ok + len('"entities":'))
        if cursor == -1:
            return None

        pos = cursor + 1
        pairs_end = pos
        while True:
            m_key_start = txt.find('"', pos)
            if m_key_start == -1:
                break
            m_key_end = txt.find('"', m_key_start + 1)
            if m_key_end == -1:
                break
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
                    break
                pairs_end = val_end + 1
                pos = val_end + 1

            comma = txt.find(",", pos)
            next_quote = txt.find('"', pos)
            if comma == -1 or (next_quote != -1 and comma > next_quote and txt[pos:next_quote].strip() not in (",", "")):
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
        # (confirmed empirically against Groq's llama models during development).
        model_forced = self.agent_model.bind_tools(tools, tool_choice="required")
        model_auto = self.agent_model.bind_tools(tools, tool_choice="auto")

        user_prompt = build_tier5_prompt(doc_text, candidate_spans)
        messages = [SystemMessage(content=SYSTEM_PROMPT_TOOL_PHASE), HumanMessage(content=user_prompt)]

        # --- phase 1: tool-gathering only, no final answer allowed here ---
        for turn in range(MAX_AGENT_TURNS):
            est_tokens = _estimate_tokens(messages)
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

        # --- phase 2: final answer. Bind the same tools but with
        # tool_choice="none" - some models (observed on gpt-oss-120b) still
        # attempt a tool call even when no tools are bound at all, because
        # they're conditioned by the tool-calling turns earlier in the
        # conversation. Explicitly forbidding tool use via tool_choice="none"
        # is a real instruction the model must honor, unlike an implicit
        # absence of tools which it can apparently still act around. ---
        model_no_tools = self.agent_model.bind_tools(tools, tool_choice="none")
        messages.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
        est_tokens = _estimate_tokens(messages)
        self.rate_budget.wait(est_tokens)
        final_response = model_no_tools.invoke(messages)
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

    # --- checkpoint/resume: load any progress from a previous (possibly
    # interrupted) run of this exact output_dir before starting ---
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_predictions.jsonl")
    preds_by_doc: dict[str, list] = {}
    processed_ids: set[str] = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                preds_by_doc[row["doc_id"]] = row["items"]
                processed_ids.add(row["doc_id"])
        print(f"[checkpoint] resuming - {len(processed_ids)} docs already done, skipping them")

    gold_all = gold_entities_by_doc(docs)

    for idx, doc in enumerate(docs, 1):
        if doc.doc_id in processed_ids:
            continue

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
            with open(checkpoint_path, "a") as f:
                f.write(json.dumps({"doc_id": doc.doc_id, "items": []}) + "\n")
            continue

        try:
            final_json, tool_call_log = orchestrator.process_document(doc.doc_id, doc_text, candidate_spans)
        except Exception as e:
            print(f"[{doc.doc_id}] agent failed: {e}")
            preds_by_doc[doc.doc_id] = []
            with open(checkpoint_path, "a") as f:
                f.write(json.dumps({"doc_id": doc.doc_id, "items": []}) + "\n")
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

        # persist progress immediately - if the process dies right after this
        # (rate limit, crash, etc.), this doc's work is not lost
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps({"doc_id": doc.doc_id, "items": items}) + "\n")

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