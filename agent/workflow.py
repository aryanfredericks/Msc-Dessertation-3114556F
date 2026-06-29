import json
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import Configs
from models import (AgentSource, BranchVote, CandidateSpan, Decision, 
                    EntityTypes, NERAgentState, OverseerOutput, ResolvedSpan, SpanExtractionOutput)
from prompts import build_span_extr_prompt
from utils.pattern_matching import match_sequence_variant
from utils.rare_agent_utils import resolve_rare_entity
from utils.common_agent_utils import predict_span_type
from utils.find_occurences import find_occurrences
from utils.rate_budget import RateBudget
from utils.bert_span_extractor import extract_spans_with_bert

from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

configs = Configs()

MAX_OCCURRENCES_PER_SPAN = 5 # covers 92.5% of BioRED gold mentions (train set analysis)

class AgentWorkflow:
    def __init__(self):
        # qwen3-32b limits  
        self.qwen_budget = RateBudget(tpm=6000, rpm=60)
        
        api_key = os.getenv("GROQ_API_KEY", "")
        
        
        self.overseer_model = ChatGroq(
            model=configs.reasoning_model,
            api_key=api_key
        )
        
        model_dir = configs.pubmed_model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.ner_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ner_model = self.ner_model.to(self.device)
        
        self.graph = self._build_graph()


    def span_extraction_node(self, state: NERAgentState) -> dict:
        doc_id = state["doc_id"]
        doc_text = state["doc_text"]
        doc = state["doc"]
        try:
            surface_strings = extract_spans_with_bert(
                doc=doc,
                doc_text=doc_text,
                tokenizer=self.tokenizer,
                model=self.ner_model,
                device=self.device,
            )
            spans = [
                CandidateSpan(text=s, doc_id=doc_id, passage_offset=0)
                for s in surface_strings
            ]
            print(f"[span_extraction] {len(spans)} candidates from BERT")
            return {"candidate_spans": spans}
        except Exception as e:
            print(f"[span_extraction] doc {doc_id} failed: {e}")
            return {
                "candidate_spans": [],
                "errors": [f"span_extraction_node: {str(e)}"]
            }

    def pattern_matching_agent(self, state: NERAgentState) -> dict:
        spans = state["candidate_spans"]
        pattern_votes: list[BranchVote] = []
        try:
            for span in spans:
                result = match_sequence_variant(span.text)
                pattern_votes.append(BranchVote(
                    entity_type=EntityTypes.SEQUENCE if result else None,
                    source=AgentSource.PATTERN,
                    span=span,
                    confidence=1.0 if result else 0.0,
                    reasoning=result or "no_match"
                ))
            return {"pattern_votes": pattern_votes}
        except Exception as e:
            return {
                "pattern_votes": [],
                "errors": [f"pattern_matching_agent: {str(e)}"]
            }

    def rare_relation_agent(self, state: NERAgentState) -> dict:
        spans = state["candidate_spans"]
        rare_votes: list[BranchVote] = []
        try:
            for span in spans:
                entity_type, conf, reasoning = resolve_rare_entity(span.text)
                rare_votes.append(BranchVote(
                    entity_type=EntityTypes(entity_type) if entity_type else None,
                    source=AgentSource.RARE,
                    span=span,
                    confidence=conf,
                    reasoning=reasoning
                ))
            return {"rare_votes": rare_votes}
        except Exception as e:
            return {
                "rare_votes": [],
                "errors": [f"rare_relation_agent: {str(e)}"]
            }
            
    def common_relation_agent(self, state: NERAgentState) -> dict:
        spans = state["candidate_spans"]
        doc_text = state["doc_text"]
        common_votes: list[BranchVote] = []

        try:
            for span in spans:
                entity_type, conf = predict_span_type(
                    span_text=span.text,
                    passage_text=doc_text,
                    passage_offset=0,
                    doc_text=doc_text,
                    tokenizer=self.tokenizer,
                    model=self.ner_model,
                    device=self.device,
                )

                # only vote if it's a common type
                if entity_type in {"GeneOrGeneProduct", 
                                "DiseaseOrPhenotypicFeature", 
                                "ChemicalEntity"}:
                    common_votes.append(BranchVote(
                        entity_type=EntityTypes(entity_type),
                        source=AgentSource.COMMON,
                        span=span,
                        confidence=conf,
                    ))
                else:
                    common_votes.append(BranchVote(
                        entity_type=None,
                        source=AgentSource.COMMON,
                        span=span,
                        confidence=0.0,
                    ))

            return {"common_votes": common_votes}

        except Exception as e:
            print(f"common_relation_agent failed: {e}")
            return {
                "common_votes": [],
                "errors": [f"common_relation_agent: {str(e)}"]
            }

    def combiner_node(self, state: NERAgentState) -> dict:
        pattern_votes = state["pattern_votes"]
        rare_votes = state["rare_votes"]
        common_votes = state["common_votes"]
        candidate_spans = state["candidate_spans"]

        # index votes by span text for quick lookup
        pattern_index: dict[str, BranchVote] = {v.span.text: v for v in pattern_votes}
        rare_index: dict[str, BranchVote] = {v.span.text: v for v in rare_votes}
        common_index: dict[str, BranchVote] = {v.span.text: v for v in common_votes}

        decisions: list[Decision] = []
        requery_spans: list[Decision] = []
        branch_sources = {"pattern": 0, "rare": 0, "common": 0, "requery_needed": 0, "dropped": 0}

        for span in candidate_spans:
            text = span.text
            pattern_vote = pattern_index.get(text)
            rare_vote = rare_index.get(text)
            common_vote = common_index.get(text)

            decision = None

            # Case 1: pattern branch wins unconditionally
            if pattern_vote and pattern_vote.entity_type is not None:
                decision = Decision(
                    span=span,
                    entity_type=pattern_vote.entity_type.value,
                    source=AgentSource.PATTERN,
                    confidence=1.0,
                    requeried=False
                )
                branch_sources["pattern"] += 1

            # Case 2: rare branch wins if external api resolved AND common branch is not already confident
            elif rare_vote and rare_vote.entity_type is not None:
                common_branch_confidence = common_vote.confidence if common_vote and common_vote.entity_type else 0.0
                if common_branch_confidence < 0.7:
                    # common branch is confident hence we trust it over the external api
                    decision = Decision(
                        span=span,
                        entity_type=rare_vote.entity_type.value,
                        source=AgentSource.RARE,
                        confidence=rare_vote.confidence,
                        requeried=False
                    )
                    branch_sources["rare"] += 1
                else:
                    decision = Decision(
                        span=span,
                        entity_type=common_vote.entity_type.value,
                        source=AgentSource.COMMON,
                        confidence=common_vote.confidence,
                        requeried=False
                    )
                    branch_sources["common"] += 1

            # Case 3: common branch wins above confidence threshold
            elif common_vote and common_vote.entity_type is not None and common_vote.confidence >= 0.5:
                decision = Decision(
                    span=span,
                    entity_type=common_vote.entity_type.value,
                    source=AgentSource.COMMON,
                    confidence=common_vote.confidence,
                    requeried=False
                )
                branch_sources["common"] += 1

            # Case 4: common vote exists but low confidence -> flag for requery
            elif common_vote and common_vote.entity_type is not None and common_vote.confidence < 0.5:
                decision = Decision(
                    span=span,
                    entity_type=None,
                    source=AgentSource.REQUERY_NEEDED,
                    confidence=common_vote.confidence,
                    requeried=False
                )
                requery_spans.append(decision)
                branch_sources["requery_needed"] += 1

            # Case 5: nothing resolved -> drop
            else:
                decision = Decision(
                    span=span,
                    entity_type=None,
                    source=AgentSource.DROPPED,
                    confidence=0.0,
                    requeried=False
                )
                branch_sources["dropped"] += 1

            decisions.append(decision)

        print(f"[combiner] pattern={branch_sources['pattern']} "
            f"rare={branch_sources['rare']} "
            f"common={branch_sources['common']} "
            f"requery_needed={branch_sources['requery_needed']} "
            f"dropped={branch_sources['dropped']}")

        return {
            "decisions": decisions,
            "requery_spans": requery_spans,
            "branch_sources": branch_sources
        }

    def overseer_node(self, state : NERAgentState) -> dict:
        requery_spans = state["requery_spans"]
        decisions = state["decisions"]
        doc_text = state["doc_text"]
        doc_id = state["doc_id"]
        
        if not requery_spans:
            return {
                "final_entities": [d for d in decisions if d.entity_type is not None],
                "requery_count" : 0
            }
            
        try:
            from utils.overseer_utils import (REQUERY_SYSTEM_PROMPT, build_requery_prompt)
            from langchain_core.messages import HumanMessage, SystemMessage
            
            requery_cases = []
            for d in requery_spans:
                # find the surrounding context around the span
                idx = doc_text.find(d.span.text)
                if idx != -1:
                    ctx_start = max(0 , idx - 50)
                    ctx_end = min(len(doc_text), idx + len(d.span.text) + 50)
                    context = doc_text[ctx_start:ctx_end].strip()
                    
                else:
                    context = ""
                    
                requery_cases.append({
                    "span_text" : d.span.text,
                    "context" : context,
                    "hint" : f"previous classifier confidence : {d.confidence:.2f}"
                })
                
            est = len(str(requery_cases)) // 4 + 500
            self.qwen_budget.wait(est)
                
            output = self._invoke_overseer([
                SystemMessage(content=REQUERY_SYSTEM_PROMPT),
                HumanMessage(content=build_requery_prompt(doc_text, requery_cases))
            ])
            
            if output is None:
                return {
                    "final_entities": [d for d in decisions if d.entity_type is not None],
                    "requery_count": 0,
                    "errors": ["overseer_node: JSON parse failed, passing decisions through"]
                }
            
            self.qwen_budget.record(est)
            
            requery_results = {e.span_text: e for e in output.entities}
            
            updated_decisions = []
            requery_count = 0
            
            for d in decisions:
                if d.source == AgentSource.REQUERY_NEEDED:
                    resolved = requery_results.get(d.span.text)
                    if resolved:
                        print(f"  [overseer] resolved '{d.span.text}' -> entity_type='{resolved.entity_type}' reasoning='{resolved.reasoning[:80]}'")
                    else:
                        print(f"  [overseer] no result for '{d.span.text}' in requery_results")
                        print(f"  [overseer] requery_results keys: {list(requery_results.keys())}")
                    if resolved and resolved.entity_type:
                        try:
                            updated = Decision(
                                span = d.span,
                                entity_type=EntityTypes(resolved.entity_type),
                                source = AgentSource.REQUERY,
                                confidence=0.75,
                                requeried=True
                            )
                            updated_decisions.append(updated)
                            requery_count+=1
                        except ValueError:
                            print(f"  [overseer] invalid type from LLM: '{resolved.entity_type}' "
                                f"for span '{d.span.text}' - dropping")
                            pass
                else:
                    updated_decisions.append(d)
                    
            final_entities = [d for d in updated_decisions if d.entity_type is not None]
            
            print(f"[overseer] : requeried -> {requery_count}\n"
                  f"resolved : {len([d for d in final_entities if d.requeried])}\n"
                  f"dropped_after_requiery : {len(requery_spans) - requery_count}\n")
                        
            return {
                "final_entities" : final_entities,
                "requery_count" : requery_count,
            }
            
        except Exception as e:
            print(f"[overseer_node] doc {doc_id} failed: {e}")
            return {
                "final_entities" : [d for d in decisions if d.entity_type is not None],
                "requery_count" : 0,
                "errors" : [f"overseer node : {str(e)}"],
            }
    
    def offset_localization_node(self, state: NERAgentState) -> dict:
        final_entities = state["final_entities"]
        doc_text = state["doc_text"]
        doc_id = state["doc_id"]

        localised: list[Decision] = []
        occurrence_count: defaultdict[str, int] = defaultdict(int)

        for d in final_entities:
            occurrences = find_occurrences(doc_text, d.span.text)

            if not occurrences:
                continue

            for start, end in occurrences:
                if occurrence_count[d.span.text] >= MAX_OCCURRENCES_PER_SPAN:
                    break
                localised.append(Decision(
                    span=d.span,
                    entity_type=d.entity_type,
                    source=d.source,
                    confidence=d.confidence,
                    requeried=d.requeried,
                    start=start,
                    end=end
                ))
                occurrence_count[d.span.text] += 1

        print(f"[offset_localisation] {len(localised)} entities localised "
            f"from {len(final_entities)} decisions "
            f"(cap={MAX_OCCURRENCES_PER_SPAN})")
        return {"final_entities": localised}
    


    def _invoke_overseer(self, messages) -> OverseerOutput | None:
        try:
            resp = self.overseer_model.invoke(messages)
            txt = resp.content.strip()

            # strip Qwen3 thinking block
            if "<think>" in txt:
                end_think = txt.rfind("</think>")
                if end_think != -1:
                    txt = txt[end_think + len("</think>"):].strip()
                else:
                    # unclosed think block - find JSON start manually
                    txt = txt[txt.find("{"):]

            # strip markdown fences
            if txt.startswith("```"):
                txt = txt[txt.find("{"):txt.rfind("}")+1]

            data = json.loads(txt)
            entities = [
                ResolvedSpan(
                    span_text=e.get("span_text", ""),
                    entity_type=e.get("entity_type"),
                    reasoning=e.get("reasoning", "")
                )
                for e in data.get("entities", [])
                if isinstance(e, dict) and e.get("span_text")
            ]
            return OverseerOutput(entities=entities)
        except json.JSONDecodeError as e:
            print(f"  [overseer] JSON parse failed: {e}")
            return None
        except Exception as e:
            print(f"  [overseer] call failed: {type(e).__name__}: {e}")
            return None
    

    
    def _build_graph(self):
        g = StateGraph(NERAgentState)
        g.add_node("span_extraction_node", self.span_extraction_node)
        g.add_node("pattern_matching_agent", self.pattern_matching_agent)
        g.add_node("rare_relation_agent", self.rare_relation_agent)
        g.add_node("common_relation_agent", self.common_relation_agent)
        g.add_node("combiner_node", self.combiner_node)
        g.add_node("overseer_node", self.overseer_node)
        g.add_node("offset_localization_node" , self.offset_localization_node)

        g.add_edge(START, "span_extraction_node")
        
        # fan out
        g.add_edge("span_extraction_node", "pattern_matching_agent")
        g.add_edge("span_extraction_node", "rare_relation_agent")
        g.add_edge("span_extraction_node", "common_relation_agent")
        
        # fan in
        g.add_edge("pattern_matching_agent", "combiner_node")
        g.add_edge("rare_relation_agent", "combiner_node")
        g.add_edge("common_relation_agent", "combiner_node")

        g.add_edge("combiner_node", "overseer_node")
        g.add_edge("overseer_node", "offset_localization_node")
        g.add_edge("offset_localization_node", END)
        return g.compile()