from langchain_protocol import TypedDict
from pydantic import BaseModel, Field
from enum import Enum
from langgraph.graph import START, StateGraph, END
from typing import List, Any, Optional

from typing import Annotated
from operator import add

class AgentSource(Enum):
    COMMON  = "common"
    RARE = "rare"
    PATTERN = "pattern"
    REQUERY = "requery"
    DROPPED = "dropped"
    REQUERY_NEEDED = "requery needed"
    
    
class EntityTypes(Enum):
    GENE  = "GeneOrGeneProduct"
    DISEASE = "DiseaseOrPhenotypicFeature"
    CHEMICAL = "ChemicalEntity"
    ORGANISM = "OrganismTaxon"
    SEQUENCE = "SequenceVariant"
    CELL = "CellLine"

class CandidateSpan(BaseModel):
    text: str = Field(description="the entity span that was extracted.")
    doc_id: str = Field(description="the id of the document from which the entity was extracted.")
    passage_offset: int = Field(description="the offset of the position of the entity in the passage.")
    
class SpanExtractionOutput(BaseModel):
    spans: List[Decision]

  
class BranchVote(BaseModel):
    span: CandidateSpan
    entity_type: EntityTypes | None
    confidence: float
    source: AgentSource

    
class ResolvedSpan(BaseModel):
    span_text: str
    entity_type: Optional[str] = None  # canonical BioRED type, or None if still uncertain
    reasoning: str = ""               # why the overseer typed it this way

class OverseerOutput(BaseModel):
    entities: List[ResolvedSpan]

class Decision(BaseModel):
    span: CandidateSpan
    entity_type: EntityTypes | None
    source: AgentSource
    confidence: float
    requeried: bool = False 
    start : int = -1
    end : int = -1

    
class NERAgentState(TypedDict):
    doc_id: str
    doc_text: str
    doc : Any
    
    candidate_spans: List[CandidateSpan]  
    
    common_votes: Annotated[List[BranchVote], add]          
    rare_votes: Annotated[List[BranchVote], add]            
    pattern_votes: Annotated[List[BranchVote], add]  
    
    decisions: List[Decision]               
    
    requery_spans: List[Decision]    
    
    requery_count: int  
                        
    skipped_spans: List[str]               
    
    branch_sources: dict                 
    
    errors: List[str]                       
    final_entities: List[Decision]

                                            
    
    