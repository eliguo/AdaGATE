from typing import Annotated, Literal
import operator
from typing_extensions import TypedDict
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "DATE", "OTHER"]


class RepairDecisionOutput(BaseModel):
    can_answer_original_query: str = Field(
        ...,
        description="Answer 'yes' if sufficient evidence exists to answer the original query, or 'no' if gaps remain"
    )
    reasoning: str = Field(
        ...,
        description="Provide a concise critical analysis: briefly state the key question(s) you investigated, your finding(s), and how this led to your decision. Keep it focused and direct."
    )

class SimpleTriplet(BaseModel):
    subject: str = Field(..., description="Source node (entity name)")
    subject_type: EntityType = Field(..., description="Source node type")
    relation: str = Field(..., description="Edge/relationship between nodes")
    object: str = Field(..., description="Target node (entity name)")
    object_type: EntityType = Field(..., description="Target node type")
    source_doc_title: str = Field(..., description="Title of Document where this triplet was found")
    confidence: int = Field(95, ge=0, le=100, description="Confidence score")
    relevance: int = Field(50, ge=0, le=100, description="Relevance score")


class StateInput(TypedDict):
    user_query: str


class State(TypedDict):
    user_query: str
    documents: List[Document]
    documents_new: Optional[List[Document]]
    relevance_documents: Optional[List[Document]]
    relevance_entities: Optional[List[SimpleTriplet]]
    micro_query: Optional[str]
    micro_query_history: Optional[List[str]]
    repair_loop_count: Optional[int]
    repair_loop_limit: Optional[int]
    cached_entities: Annotated[List[SimpleTriplet], operator.add]
    repair_decision: Optional[RepairDecisionOutput]
    final_answer: Optional[str]
    # AdaGATE new fields
    utility_scores: Optional[List[float]]      # per-doc utility scores
    token_budget_used: Optional[int]           # tokens used in current evidence set
    token_budget: Optional[int]                # global token budget B