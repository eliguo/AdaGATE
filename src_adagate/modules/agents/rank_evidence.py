from typing import List, Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

import tiktoken
import numpy as np

from src_adagate.states import State, SimpleTriplet
from src_adagate.modules.configuration import Configuration

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


class RelevanceScoringOutput(BaseModel):
    entities_to_keep: List[int] = []


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text or ""))


def _get_title(doc: Document) -> str:
    return (doc.metadata or {}).get("title", "Document")


def _estimate_effective_capacity(
    docs: List[Document],
    utility_scores: List[float],
    buffer: int = 2,
) -> int:
    """
    AdaGATE adaptive capacity estimation via largest utility score drop.
    K_eff = i* + buffer where i* = argmax of first differences.
    """
    if len(utility_scores) < 2:
        return len(docs)
    diffs = [
        utility_scores[i] - utility_scores[i + 1]
        for i in range(len(utility_scores) - 1)
    ]
    i_star = int(np.argmax(diffs)) if diffs else 0
    return min(i_star + 1 + buffer, len(docs))


def _estimate_effective_capacity_by_length(
    docs: List[Document],
    buffer: int = 2,
) -> int:
    """Fallback: use token length differences when utility scores unavailable."""
    if len(docs) < 2:
        return len(docs)
    lengths = [_count_tokens(d.page_content) for d in docs]
    diffs = [abs(lengths[i] - lengths[i + 1]) for i in range(len(lengths) - 1)]
    i_star = int(np.argmax(diffs)) if diffs else 0
    return min(i_star + 1 + buffer, len(docs))


def _apply_token_budget(
    docs: List[Document],
    budget: int,
) -> tuple[List[Document], int]:
    """
    Greedily select documents until token budget B is exhausted.
    Returns (selected_docs, tokens_used).
    """
    selected = []
    used = 0
    for doc in docs:
        doc_tokens = _count_tokens(doc.page_content)
        if used + doc_tokens <= budget:
            selected.append(doc)
            used += doc_tokens
        else:
            break
    return selected, used


def _format_indexed_entities(entities: List[SimpleTriplet], max_n: int = 50) -> str:
    shown = entities[:max_n]
    lines = []
    for i, e in enumerate(shown, 1):
        lines.append(
            f"[{i}] [{e.subject_type}] {e.subject} --{e.relation}--> [{e.object_type}] {e.object}"
        )
    return "\n".join(lines) if lines else "None"


def _format_indexed_documents(
    documents: List[Document],
    max_k: int = 50,
    snippet_len: int = 6000,
) -> str:
    shown = documents[:max_k]
    blocks = []
    for i, d in enumerate(shown, 1):
        title = _get_title(d)
        snippet = (d.page_content or "")[:snippet_len]
        blocks.append(f"[{i}] {title}\n{snippet}")
    return "\n\n".join(blocks) if blocks else "None"


def _create_rank_prompt() -> ChatPromptTemplate:
    system = """You are a relevance ranker. Be conservative and prefer fewer, higher-quality items."""
    human = """QUESTION:
<QUESTION>
{question}
</QUESTION>

REPAIR REASONING (guides which chain is sufficient):
<REPAIR_REASONING>
{repair_reasoning}
</REPAIR_REASONING>

ENTITIES (indexed):
<ENTITIES>
{indexed_entities}
</ENTITIES>

DOCUMENTS (indexed):
<DOCUMENTS>
{indexed_documents}
</DOCUMENTS>

TASKS:
1) Prioritize only items that support the minimal sufficient chain implied by <REPAIR_REASONING>.
2) Select the minimal sufficient subset of ENTITIES (<= 10) as a list of indices. Choose the FEWEST entities needed to answer.

Return JSON with:
- entities_to_keep: [indices]"""
    return ChatPromptTemplate.from_messages([("system", system), ("human", human)])


async def rank_evidence(state: State, config: RunnableConfig) -> State:
    print("---RANK EVIDENCE (AdaGATE)---")

    user_query     = state["user_query"]
    documents      = state.get("documents", []) or []      # already utility-sorted from to_repair
    entities       = state.get("cached_entities", []) or []
    utility_scores = state.get("utility_scores") or []

    configurable  = config.get("configurable", {})
    configuration = Configuration(**configurable)
    token_budget  = configuration.token_budget

    # Adaptive capacity on all utility-sorted docs
    if utility_scores and len(utility_scores) >= 2:
        k_eff = _estimate_effective_capacity(documents, utility_scores)
        print(f"  adaptive capacity K_eff={k_eff} (from utility scores)")
    else:
        k_eff = _estimate_effective_capacity_by_length(documents)
        print(f"  adaptive capacity K_eff={k_eff} (fallback: token lengths)")

    candidate_docs = documents[:k_eff]

    # Token budget, docs stay in utility score order
    relevance_documents, tokens_used = _apply_token_budget(candidate_docs, token_budget)

    if not relevance_documents and documents:
        relevance_documents = documents[:1]
        tokens_used         = _count_tokens(documents[0].page_content)

    print(f"  token_budget={token_budget} | tokens_used={tokens_used} | docs={len(relevance_documents)}")

    # LLM entity selection, only affects relevance_entities not document ordering
    repair_reasoning = ""
    repair_decision  = state.get("repair_decision")
    if repair_decision and getattr(repair_decision, "reasoning", None):
        repair_reasoning = repair_decision.reasoning

    llm        = configuration.build_llm()
    structured = llm.with_structured_output(RelevanceScoringOutput)
    chain      = _create_rank_prompt() | structured

    result = await chain.ainvoke({
        "question":          user_query,
        "repair_reasoning":  repair_reasoning,
        "indexed_entities":  _format_indexed_entities(entities, max_n=30),
        "indexed_documents": _format_indexed_documents(relevance_documents, max_k=50, snippet_len=600),
    })

    entities_to_keep = [
        idx for idx in (result.entities_to_keep or [])
        if 1 <= idx <= len(entities)
    ]
    if not entities_to_keep and entities:
        entities_to_keep = [1]

    relevance_entities = [entities[i - 1] for i in entities_to_keep if 1 <= i <= len(entities)]

    print(f"  relevance_entities={len(relevance_entities)} | relevance_docs={len(relevance_documents)}")

    return {
        "relevance_documents": relevance_documents,
        "relevance_entities":  relevance_entities,
        "token_budget_used":   tokens_used,
        "token_budget":        token_budget,
    }