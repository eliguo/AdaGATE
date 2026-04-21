### AdaGATE utility scoring, repair decision, and routing

from typing import List
from src_adagate.states import State, SimpleTriplet, RepairDecisionOutput
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from src_adagate.modules.configuration import Configuration
import re


def _compute_utility_scores(
    documents: List[Document],
    user_query: str,
    cached_entities: List[SimpleTriplet],
    gap_reasoning: str,
    tau_conf: float = 0.70,
) -> List[tuple]:
    # S_t(c) = lambda1*GapCov + lambda2*Corr + lambda3*Nov - lambda4*Red + lambda5*RelQ
    gap_terms = {
        t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", gap_reasoning or "")
        if len(t) > 3
    }
    existing_entities = {
        e.subject.lower() for e in cached_entities if e.subject
    } | {
        e.object.lower() for e in cached_entities if e.object
    }
    query_terms = {
        t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", user_query or "")
        if len(t) > 3
    }

    # Uncertain triplets for Corr term (confidence < tau_conf * 100)
    uncertain_doc_titles = {}
    tau_int = int(tau_conf * 100)
    for e in cached_entities:
        if e.confidence < tau_int:
            title = getattr(e, "source_doc_title", "") or ""
            uncertain_doc_titles[title] = uncertain_doc_titles.get(title, 0) + 1

    scored = []
    for doc in documents:
        content       = (doc.page_content or "").lower()
        content_terms = {
            t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", content)
            if len(t) > 3
        }

        gap_cov    = len(gap_terms & content_terms) / len(gap_terms) if gap_terms else 0.0
        doc_title  = (doc.metadata or {}).get("title", "")
        corr       = min(uncertain_doc_titles.get(doc_title, 0) / max(len(cached_entities), 1), 1.0)
        novelty    = len(content_terms - existing_entities) / len(content_terms) if content_terms else 0.0
        redundancy = len(content_terms & existing_entities) / len(content_terms) if content_terms else 0.0
        rel_q      = len(query_terms & content_terms) / len(query_terms) if query_terms else 0.0

        score = (
            0.35 * gap_cov +
            0.20 * corr +
            0.15 * novelty -
            0.15 * redundancy +
            0.15 * rel_q
        )
        scored.append((doc, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)


def _format_cached_entities_for_repair(cached_entities: List[SimpleTriplet]) -> str:
    if not cached_entities:
        return "CACHED ENTITIES: None available"
    entities_to_show = cached_entities[:100]
    formatted = "CACHED ENTITIES (Knowledge from previous iterations):\n"
    for i, e in enumerate(entities_to_show, 1):
        formatted += (
            f"[{i}] [{e.subject_type}] {e.subject} → {e.relation} → [{e.object_type}] {e.object}\n\n"
        )
    if len(cached_entities) > 100:
        formatted += f"... and {len(cached_entities) - 100} more entities\n"
    return formatted


def _format_documents_for_repair(documents: List[Document], max_docs: int = 20, snippet_len: int = 1000) -> str:
    if not documents:
        return "No documents available."
    out = []
    for d in documents[:max_docs]:
        title   = (d.metadata or {}).get("title", "Document")
        snippet = (d.page_content or "")[:snippet_len]
        out.append(f"--- {title} ---\n{snippet}")
    return "\n\n".join(out)


_ASSOCIATION_RELATIONS = {
    "associated_with",
    "improves_relations_between",
    "aims_to_improve_relations_between",
    "produced_to_improve_relations_between",
}


def _extract_candidates_for_repair(cached_entities: List[SimpleTriplet]) -> str:
    if not cached_entities:
        return "None"
    raw: list[str] = []
    for e in cached_entities:
        rel_norm = (e.relation or "").strip().lower().replace(" ", "_")
        if rel_norm in _ASSOCIATION_RELATIONS:
            if e.object:
                raw.append(e.object)
    if not raw:
        return "None"
    pattern          = re.compile(r"\s*(?:,|/|&|\band\b)\s*", re.IGNORECASE)
    split_candidates = []
    for item in raw:
        parts = [p.strip(" \t\n\r,/&") for p in pattern.split(item) if p.strip()]
        split_candidates.extend(parts if parts else [item])
    seen, ordered = set(), []
    for c in split_candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(c)
    if not ordered:
        return "None"
    return "\n".join(f"[{i}] {name}" for i, name in enumerate(ordered, 1))


def _create_repair_decision_prompt() -> ChatPromptTemplate:
    system_prompt = """You are a logical chain checker. Use the template exactly.

RULE: Check if connections exist, don't create them."""

    human_prompt = """QUERY:
<QUERY>
{original_query}
</QUERY>

ENTITIES:
<ENTITIES>
{cached_entities_context}
</ENTITIES>

FORMAT ENTITIES CLEARLY:
- Insert a blank line between each entity.
- Keep each entity on a single line: [index] [TYPE] Subject → relation → [TYPE] Object.

DOCUMENTS (snippets to use only as explicit evidence; do not invent aliases):
<DOCUMENTS>
{documents_context}
</DOCUMENTS>

CANDIDATES (parsed from association relations; deduplicated):
<CANDIDATES>
{candidates_context}
</CANDIDATES>

SYSTEMATIC REASONING TEMPLATE:

1. QUERY DECOMPOSITION:
   - Core question: [What specific information is being requested?]
   - Scope requirement: [a/an/any vs all/every vs the (specific)]
   - Success criteria: [What constitutes a complete answer?]

2. PATH DISCOVERY:
   - Starting entities: [Identify plausible starting points]
   - Possible chains: [For each starting entity, list reasoning paths]
   - Chain completeness: [Which paths lead from question to answer?]

3. RELATIONSHIP ANALYSIS:
   - Direct matches: [Exact entity and relationship matches]
   - Missing links: [Only for entities required under the scope]

4. SUFFICIENCY ASSESSMENT:
   - Are all required chains complete?

5. DECISION SYNTHESIS:
   - Answer feasibility: [Yes/No]
   - Minimal justification: [1-2 short sentences]

DECISION: [Yes/No] based on query validity AND found connections."""

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])


async def to_repair(state: State, config: RunnableConfig) -> State:
    print("---TO REPAIR (AdaGATE)---")

    configurable   = config.get("configurable", {})
    configuration  = Configuration(**configurable)
    loop_limit     = configuration.repair_loop_limit
    tau_conf       = configuration.tau_conf

    user_query      = state["user_query"]
    cached_entities = state.get("cached_entities", [])
    documents       = state.get("documents", [])

    print(f"  {len(cached_entities)} cached entities | {len(documents)} documents")

    # Utility scoring and document re-ranking
    repair_decision_prev = state.get("repair_decision")
    gap_reasoning        = getattr(repair_decision_prev, "reasoning", "") if repair_decision_prev else ""

    scored         = _compute_utility_scores(documents, user_query, cached_entities, gap_reasoning, tau_conf)
    utility_scores = [round(s, 4) for _, s in scored]
    documents      = [doc for doc, _ in scored]

    if utility_scores:
        print(f"  utility scores (top 5): {utility_scores[:5]}")

    # Repair decision
    llm            = configuration.build_llm()
    structured_llm = llm.with_structured_output(RepairDecisionOutput)
    decision_chain = _create_repair_decision_prompt() | structured_llm

    try:
        print("Making repair decision with LLM...")
        decision_result = await decision_chain.ainvoke({
            "original_query":          user_query,
            "cached_entities_context": _format_cached_entities_for_repair(cached_entities),
            "documents_context":       _format_documents_for_repair(documents),
            "candidates_context":      _extract_candidates_for_repair(cached_entities),
        })

        print(f"Decision: {decision_result.can_answer_original_query}")
        concise_reason = decision_result.reasoning
        if concise_reason and len(concise_reason) > 400:
            concise_reason = concise_reason[:400] + "..."
        print(f"Reasoning: {concise_reason}")

        return {
            "user_query":        user_query,
            "documents":         documents,
            "utility_scores":    utility_scores,
            "repair_decision":   decision_result,
            "repair_loop_limit": loop_limit,
        }

    except Exception as e:
        print(f"Error in repair decision: {e}")
        return {
            "user_query":        user_query,
            "documents":         documents,
            "utility_scores":    utility_scores,
            "repair_decision":   RepairDecisionOutput(
                can_answer_original_query="no",
                reasoning=f"Error occurred during repair decision: {str(e)}"
            ),
            "repair_loop_limit": loop_limit,
        }


def route_after_repair_decision(state: State):
    print("---ROUTING AFTER REPAIR DECISION---")

    repair_decision = state.get("repair_decision")
    if repair_decision and repair_decision.can_answer_original_query.lower() == "yes":
        print("✅ Ready to answer - proceeding to rank_evidence")
        return "rank_evidence"

    loop_count = state.get("repair_loop_count") or 0
    loop_limit = state.get("repair_loop_limit")
    print(f"Loop count: {loop_count}, Loop limit: {loop_limit}")
    if loop_count >= loop_limit:
        print("🔁 Loop limit reached, proceeding to rank_evidence")
        return "rank_evidence"

    print("❌ Not ready - proceeding to micro_query_agent")
    return "micro_query_agent"