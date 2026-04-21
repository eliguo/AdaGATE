### Generate final answer from curated entities and documents

from langchain_core.runnables import RunnableConfig
from src_adagate.states import State
from src_adagate.modules.configuration import Configuration
from langchain_core.prompts import ChatPromptTemplate


def _create_answer_generation_prompt() -> ChatPromptTemplate:
    system = """\
You are an expert answer synthesizer for a multi-hop question answering system. Your job is to provide SHORT, ACCURATE answers based ONLY on the provided entities and documents.

You have access to:
1. **CURATED ENTITIES**: Key facts extracted and filtered for relevance across multiple reasoning steps.
2. **SUPPORTING DOCUMENTS**: Additional context and verification.
3. **ORIGINAL QUESTION**: The complex question requiring multi-step reasoning.
4. **REPAIR REASONING**: The minimal sufficient chain guidance from the decision step.

CRITICAL RULES:
1. **BE PRECISE**: Answer in 2-4 words maximum.
2. **BE CONCISE**: Only provide the essential information needed to directly answer the question.
3. **BE STRICT**: Only use facts explicitly stated in the entities and documents.
4. **BE HONEST**: If you cannot fully answer based on the given data, return "I don't know."
5. **OMIT UNNECESSARY DETAILS**: Do NOT include extra facts, side notes, or additional context, even if they are correct, unless they are strictly required to answer the question correctly.
6. **GIVE THE BEST ANSWER**: Give the best specific answer possible. If not possible, give the best answer you can at the scope.

SYNTHESIS STRATEGY:
- **Start with entities**: These represent the most important, curated facts.
- **Use documents for context**: Only when the entities alone do not fully resolve the question.
- **Connect the dots**: Only include facts necessary to complete the reasoning chain. Avoid adding extra facts that are not needed for the specific answer.
- **Multi-hop reasoning**: Connect the minimum number of steps required to answer the question correctly.
- **Missing links**: If any essential link is missing, return "I don't know."
- **Focus on minimality**: Extra details that do not contribute to answering the question should be excluded.

OUTPUT REQUIREMENTS:
- **Length**: 2-4 words maximum.
- **Format**: Direct answer only. Do not provide explanations unless the reasoning is inherently multi-hop and all steps are required.
- **Unknown answers**: If insufficient evidence is provided, respond with exactly: "I don't know."

EXAMPLES:
- Good: "Paris."
- Good: "Yes."
- Good: "I don't know."
- Bad: "The capital of France is Paris, which is a popular tourist destination." → Extra information not required.
- Bad: "Based on the provided evidence, it seems that..." → Too verbose.
- Bad: "The answer is Paris because..." → Explanatory phrases are unnecessary.

The goal is to deliver the **most minimal, factually complete answer** that directly resolves the question.
"""

    human = """\
**QUESTION:**
{question}

**REPAIR REASONING:**
{repair_reasoning}

**CURATED ENTITIES:**
{cached_entities}

**DOCUMENTS:**
{relevant_documents}

Provide a precise 2-4 words final answer using only the above information."""

    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human)
    ])


def _format_documents(docs):
    # Handles both Document objects (page_content) and GradeDocuments objects (content)
    if not docs:
        return "No documents available."
    formatted_docs = []
    for i, doc in enumerate(docs):
        if hasattr(doc, 'page_content'):
            content = doc.page_content
        elif hasattr(doc, 'content'):
            content = doc.content
        else:
            content = str(doc)
        formatted_docs.append(f"[Doc {i+1}]: {content}")
    return "\n\n".join(formatted_docs)


async def generate_agent(state: State, config: RunnableConfig) -> State:
    print("---GENERATE---")

    user_query          = state["user_query"]
    documents           = state.get("documents", [])
    relevance_documents = state.get("relevance_documents", [])
    cached_entities     = state.get("cached_entities", [])
    relevance_entities  = state.get("relevance_entities", [])
    repair_decision     = state.get("repair_decision")
    repair_reasoning    = getattr(repair_decision, "reasoning", "") if repair_decision else ""

    # Prefer relevance_documents if available, fall back to raw documents
    documents_for_generation = relevance_documents if relevance_documents else documents

    configurable  = config.get("configurable", {})
    configuration = Configuration(**configurable)
    llm           = configuration.build_llm()
    prompt        = _create_answer_generation_prompt()

    chosen_entities = relevance_entities if relevance_entities else cached_entities
    if chosen_entities:
        entities_str = "\n".join(
            f"• [{e.subject_type}] {e.subject} --{e.relation}--> [{e.object_type}] {e.object} \n"
            for e in chosen_entities
        )
    else:
        entities_str = "No curated entities available - relying solely on documents."

    formatted_documents = (
        _format_documents(documents_for_generation)
        if documents_for_generation else "No documents available."
    )

    generation = await (prompt | llm).ainvoke({
        "question":           user_query,
        "repair_reasoning":   repair_reasoning,
        "cached_entities":    entities_str,
        "relevant_documents": formatted_documents,
    })

    return {"final_answer": generation.content}