### Retrieve docs
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/gpfs/scratch/yg3030/AdaGATE/SEAL-RAG/.env")

from langgraph.constants import Send

from src_adagate.states import State
from src_adagate.modules.configuration import Configuration
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

import asyncio


def get_document_title(document: Document) -> str:
    metadata = document.metadata or {}
    title = metadata.get("title") or metadata.get("Title") or metadata.get("name") or metadata.get("source")
    if isinstance(title, str) and title.strip():
        return title.strip()
    fallback = metadata.get("id") or document.page_content[:80]
    return str(fallback).strip()


def merge_documents_by_title(existing_documents: list[Document], new_documents: list[Document]) -> list[Document]:
    merged: list[Document] = []
    seen_titles: set[str] = set()
    for doc in existing_documents + new_documents:
        key = get_document_title(doc)
        if key and key not in seen_titles:
            seen_titles.add(key)
            merged.append(doc)
    return merged


async def retrieve_docs(state: State, config: RunnableConfig) -> State:
    print("---RETRIEVE (AdaGATE)---")
    configurable = config.get("configurable", {})
    configuration = Configuration(**configurable)

    micro_q = state.get("micro_query")
    user_query = state["user_query"]

    retriever = await asyncio.to_thread(configuration.build_retriever)

    # H_gap: gap-anchored micro-query (or original query on first iteration)
    gap_query = micro_q or user_query
    gap_docs: list[Document] = await retriever.ainvoke(gap_query) or []

    # H_q: explicit question-anchored fallback (AdaGATE innovation)
    # always retrieve using the original question as a parallel channel
    if micro_q:
        fallback_docs: list[Document] = await retriever.ainvoke(user_query) or []
        print(f"  H_gap: {len(gap_docs)} docs | H_q fallback: {len(fallback_docs)} docs")
    else:
        fallback_docs = []

    # Merge H_gap and H_q, deduplicate by title
    new_docs = merge_documents_by_title(gap_docs, fallback_docs)

    existing_docs: list[Document] = state.get("documents", []) or []
    existing_titles = {get_document_title(d) for d in existing_docs}
    docs_new_only = [d for d in new_docs if get_document_title(d) not in existing_titles]
    merged = merge_documents_by_title(existing_docs, new_docs)

    return {
        "documents":          merged,
        "documents_new":      docs_new_only,
        "repair_loop_limit":  configuration.repair_loop_limit,
    }


def continue_to_cached_entities_update_agent(state: State):
    print("---CONTINUE TO CACHED ENTITIES UPDATE AGENT---")

    user_query = state["user_query"]
    documents = state.get("documents_new") or []
    micro_query = state.get("micro_query", None)
    cached_entities = state.get("cached_entities", [])
    repair_decision = state.get("repair_decision", None)

    print(f"Dispatching {len(documents)} documents for entity extraction")

    if not documents:
        loop_count = state.get("repair_loop_count") or 0
        loop_limit = state.get("repair_loop_limit") or 3
        if loop_count < loop_limit:
            return "micro_query_agent"
        else:
            return "to_repair"

    send_commands = [Send("cached_entities_update", {
        "doc_to_extract":  doc,
        "user_query":      user_query,
        "micro_query":     micro_query,
        "cached_entities": cached_entities or [],
        "repair_decision": repair_decision,
    }) for doc in documents]

    return send_commands