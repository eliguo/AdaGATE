"""
run_evaluate.py

batch evaluation script for RAG controllers on HotpotQA.
runs a specified graph over N examples, computes answer correctness
and evidence retrieval metrics, and saves results to a JSON file.

usage:
    uv run python run_evaluate.py

all experiment parameters are set in the CONFIG section below.
to run a different experiment, change only that section.
"""

import os
import ast
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path="/gpfs/scratch/yg3030/AdaGATE/SEAL-RAG/.env")

from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# ============================================================
# CONFIG — change only this section between experiments
# ============================================================

N               = 20
RETRIEVER_K     = 3                 # fixed context size (k slots)
REPAIR_LOOP     = 1                 # max repair iterations (L)
MODEL           = "openai:gpt-4o-mini"   # backbone LLM
EVAL_MODEL      = "gpt-4o-mini"    # judge model for correctness
PINECONE_INDEX  = "seal-v3-hard"   # Pinecone index to query
CONTROLLER      = "seal_rag"       # label for output filename

# select which graph to use — comment/uncomment as needed
from src.workflow_manager import build_seal_rag_graph
# from src.other_rags.basic_rag import get_basic_rag_graph
# from src.other_rags.self_rag import get_self_rag_graph
# from src.other_rags.crag import get_crag_graph
# from src.other_rags.adaptive_rag import get_adaptive_rag_graph

def build_graph():
    return build_seal_rag_graph()
    # return get_basic_rag_graph()
    # return get_self_rag_graph()
    # return get_crag_graph()
    # return get_adaptive_rag_graph()

# ============================================================
# CORRECTNESS JUDGE
# ============================================================

class CorrectnessResult(BaseModel):
    """structured output from the LLM judge."""
    reasoning: str = Field(description="brief explanation of the correctness decision")
    score: bool = Field(description="true if answer is correct, false otherwise")

correctness_prompt = ChatPromptTemplate.from_messages([
    ("system", """you are an expert evaluator for question answering systems.
a correct answer:
- matches the ground truth semantically (exact wording not required)
- contains no factual errors relative to the reference
- directly addresses the question asked

penalize:
- factual contradictions with the ground truth
- "i don't know" responses when the ground truth is clear
- answers that are technically true but miss the point of the question"""),
    ("human", """question: {question}
agent answer: {agent_answer}
ground truth: {ground_truth}

is the agent answer correct? provide your reasoning and a boolean score.""")
])

def build_judge(model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0)
    return correctness_prompt | llm.with_structured_output(CorrectnessResult)

# ============================================================
# RETRIEVAL METRICS
# ============================================================

def compute_retrieval_metrics(supporting_facts, retrieved_docs) -> dict:
    """
    compute precision, recall, and F1 at k against gold supporting titles.
    supporting_facts: dict with key 'title' (list of gold titles)
    retrieved_docs: list of langchain Document objects
    """
    gold_titles = set(supporting_facts["title"])

    # extract titles from document page_content (format: "Title: sentence...")
    # deduplicate to avoid inflating precision
    seen, deduped_titles = set(), []
    for doc in retrieved_docs:
        text = getattr(doc, "page_content", str(doc))
        title = text.split(":", 1)[0].strip()
        if title not in seen:
            seen.add(title)
            deduped_titles.append(title)

    tp = len(gold_titles & set(deduped_titles))
    precision = tp / len(deduped_titles) if deduped_titles else 0.0
    recall    = tp / len(gold_titles)    if gold_titles    else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "gold_titles": list(gold_titles),
            "retrieved_titles": deduped_titles}

# ============================================================
# GRAPH INVOCATION
# ============================================================

async def run_graph(graph, question: str) -> tuple[str, list]:
    """
    invoke the RAG graph on a single question.
    returns (answer_string, list_of_retrieved_documents).
    """
    result = await graph.ainvoke(
        {"user_query": question},
        config={
            "configurable": {
                "model":              MODEL,
                "retriever_k":        RETRIEVER_K,
                "repair_loop_limit":  REPAIR_LOOP,
                "pinecone_index_name": PINECONE_INDEX,
            },
            "recursion_limit": 100,
        }
    )
    # handle different output key names across graph implementations
    docs   = result.get("relevance_documents") or result.get("documents", [])
    answer = result.get("final_answer")        or result.get("generation", "")
    return answer, docs

# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

async def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results_{CONTROLLER}_k{RETRIEVER_K}_L{REPAIR_LOOP}_n{N}_{timestamp}.json"

    print(f"loading {N} HotpotQA distractor-setting examples...")
    dataset = load_dataset("hotpot_qa", "distractor", split=f"validation[:{N}]")

    print(f"building graph: {CONTROLLER}")
    graph = build_graph()
    judge = build_judge(EVAL_MODEL)

    results = []

    for i, example in enumerate(dataset):
        print(f"\n[{i+1}/{N}] {example['question'][:75]}...")

        try:
            # --- run the RAG graph ---
            answer, docs = await run_graph(graph, example["question"])

            # --- judge correctness ---
            eval_result = judge.invoke({
                "question":     example["question"],
                "agent_answer": answer,
                "ground_truth": example["answer"],
            })

            # --- retrieval metrics ---
            sf = example["supporting_facts"]
            metrics = compute_retrieval_metrics(sf, docs)

            row = {
                "id":           example["id"],
                "question":     example["question"],
                "ground_truth": example["answer"],
                "predicted":    answer,
                "correct":      eval_result.score,
                "reasoning":    eval_result.reasoning,
                **metrics,
            }
            results.append(row)

            status = "✓" if eval_result.score else "✗"
            print(f"  {status} pred='{answer}' | gt='{example['answer']}'")
            print(f"    P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id":       example["id"],
                "question": example["question"],
                "error":    str(e),
            })

    # --------------------------------------------------------
    # aggregate metrics (exclude errored rows)
    # --------------------------------------------------------
    valid = [r for r in results if "correct" in r]
    n_valid = len(valid)

    if n_valid == 0:
        print("\nno valid results to aggregate.")
        return

    accuracy  = sum(r["correct"]   for r in valid) / n_valid
    avg_prec  = sum(r["precision"] for r in valid) / n_valid
    avg_rec   = sum(r["recall"]    for r in valid) / n_valid
    avg_f1    = sum(r["f1"]        for r in valid) / n_valid

    summary = {
        "controller":     CONTROLLER,
        "model":          MODEL,
        "retriever_k":    RETRIEVER_K,
        "repair_loop":    REPAIR_LOOP,
        "n_total":        N,
        "n_valid":        n_valid,
        "n_errors":       N - n_valid,
        "accuracy":       round(accuracy,  4),
        "avg_precision":  round(avg_prec,  4),
        "avg_recall":     round(avg_rec,   4),
        "avg_f1":         round(avg_f1,    4),
        "timestamp":      timestamp,
    }

    output = {"summary": summary, "results": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*55}")
    print(f"controller : {CONTROLLER}")
    print(f"model      : {MODEL}  |  k={RETRIEVER_K}  L={REPAIR_LOOP}  n={n_valid}")
    print(f"accuracy   : {accuracy:.1%}  ({sum(r['correct'] for r in valid)}/{n_valid})")
    print(f"precision  : {avg_prec:.1%}")
    print(f"recall     : {avg_rec:.1%}")
    print(f"f1         : {avg_f1:.1%}")
    print(f"{'='*55}")
    print(f"results saved → {output_path}")

asyncio.run(main())
