### Batch evaluation for RAG controllers on HotpotQA

import os
import ast
import asyncio
import json
import time
from datetime import datetime
from dotenv import load_dotenv

import tiktoken

load_dotenv(dotenv_path="/gpfs/scratch/yg3030/AdaGATE/SEAL-RAG/.env")

from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Experiment parameters, change only this section between runs
N                  = 200
RETRIEVER_K        = 3
REPAIR_LOOP        = 1
MODEL              = "openai:gpt-4o-mini"
EVAL_MODEL         = "gpt-4o"
PINECONE_INDEX     = "seal-v3-hard"
PINECONE_NAMESPACE = None
CONTROLLER         = "adagate_clean"
TOKEN_BUDGET       = 3000

SLEEP_BETWEEN_QUERIES = 1.0
MAX_RETRIES           = 5
RETRY_BASE_WAIT       = 60

from src_adagate.workflow_manager import build_adagate_graph

def build_graph():
    return build_adagate_graph()


# Correctness judge

class CorrectnessResult(BaseModel):
    reasoning: str = Field(description="brief explanation of the correctness decision")
    score: bool    = Field(description="true if correct, false otherwise")

correctness_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert evaluator for question answering systems.
A correct answer:
- matches the ground truth semantically (exact wording not required)
- contains no factual errors relative to the reference
- directly addresses the question asked

Penalize:
- factual contradictions with the ground truth
- "I don't know" responses when the ground truth is clear
- answers that are technically true but miss the point"""),
    ("human", """Question: {question}
Agent answer: {agent_answer}
Ground truth: {ground_truth}

Is the agent answer correct? Provide reasoning and a boolean score.""")
])

def build_judge(model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0)
    return correctness_prompt | llm.with_structured_output(CorrectnessResult)


# Retrieval metrics

def compute_retrieval_metrics(supporting_facts, retrieved_docs) -> dict:
    # Precision/recall/F1 against gold supporting titles
    gold_titles = set(supporting_facts["title"])

    seen, deduped_titles = set(), []
    for doc in retrieved_docs:
        text  = getattr(doc, "page_content", str(doc))
        title = text.split(":", 1)[0].strip()
        if title not in seen:
            seen.add(title)
            deduped_titles.append(title)

    tp        = len(gold_titles & set(deduped_titles))
    precision = tp / len(deduped_titles) if deduped_titles else 0.0
    recall    = tp / len(gold_titles)    if gold_titles    else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)

    return {
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "gold_titles":      list(gold_titles),
        "retrieved_titles": deduped_titles,
    }


# Token counting

_tokenizer = tiktoken.get_encoding("cl100k_base")

def count_input_tokens(question: str, docs: list) -> int:
    # Rough estimate of tokens sent to the generator
    text = question + " " + " ".join(
        getattr(d, "page_content", str(d)) for d in docs
    )
    return len(_tokenizer.encode(text))


# Graph invocation

async def run_graph(graph, question: str) -> tuple[str, list]:
    result = await graph.ainvoke(
        {"user_query": question},
        config={
            "configurable": {
                "model":               MODEL,
                "retriever_k":         RETRIEVER_K,
                "repair_loop_limit":   REPAIR_LOOP,
                "pinecone_index_name": PINECONE_INDEX,
                "pinecone_namespace":  PINECONE_NAMESPACE,
                "token_budget":        TOKEN_BUDGET,
            },
            "recursion_limit": 100,
        }
    )
    docs   = result.get("relevance_documents") or result.get("documents", [])
    answer = result.get("final_answer")        or result.get("generation", "")
    return answer, docs


# Main loop

async def main():
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results_{CONTROLLER}_k{RETRIEVER_K}_L{REPAIR_LOOP}_n{N}_B{TOKEN_BUDGET}_{timestamp}.json"

    print(f"loading {N} HotpotQA distractor-setting examples...")
    dataset = load_dataset("hotpot_qa", "distractor", split=f"validation[:{N}]")

    print(f"building graph : {CONTROLLER}")
    print(f"backbone model : {MODEL}")
    print(f"judge model    : {EVAL_MODEL}")
    print(f"token budget   : {TOKEN_BUDGET}")
    print(f"k={RETRIEVER_K}  L={REPAIR_LOOP}  N={N}\n")

    graph   = build_graph()
    judge   = build_judge(EVAL_MODEL)
    results = []

    for i, example in enumerate(dataset):
        print(f"\n[{i+1}/{N}] {example['question'][:75]}...")

        for attempt in range(MAX_RETRIES):
            try:
                answer, docs = await run_graph(graph, example["question"])

                eval_result = judge.invoke({
                    "question":     example["question"],
                    "agent_answer": answer,
                    "ground_truth": example["answer"],
                })

                metrics      = compute_retrieval_metrics(example["supporting_facts"], docs)
                input_tokens = count_input_tokens(example["question"], docs)

                row = {
                    "id":            example["id"],
                    "question":      example["question"],
                    "ground_truth":  example["answer"],
                    "predicted":     answer,
                    "correct":       eval_result.score,
                    "reasoning":     eval_result.reasoning,
                    "input_tokens":  input_tokens,
                    "doc_count":     len(docs),
                    **metrics,
                }
                results.append(row)

                status      = "✓" if eval_result.score else "✗"
                budget_util = input_tokens / TOKEN_BUDGET if TOKEN_BUDGET else 0
                print(f"  {status} pred='{answer}' | gt='{example['answer']}'")
                print(f"    P={metrics['precision']:.2f}  R={metrics['recall']:.2f}"
                      f"  F1={metrics['f1']:.2f}  tokens={input_tokens}  docs={len(docs)}")
                break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BASE_WAIT * (2 ** attempt)
                    print(f"  attempt {attempt+1} failed: {e}, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  ERROR after {MAX_RETRIES} attempts: {e}")
                    results.append({"id": example["id"], "question": example["question"], "error": str(e)})

        await asyncio.sleep(SLEEP_BETWEEN_QUERIES)

    # Aggregate
    valid   = [r for r in results if "correct" in r]
    n_valid = len(valid)

    if n_valid == 0:
        print("\nno valid results.")
        return

    accuracy           = sum(r["correct"]      for r in valid) / n_valid
    avg_prec           = sum(r["precision"]    for r in valid) / n_valid
    avg_rec            = sum(r["recall"]       for r in valid) / n_valid
    avg_f1             = sum(r["f1"]           for r in valid) / n_valid
    avg_input_tokens   = sum(r["input_tokens"] for r in valid) / n_valid
    avg_doc_count      = sum(r["doc_count"]    for r in valid) / n_valid
    avg_budget_util    = avg_input_tokens / TOKEN_BUDGET if TOKEN_BUDGET else 0
    correct_rows       = [r for r in valid if r["correct"]]
    tokens_per_correct = (sum(r["input_tokens"] for r in correct_rows) / len(correct_rows)
                          if correct_rows else 0)

    summary = {
        "controller":         CONTROLLER,
        "dataset_name":       PINECONE_NAMESPACE or "clean",
        "model":              MODEL,
        "eval_model":         EVAL_MODEL,
        "retriever_k":        RETRIEVER_K,
        "repair_loop":        REPAIR_LOOP,
        "token_budget":       TOKEN_BUDGET,
        "n_total":            N,
        "n_valid":            n_valid,
        "n_errors":           N - n_valid,
        "accuracy":           round(accuracy,           4),
        "avg_precision":      round(avg_prec,           4),
        "avg_recall":         round(avg_rec,            4),
        "avg_f1":             round(avg_f1,             4),
        "avg_input_tokens":   round(avg_input_tokens,   2),
        "avg_doc_count":      round(avg_doc_count,      2),
        "avg_budget_util":    round(avg_budget_util,    4),
        "tokens_per_correct": round(tokens_per_correct, 2),
        "timestamp":          timestamp,
    }

    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*55}")
    print(f"controller      : {CONTROLLER}")
    print(f"backbone        : {MODEL}")
    print(f"judge           : {EVAL_MODEL}")
    print(f"k={RETRIEVER_K}  L={REPAIR_LOOP}  n={n_valid}")
    print(f"accuracy        : {accuracy:.1%}  ({sum(r['correct'] for r in valid)}/{n_valid})")
    print(f"precision       : {avg_prec:.1%}")
    print(f"recall          : {avg_rec:.1%}")
    print(f"f1              : {avg_f1:.1%}")
    print(f"avg tokens      : {avg_input_tokens:.0f}")
    print(f"avg docs        : {avg_doc_count:.1f}")
    print(f"budget util     : {avg_budget_util:.1%}")
    print(f"tokens/correct  : {tokens_per_correct:.2f}")
    print(f"{'='*55}")
    print(f"results saved → {output_path}")


asyncio.run(main())