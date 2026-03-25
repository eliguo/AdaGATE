"""
run_ares.py

takes the JSON output from evaluate.py, converts it to the TSV format
required by ARES, then runs ARES ues_idp scoring for:
  - context relevance  (is the retrieved document relevant to the query?)
  - answer faithfulness (is the answer grounded in the retrieved document?)
  - answer relevance   (does the answer address the question?)

usage:
    uv run python run_ares.py --input <path_to_results_json>

example:
    uv run python run_ares.py --input results_seal_rag_k3_L1_n20_20260324_190000.json
"""

import os
import csv
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path="/gpfs/scratch/yg3030/AdaGATE/SEAL-RAG/.env")

# ============================================================
# CONFIG
# ============================================================

FEW_SHOT_PROMPT = "/gpfs/scratch/yg3030/AdaGATE/nq_few_shot_prompt_for_judge_scoring.tsv"
ARES_JUDGE_MODEL = "gpt-3.5-turbo-0125"   # cheaper than 4o, sufficient for ues_idp

# ============================================================
# STEP 1: convert evaluate.py JSON output → ARES TSV format
# ============================================================

def json_to_ares_tsv(json_path: str) -> str:
    """
    converts evaluate.py results JSON to ARES unlabeled TSV.
    ARES expects columns: Query, Document, Answer
    returns the path to the generated TSV file.
    """
    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
    valid = [r for r in results if "error" not in r]

    tsv_path = json_path.replace(".json", "_ares_input.tsv")

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Query", "Document", "Answer"],
            delimiter="\t"
        )
        writer.writeheader()
        for r in valid:
            # use retrieved titles joined as the document field
            # fallback to empty string if not present
            doc_text = " | ".join(r.get("retrieved_titles", []))
            writer.writerow({
                "Query":    r["question"],
                "Document": doc_text,
                "Answer":   r["predicted"],
            })

    print(f"converted {len(valid)} results → {tsv_path}")
    return tsv_path

# ============================================================
# STEP 2: run ARES ues_idp scoring
# ============================================================

def run_ares_scoring(tsv_path: str, few_shot_path: str, model: str) -> dict:
    """
    runs ARES ues_idp (UES + in-domain prompting) scoring.
    this uses an LLM judge directly — no fine-tuning or GPU needed.
    returns the ARES scores dict.
    """
    from ares import ARES

    ues_idp_config = {
        "in_domain_prompts_dataset": few_shot_path,
        "unlabeled_evaluation_set":  tsv_path,
        "model_choice":              model,
    }

    ares = ARES(ues_idp=ues_idp_config)
    results = ares.ues_idp()
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True,
        help="path to results JSON file output by evaluate.py"
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"input file not found: {args.input}")

    # load summary info for display
    with open(args.input) as f:
        data = json.load(f)
    summary = data.get("summary", {})

    print(f"\nrunning ARES on: {args.input}")
    print(f"controller : {summary.get('controller', 'unknown')}")
    print(f"n_valid    : {summary.get('n_valid', '?')}")
    print(f"judge model: {ARES_JUDGE_MODEL}\n")

    # step 1: convert to TSV
    tsv_path = json_to_ares_tsv(args.input)

    # step 2: run ARES
    print("running ARES ues_idp scoring (this may take a few minutes)...")
    ares_results = run_ares_scoring(tsv_path, FEW_SHOT_PROMPT, ARES_JUDGE_MODEL)

    # step 3: display and save
    print(f"\n{'='*55}")
    print("ARES evaluation results")
    print(f"{'='*55}")
    for key, value in ares_results.items():
        print(f"  {key}: {value}")
    print(f"{'='*55}")

    # save ARES results alongside the original JSON
    ares_output_path = args.input.replace(".json", "_ares_scores.json")
    output = {
        "evaluate_py_summary": summary,
        "ares_scores":         ares_results,
    }
    with open(ares_output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nARES scores saved → {ares_output_path}")

if __name__ == "__main__":
    main()
