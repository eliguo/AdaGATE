### Convert evaluate.py output to ARES TSV and run ues_idp scoring

import os
import csv
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path="/gpfs/scratch/yg3030/AdaGATE/SEAL-RAG/.env")

FEW_SHOT_PROMPT  = "/gpfs/scratch/yg3030/AdaGATE/nq_few_shot_prompt_for_judge_scoring.tsv"
ARES_JUDGE_MODEL = "gpt-4o"


def json_to_ares_tsv(json_path: str) -> str:
    # Convert evaluate.py results to ARES unlabeled TSV (Query / Document / Answer)
    with open(json_path) as f:
        data = json.load(f)

    valid    = [r for r in data["results"] if "error" not in r]
    tsv_path = json_path.replace(".json", "_ares_input.tsv")

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Query", "Document", "Answer"], delimiter="\t")
        writer.writeheader()
        for r in valid:
            writer.writerow({
                "Query":    r["question"],
                "Document": " | ".join(r.get("retrieved_titles", [])),
                "Answer":   r["predicted"],
            })

    print(f"converted {len(valid)} results → {tsv_path}")
    return tsv_path


def run_ares_scoring(tsv_path: str, few_shot_path: str, model: str) -> dict:
    # Run ues_idp directly, no fine-tuning or GPU needed
    from ares import ARES

    ares = ARES(ues_idp={
        "in_domain_prompts_dataset": few_shot_path,
        "unlabeled_evaluation_set":  tsv_path,
        "model_choice":              model,
    })
    return ares.ues_idp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to evaluate.py results JSON")
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"input file not found: {args.input}")

    with open(args.input) as f:
        data = json.load(f)
    summary = data.get("summary", {})

    print(f"\nrunning ARES on: {args.input}")
    print(f"controller : {summary.get('controller', 'unknown')}")
    print(f"n_valid    : {summary.get('n_valid', '?')}")
    print(f"judge model: {ARES_JUDGE_MODEL}\n")

    tsv_path     = json_to_ares_tsv(args.input)
    ares_results = run_ares_scoring(tsv_path, FEW_SHOT_PROMPT, ARES_JUDGE_MODEL)

    print(f"\n{'='*50}")
    print("ARES scores")
    print(f"{'='*50}")
    for k, v in ares_results.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")

    out_path = args.input.replace(".json", "_ares_scores.json")
    with open(out_path, "w") as f:
        json.dump({"evaluate_py_summary": summary, "ares_scores": ares_results}, f, indent=2)
    print(f"\nARES scores saved → {out_path}")


if __name__ == "__main__":
    main()