#!/usr/bin/env python3
"""
Generate natural redundancy variants without exact duplicate passages.

This script mirrors the redundancy side of stress_test_variants.py, but
replaces exact duplicate injection with lightly rewritten paraphrases and
partial-overlap copies.

Usage:
    python3 natural_redundancy_variants.py \
        --input dataset/hotpotqa/hotpot_dev_distractor_v1.json \
        --dataset hotpotqa \
        --output-dir generated_stress_tests/hotpotqa_natural_redundancy
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Sequence

import stress_test_variants as stv


PHRASE_REWRITES = [
    (re.compile(r"\bis a\b"), "is described as a"),
    (re.compile(r"\bis an\b"), "is described as an"),
    (re.compile(r"\bis the\b"), "is described as the"),
    (re.compile(r"\bwas a\b"), "was described as a"),
    (re.compile(r"\bwas an\b"), "was described as an"),
    (re.compile(r"\bwas the\b"), "was described as the"),
    (re.compile(r"\bare a\b"), "are described as a"),
    (re.compile(r"\bwere a\b"), "were described as a"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate natural redundancy variants without exact duplicates."
    )
    parser.add_argument("--input", required=True, help="Path to the source dataset JSON/JSONL file.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["hotpotqa", "2wikimultihopqa", "auto"],
        help="Dataset name. 'auto' uses the same lightweight detection as stress_test_variants.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where generated natural-redundancy files and the manifest will be written.",
    )
    parser.add_argument(
        "--redundancy-ratios",
        nargs="*",
        type=float,
        default=[0.1, 0.3, 0.5],
        help="Redundancy ratios rho where rho = N_redundant / (N_orig + N_redundant).",
    )
    parser.add_argument(
        "--redundancy-source",
        choices=["support_only", "all_context"],
        default="support_only",
        help="Where redundant passages should be derived from.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N records for each ratio. Use 0 to disable progress logging.",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "json", "jsonl"],
        default="auto",
        help="Output format. 'auto' mirrors the input format.",
    )
    return parser.parse_args()


def lower_first(text: str) -> str:
    if not text:
        return text
    return text[:1].lower() + text[1:]


def light_paraphrase_sentence(sentence: str) -> str:
    sentence = stv.normalize_text(sentence).strip()
    rewritten = stv.apply_synonym_rewrites(sentence)
    if rewritten != sentence:
        return rewritten

    for pattern, replacement in PHRASE_REWRITES:
        updated = pattern.sub(replacement, sentence, count=1)
        if updated != sentence:
            return updated

    if "," in sentence:
        head, tail = sentence.split(",", 1)
        tail = tail.strip()
        head = head.strip()
        if tail and head:
            return "%s, %s" % (tail[:1].upper() + tail[1:], head)

    return "In related wording, %s" % lower_first(sentence)


def trim_to_partial_sentence(sentence: str) -> str:
    sentence = stv.normalize_text(sentence).strip()
    for delimiter in ["; ", ", ", " and ", " which ", " that "]:
        if delimiter in sentence:
            fragment = sentence.split(delimiter, 1)[0].strip()
            if len(fragment.split()) >= 5:
                return fragment.rstrip(".") + "."

    words = sentence.split()
    if len(words) <= 8:
        return light_paraphrase_sentence(sentence)
    keep = max(6, int(round(len(words) * 0.65)))
    return " ".join(words[:keep]).rstrip(".,;:") + "."


def make_partial_overlap_copy(sentences: Sequence[str]) -> List[str]:
    clean_sentences = [stv.normalize_text(sentence).strip() for sentence in sentences if stv.normalize_text(sentence).strip()]
    if len(clean_sentences) >= 3:
        return [
            clean_sentences[0],
            light_paraphrase_sentence(clean_sentences[-1]),
        ]
    if len(clean_sentences) == 2:
        return [
            light_paraphrase_sentence(clean_sentences[0]),
            trim_to_partial_sentence(clean_sentences[1]),
        ]
    if len(clean_sentences) == 1:
        return [trim_to_partial_sentence(clean_sentences[0])]
    return []


def make_light_paraphrase(sentences: Sequence[str]) -> List[str]:
    clean_sentences = [stv.normalize_text(sentence).strip() for sentence in sentences if stv.normalize_text(sentence).strip()]
    paraphrased = [light_paraphrase_sentence(sentence) for sentence in clean_sentences]
    if len(paraphrased) > 1:
        paraphrased = paraphrased[1:] + paraphrased[:1]
    return paraphrased


def ensure_not_exact_copy(
    new_sentences: List[str],
    base_passage: Dict[str, Any],
) -> List[str]:
    if not new_sentences:
        new_sentences = list(base_passage["sentences"])

    new_text = " ".join(new_sentences).strip()
    if new_text != base_passage["text"]:
        return new_sentences

    updated = list(new_sentences)
    updated[0] = light_paraphrase_sentence(updated[0])
    if " ".join(updated).strip() != base_passage["text"]:
        return updated

    updated[0] = "In related wording, %s" % lower_first(updated[0])
    return updated


def make_natural_redundant_passage(
    base_passage: Dict[str, Any],
    duplicate_index: int,
    style: str,
) -> Dict[str, Any]:
    title = base_passage["title"]
    sentences = list(base_passage["sentences"])

    if style == "light_paraphrase":
        new_sentences = make_light_paraphrase(sentences)
    elif style == "partial_overlap":
        new_sentences = make_partial_overlap_copy(sentences)
    else:
        raise ValueError("Unsupported natural redundancy style: %s" % style)

    new_sentences = ensure_not_exact_copy(new_sentences, base_passage)
    return {
        "title": "%s [natural-redundant-%d]" % (title, duplicate_index),
        "sentences": new_sentences,
        "text": " ".join(new_sentences).strip(),
        "source_title": title,
        "redundancy_style": style,
    }


def inject_natural_redundancy(
    example: Dict[str, Any],
    ratio: float,
    rng: random.Random,
    source_mode: str,
) -> Dict[str, Any]:
    mutated = copy.deepcopy(example)
    original_context = [stv.normalize_context_entry(entry) for entry in stv.get_context_entries(example)]
    support_titles = stv.extract_support_titles(example)

    inject_count = stv.compute_injected_count(len(original_context), ratio)
    bases = stv.choose_redundancy_bases(original_context, support_titles, source_mode)
    if not bases:
        bases = list(original_context)

    redundant_passages = []
    styles = ["light_paraphrase", "partial_overlap"]
    for duplicate_index in range(inject_count):
        base = bases[duplicate_index % len(bases)]
        style = styles[duplicate_index % len(styles)]
        redundant_passages.append(
            make_natural_redundant_passage(base, duplicate_index + 1, style)
        )

    combined = list(original_context) + redundant_passages
    rng.shuffle(combined)

    provenance = []
    original_lookup = Counter((passage["title"], passage["text"]) for passage in original_context)
    used_original = Counter()
    redundant_lookup = {
        (passage["title"], passage["text"]): passage for passage in redundant_passages
    }

    for passage in combined:
        key = (passage["title"], passage["text"])
        if used_original[key] < original_lookup[key]:
            provenance.append({"source": "original", "variant": "clean"})
            used_original[key] += 1
        else:
            source_passage = redundant_lookup.get(key, passage)
            provenance.append(
                {
                    "source": "redundancy",
                    "variant": source_passage.get("redundancy_style", "natural_rewrite"),
                    "source_title": source_passage.get("source_title"),
                }
            )

    stv.set_context_entries(mutated, example, combined)
    mutated["stress_test_type"] = "natural_redundancy_injection"
    mutated["stress_test_family"] = "redundancy_injection"
    mutated["stress_test_level"] = ratio
    mutated["stress_test_metadata"] = {
        "ratio": ratio,
        "original_passages": len(original_context),
        "injected_passages": len(redundant_passages),
        "formula": "rho = N_redundant / (N_orig + N_redundant)",
        "source_mode": source_mode,
        "redundancy_strategy": "paraphrase_and_partial_overlap",
        "injected_titles": [item["title"] for item in redundant_passages],
        "source_titles": [item.get("source_title") for item in redundant_passages],
    }
    mutated["stress_test_passage_metadata"] = stv.build_passage_metadata(
        combined,
        support_titles=support_titles,
        provenance=provenance,
    )
    return mutated


def generate_natural_redundancy_variants(
    records: Sequence[Dict[str, Any]],
    dataset_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    output_format = stv.choose_output_format(args.output_format, args.input_format)

    manifest = {
        "dataset": dataset_name,
        "input_path": os.path.abspath(args.input),
        "record_count": len(records),
        "output_format": output_format,
        "variant_family": "natural_redundancy_injection",
        "natural_redundancy_variants": [],
    }

    progress_every = max(0, int(getattr(args, "progress_every", 0)))
    total_records = len(records)
    for ratio in args.redundancy_ratios:
        transformed = []
        local_rng = random.Random(rng.randint(0, 10 ** 9))
        if progress_every:
            print(
                "Starting natural redundancy ratio %.2f on %d records" % (ratio, total_records),
                flush=True,
            )
        for index, record in enumerate(records):
            record_number = index + 1
            if progress_every and (
                record_number == 1
                or record_number == total_records
                or record_number % progress_every == 0
            ):
                print(
                    "Natural redundancy ratio %.2f: processing record %d/%d, id=%s"
                    % (
                        ratio,
                        record_number,
                        total_records,
                        stv.get_example_id(record, index),
                    ),
                    flush=True,
                )
            transformed.append(
                inject_natural_redundancy(
                    example=record,
                    ratio=ratio,
                    rng=local_rng,
                    source_mode=args.redundancy_source,
                )
            )

        file_name = "%s_natural_redundancy_rho_%0.2f.%s" % (
            dataset_name,
            ratio,
            output_format,
        )
        output_path = os.path.join(args.output_dir, file_name)
        stv.save_records(output_path, transformed, output_format)
        if progress_every:
            print(
                "Finished natural redundancy ratio %.2f; wrote %s" % (ratio, os.path.abspath(output_path)),
                flush=True,
            )
        manifest["natural_redundancy_variants"].append(
            {
                "ratio": ratio,
                "path": os.path.abspath(output_path),
            }
        )

    manifest_path = os.path.join(args.output_dir, "%s_natural_redundancy_manifest.json" % dataset_name)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    args = parse_args()
    records, input_format = stv.load_records(args.input)
    if not records:
        raise ValueError("The input dataset is empty: %s" % args.input)

    args.input_format = input_format
    dataset_name = stv.normalize_dataset_name(args.dataset, records[0])
    manifest = generate_natural_redundancy_variants(records, dataset_name, args)

    print("Generated natural redundancy variants for %s" % dataset_name)
    print("Examples: %d" % manifest["record_count"])
    print("Output directory: %s" % os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
