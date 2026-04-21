#!/usr/bin/env python3
"""
Generate mixed noise passages for HotpotQA-style datasets.

The injected noise is split between two sources:
1. Cross-query duplicates: random documents copied from other examples.
2. Local syntax noise: documents from the current example with stronger word
   deletion, repeated swaps, text distortion, lexical modification, and
   occasional sentence-order changes.
3. Hybrid syntax/random-partial noise: syntax-modified local documents with
   more than half of the modified text deleted, then mixed with a partial
   document from another query.

The script mirrors the input/output conventions in stress_test_variants.py.

Usage:
    python3 mixed_noise_variants.py \
        --input dataset/hotpotqa/hotpot_dev_distractor_v1.json \
        --dataset hotpotqa \
        --output-dir generated_stress_tests/hotpotqa_mixed_noise
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import stress_test_variants as stv


YEAR_RE = re.compile(r"\b(1[5-9][0-9]{2}|20[0-9]{2})\b")
NUMBER_RE = re.compile(r"\b([2-9][0-9]?|[1-9][0-9]{2,})\b")
WORD_RE = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mixed HotpotQA noise from cross-query duplicates and local syntax edits."
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
        help="Directory where generated mixed-noise files and the manifest will be written.",
    )
    parser.add_argument(
        "--noise-ratios",
        nargs="*",
        type=float,
        default=[0.1, 0.3, 0.5],
        help="Noise ratios rho where rho = N_noise / (N_orig + N_noise).",
    )
    parser.add_argument(
        "--noise-source",
        choices=["mixed", "syntax_only", "syntax_random_partial"],
        default="mixed",
        help=(
            "Use 'mixed' for half cross-query duplicates and half syntax noise, "
            "'syntax_only' for all syntax noise, or 'syntax_random_partial' "
            "for syntax noise truncated and mixed with partial other-query docs."
        ),
    )
    parser.add_argument(
        "--syntax-source",
        choices=["support_only", "all_context"],
        default="all_context",
        help="Which current-query documents should be used as templates for syntax noise.",
    )
    parser.add_argument(
        "--syntax-operations",
        nargs="*",
        choices=["delete", "swap", "distortion", "modification"],
        default=["delete", "swap", "distortion", "modification"],
        help="Allowed local syntax-noise operations.",
    )
    parser.add_argument(
        "--syntax-edit-passes",
        type=int,
        default=3,
        help="Number of syntax operations to apply to each sentence in syntax-modified passages.",
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


def split_noise_count(total_count: int) -> Tuple[int, int]:
    """Return cross-query and local-syntax counts, keeping the split near half."""
    cross_query_count = total_count // 2
    syntax_count = total_count - cross_query_count
    return cross_query_count, syntax_count


def choose_noise_counts(total_count: int, noise_source: str) -> Tuple[int, int]:
    if noise_source in ["syntax_only", "syntax_random_partial"]:
        return 0, total_count
    if noise_source == "mixed":
        return split_noise_count(total_count)
    raise ValueError("Unsupported noise source: %s" % noise_source)


def clean_join_tokens(tokens: Sequence[str]) -> str:
    text = ""
    for token in tokens:
        if not token:
            continue
        if re.match(r"^[,.;:!?)]$", token):
            text = text.rstrip() + token
        elif token in ["'s", "n't"]:
            text = text.rstrip() + token
        elif token == "(":
            text += (" " if text and not text.endswith(" ") else "") + token
        else:
            text += (" " if text and not text.endswith((" ", "(")) else "") + token
    return text.strip()


def delete_words(sentence: str, rng: random.Random) -> Tuple[str, str]:
    tokens = WORD_RE.findall(stv.normalize_text(sentence))
    word_positions = [index for index, token in enumerate(tokens) if re.match(r"^[A-Za-z0-9]+$", token)]
    if len(word_positions) <= 4:
        return sentence, "delete_noop"

    delete_total = max(1, int(math.ceil(len(word_positions) * 0.35)))
    delete_total = min(delete_total, max(1, len(word_positions) - 3))
    to_delete = set(rng.sample(word_positions, delete_total))
    kept = [token for index, token in enumerate(tokens) if index not in to_delete]
    updated = clean_join_tokens(kept)
    return updated or sentence, "delete"


def swap_words(sentence: str, rng: random.Random) -> Tuple[str, str]:
    tokens = WORD_RE.findall(stv.normalize_text(sentence))
    word_positions = [index for index, token in enumerate(tokens) if re.match(r"^[A-Za-z0-9]+$", token)]
    if len(word_positions) < 2:
        return sentence, "swap_noop"

    swap_total = max(1, int(math.ceil(len(word_positions) * 0.18)))
    for _ in range(min(swap_total, len(word_positions) - 1)):
        left_position_index = rng.randrange(0, len(word_positions) - 1)
        left = word_positions[left_position_index]
        right = word_positions[left_position_index + 1]
        tokens[left], tokens[right] = tokens[right], tokens[left]
    updated = clean_join_tokens(tokens)
    return updated or sentence, "swap"


def distort_word(word: str, rng: random.Random) -> str:
    if len(word) <= 3:
        return word
    operation = rng.choice(["drop_char", "swap_char", "repeat_char"])
    chars = list(word)
    index = rng.randrange(1, len(chars) - 1)
    if operation == "drop_char":
        del chars[index]
    elif operation == "swap_char" and index + 1 < len(chars):
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
    else:
        chars.insert(index, chars[index])
    return "".join(chars)


def distort_sentence(sentence: str, rng: random.Random) -> Tuple[str, str]:
    tokens = WORD_RE.findall(stv.normalize_text(sentence))
    word_positions = [
        index
        for index, token in enumerate(tokens)
        if re.match(r"^[A-Za-z]{4,}$", token)
    ]
    if not word_positions:
        return sentence, "distortion_noop"

    distort_total = max(1, int(math.ceil(len(word_positions) * 0.28)))
    for index in rng.sample(word_positions, min(distort_total, len(word_positions))):
        tokens[index] = distort_word(tokens[index], rng)
    updated = clean_join_tokens(tokens)
    return updated or sentence, "distortion"


def modify_sentence(sentence: str, rng: random.Random) -> Tuple[str, str]:
    sentence = stv.normalize_text(sentence).strip()
    rewritten = stv.apply_synonym_rewrites(sentence)
    if rewritten != sentence:
        return rewritten, "modification_synonym"

    def shift_year(match: re.Match) -> str:
        year = int(match.group(1))
        return str(max(1000, year + rng.choice([-4, -2, 2, 4])))

    updated = YEAR_RE.sub(shift_year, sentence, count=1)
    if updated != sentence:
        return updated, "modification_year"

    def shift_number(match: re.Match) -> str:
        value = int(match.group(1))
        return str(max(0, value + rng.choice([-2, -1, 1, 2])))

    updated = NUMBER_RE.sub(shift_number, sentence, count=1)
    if updated != sentence:
        return updated, "modification_number"

    if sentence.endswith("."):
        return sentence[:-1] + " in related records.", "modification_append"
    return sentence + " in related records.", "modification_append"


def apply_syntax_operation(
    sentence: str,
    operation: str,
    rng: random.Random,
) -> Tuple[str, str]:
    if operation == "delete":
        return delete_words(sentence, rng)
    if operation == "swap":
        return swap_words(sentence, rng)
    if operation == "distortion":
        return distort_sentence(sentence, rng)
    if operation == "modification":
        return modify_sentence(sentence, rng)
    raise ValueError("Unsupported syntax operation: %s" % operation)


def choose_operation_sequence(
    operations: Sequence[str],
    edit_passes: int,
    rng: random.Random,
    offset: int,
) -> List[str]:
    available = list(operations)
    if not available:
        raise ValueError("At least one syntax operation must be provided.")

    rng.shuffle(available)
    passes = max(1, edit_passes)
    sequence = []
    for index in range(passes):
        if index < len(available):
            sequence.append(available[(index + offset) % len(available)])
        else:
            sequence.append(rng.choice(available))
    return sequence


def apply_significant_syntax_noise(
    sentence: str,
    operations: Sequence[str],
    edit_passes: int,
    rng: random.Random,
    offset: int,
) -> Tuple[str, List[str]]:
    updated = stv.normalize_text(sentence).strip()
    variants = []
    for operation in choose_operation_sequence(operations, edit_passes, rng, offset):
        candidate, variant = apply_syntax_operation(updated, operation, rng)
        variants.append(variant)
        if candidate != updated:
            updated = candidate
    return updated or sentence, variants


def choose_cross_query_duplicates(
    example: Dict[str, Any],
    pool: Sequence[Dict[str, Any]],
    count: int,
    rng: random.Random,
    example_index: int,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []

    current_example_id = stv.get_example_id(example, example_index)
    existing_titles = set(
        stv.normalize_context_entry(entry)["title"]
        for entry in stv.get_context_entries(example)
    )
    candidates = [
        candidate
        for candidate in pool
        if candidate.get("source_example_id") != current_example_id
        and candidate.get("title") not in existing_titles
    ]
    if not candidates:
        return []

    rng.shuffle(candidates)
    selected = []
    for index in range(count):
        candidate = copy.deepcopy(candidates[index % len(candidates)])
        candidate["noise_kind"] = "cross_query_duplicate"
        candidate["source_title"] = candidate.get("title")
        selected.append(candidate)
    return selected


def choose_random_partial_sources(
    example: Dict[str, Any],
    pool: Sequence[Dict[str, Any]],
    count: int,
    rng: random.Random,
    example_index: int,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []

    current_example_id = stv.get_example_id(example, example_index)
    existing_titles = set(
        stv.normalize_context_entry(entry)["title"]
        for entry in stv.get_context_entries(example)
    )
    candidates = [
        candidate
        for candidate in pool
        if candidate.get("source_example_id") != current_example_id
        and candidate.get("title") not in existing_titles
        and stv.normalize_text(candidate.get("text")).strip()
    ]
    if not candidates:
        return []

    rng.shuffle(candidates)
    selected = []
    for index in range(count):
        selected.append(copy.deepcopy(candidates[index % len(candidates)]))
    return selected


def choose_syntax_noise_bases(
    example: Dict[str, Any],
    count: int,
    rng: random.Random,
    source_mode: str,
    example_index: int,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []

    normalized_context = []
    for context_index, entry in enumerate(stv.get_context_entries(example)):
        normalized = stv.normalize_context_entry(entry)
        normalized["source_context_index"] = context_index
        normalized["source_example_id"] = stv.get_example_id(example, example_index)
        normalized_context.append(normalized)

    support_titles = set(stv.extract_support_titles(example))
    if source_mode == "support_only":
        bases = [item for item in normalized_context if item["title"] in support_titles]
        if not bases:
            bases = list(normalized_context)
    elif source_mode == "all_context":
        bases = list(normalized_context)
    else:
        raise ValueError("Unsupported syntax source: %s" % source_mode)

    if not bases:
        return []

    rng.shuffle(bases)
    selected = []
    for index in range(count):
        selected.append(copy.deepcopy(bases[index % len(bases)]))
    return selected


def truncate_to_less_than_half(text: str, rng: random.Random) -> Tuple[str, float]:
    tokens = WORD_RE.findall(stv.normalize_text(text))
    if not tokens:
        return "", 1.0

    keep_count = max(1, int(math.floor(len(tokens) * rng.uniform(0.30, 0.45))))
    keep_count = min(keep_count, max(1, (len(tokens) - 1) // 2))
    if keep_count >= len(tokens):
        keep_count = max(1, len(tokens) // 2)

    if len(tokens) <= keep_count:
        kept = tokens
    else:
        start = rng.randrange(0, len(tokens) - keep_count + 1)
        kept = tokens[start : start + keep_count]

    truncated = clean_join_tokens(kept)
    deleted_fraction = 1.0 - (float(len(kept)) / float(len(tokens)))
    return truncated, deleted_fraction


def make_partial_random_text(
    random_passage: Dict[str, Any],
    rng: random.Random,
) -> Tuple[str, float]:
    text = stv.normalize_text(random_passage.get("text")).strip()
    tokens = WORD_RE.findall(text)
    if not tokens:
        return "", 0.0

    keep_count = max(1, int(math.ceil(len(tokens) * rng.uniform(0.25, 0.45))))
    keep_count = min(keep_count, len(tokens))
    start = 0
    if len(tokens) > keep_count:
        start = rng.randrange(0, len(tokens) - keep_count + 1)
    kept = tokens[start : start + keep_count]
    return clean_join_tokens(kept), float(len(kept)) / float(len(tokens))


def make_syntax_noise_passage(
    base_passage: Dict[str, Any],
    noise_index: int,
    rng: random.Random,
    operations: Sequence[str],
    edit_passes: int,
) -> Dict[str, Any]:
    if not operations:
        raise ValueError("At least one syntax operation must be provided.")

    sentences = list(base_passage.get("sentences") or [])
    if not sentences and base_passage.get("text"):
        sentences = [base_passage["text"]]
    if not sentences:
        sentences = ["The source document contains no usable sentence text."]

    operation_plan = []
    changed_sentences = []
    for sentence_index, sentence in enumerate(sentences):
        updated, variants = apply_significant_syntax_noise(
            sentence=sentence,
            operations=operations,
            edit_passes=edit_passes,
            rng=rng,
            offset=noise_index + sentence_index - 1,
        )
        changed_sentences.append(updated)
        operation_plan.extend(variants)

    if len(changed_sentences) >= 3 and rng.random() < 0.65:
        left_index = rng.randrange(0, len(changed_sentences) - 1)
        right_index = left_index + 1
        changed_sentences[left_index], changed_sentences[right_index] = (
            changed_sentences[right_index],
            changed_sentences[left_index],
        )
        operation_plan.append("sentence_swap")

    original_text = " ".join(sentences).strip()
    changed_text = " ".join(changed_sentences).strip()
    if changed_text == original_text:
        changed_sentences[0] = "%s Syntax noise marker %d." % (
            changed_sentences[0].rstrip("."),
            noise_index,
        )
        changed_text = " ".join(changed_sentences).strip()
        operation_plan.append("fallback_marker")

    base_title = stv.normalize_text(base_passage.get("title")).strip() or "Untitled"
    return {
        "title": "%s [syntax-noise-%d]" % (base_title, noise_index),
        "sentences": changed_sentences,
        "text": changed_text,
        "source_title": base_title,
        "source_context_index": base_passage.get("source_context_index"),
        "source_example_id": base_passage.get("source_example_id"),
        "noise_kind": "syntax_modified",
        "syntax_operations": sorted(set(operation_plan)),
        "lexical_overlap_with_source": stv.jaccard_similarity(
            stv.unique_tokens(original_text),
            stv.unique_tokens(changed_text),
        ),
    }


def make_syntax_random_partial_passage(
    base_passage: Dict[str, Any],
    random_passage: Dict[str, Any],
    noise_index: int,
    rng: random.Random,
    operations: Sequence[str],
    edit_passes: int,
) -> Dict[str, Any]:
    syntax_passage = make_syntax_noise_passage(
        base_passage=base_passage,
        noise_index=noise_index,
        rng=rng,
        operations=operations,
        edit_passes=edit_passes,
    )
    syntax_text = syntax_passage["text"]
    truncated_syntax, deleted_fraction = truncate_to_less_than_half(syntax_text, rng)
    random_partial, random_partial_fraction = make_partial_random_text(random_passage, rng)

    combined_sentences = []
    if truncated_syntax:
        combined_sentences.append(truncated_syntax)
    if random_partial:
        combined_sentences.append(random_partial)
    if not combined_sentences:
        combined_sentences = ["The hybrid noise document could not be constructed from available text."]

    source_title = syntax_passage.get("source_title") or syntax_passage["title"]
    random_title = stv.normalize_text(random_passage.get("title")).strip()
    combined_text = " ".join(combined_sentences).strip()
    return {
        "title": "%s [syntax-random-partial-%d]" % (source_title, noise_index),
        "sentences": combined_sentences,
        "text": combined_text,
        "source_title": source_title,
        "source_context_index": syntax_passage.get("source_context_index"),
        "source_example_id": syntax_passage.get("source_example_id"),
        "noise_kind": "syntax_random_partial",
        "syntax_operations": sorted(
            set(
                list(syntax_passage.get("syntax_operations", []))
                + ["major_deletion", "random_partial_append"]
            )
        ),
        "syntax_deleted_fraction": deleted_fraction,
        "random_partial_title": random_title,
        "random_partial_source_example_id": random_passage.get("source_example_id"),
        "random_partial_fraction": random_partial_fraction,
        "lexical_overlap_with_source": stv.jaccard_similarity(
            stv.unique_tokens(stv.normalize_text(base_passage.get("text"))),
            stv.unique_tokens(combined_text),
        ),
    }


def inject_mixed_noise(
    example: Dict[str, Any],
    pool: Sequence[Dict[str, Any]],
    ratio: float,
    rng: random.Random,
    syntax_source: str,
    syntax_operations: Sequence[str],
    syntax_edit_passes: int,
    example_index: int,
    noise_source: str = "mixed",
) -> Dict[str, Any]:
    mutated = copy.deepcopy(example)
    original_context = [stv.normalize_context_entry(entry) for entry in stv.get_context_entries(example)]
    support_titles = stv.extract_support_titles(example)

    inject_count = stv.compute_injected_count(len(original_context), ratio)
    cross_count, syntax_count = choose_noise_counts(inject_count, noise_source)

    cross_query_passages = choose_cross_query_duplicates(
        example=example,
        pool=pool,
        count=cross_count,
        rng=rng,
        example_index=example_index,
    )
    syntax_count += cross_count - len(cross_query_passages)

    syntax_bases = choose_syntax_noise_bases(
        example=example,
        count=syntax_count,
        rng=rng,
        source_mode=syntax_source,
        example_index=example_index,
    )
    if noise_source == "syntax_random_partial":
        random_partial_sources = choose_random_partial_sources(
            example=example,
            pool=pool,
            count=len(syntax_bases),
            rng=rng,
            example_index=example_index,
        )
        if len(random_partial_sources) < len(syntax_bases):
            syntax_bases = syntax_bases[: len(random_partial_sources)]
        syntax_passages = [
            make_syntax_random_partial_passage(
                base_passage=base,
                random_passage=random_partial_sources[noise_index],
                noise_index=noise_index + 1,
                rng=rng,
                operations=syntax_operations,
                edit_passes=syntax_edit_passes,
            )
            for noise_index, base in enumerate(syntax_bases)
        ]
    else:
        syntax_passages = [
            make_syntax_noise_passage(
                base_passage=base,
                noise_index=noise_index + 1,
                rng=rng,
                operations=syntax_operations,
                edit_passes=syntax_edit_passes,
            )
            for noise_index, base in enumerate(syntax_bases)
        ]

    noise_passages = cross_query_passages + syntax_passages
    combined = list(original_context) + noise_passages
    rng.shuffle(combined)

    provenance = []
    original_lookup = Counter((passage["title"], passage["text"]) for passage in original_context)
    used_original = Counter()
    noise_lookup = {
        (passage["title"], passage["text"]): passage for passage in noise_passages
    }

    for passage in combined:
        key = (passage["title"], passage["text"])
        if used_original[key] < original_lookup[key]:
            provenance.append({"source": "original", "variant": "clean"})
            used_original[key] += 1
        else:
            source_passage = noise_lookup.get(key, passage)
            variant = source_passage.get("noise_kind", "mixed_noise")
            item = {
                "source": "noise",
                "variant": variant,
                "source_title": source_passage.get("source_title"),
                "source_example_id": source_passage.get("source_example_id"),
            }
            if variant in ["syntax_modified", "syntax_random_partial"]:
                item["syntax_operations"] = source_passage.get("syntax_operations", [])
                item["lexical_overlap_with_source"] = source_passage.get(
                    "lexical_overlap_with_source"
                )
            if variant == "syntax_random_partial":
                item["syntax_deleted_fraction"] = source_passage.get("syntax_deleted_fraction")
                item["random_partial_title"] = source_passage.get("random_partial_title")
                item["random_partial_source_example_id"] = source_passage.get(
                    "random_partial_source_example_id"
                )
                item["random_partial_fraction"] = source_passage.get("random_partial_fraction")
            provenance.append(item)

    stv.set_context_entries(mutated, example, combined)
    mutated["stress_test_type"] = "mixed_noise_injection"
    mutated["stress_test_family"] = "noise_injection"
    mutated["stress_test_level"] = ratio
    mutated["stress_test_metadata"] = {
        "ratio": ratio,
        "original_passages": len(original_context),
        "injected_passages": len(noise_passages),
        "formula": "rho = N_noise / (N_orig + N_noise)",
        "noise_source": noise_source,
        "noise_strategy": (
            "syntax_modified_major_deletion_plus_random_partial"
            if noise_source == "syntax_random_partial"
            else
            "all_local_syntax_noise"
            if noise_source == "syntax_only"
            else "half_cross_query_duplicates_half_local_syntax_noise"
        ),
        "requested_cross_query_passages": cross_count,
        "cross_query_passages": len(cross_query_passages),
        "syntax_modified_passages": len(syntax_passages),
        "syntax_random_partial_passages": (
            len(syntax_passages) if noise_source == "syntax_random_partial" else 0
        ),
        "major_deletion_rule": (
            "delete more than half of the syntax-modified text before appending partial random text"
            if noise_source == "syntax_random_partial"
            else None
        ),
        "syntax_source": syntax_source,
        "syntax_operations": list(syntax_operations),
        "syntax_edit_passes": max(1, syntax_edit_passes),
        "injected_titles": [item["title"] for item in noise_passages],
        "source_titles": [item.get("source_title") for item in noise_passages],
    }
    mutated["stress_test_passage_metadata"] = stv.build_passage_metadata(
        combined,
        support_titles=support_titles,
        provenance=provenance,
    )
    return mutated


def generate_mixed_noise_variants(
    records: Sequence[Dict[str, Any]],
    dataset_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    pool = stv.build_global_passage_pool(records)
    output_format = stv.choose_output_format(args.output_format, args.input_format)

    manifest = {
        "dataset": dataset_name,
        "input_path": os.path.abspath(args.input),
        "record_count": len(records),
        "global_passage_pool_size": len(pool),
        "output_format": output_format,
        "variant_family": "mixed_noise_injection",
        "mixed_noise_variants": [],
    }

    progress_every = max(0, int(getattr(args, "progress_every", 0)))
    total_records = len(records)
    for ratio in args.noise_ratios:
        transformed = []
        local_rng = random.Random(rng.randint(0, 10 ** 9))
        if progress_every:
            print(
                "Starting mixed noise ratio %.2f on %d records" % (ratio, total_records),
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
                    "Mixed noise ratio %.2f: processing record %d/%d, id=%s"
                    % (
                        ratio,
                        record_number,
                        total_records,
                        stv.get_example_id(record, index),
                    ),
                    flush=True,
                )
            transformed.append(
                inject_mixed_noise(
                    example=record,
                    pool=pool,
                    ratio=ratio,
                    rng=local_rng,
                    syntax_source=args.syntax_source,
                    syntax_operations=args.syntax_operations,
                    syntax_edit_passes=getattr(args, "syntax_edit_passes", 3),
                    example_index=index,
                    noise_source=getattr(args, "noise_source", "mixed"),
                )
            )

        file_name = "%s_mixed_noise_rho_%0.2f.%s" % (
            dataset_name,
            ratio,
            output_format,
        )
        output_path = os.path.join(args.output_dir, file_name)
        stv.save_records(output_path, transformed, output_format)
        if progress_every:
            print(
                "Finished mixed noise ratio %.2f; wrote %s"
                % (ratio, os.path.abspath(output_path)),
                flush=True,
            )
        manifest["mixed_noise_variants"].append(
            {
                "ratio": ratio,
                "path": os.path.abspath(output_path),
            }
        )

    manifest_path = os.path.join(args.output_dir, "%s_mixed_noise_manifest.json" % dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    args = parse_args()
    records, input_format = stv.load_records(args.input)
    if not records:
        raise ValueError("The input dataset is empty: %s" % args.input)
    if not args.syntax_operations:
        raise ValueError("At least one syntax operation is required.")
    if args.syntax_edit_passes < 1:
        raise ValueError("--syntax-edit-passes must be at least 1.")

    args.input_format = input_format
    dataset_name = stv.normalize_dataset_name(args.dataset, records[0])
    manifest = generate_mixed_noise_variants(records, dataset_name, args)

    print("Generated mixed noise variants for %s" % dataset_name)
    print("Examples: %d" % manifest["record_count"])
    print("Global passage pool: %d" % manifest["global_passage_pool_size"])
    print("Output directory: %s" % os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
