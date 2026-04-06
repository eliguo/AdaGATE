#!/usr/bin/env python3
"""
Generate section 4.2 stress-test variants of evidence pools for
HotpotQA and 2WikiMultiHopQA.

The script creates three families of artificial evidence pools:
1. Noise injection
2. Redundancy injection
3. Position perturbations

It is designed to work on HotpotQA-style and 2WikiMultiHopQA-style
examples that expose:
  - question
  - answer (optional but helpful for off-topic filtering)
  - context
  - supporting_facts

The output preserves the original example fields and adds:
  - stress_test_type
  - stress_test_level
  - stress_test_metadata
  - stress_test_passage_metadata

Usage:
    python3 stress_test_variants.py \
        --input hotpot_dev_distractor_v1.json \
        --dataset hotpotqa \
        --output-dir outputs/hotpotqa
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
from typing import Any, Dict, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


SYNONYM_MAP = {
    "also": "as well",
    "american": "u.s.",
    "artist": "performer",
    "author": "writer",
    "became": "later became",
    "began": "started",
    "best": "strongest",
    "born": "was born",
    "british": "uk",
    "called": "known as",
    "city": "municipality",
    "country": "nation",
    "created": "developed",
    "director": "filmmaker",
    "famous": "well-known",
    "founded": "established",
    "is": "is currently",
    "located": "situated",
    "made": "produced",
    "member": "participant",
    "movie": "film",
    "named": "titled",
    "novel": "book",
    "part": "portion",
    "released": "issued",
    "river": "waterway",
    "served": "worked",
    "singer": "vocalist",
    "stars": "features",
    "television": "tv",
    "university": "college",
    "village": "settlement",
    "was": "was previously",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stress-test evidence pools for HotpotQA and 2WikiMultiHopQA."
    )
    parser.add_argument("--input", required=True, help="Path to the source dataset JSON/JSONL file.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["hotpotqa", "2wikimultihopqa", "auto"],
        help="Dataset name. 'auto' uses lightweight schema detection.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where generated variant files and the manifest will be written.",
    )
    parser.add_argument(
        "--noise-ratios",
        nargs="*",
        type=float,
        default=[0.1, 0.3, 0.5],
        help="Noise ratios rho where rho = N_noise / (N_orig + N_noise).",
    )
    parser.add_argument(
        "--redundancy-ratios",
        nargs="*",
        type=float,
        default=[0.1, 0.3, 0.5],
        help="Redundancy ratios rho where rho = N_redundant / (N_orig + N_redundant).",
    )
    parser.add_argument(
        "--position-modes",
        nargs="*",
        default=["shuffle_all", "support_front", "support_middle", "support_back"],
        choices=["shuffle_all", "support_front", "support_middle", "support_back", "reverse"],
        help="Position perturbation modes to generate.",
    )
    parser.add_argument(
        "--noise-overlap-threshold",
        type=float,
        default=0.10,
        help="Maximum lexical overlap allowed between injected noise and the question/answer text.",
    )
    parser.add_argument(
        "--redundancy-source",
        choices=["support_only", "all_context"],
        default="support_only",
        help="Where redundant passages should be copied from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "json", "jsonl"],
        default="auto",
        help="Output format. 'auto' mirrors the input format.",
    )
    return parser.parse_args()


def load_records(path: str) -> Tuple[List[Dict[str, Any]], str]:
    with open(path, "r", encoding="utf-8") as handle:
        prefix = handle.read(1)
        while prefix and prefix.isspace():
            prefix = handle.read(1)
        handle.seek(0)
        if prefix == "[":
            data = json.load(handle)
            return data, "json"

        records = []
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records, "jsonl"


def save_records(path: str, records: Sequence[Dict[str, Any]], fmt: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        if fmt == "json":
            json.dump(list(records), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        elif fmt == "jsonl":
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        else:
            raise ValueError("Unsupported output format: %s" % fmt)


def normalize_dataset_name(name: str, sample: Dict[str, Any]) -> str:
    if name != "auto":
        return name
    if "supporting_facts" in sample and "context" in sample:
        return "hotpotqa"
    if "ctxs" in sample:
        return "hotpotqa"
    return "2wikimultihopqa"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(normalize_text(text))]


def unique_tokens(text: str) -> set:
    return set(tokenize(text))


def jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def normalize_context_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, dict):
        title = normalize_text(entry.get("title") or entry.get("page_title") or entry.get("name"))
        if "sentences" in entry and isinstance(entry["sentences"], list):
            sentences = [normalize_text(sentence) for sentence in entry["sentences"]]
        elif "text" in entry:
            raw_text = entry["text"]
            if isinstance(raw_text, list):
                sentences = [normalize_text(sentence) for sentence in raw_text]
            else:
                sentences = [normalize_text(raw_text)]
        else:
            sentences = []
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        title = normalize_text(entry[0])
        raw_sentences = entry[1]
        if isinstance(raw_sentences, list):
            sentences = [normalize_text(sentence) for sentence in raw_sentences]
        else:
            sentences = [normalize_text(raw_sentences)]
    else:
        title = ""
        sentences = [normalize_text(entry)]

    sentences = [sentence.strip() for sentence in sentences if normalize_text(sentence).strip()]
    return {
        "title": title.strip(),
        "sentences": sentences,
        "text": " ".join(sentences).strip(),
    }


def get_context_field_name(example: Dict[str, Any]) -> str:
    if "context" in example:
        return "context"
    if "ctxs" in example:
        return "ctxs"
    return "context"


def get_context_entries(example: Dict[str, Any]) -> List[Any]:
    return list(example.get(get_context_field_name(example), []) or [])


def get_answer_text(example: Dict[str, Any]) -> str:
    if "answer" in example:
        return normalize_text(example.get("answer"))
    answers = example.get("answers")
    if isinstance(answers, list) and answers:
        return normalize_text(answers[0])
    return ""


def denormalize_context_entry(entry: Dict[str, Any], output_field: str) -> Any:
    if output_field == "ctxs":
        return {
            "title": entry["title"],
            "text": entry["text"],
        }
    return [entry["title"], list(entry["sentences"])]


def set_context_entries(mutated: Dict[str, Any], original_example: Dict[str, Any], entries: Sequence[Dict[str, Any]]) -> None:
    output_field = get_context_field_name(original_example)
    mutated[output_field] = [denormalize_context_entry(item, output_field) for item in entries]
    alternate_field = "ctxs" if output_field == "context" else "context"
    if alternate_field in mutated:
        del mutated[alternate_field]


def extract_support_titles(example: Dict[str, Any]) -> List[str]:
    titles = []
    for fact in example.get("supporting_facts", []) or []:
        if isinstance(fact, dict):
            title = normalize_text(fact.get("title"))
        elif isinstance(fact, (list, tuple)) and fact:
            title = normalize_text(fact[0])
        else:
            title = ""
        if title:
            titles.append(title)

    if titles:
        return sorted(set(titles))

    if "supporting_context" in example and isinstance(example["supporting_context"], list):
        inferred = []
        for item in example["supporting_context"]:
            normalized = normalize_context_entry(item)
            if normalized["title"]:
                inferred.append(normalized["title"])
        return sorted(set(inferred))

    return []


def get_example_id(example: Dict[str, Any], fallback_index: int) -> str:
    for key in ("_id", "id", "qid", "question_id"):
        if key in example:
            return normalize_text(example[key])
    return "example_%06d" % fallback_index


def compute_injected_count(original_size: int, ratio: float) -> int:
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError("Ratios must satisfy 0 <= rho < 1.0, got %.4f" % ratio)
    if original_size <= 0 or ratio == 0.0:
        return 0
    return int(math.ceil((ratio * float(original_size)) / (1.0 - ratio)))


def build_global_passage_pool(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pool = []
    seen = set()

    for example_index, example in enumerate(records):
        for context_index, entry in enumerate(get_context_entries(example)):
            normalized = normalize_context_entry(entry)
            if not normalized["text"]:
                continue
            key = (normalized["title"].lower(), normalized["text"].lower())
            if key in seen:
                continue
            seen.add(key)
            pool.append(
                {
                    "title": normalized["title"],
                    "sentences": normalized["sentences"],
                    "text": normalized["text"],
                    "source_example_id": get_example_id(example, example_index),
                    "source_context_index": context_index,
                    "token_set": unique_tokens(normalized["title"] + " " + normalized["text"]),
                }
            )
    return pool


def build_passage_metadata(
    passages: Sequence[Dict[str, Any]],
    support_titles: Sequence[str],
    provenance: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    support_title_set = set(support_titles)
    metadata = []
    for index, passage in enumerate(passages):
        item = {
            "position": index,
            "title": passage["title"],
            "is_supporting_title": passage["title"] in support_title_set,
        }
        if provenance is not None and index < len(provenance):
            item.update(provenance[index])
        metadata.append(item)
    return metadata


def choose_noise_candidates(
    example: Dict[str, Any],
    pool: Sequence[Dict[str, Any]],
    count: int,
    rng: random.Random,
    overlap_threshold: float,
    example_index: int,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []

    existing_titles = set()
    existing_titles.update(extract_support_titles(example))
    for context_entry in get_context_entries(example):
        existing_titles.add(normalize_context_entry(context_entry)["title"])

    text_anchor = " ".join(
        [
            normalize_text(example.get("question")),
            get_answer_text(example),
        ]
    )
    anchor_tokens = unique_tokens(text_anchor)
    current_example_id = get_example_id(example, example_index)

    strict = []
    relaxed = []
    for candidate in pool:
        if candidate["source_example_id"] == current_example_id:
            continue
        if candidate["title"] in existing_titles:
            continue
        overlap = jaccard_similarity(anchor_tokens, candidate["token_set"])
        if overlap <= overlap_threshold:
            strict.append(candidate)
        else:
            relaxed.append(candidate)

    rng.shuffle(strict)
    rng.shuffle(relaxed)

    selected = strict[:count]
    if len(selected) < count:
        needed = count - len(selected)
        selected.extend(relaxed[:needed])
    return [copy.deepcopy(item) for item in selected]


def apply_synonym_rewrites(text: str) -> str:
    words = text.split()
    rewritten = []
    changes = 0
    for word in words:
        match = re.match(r"^([A-Za-z]+)([^A-Za-z]*)$", word)
        if not match:
            rewritten.append(word)
            continue
        core = match.group(1)
        suffix = match.group(2)
        replacement = SYNONYM_MAP.get(core.lower())
        if replacement is None:
            rewritten.append(word)
            continue
        changes += 1
        if core[0].isupper():
            replacement = replacement.capitalize()
        rewritten.append(replacement + suffix)
    if changes == 0:
        return text
    return " ".join(rewritten)


def make_redundant_passage(
    base_passage: Dict[str, Any],
    duplicate_index: int,
    style: str,
) -> Dict[str, Any]:
    title = base_passage["title"]
    sentences = list(base_passage["sentences"])

    if style == "exact_duplicate":
        new_sentences = list(sentences)
    elif style == "near_duplicate":
        new_sentences = []
        for sentence in sentences:
            new_sentence = apply_synonym_rewrites(sentence)
            if new_sentence == sentence and len(sentences) > 1:
                new_sentence = sentence
            new_sentences.append(new_sentence)
        if len(new_sentences) > 1:
            new_sentences = new_sentences[1:] + new_sentences[:1]
    elif style == "overlapping_window":
        if len(sentences) <= 1:
            new_sentences = list(sentences)
        else:
            new_sentences = list(sentences[1:])
    else:
        raise ValueError("Unsupported redundancy style: %s" % style)

    if not new_sentences:
        new_sentences = list(sentences)

    return {
        "title": "%s [redundant-%d]" % (title, duplicate_index),
        "sentences": new_sentences,
        "text": " ".join(new_sentences).strip(),
    }


def choose_redundancy_bases(
    normalized_context: Sequence[Dict[str, Any]],
    support_titles: Sequence[str],
    source_mode: str,
) -> List[Dict[str, Any]]:
    if source_mode == "support_only":
        support_set = set(support_titles)
        bases = [passage for passage in normalized_context if passage["title"] in support_set]
        if bases:
            return bases
    return list(normalized_context)


def inject_noise(
    example: Dict[str, Any],
    pool: Sequence[Dict[str, Any]],
    ratio: float,
    rng: random.Random,
    overlap_threshold: float,
    example_index: int,
) -> Dict[str, Any]:
    mutated = copy.deepcopy(example)
    original_context = [normalize_context_entry(entry) for entry in get_context_entries(example)]
    support_titles = extract_support_titles(example)

    inject_count = compute_injected_count(len(original_context), ratio)
    noise_passages = choose_noise_candidates(
        example=example,
        pool=pool,
        count=inject_count,
        rng=rng,
        overlap_threshold=overlap_threshold,
        example_index=example_index,
    )

    combined = list(original_context) + noise_passages
    rng.shuffle(combined)

    provenance = []
    original_lookup = Counter(
        (passage["title"], passage["text"]) for passage in original_context
    )
    used_original = Counter()

    for passage in combined:
        key = (passage["title"], passage["text"])
        if used_original[key] < original_lookup[key]:
            provenance.append({"source": "original", "variant": "clean"})
            used_original[key] += 1
        else:
            provenance.append(
                {
                    "source": "noise",
                    "variant": "off_topic",
                    "source_example_id": passage.get("source_example_id"),
                }
            )

    set_context_entries(mutated, example, combined)
    mutated["stress_test_type"] = "noise_injection"
    mutated["stress_test_level"] = ratio
    mutated["stress_test_metadata"] = {
        "ratio": ratio,
        "original_passages": len(original_context),
        "injected_passages": len(noise_passages),
        "formula": "rho = N_noise / (N_orig + N_noise)",
        "injected_titles": [item["title"] for item in noise_passages],
    }
    mutated["stress_test_passage_metadata"] = build_passage_metadata(
        combined,
        support_titles=support_titles,
        provenance=provenance,
    )
    return mutated


def inject_redundancy(
    example: Dict[str, Any],
    ratio: float,
    rng: random.Random,
    source_mode: str,
) -> Dict[str, Any]:
    mutated = copy.deepcopy(example)
    original_context = [normalize_context_entry(entry) for entry in get_context_entries(example)]
    support_titles = extract_support_titles(example)

    inject_count = compute_injected_count(len(original_context), ratio)
    bases = choose_redundancy_bases(original_context, support_titles, source_mode)
    if not bases:
        bases = list(original_context)

    redundant_passages = []
    styles = ["exact_duplicate", "near_duplicate", "overlapping_window"]
    for duplicate_index in range(inject_count):
        base = bases[duplicate_index % len(bases)]
        style = styles[duplicate_index % len(styles)]
        redundant_passages.append(make_redundant_passage(base, duplicate_index + 1, style))

    combined = list(original_context) + redundant_passages
    rng.shuffle(combined)

    provenance = []
    original_lookup = Counter(
        (passage["title"], passage["text"]) for passage in original_context
    )
    used_original = Counter()
    redundant_lookup = {}
    for redundant_index, passage in enumerate(redundant_passages):
        redundant_lookup[(passage["title"], passage["text"])] = styles[redundant_index % len(styles)]

    for passage in combined:
        key = (passage["title"], passage["text"])
        if used_original[key] < original_lookup[key]:
            provenance.append({"source": "original", "variant": "clean"})
            used_original[key] += 1
        else:
            provenance.append(
                {
                    "source": "redundancy",
                    "variant": redundant_lookup.get(key, "duplicate"),
                }
            )

    set_context_entries(mutated, example, combined)
    mutated["stress_test_type"] = "redundancy_injection"
    mutated["stress_test_level"] = ratio
    mutated["stress_test_metadata"] = {
        "ratio": ratio,
        "original_passages": len(original_context),
        "injected_passages": len(redundant_passages),
        "formula": "rho = N_redundant / (N_orig + N_redundant)",
        "source_mode": source_mode,
        "injected_titles": [item["title"] for item in redundant_passages],
    }
    mutated["stress_test_passage_metadata"] = build_passage_metadata(
        combined,
        support_titles=support_titles,
        provenance=provenance,
    )
    return mutated


def reorder_for_position_mode(
    normalized_context: Sequence[Dict[str, Any]],
    support_titles: Sequence[str],
    mode: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    support_set = set(support_titles)
    support_passages = [passage for passage in normalized_context if passage["title"] in support_set]
    other_passages = [passage for passage in normalized_context if passage["title"] not in support_set]

    if mode == "shuffle_all":
        combined = list(normalized_context)
        rng.shuffle(combined)
        return combined
    if mode == "reverse":
        return list(reversed(normalized_context))
    if mode == "support_front":
        rng.shuffle(support_passages)
        rng.shuffle(other_passages)
        return support_passages + other_passages
    if mode == "support_back":
        rng.shuffle(support_passages)
        rng.shuffle(other_passages)
        return other_passages + support_passages
    if mode == "support_middle":
        rng.shuffle(support_passages)
        rng.shuffle(other_passages)
        midpoint = len(other_passages) // 2
        return other_passages[:midpoint] + support_passages + other_passages[midpoint:]
    raise ValueError("Unsupported position mode: %s" % mode)


def perturb_positions(
    example: Dict[str, Any],
    mode: str,
    rng: random.Random,
) -> Dict[str, Any]:
    mutated = copy.deepcopy(example)
    original_context = [normalize_context_entry(entry) for entry in get_context_entries(example)]
    support_titles = extract_support_titles(example)

    reordered = reorder_for_position_mode(
        normalized_context=original_context,
        support_titles=support_titles,
        mode=mode,
        rng=rng,
    )

    original_positions = {}
    for index, passage in enumerate(original_context):
        key = (passage["title"], passage["text"])
        original_positions.setdefault(key, []).append(index)

    consumed = Counter()
    provenance = []
    for index, passage in enumerate(reordered):
        key = (passage["title"], passage["text"])
        original_index_list = original_positions.get(key, [])
        source_index = original_index_list[consumed[key]] if consumed[key] < len(original_index_list) else None
        consumed[key] += 1
        provenance.append(
            {
                "source": "original",
                "variant": "reordered",
                "original_position": source_index,
                "new_position": index,
            }
        )

    set_context_entries(mutated, example, reordered)
    mutated["stress_test_type"] = "position_perturbation"
    mutated["stress_test_level"] = mode
    mutated["stress_test_metadata"] = {
        "mode": mode,
        "support_titles": support_titles,
        "original_passages": len(original_context),
    }
    mutated["stress_test_passage_metadata"] = build_passage_metadata(
        reordered,
        support_titles=support_titles,
        provenance=provenance,
    )
    return mutated


def choose_output_format(requested_format: str, input_format: str) -> str:
    if requested_format == "auto":
        return input_format
    return requested_format


def generate_variants(
    records: Sequence[Dict[str, Any]],
    dataset_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    pool = build_global_passage_pool(records)
    output_format = choose_output_format(args.output_format, args.input_format)

    manifest = {
        "dataset": dataset_name,
        "input_path": os.path.abspath(args.input),
        "record_count": len(records),
        "global_passage_pool_size": len(pool),
        "output_format": output_format,
        "noise_variants": [],
        "redundancy_variants": [],
        "position_variants": [],
    }

    for ratio in args.noise_ratios:
        transformed = []
        local_rng = random.Random(rng.randint(0, 10 ** 9))
        for index, record in enumerate(records):
            transformed.append(
                inject_noise(
                    example=record,
                    pool=pool,
                    ratio=ratio,
                    rng=local_rng,
                    overlap_threshold=args.noise_overlap_threshold,
                    example_index=index,
                )
            )
        file_name = "%s_noise_rho_%0.2f.%s" % (
            dataset_name,
            ratio,
            output_format,
        )
        output_path = os.path.join(args.output_dir, file_name)
        save_records(output_path, transformed, output_format)
        manifest["noise_variants"].append(
            {
                "ratio": ratio,
                "path": os.path.abspath(output_path),
            }
        )

    for ratio in args.redundancy_ratios:
        transformed = []
        local_rng = random.Random(rng.randint(0, 10 ** 9))
        for record in records:
            transformed.append(
                inject_redundancy(
                    example=record,
                    ratio=ratio,
                    rng=local_rng,
                    source_mode=args.redundancy_source,
                )
            )
        file_name = "%s_redundancy_rho_%0.2f.%s" % (
            dataset_name,
            ratio,
            output_format,
        )
        output_path = os.path.join(args.output_dir, file_name)
        save_records(output_path, transformed, output_format)
        manifest["redundancy_variants"].append(
            {
                "ratio": ratio,
                "path": os.path.abspath(output_path),
            }
        )

    for mode in args.position_modes:
        transformed = []
        local_rng = random.Random(rng.randint(0, 10 ** 9))
        for record in records:
            transformed.append(
                perturb_positions(
                    example=record,
                    mode=mode,
                    rng=local_rng,
                )
            )
        file_name = "%s_position_%s.%s" % (
            dataset_name,
            mode,
            output_format,
        )
        output_path = os.path.join(args.output_dir, file_name)
        save_records(output_path, transformed, output_format)
        manifest["position_variants"].append(
            {
                "mode": mode,
                "path": os.path.abspath(output_path),
            }
        )

    manifest_path = os.path.join(args.output_dir, "%s_manifest.json" % dataset_name)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    args = parse_args()
    records, input_format = load_records(args.input)
    if not records:
        raise ValueError("The input dataset is empty: %s" % args.input)

    args.input_format = input_format
    dataset_name = normalize_dataset_name(args.dataset, records[0])
    manifest = generate_variants(records, dataset_name, args)

    print("Generated stress-test variants for %s" % dataset_name)
    print("Examples: %d" % manifest["record_count"])
    print("Global passage pool: %d" % manifest["global_passage_pool_size"])
    print("Output directory: %s" % os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
