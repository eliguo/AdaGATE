# AdaGATE: Adaptive Gap-Aware Token-Efficient Evidence Assembly

AdaGATE is a **training-free evidence controller** for multi-hop Retrieval-Augmented Generation (RAG). It extends SEAL-RAG-style gap-aware repair to a **token-budgeted** setting and is designed to sit on top of any black-box LLM and dense retriever.

Instead of simply taking the top-k retrieved passages or blindly adding more context, AdaGATE:

- Maintains an **entity-centric ledger** and explicit **gap specification** over the current evidence set  
- Issues **gap-targeted micro-queries** to repair missing bridge facts and qualifiers  
- Scores candidates with a **utility function** that balances gap coverage, corroboration, novelty, redundancy, and question-aware relevance  
- Performs **replacement-based updates** under a global **token budget**, with an **Adaptive-k style capacity heuristic** over utility scores

The goal is to assemble a small, high-precision evidence set that is sufficient for multi-hop reasoning while respecting realistic deployment constraints (latency, API cost, and context length).

---

## 1. Problem Setup

Given:

- A query `q`
- A corpus `D` indexed by a dense (or hybrid) retriever
- A global evidence token budget `B`
- A generator LLM treated as a black box (API model or open-weight model)

We first retrieve a candidate pool:

- `C = {c₁, …, c_N}` passages from `D`, each with token length `len(c)`

AdaGATE must construct an evidence set `E ⊆ C` such that:

- The **token constraint** holds:  
  `∑_{c ∈ E} len(c) ≤ B`
- `E` contains the **bridge entities, relations, and qualifiers** needed for multi-hop reasoning
- `E` avoids **redundant** and **off-topic** passages that waste tokens and dilute context

AdaGATE runs a bounded number of **Search → Extract → Assess → Replace** loops to iteratively repair `E` under this budget.

---

## 2. High-Level Pipeline

The controller loop can be summarized as:

1. **Initial Retrieval**
   - Use the base retriever on `q` to obtain an initial candidate pool `C₀`
   - Build an initial evidence set `E₀` by greedily adding top-similarity passages until the token budget `B` is reached

2. **Extract (Ledger + Gaps)**
   - Run an IE / LLM extraction prompt on `(q, E_t)` to build:
     - An **entity–relation ledger** `U_t` (supported facts with confidence)
     - A **gap set** `G_t` describing missing entities/relations/values needed to answer `q`

3. **Assess Sufficiency**
   - If `G_t` is empty, or only contains very abstract, non-repairable gaps, or no candidate can improve utility under the budget, **stop** and return `E_t`
   - Otherwise, proceed to targeted search

4. **Search (Micro-Queries)**
   - From `G_t`, generate **gap-anchored micro-queries**
   - From `q`, generate a small set of **question-anchored queries** (fallback path)
   - Retrieve a new candidate pool `C_t` using these micro-queries

5. **Score (Utility + Adaptive Capacity)**
   - For each candidate `c ∈ C_t`, compute a **utility score** `S_t(c)` that combines:
     - Gap coverage
     - Corroboration
     - Non-lateral novelty
     - Redundancy penalty
     - Question-aware relevance
   - Sort candidates by utility and apply a **similarity-gap heuristic** (inspired by Adaptive-k) over the utility scores to estimate an **effective capacity** `K_eff` (how many candidates are truly “worth considering”)

6. **Replace Under Token Budget**
   - Among current evidence `E_t`, identify the **lowest-utility** passage `v_t`
   - Among top-utility candidates (up to `K_eff`), identify the **highest-utility** candidate `c*_t`
   - If replacing `v_t` with `c*_t`:
     - (a) improves utility by at least a small margin `ε`, and  
     - (b) keeps total tokens within `B`,  
     then perform the swap: `E_{t+1} = (E_t \ {v_t}) ∪ {c*_t}`
   - Protect newly inserted passages for one iteration (dwell-time guard), then go back to Extract

After a small number of iterations, the final `E_T` is concatenated with `q` and fed to the LLM.

---

## 3. Core Components

### 3.1 Entity Ledger and Gap Specification

AdaGATE reuses SEAL-RAG style primitives for **structured reasoning state**:

- **Ledger `U_t`**  
  Extracted from `(q, E_t)` using an IE / LLM prompt.  
  Contains tuples `(entity, relation, value, confidence)` describing what `E_t` currently supports.

- **Gap set `G_t`**  
  Computed from `(q, U_t)` via another prompt.  
  Each gap `g ∈ G_t` describes **missing information** needed to answer `q`, e.g.:
  - “founding date of Organization X”
  - “birthplace of Person Y”
  - “relation between Entity A and Entity B”

AdaGATE does **not** modify the internal architecture of these extractors; it uses them as black-box primitives.

---

### 3.2 Micro-Query Generation

AdaGATE combines two types of micro-queries:

1. **Gap-anchored queries**  
   For each gap `g ∈ G_t`, generate one or more “atomic” queries, such as:
   - `"founding date of Organization X"`
   - `"city where Person Y was born"`

2. **Question-anchored queries**  
   A small set of rewrites or keyword subqueries derived directly from `q`, independent of `G_t`.

All micro-queries are sent to the retriever, yielding a candidate pool `C_t`.  
Gap-anchored queries target **bridge facts**, while question-anchored queries provide a **fallback** when gap extraction is noisy or underspecified.

---

### 3.3 Utility Function

For each candidate passage `c ∈ C_t`, AdaGATE computes a utility score:

```text
S_t(c) = λ₁ · GapCov(c, G_t)
       + λ₂ · Corr(c, U_t)
       + λ₃ · Nov(c, U_t)
       – λ₄ · Red(c, E_t)
       + λ₅ · Rel_Q(c, q)
```

Where:

- `GapCov(c, G_t)`  
  How well `c` **covers missing gaps** (`G_t`), using soft indicators that `c` contains the requested entity/attribute/relation.

- `Corr(c, U_t)`  
  How much `c` **corroborates uncertain facts** in the ledger (e.g., reinforces low-confidence triples).

- `Nov(c, U_t)`  
  **Non-lateral novelty**: does `c` introduce new entities/relations not in `U_t`, rather than just paraphrasing existing facts?

- `Red(c, E_t)`  
  **Local redundancy penalty**: similarity between `c` and existing passages in `E_t`, discouraging near duplicates.

- `Rel_Q(c, q)`  
  **Question-aware relevance**: direct relevance of `c` to `q` (e.g., retriever similarity or cross-encoder score).  
  This term serves as a **fallback** when gap extraction is imperfect, preventing the controller from chasing spurious gaps.

The weights `λ₁,…,λ₅` can be tuned per query type (e.g., more gap-focused for structured factoid questions, more `Rel_Q`-heavy for abstract or opinion-like queries).

---

### 3.4 Token-Budgeted Selection and Replacement

We enforce a **token-level budget**:

```text
∑_{c ∈ E_t} len(c) ≤ B
```

This is more realistic than a fixed number of passages `k`, because chunk lengths vary significantly in practice.

AdaGATE conceptually optimizes a set-level objective:

```text
F(E) = α_rel · R(E, q) – β · Redundancy(E)
```

- `R(E, q)` aggregates gap-aware and question-aware relevance over all `c ∈ E`
- `Redundancy(E)` measures pairwise similarity within `E` (inspired by AdaGReS)

Directly maximizing `F(E)` is combinatorial, so AdaGATE uses `S_t(c)` as a **surrogate for marginal gain** and performs **local swaps**:

1. Find lowest-utility passage in current evidence:  
   `v_t = argmin_{e ∈ E_t} S_t(e)`
2. Find highest-utility candidate in `C_t`:  
   `c*_t = argmax_{c ∈ C_t} S_t(c)`
3. Replace `v_t` with `c*_t` if:
   - `S_t(c*_t) > S_t(v_t) + ε` (utility gain), and  
   - New total tokens stay within budget `B`

This yields a **monotonic increase in information density** while respecting the token constraint.

---

### 3.5 Utility-Adaptive Capacity (Adaptive-k Style)

Even with a budget, naively filling all available tokens can reintroduce **context dilution**. AdaGATE borrows the **largest-gap** idea from Adaptive-k, but applies it to the **utility scores** instead of raw similarity:

1. Sort utilities:  
   `S_t^{(1)} ≥ S_t^{(2)} ≥ … ≥ S_t^{(M)}` for `M = |C_t|`
2. Compute differences:  
   `Δ_i = S_t^{(i)} – S_t^{(i+1)}`
3. Restrict to the top `ρM` candidates (e.g., `ρ = 0.9`) and find:
   `i* = argmax Δ_i`
4. Define an **effective capacity**:  
   `K_eff = i* + buffer` (small buffer like +3~5)

AdaGATE then **focuses replacement** on candidates within the top `K_eff` utilities, instead of exhausting the budget with mediocre passages. This gives a **data-driven notion of how many passages are worth including**.

---

### 3.6 Loop Controller and Stopping

The loop repeats until one of the stopping conditions holds:

- The gap set `G_t` is empty or only contains abstract gaps that are not meaningfully retrievable
- No candidate can improve utility under the budget (no valid swap satisfying the gain + budget conditions)
- A maximum number of iterations `L_max` is reached

At that point, `E_T` is considered **sufficient** and is passed (with `q`) to the generator LLM.

---

## 4. Relation to Prior Work

AdaGATE connects and extends several lines of work:

- **SEAL-RAG**  
  AdaGATE adopts the **entity ledger + explicit gap specification + “replace, don’t expand”** philosophy, but:
  - Moves from **fixed-k** to **token-budgeted** evidence sets  
  - Adds **question-aware relevance** as a fallback for the extraction bottleneck  
  - Explicitly incorporates **redundancy penalties** at both local and set level  
  - Uses an **Adaptive-k style utility-gap heuristic** to decide effective capacity

- **Adaptive-k Retrieval**  
  AdaGATE uses a similar **largest-gap heuristic**, but over **utility scores** rather than bare similarity, and within a **multi-hop repair loop** instead of a single pass.

- **AdaGReS**  
  AdaGATE inherits the idea of **relevance vs. redundancy under a token budget**, but extends it to a **gap-aware repair controller** with micro-queries and replacement, rather than a single-shot selector.

- **Self-RAG / Adaptive-RAG / CRAG**  
  Unlike these methods, AdaGATE is **fully training-free**, does **not** require finetuning or extra critics, and is designed to be **plug-and-play** on top of black-box API LLMs.

---

## 5. Module Overview

A high-level mapping from modules to models:

| Module                         | Role                                        | Typical Implementation                                      |
|--------------------------------|--------------------------------------------|-------------------------------------------------------------|
| Base retriever                 | Initial and micro-query retrieval          | Dense retriever (e.g., BGE, Contriever, etc.)              |
| Chunker                        | Split docs into passages                   | Semantic / sliding-window chunking                         |
| Ledger extractor               | Build entity–relation ledger `U_t`         | LLM prompt or IE model                                     |
| Gap extractor                  | Build gap set `G_t` from `(q, U_t)`        | LLM prompt with structured schema                          |
| Micro-query generator         | Create gap- and question-anchored queries  | LLM prompt                                                 |
| Utility scorer `S_t(c)`       | Compute gap-aware utility per candidate    | Combines IE outputs + embeddings + retriever similarities  |
| Adaptive capacity estimator   | Compute `K_eff` from utility gaps          | Simple numeric heuristic over `S_t(c)`                     |
| Replacement controller        | Swap evidence under token budget           | Local greedy algorithm using `S_t` and token counts        |
| Generator LLM                 | Answer the question from `(q, E_T)`        | API model (e.g., GPT-4.x family) or open-weight LLM        |

This design keeps the **controller logic (AdaGATE)** separate from the **backbone models**, making it easy to swap in different retrievers or generators while reusing the same evidence-assembly pipeline.
