---
name: meta-harness-mm
description: Run one iteration of Meta-Meta-Harness memory system evolution with lineage, vector reward, node memory, and edge observations.
---

# Meta-Harness (Memory System Evolution)

Run ONE iteration of memory system evolution. Do all work in the main session - do NOT delegate to subagents. Constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping.

**You do NOT run benchmarks.** You analyze results + prediction traces, prototype changes, and implement new systems. The outer loop (`meta_harness.py`) handles benchmarking separately.

## CRITICAL CONSTRAINTS

- You MUST implement 2 new memory systems every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design exactly 2 candidates per iteration: mix of exploitation and exploration.

### Anti-parameter-tuning rules

The most common failure mode is creating systems that are just parameter variants of existing ones. Check `evolution_summary.jsonl` for what's been tried. Parameter sweeps (pool sizes, retrieval counts, context budgets, similarity metrics) often regress or tie.

**Good candidates change a fundamental mechanism:**

- A new retrieval algorithm (e.g. contrastive pairs, diversity-aware selection, graph-based traversal)
- A new prompt architecture (e.g. organize by confusion clusters instead of listing examples sequentially)
- A new learning strategy (e.g. LLM-generated lesson summaries instead of raw example storage)
- A new memory structure (e.g. separate fast/slow pools, hierarchical organization, compressed representations)

**Bad candidates just tune numbers.** If the logic in `predict()` and `learn_from_batch()` is identical to the base except for constants, it's a parameter variant. Rewrite with a genuinely novel mechanism.

**Combining systems is valid.** Take the retrieval strategy from system A and the memory format from system B, or draw on published approaches (DSPy, OPRO, Reflexion, CEIL, etc.).

Exploitation axes: A=Prompt template, B=Memory content, C=Selection algorithm, D=Memory sizing, E=Learning trigger, F=LLM usage in learning. If last 3 iterations explored the same axis, pick different ones.

### Anti-overfitting rules

- **No dataset-specific hints.** Do not hardcode knowledge about specific datasets. Memory systems must be general-purpose.
- **Never mention dataset names** in system code, prompts, or comments.
- **General patterns are OK.** Rules like "prioritize recent errors" or "balance label coverage" are fine; they apply broadly.

## WORKFLOW

**Do ALL steps yourself in the main session.**

### Step 0: Post-eval reports (write if missing)

Check the reports directory (path in the task prompt's "Run directories" section). For each past iteration that has results in `evolution_summary.jsonl` but NO report, write one. Each report should be **<=30 lines** covering: what changed, which datasets improved/regressed and why, and a takeaway for future iterations.

### Step 1: Analyze

1. **Read all state files:**
   - `evolution_summary.jsonl` - what's been tried (one JSON per candidate)
   - `frontier_val.json` - current best per dataset (val accuracy)
   - active config for current datasets and baselines
   - recent `logs/<dataset>/<agent>/<model>/log.jsonl` traces if they exist
   - when present, `nodes.jsonl`, `frontier_vec.json`, and recent `traces/edge_*.jsonl`

2. Compare mechanisms, not only scores. Use the current run's agents and the
   Meta-Meta graph to identify which code changes caused which objective changes.

3. Formulate 3 falsifiable hypotheses. Each should target a concrete mechanism,
   not just constants such as pool size, top-k, or context budget.

4. Choose exactly 2 candidates. One should exploit a strong parent/frontier
   mechanism. One should explore a meaningfully different mechanism.

### Step 2: Prototype - MANDATORY

**You MUST prototype your mechanism before writing the final system.** Do NOT skip this step. Candidates that skip prototyping tend to have bugs or produce no improvement.

For each candidate:

1. Write a test script in `/tmp/` that exercises the core retrieval/learning logic in isolation.
2. Pull real examples from `logs/<dataset>/<memory>/<model>/log.jsonl` to test against.
3. Try 2-3 variants and compare before picking the best one.
4. Delete scripts when done.

### Step 3: Implement

For each of the 2 candidates:

1. Copy a top-performing base system to `agents/<name>.py`, then make targeted modifications. This copy-then-edit approach ensures correct imports and proven patterns.
2. Implement the new mechanism according to your hypothesis.
3. Self-critique: after implementing, re-read the file and check whether this system introduces a genuinely new mechanism rather than only number changes.
4. Validate: `uv run python -c "from text_classification.agents.<name> import *; print('OK')"`

Do not edit `config.yaml` just to register candidates. The benchmark auto-discovers files in `agents/`.

### Step 4: Write pending_eval.json

Write to the path specified in the task prompt (NOT hardcoded - it may be in a run-specific subdirectory):

```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "<snake_case_name>",
      "file": "agents/<name>.py",
      "hypothesis": "<falsifiable claim>",
      "predicted_delta_r": [<expected_delta_for_dim_1>, <expected_delta_for_dim_2>, "..."],
      "axis": "exploitation|exploration",
      "base_system": "<what it builds on>",
      "components": ["tag1", "tag2", "..."]
    }
  ]
}
```

Output: `CANDIDATES: <name1>, <name2>`

## MemorySystem Interface

```python
class MemorySystem(ABC):
    def __init__(self, llm: LLMCallable): ...
    def predict(self, input: str) -> tuple[str, dict[str, Any]]: ...
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None: ...
    def get_state(self) -> str: ...       # JSON-serializable
    def set_state(self, state: str) -> None: ...
```

- Extend `MemorySystem` from `..memory_system`
- Import `LLMCallable` from `..llm`, `extract_json_field` from `..memory_system`
- Use `extract_json_field(response, "final_answer")` for answer extraction (NOT custom regex)
- Use `self.call_llm(prompt)` for LLM calls (NOT `self._llm` directly)
- `predict` must work without any prior learning (cold start)
- `learn_from_batch` receives list of dicts with keys: input, prediction, ground_truth, was_correct, metadata

## Directory Structure

- Val results: `logs/<dataset>/<memory>/<model>/val.json` (accuracy field)
- Training logs: `logs/<dataset>/<memory>/<model>/log.jsonl`
- Memory state: `logs/<dataset>/<memory>/<model>/memory.json`
- Test results: `results/<dataset>/<memory>/<model>/test.json` (separate dir, never exposed during evolution)

## evolution_summary.jsonl Format

One JSON object per line, one line per evaluated candidate:

```json
{"iteration": 1, "system": "example_system", "avg_val": 45.0, "axis": "exploitation", "hypothesis": "...", "delta": +2.1, "outcome": "45.0% (+2.1)", "components": ["tag1", "tag2", "tag3"]}
```

## Meta-Meta Extension

When the task prompt contains `# Meta-Meta evolution state`, use it as first-class
state, not as background decoration.

- Use the selected parent node, frontier node memories, `r_vec`, and recent
  `(diff -> delta_r)` observations as extra evidence for choosing what to build.
- Prefer mechanisms whose observed effects target weak dimensions, but keep the
  official Meta-Harness goal: produce two strong candidate memory systems every
  iteration.
- Always include `predicted_delta_r` in each `pending_eval.json` candidate. It is
  a numeric list in the same dimension order shown in the Meta-Meta state.
