"""Reaction-stratified retrieval with MMR diversity instead of hard label caps.

Builds on reaction_stratified_retrieval (USPTO 0.267, S2D 0.780, LawBench 0.160).
Replaces hard 3-per-label cap with MMR reranking in the fallback path to improve
LawBench label coverage while maintaining reaction-type stratification for USPTO.
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000
TOP_K = 16
MMR_LAMBDA = 0.7


class ReactionStratifiedMmr(MemorySystem):
    """Stratified retrieval with MMR diversity instead of hard label caps."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.reaction_groups: defaultdict[str, list[int]] = defaultdict(list)

    def _extract_reaction_type(self, text: str) -> str | None:
        match = re.search(r'reaction type is (\w+)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_smiles(self, text: str) -> str:
        match = re.search(r'Input:\s*([^\n]+)', text)
        return match.group(1).strip() if match else ""

    def _char_ngrams(self, text: str, n: int = 3) -> set[str]:
        return set(text[i:i+n] for i in range(len(text)-n+1))

    def _jaccard(self, a: set, b: set) -> float:
        return len(a & b) / len(a | b) if a | b else 0.0

    def _mmr_rerank(self, scored: list[tuple[float, int]], max_k: int) -> list[int]:
        """MMR reranking for diversity."""
        if len(scored) <= max_k:
            return [idx for _, idx in scored]

        selected_indices = []
        remaining = list(scored)

        while remaining and len(selected_indices) < max_k:
            best_idx = -1
            best_mmr = -float('inf')

            for i, (score, idx) in enumerate(remaining):
                relevance = score
                if selected_indices:
                    ex_text = self.examples[idx]['input']
                    ex_ngrams = self._char_ngrams(ex_text)
                    max_sim = max(
                        self._jaccard(ex_ngrams, self._char_ngrams(self.examples[s_idx]['input']))
                        for s_idx in selected_indices
                    )
                    diversity = 1 - max_sim
                else:
                    diversity = 1.0

                mmr = MMR_LAMBDA * relevance + (1 - MMR_LAMBDA) * diversity
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected_indices.append(remaining.pop(best_idx)[1])

        return selected_indices

    def _retrieve_examples(self, input: str, max_k: int = 40) -> list[dict]:
        if not self.examples:
            return []

        reaction_type = self._extract_reaction_type(input)

        if reaction_type and reaction_type in self.reaction_groups:
            # Chemical domain: stratify by reaction type + SMILES similarity
            candidate_indices = self.reaction_groups[reaction_type]
            query_smiles = self._extract_smiles(input)
            query_ngrams = self._char_ngrams(query_smiles)

            scored = []
            for idx in candidate_indices:
                ex_smiles = self._extract_smiles(self.examples[idx]['input'])
                ex_ngrams = self._char_ngrams(ex_smiles)
                overlap = len(query_ngrams & ex_ngrams)
                scored.append((overlap / 100.0, idx))

            scored.sort(reverse=True, key=lambda x: x[0])
            selected_indices = self._mmr_rerank(scored, TOP_K)
        else:
            # Non-chemical: use TF-IDF + MMR
            query_tokens = input.lower().split()[:200]
            query_counts = Counter(query_tokens)

            scored = []
            for idx, ex in enumerate(self.examples):
                doc_tokens = ex['input'].lower().split()[:200]
                doc_counts = Counter(doc_tokens)
                score = sum(
                    query_counts[term] * doc_counts[term]
                    for term in query_counts if term in doc_counts
                )
                scored.append((score, idx))

            scored.sort(reverse=True, key=lambda x: x[0])
            # Normalize scores to [0, 1]
            max_score = scored[0][0] if scored and scored[0][0] > 0 else 1.0
            scored = [(s / max_score, idx) for s, idx in scored]
            selected_indices = self._mmr_rerank(scored[:64], TOP_K)

        return [self.examples[idx] for idx in selected_indices]

    def _format_examples_section(self, examples: list[dict]) -> str:
        if not examples:
            return ""

        parts = []
        total_chars = 0
        for ex in examples:
            question = ex.get("raw_question", ex["input"])
            part = f"Q: {question}\nA: {ex['target']}"
            if total_chars + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total_chars += len(part) + 2

        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._retrieve_examples(input)
        examples_section = self._format_examples_section(examples)
        prompt = PROMPT_TEMPLATE.format(
            examples_section=examples_section,
            input=input,
        )

        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")

        return answer, {"full_response": response, "num_examples": len(examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            idx = len(self.examples)
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            reaction_type = self._extract_reaction_type(ex["input"])
            if reaction_type:
                self.reaction_groups[reaction_type].append(idx)

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "reaction_groups": {k: list(v) for k, v in self.reaction_groups.items()}
        }, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.reaction_groups = defaultdict(list, {
            k: v for k, v in data.get("reaction_groups", {}).items()
        })
