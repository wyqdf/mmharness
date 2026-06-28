"""Reaction soft-blending retrieval: weighted combination of SMILES and term signals.

Builds on reaction_stratified_retrieval but replaces binary switch (reaction → SMILES,
else → TF-IDF) with soft blending. Computes both SMILES char-ngram and term-overlap
scores for ALL inputs, then blends with domain-adaptive weights (0.8/0.2 for chemical,
0.3/0.7 for non-chemical). Reduces failure modes from hard switching.
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


class ReactionSoftBlend(MemorySystem):
    """Soft-blending retrieval combining SMILES and term signals."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []

    def _extract_reaction_type(self, text: str) -> str | None:
        match = re.search(r'reaction type is (\w+)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_smiles(self, text: str) -> str:
        match = re.search(r'Input:\s*([^\n]+)', text)
        return match.group(1).strip() if match else ""

    def _char_ngrams(self, text: str, n: int = 3) -> set[str]:
        return set(text[i:i+n] for i in range(len(text)-n+1))

    def _retrieve_examples(self, input: str) -> list[dict]:
        if not self.examples:
            return []

        is_chemical = self._extract_reaction_type(input) is not None

        # Compute SMILES scores for all examples
        query_smiles = self._extract_smiles(input)
        query_ngrams = self._char_ngrams(query_smiles) if query_smiles else set()

        # Compute term-overlap scores for all examples
        query_tokens = input.lower().split()[:200]
        query_counts = Counter(query_tokens)

        scored = []
        for idx, ex in enumerate(self.examples):
            # SMILES char-ngram score
            ex_smiles = self._extract_smiles(ex['input'])
            ex_ngrams = self._char_ngrams(ex_smiles) if ex_smiles else set()
            smiles_score = len(query_ngrams & ex_ngrams) if query_ngrams and ex_ngrams else 0

            # Term-overlap score
            doc_tokens = ex['input'].lower().split()[:200]
            doc_counts = Counter(doc_tokens)
            term_score = sum(
                query_counts[term] * doc_counts[term]
                for term in query_counts if term in doc_counts
            )

            # Blend with domain-adaptive weights
            if is_chemical:
                weight_smiles = 0.8
                weight_term = 0.2
            else:
                weight_smiles = 0.3
                weight_term = 0.7

            blended = weight_smiles * smiles_score + weight_term * term_score
            scored.append((blended, idx))

        scored.sort(reverse=True, key=lambda x: x[0])

        # Diversify by label
        selected = []
        label_counts = Counter()
        for _, idx in scored:
            label = self.examples[idx]['target']
            if label_counts[label] < 3 and len(selected) < TOP_K:
                selected.append(self.examples[idx])
                label_counts[label] += 1

        return selected

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
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
