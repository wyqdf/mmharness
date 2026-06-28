"""Functional group matching for USPTO: extract chemical substructures.

Builds on reaction_stratified_retrieval (USPTO 0.267) by replacing character
n-grams with functional group extraction (carbonyl, amine, halogen, benzene rings).
Jaccard overlap on functional group sets provides finer-grained chemical similarity.
"""

import json
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


class FunctionalGroupRetrieval(MemorySystem):
    """Retrieval using functional group substructure matching."""

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

    def _extract_functional_groups(self, smiles: str) -> set[str]:
        """Extract functional groups from SMILES."""
        groups = set()
        patterns = [
            (r'C\(=O\)O[^=]', 'COOH'),
            (r'C\(=O\)N', 'CONR'),
            (r'C=O', 'C=O'),
            (r'O[^=]', 'OR'),
            (r'N[^=]', 'NR'),
            (r'S', 'S'),
            (r'Cl', 'Cl'),
            (r'Br', 'Br'),
            (r'I', 'I'),
            (r'F', 'F'),
            (r'c1ccccc1', 'Ph'),
            (r'#', 'C#'),
        ]
        for pattern, name in patterns:
            if re.search(pattern, smiles):
                groups.add(name)
        return groups

    def _functional_group_similarity(self, query_smiles: str, ex_smiles: str) -> float:
        """Jaccard similarity on functional groups."""
        q_groups = self._extract_functional_groups(query_smiles)
        ex_groups = self._extract_functional_groups(ex_smiles)
        if not q_groups and not ex_groups:
            return 0.0
        union = q_groups | ex_groups
        if not union:
            return 0.0
        return len(q_groups & ex_groups) / len(union)

    def _retrieve_examples(self, input: str, max_k: int = 40) -> list[dict]:
        if not self.examples:
            return []

        reaction_type = self._extract_reaction_type(input)

        if reaction_type and reaction_type in self.reaction_groups:
            candidate_indices = self.reaction_groups[reaction_type]
            query_smiles = self._extract_smiles(input)

            scored = []
            for idx in candidate_indices:
                ex_smiles = self._extract_smiles(self.examples[idx]['input'])
                sim = self._functional_group_similarity(query_smiles, ex_smiles)
                scored.append((sim, idx))

            scored.sort(reverse=True, key=lambda x: x[0])
            selected_indices = [idx for _, idx in scored[:max_k]]
        else:
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
            selected_indices = [idx for _, idx in scored[:max_k]]

        selected = []
        label_counts = Counter()
        for idx in selected_indices:
            label = self.examples[idx]['target']
            if label_counts[label] < 3:
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
