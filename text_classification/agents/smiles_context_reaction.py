"""SMILES-context reaction retrieval: enhanced chemical structure matching.

Builds on reaction_stratified_retrieval (USPTO frontier 0.267) by adding:
1. Explicit SMILES tokenization (functional groups, rings, bonds)
2. Reaction context extraction (reagents, conditions)
3. Dual scoring: structure similarity + context relevance

Hypothesis: USPTO needs finer-grained SMILES matching beyond char-ngrams.
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


def _extract_smiles(text: str) -> str:
    match = re.search(r'Input:\s*([^\n]+)', text)
    if match:
        return match.group(1).strip()
    candidates = re.findall(r'[A-Z][A-Za-z0-9\(\)\[\]=#@\+\-\.]{4,}', text)
    return max(candidates, key=len) if candidates else ""


def _smiles_tokens(smiles: str) -> list[str]:
    """Tokenize SMILES into functional groups, rings, bonds."""
    tokens = []
    # Ring numbers
    tokens.extend(re.findall(r'[0-9]', smiles))
    # Bonds
    tokens.extend(re.findall(r'[=#\-]', smiles))
    # Functional groups (2-4 char)
    for n in [4, 3, 2]:
        i = 0
        while i < len(smiles) - n + 1:
            chunk = smiles[i:i+n]
            if re.match(r'[A-Z][A-Za-z0-9@\(\)\[\]]{' + str(n-1) + '}', chunk):
                tokens.append(chunk)
            i += 1
    return tokens


def _extract_reaction_type(text: str) -> str:
    match = re.search(r'reaction type is (\w+)', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    if any(x in text for x in ['TBS', 'TBDMS', 'Boc', 'Fmoc']):
        return 'protecting'
    if any(x in text for x in ['Br', 'Cl', 'I']):
        return 'halogenation'
    return 'unknown'


class SmilesContextReaction(MemorySystem):
    """SMILES-aware reaction retrieval with context."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict] = []
        self.reaction_groups: defaultdict[str, list[int]] = defaultdict(list)

    def _score_example(self, query: str, ex_idx: int) -> float:
        ex = self.examples[ex_idx]
        q_smiles = _extract_smiles(query)
        ex_smiles = _extract_smiles(ex["input"])

        # Structure similarity
        q_toks = _smiles_tokens(q_smiles)
        ex_toks = _smiles_tokens(ex_smiles)
        q_set = set(q_toks)
        ex_set = set(ex_toks)
        struct_score = len(q_set & ex_set) / max(len(q_set | ex_set), 1) if q_set or ex_set else 0

        # Context similarity (non-SMILES text)
        q_ctx = re.sub(r'[A-Z][A-Za-z0-9\(\)\[\]=#@\+\-\.]+', '', query).lower()
        ex_ctx = re.sub(r'[A-Z][A-Za-z0-9\(\)\[\]=#@\+\-\.]+', '', ex["input"]).lower()
        q_ctx_toks = set(re.findall(r'\w+', q_ctx))
        ex_ctx_toks = set(re.findall(r'\w+', ex_ctx))
        ctx_score = len(q_ctx_toks & ex_ctx_toks) / max(len(q_ctx_toks | ex_ctx_toks), 1) if q_ctx_toks or ex_ctx_toks else 0

        return 0.7 * struct_score + 0.3 * ctx_score

    def _retrieve_indices(self, query: str) -> list[int]:
        if not self.examples:
            return []

        rtype = _extract_reaction_type(query)
        candidates = self.reaction_groups.get(rtype, list(range(len(self.examples))))

        if not candidates:
            candidates = list(range(len(self.examples)))

        scored = [(self._score_example(query, idx), idx) for idx in candidates]
        scored.sort(reverse=True)
        return [idx for _, idx in scored[:TOP_K]]

    def _format_examples(self, idxs: list[int]) -> str:
        parts = []
        total = 0
        for idx in idxs:
            ex = self.examples[idx]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        idxs = self._retrieve_indices(input)
        examples_section = self._format_examples(idxs)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]):
        for r in batch_results:
            idx = len(self.examples)
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            rtype = _extract_reaction_type(ex["input"])
            self.reaction_groups[rtype].append(idx)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
