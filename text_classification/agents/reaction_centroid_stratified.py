"""Reaction-stratified retrieval with label-centroid priors.

Combines reaction_stratified_retrieval (USPTO frontier 0.267) with label_centroid_bm25
(LawBench frontier 0.380). Stratifies by reaction type, then scores with BM25 + centroid prior.
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
CENTROID_ALPHA = 0.6


def _tokenize(s: str) -> list[str]:
    lower = s.lower()
    words = re.findall(r"[a-z0-9]+|[\(\)=#\[\]/\\@\+\-\.]", lower)
    compact = re.sub(r"\s+", "", lower)
    ngrams = []
    for n in range(2, 4):
        if len(compact) >= n:
            for i in range(len(compact) - n + 1):
                ngrams.append(compact[i : i + n])
    return words + ngrams


def _bm25_idf(docs_tokens: list[list[str]]) -> dict[str, float]:
    n = len(docs_tokens)
    df = Counter()
    for d in docs_tokens:
        for t in set(d):
            df[t] += 1
    return {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}


def _bm25_score(qtoks, tf, dl, avgdl, idf, k1=1.5, b=0.75):
    s = 0.0
    for t in qtoks:
        f = tf.get(t)
        if not f:
            continue
        denom = f + k1 * (1 - b + b * dl / max(1.0, avgdl))
        s += idf.get(t, 0.0) * f * (k1 + 1) / denom
    return s


class ReactionCentroidStratified(MemorySystem):
    """Reaction-type stratification + label-centroid BM25."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict] = []
        self.reaction_groups: defaultdict[str, list[int]] = defaultdict(list)
        self._doc_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._label_centroid: defaultdict[str, defaultdict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self._dirty = True

    def _extract_reaction_type(self, text: str) -> str | None:
        match = re.search(r'reaction type is (\w+)', text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        if 'protect' in text.lower() or 'TBS' in text or 'TBDMS' in text:
            return 'protecting'
        if 'halogen' in text.lower() or 'Br' in text or 'Cl' in text or 'I' in text:
            return 'halogenation'
        return None

    def _ensure_index(self):
        if not self._dirty or not self.examples:
            return
        self._doc_tokens = [_tokenize(ex["input"]) for ex in self.examples]
        self._idf = _bm25_idf(self._doc_tokens)
        self._avgdl = sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens) if self._doc_tokens else 1.0

        # Build per-reaction-type label centroids
        self._label_centroid.clear()
        for i, ex in enumerate(self.examples):
            rtype = self._extract_reaction_type(ex["input"]) or "unknown"
            label = ex["target"]
            self._label_centroid[rtype][label].update(self._doc_tokens[i])

        self._dirty = False

    def _retrieve_indices(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []

        q_tok = _tokenize(query)
        rtype = self._extract_reaction_type(query) or "unknown"

        # Get candidates from same reaction type
        candidates = self.reaction_groups.get(rtype, [])
        if not candidates:
            # Fallback: use all examples
            candidates = list(range(len(self.examples)))

        scores = []
        for idx in candidates:
            doc_tok = self._doc_tokens[idx]
            tf = Counter(doc_tok)
            inst_score = _bm25_score(q_tok, tf, len(doc_tok), self._avgdl, self._idf)

            # Add centroid prior if label exists
            label = self.examples[idx]["target"]
            if label in self._label_centroid[rtype]:
                centroid_tf = self._label_centroid[rtype][label]
                centroid_len = sum(centroid_tf.values())
                centroid_score = _bm25_score(q_tok, centroid_tf, centroid_len, self._avgdl, self._idf)
                total_score = inst_score + CENTROID_ALPHA * centroid_score
            else:
                total_score = inst_score

            scores.append((total_score, idx))

        scores.sort(reverse=True)
        return [idx for _, idx in scores[:TOP_K]]

    def _format_examples(self, idxs: list[int]) -> str:
        parts = []
        total = 0
        for i in idxs:
            ex = self.examples[i]
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

            rtype = self._extract_reaction_type(ex["input"]) or "unknown"
            self.reaction_groups[rtype].append(idx)

        self._dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._dirty = True
