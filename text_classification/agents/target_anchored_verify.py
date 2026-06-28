"""Target-anchored verification: verify predictions via target-side retrieval.

Stage 1: label-centroid BM25 retrieval → tentative answer
Stage 2 (gated): retrieve training examples whose TARGETS match the tentative,
then verify via LLM with sibling inputs. Gate fires when tentative has token
overlap with training targets (classification datasets).

Targets S2D/LawBench improvements via output-anchored verification.
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

VERIFY_PROMPT = """You gave a tentative answer: "{tentative}"

Below are training cases whose TARGET matches your tentative:
{sibling_section}

**Problem:**
{input}

Do these sibling inputs resemble the problem? If yes, commit to "{tentative}". If no, revise.
Respond in JSON: {{"reasoning": "[reasoning]", "final_answer": "[your final answer]"}}"""

MAX_CHARS = 30000
TOP_K = 16
CENTROID_ALPHA = 0.7
GATE_OVERLAP = 0.3
MIN_SIBLINGS = 2


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


class TargetAnchoredVerify(MemorySystem):
    """Target-side retrieval for output verification."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict] = []
        self._doc_tokens: list[list[str]] = []
        self._target_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._label_centroid: defaultdict[str, Counter] = defaultdict(Counter)
        self._dirty = True

    def _ensure_index(self):
        if not self._dirty or not self.examples:
            return
        self._doc_tokens = [_tokenize(ex["input"]) for ex in self.examples]
        self._target_tokens = [_tokenize(ex["target"]) for ex in self.examples]
        self._idf = _bm25_idf(self._doc_tokens)
        self._avgdl = sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens) if self._doc_tokens else 1.0

        self._label_centroid.clear()
        for i, ex in enumerate(self.examples):
            self._label_centroid[ex["target"]].update(self._doc_tokens[i])

        self._dirty = False

    def _retrieve_indices(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []

        q_tok = _tokenize(query)
        scores = []
        for idx in range(len(self.examples)):
            doc_tok = self._doc_tokens[idx]
            tf = Counter(doc_tok)
            inst_score = _bm25_score(q_tok, tf, len(doc_tok), self._avgdl, self._idf)

            label = self.examples[idx]["target"]
            if label in self._label_centroid:
                centroid_tf = self._label_centroid[label]
                centroid_len = sum(centroid_tf.values())
                centroid_score = _bm25_score(q_tok, centroid_tf, centroid_len, self._avgdl, self._idf)
                total_score = inst_score + CENTROID_ALPHA * centroid_score
            else:
                total_score = inst_score

            scores.append((total_score, idx))

        scores.sort(reverse=True)
        return [idx for _, idx in scores[:TOP_K]]

    def _retrieve_siblings(self, tentative: str) -> list[int]:
        """Retrieve training examples whose targets match tentative."""
        self._ensure_index()
        if not self.examples:
            return []

        tent_toks = set(_tokenize(tentative))
        siblings = []
        for idx, target_toks in enumerate(self._target_tokens):
            target_set = set(target_toks)
            overlap = len(tent_toks & target_set) / max(len(tent_toks | target_set), 1) if tent_toks or target_set else 0
            if overlap > GATE_OVERLAP:
                siblings.append((overlap, idx))

        siblings.sort(reverse=True)
        return [idx for _, idx in siblings[:8]]

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
        # Stage 1: standard retrieval
        idxs = self._retrieve_indices(input)
        examples_section = self._format_examples(idxs)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        tentative = extract_json_field(response, "final_answer")

        # Stage 2: gate check
        sibling_idxs = self._retrieve_siblings(tentative)
        if len(sibling_idxs) < MIN_SIBLINGS:
            return tentative, {"full_response": response, "verified": False}

        # Verification LLM call
        sibling_section = self._format_examples(sibling_idxs[:6])
        verify_prompt = VERIFY_PROMPT.format(
            tentative=tentative,
            sibling_section=sibling_section,
            input=input
        )
        verify_response = self.call_llm(verify_prompt)
        final_answer = extract_json_field(verify_response, "final_answer")

        return final_answer, {"stage1": tentative, "verified": True}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]):
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

        self._dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._dirty = True
