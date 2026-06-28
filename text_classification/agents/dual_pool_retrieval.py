"""Dual-pool retrieval: fast (recent) + slow (centroid) memory pools.

Maintains two separate memory pools with different retrieval strategies:
- Fast pool: Recent 30 examples, BM25 with recency boost (weight by 1/age)
- Slow pool: All examples, label-centroid BM25 for stable prototypes

At predict time, retrieves from both pools and interleaves results (alternating
fast/slow examples). Fast pool captures recent patterns and evolution; slow pool
maintains stable label prototypes. Targets S2D +0.02, LawBench +0.04-0.06.
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
FAST_POOL_SIZE = 30
TOP_K = 16
CENTROID_ALPHA = 0.7


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
        f = tf.get(t, 0)
        if not f:
            continue
        denom = f + k1 * (1 - b + b * dl / max(1.0, avgdl))
        s += idf.get(t, 0.0) * f * (k1 + 1) / denom
    return s


class DualPoolRetrieval(MemorySystem):
    """Dual-pool: fast (recent+recency) + slow (all+centroid)."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict] = []
        self._docs_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._label_centroid_tf: dict[str, Counter] = {}
        self._label_centroid_len: dict[str, int] = {}
        self._dirty = True

    def _ensure_index(self):
        if not self._dirty or not self.examples:
            return

        self._docs_tokens = [_tokenize(ex["input"]) for ex in self.examples]
        self._idf = _bm25_idf(self._docs_tokens)
        self._avgdl = sum(len(t) for t in self._docs_tokens) / len(self._docs_tokens) if self._docs_tokens else 1.0

        label_docs = defaultdict(list)
        for ex, tokens in zip(self.examples, self._docs_tokens):
            label_docs[ex["target"]].append(tokens)

        for label, docs in label_docs.items():
            all_toks = [t for doc in docs for t in doc]
            self._label_centroid_tf[label] = Counter(all_toks)
            self._label_centroid_len[label] = len(all_toks)

        self._dirty = False

    def _retrieve_fast(self, input: str, k: int) -> list[int]:
        """Recent pool with recency weighting."""
        q_toks = _tokenize(input)
        fast_pool = self.examples[-FAST_POOL_SIZE:] if len(self.examples) > FAST_POOL_SIZE else self.examples
        fast_start = max(0, len(self.examples) - FAST_POOL_SIZE)

        scores = []
        for i, ex in enumerate(fast_pool):
            global_idx = fast_start + i
            tokens = self._docs_tokens[global_idx]
            tf = Counter(tokens)
            bm25 = _bm25_score(q_toks, tf, len(tokens), self._avgdl, self._idf)

            age = len(self.examples) - global_idx
            recency_boost = 1.0 / (1.0 + age * 0.1)
            score = bm25 * (1.0 + recency_boost)
            scores.append((score, global_idx))

        scores.sort(reverse=True)
        return [idx for _, idx in scores[:k]]

    def _retrieve_slow(self, input: str, k: int) -> list[int]:
        """Full pool with label-centroid prior."""
        q_toks = _tokenize(input)
        scores = []

        for i, ex in enumerate(self.examples):
            tokens = self._docs_tokens[i]
            tf = Counter(tokens)
            inst_score = _bm25_score(q_toks, tf, len(tokens), self._avgdl, self._idf)

            label = ex["target"]
            if label in self._label_centroid_tf:
                cent_score = _bm25_score(q_toks, self._label_centroid_tf[label],
                                          self._label_centroid_len[label], self._avgdl, self._idf)
                total = inst_score + CENTROID_ALPHA * cent_score
            else:
                total = inst_score
            scores.append((total, i))

        scores.sort(reverse=True)
        return [idx for _, idx in scores[:k]]

    def _format_examples(self, idxs: list[int]) -> str:
        parts = []
        total = 0
        for i in idxs:
            ex = self.examples[i]
            part = f"Q: {ex['input'][:200]}\nA: {ex['target']}"
            if total + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._ensure_index()

        if not self.examples:
            prompt = PROMPT_TEMPLATE.format(examples_section="", input=input)
            resp = self.call_llm(prompt)
            answer = extract_json_field(resp, "final_answer")
            return answer, {"full_response": resp}

        k_per_pool = TOP_K // 2
        fast_idxs = self._retrieve_fast(input, k_per_pool)
        slow_idxs = self._retrieve_slow(input, k_per_pool)

        combined = []
        for i in range(max(len(fast_idxs), len(slow_idxs))):
            if i < len(fast_idxs):
                combined.append(fast_idxs[i])
            if i < len(slow_idxs) and slow_idxs[i] not in combined:
                combined.append(slow_idxs[i])

        examples_text = self._format_examples(combined[:TOP_K])
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_text, input=input)
        resp = self.call_llm(prompt)
        answer = extract_json_field(resp, "final_answer")

        return answer, {"full_response": resp, "fast": len(fast_idxs), "slow": len(slow_idxs)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]):
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            self.examples.append(ex)
        self._dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._dirty = True
