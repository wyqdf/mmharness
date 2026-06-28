"""Multilabel-aware centroid retrieval: boost multi-label examples for multi-label queries.

Builds on label_centroid_bm25 (LawBench 0.380) by adding multi-stage reranking:
1. Label-centroid BM25 retrieval (top 40)
2. Detect multi-label indicators in query (；;、and,)
3. Boost examples with multi-label targets (1.5x) when query is multi-label
4. Select top 16 after reranking

Targets LawBench's 58% multi-label cases.
"""

import json
import math
import re
from collections import Counter, defaultdict
from threading import Lock
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
CANDIDATE_POOL = 40
CENTROID_ALPHA = 0.7
MULTILABEL_BOOST = 1.5


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
    df: Counter = Counter()
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


def _has_multilabel_indicator(text: str) -> bool:
    """Check if text suggests multiple items."""
    return any(sep in text for sep in ['；', ';', '、', ' and ', ',', '且'])


class MultilabelCentroidRetrieval(MemorySystem):
    """Label-centroid BM25 with multi-label-aware reranking."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._docs_tokens: list[list[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._label_centroid_tf: dict[str, Counter] = {}
        self._label_centroid_len: dict[str, int] = {}
        self._avg_centroid_len: float = 0.0
        self._index_dirty = True
        self._index_lock = Lock()

    def _ensure_index(self):
        if not self._index_dirty:
            return
        with self._index_lock:
            if not self._index_dirty:
                return

            self._docs_tokens = [_tokenize(ex['input']) for ex in self.examples]
            self._doc_tfs = [Counter(toks) for toks in self._docs_tokens]
            self._doc_lens = [len(toks) for toks in self._docs_tokens]
            self._avgdl = sum(self._doc_lens) / max(1, len(self._doc_lens))
            self._idf = _bm25_idf(self._docs_tokens)

            label_docs = defaultdict(list)
            for i, ex in enumerate(self.examples):
                label_docs[ex['target']].append(self._docs_tokens[i])

            self._label_centroid_tf = {}
            self._label_centroid_len = {}
            for label, docs in label_docs.items():
                merged = Counter()
                for d in docs:
                    merged.update(d)
                self._label_centroid_tf[label] = merged
                self._label_centroid_len[label] = sum(merged.values())

            if self._label_centroid_len:
                self._avg_centroid_len = sum(self._label_centroid_len.values()) / len(self._label_centroid_len)
            else:
                self._avg_centroid_len = 1.0

            self._index_dirty = False

    def _retrieve_examples(self, input: str) -> list[dict]:
        if not self.examples:
            return []

        self._ensure_index()
        qtoks = _tokenize(input)

        scored = []
        for i in range(len(self.examples)):
            inst_score = _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)

            label = self.examples[i]['target']
            centroid_tf = self._label_centroid_tf.get(label, Counter())
            centroid_len = self._label_centroid_len.get(label, 1)
            centroid_score = _bm25_score(qtoks, centroid_tf, centroid_len, self._avg_centroid_len, self._idf)

            combined = inst_score + CENTROID_ALPHA * centroid_score
            scored.append((combined, i))

        scored.sort(reverse=True, key=lambda x: x[0])
        candidates = scored[:CANDIDATE_POOL]

        query_multilabel = _has_multilabel_indicator(input)
        if query_multilabel:
            reranked = []
            for score, idx in candidates:
                boost = MULTILABEL_BOOST if _has_multilabel_indicator(self.examples[idx]['target']) else 1.0
                reranked.append((score * boost, idx))
            reranked.sort(reverse=True, key=lambda x: x[0])
            selected_indices = [idx for _, idx in reranked[:TOP_K]]
        else:
            selected_indices = [idx for _, idx in candidates[:TOP_K]]

        return [self.examples[i] for i in selected_indices]

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
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._index_dirty = True
