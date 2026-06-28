"""Label-centroid prior retrieval: BM25 + prototype-level prior.

Builds on charngram_bm25_mmr by adding per-label centroid term frequencies.
At predict time, scores = instance BM25 + alpha * centroid BM25, boosting
examples whose label prototype matches the query. Helps classification
datasets (S2D, LawBench); degrades gracefully on unique-label datasets (USPTO).
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
CANDIDATE_POOL = 64
RERANK_POOL = 80
MMR_LAMBDA = 0.7
CENTROID_ALPHA = 0.7


def _tokenize(s: str) -> list[str]:
    lower = s.lower()
    words = re.findall(r"[a-z0-9]+|[\(\)=#\[\]/\\@\+\-\.]", lower)
    compact = re.sub(r"\s+", "", lower)
    ngrams: list[str] = []
    for n in range(2, 4):
        if len(compact) < n:
            continue
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


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class LabelCentroidBM25(MemorySystem):
    """BM25 with label-centroid prior."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._docs_tokens: list[list[str]] = []
        self._doc_token_sets: list[set[str]] = []
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
            questions = [ex.get("raw_question") or ex["input"] for ex in self.examples]
            docs_tokens = [_tokenize(q) for q in questions]
            doc_token_sets = [set(t) for t in docs_tokens]
            doc_tfs = [Counter(t) for t in docs_tokens]
            doc_lens = [len(t) for t in docs_tokens]
            n = len(docs_tokens)
            avgdl = (sum(doc_lens) / n) if n else 0.0
            idf = _bm25_idf(docs_tokens)
            label_groups: dict[str, list[int]] = defaultdict(list)
            for i, ex in enumerate(self.examples):
                label_groups[ex["target"]].append(i)
            label_centroid_tf: dict[str, Counter] = {}
            label_centroid_len: dict[str, int] = {}
            for label, idxs in label_groups.items():
                centroid = Counter()
                for i in idxs:
                    centroid.update(doc_tfs[i])
                label_centroid_tf[label] = centroid
                label_centroid_len[label] = sum(centroid.values())
            avg_centroid_len = (
                sum(label_centroid_len.values()) / len(label_centroid_len)
                if label_centroid_len
                else 0.0
            )

            self._docs_tokens = docs_tokens
            self._doc_token_sets = doc_token_sets
            self._doc_tfs = doc_tfs
            self._doc_lens = doc_lens
            self._avgdl = avgdl
            self._idf = idf
            self._label_centroid_tf = label_centroid_tf
            self._label_centroid_len = label_centroid_len
            self._avg_centroid_len = avg_centroid_len
            self._index_dirty = False

    def _select(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        qtoks = _tokenize(query)
        n = len(self.examples)
        doc_scores = [
            _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            for i in range(n)
        ]
        max_doc_score = max(doc_scores) or 1.0
        centroid_scores = {}
        for label, ctf in self._label_centroid_tf.items():
            centroid_len = self._label_centroid_len.get(label, sum(ctf.values()))
            centroid_scores[label] = _bm25_score(qtoks, ctf, centroid_len, self._avg_centroid_len, self._idf)
        max_centroid_score = max(centroid_scores.values()) if centroid_scores else 1.0
        max_centroid_score = max_centroid_score or 1.0
        scores = []
        for i in range(n):
            doc_score_norm = doc_scores[i] / max_doc_score
            label = self.examples[i]["target"]
            centroid_score_norm = centroid_scores.get(label, 0.0) / max_centroid_score
            final_score = doc_score_norm + CENTROID_ALPHA * centroid_score_norm
            scores.append((final_score, i))
        scores.sort(key=lambda x: -x[0])
        pool = scores[:RERANK_POOL]
        selected: list[int] = []
        selected_sets: list[set[str]] = []
        max_rel = max((s for s, _ in pool), default=1.0) or 1.0
        remaining = list(pool)
        while remaining and len(selected) < TOP_K:
            best_j, best_val = -1, -1e18
            for j, (rel, di) in enumerate(remaining):
                rel_norm = rel / max_rel
                if not selected_sets:
                    div = 0.0
                else:
                    div = max(_jaccard(self._doc_token_sets[di], s) for s in selected_sets)
                val = MMR_LAMBDA * rel_norm - (1 - MMR_LAMBDA) * div
                if val > best_val:
                    best_val = val
                    best_j = j
            rel, di = remaining.pop(best_j)
            selected.append(di)
            selected_sets.append(self._doc_token_sets[di])
        return selected

    def _format_examples(self, query: str) -> str:
        idxs = self._select(query)
        if not idxs:
            return ""
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
        examples_section = self._format_examples(input)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_examples": len(self.examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]):
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._index_dirty = True
