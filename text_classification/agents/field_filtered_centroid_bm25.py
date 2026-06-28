"""Field-filtered label-centroid BM25: auto-detect categorical field, pre-filter, then centroid-boosted retrieval.

Combines field_filtered_retrieval (official pattern) with label_centroid_bm25 (current LawBench frontier).
Auto-detects ONE discriminative categorical field from training examples, pre-filters candidates by field-value
match at predict time, then runs label-centroid BM25 within the filtered pool.
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
CANDIDATE_POOL = 64
MMR_LAMBDA = 0.7
CENTROID_ALPHA = 0.7
FIELD_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _\-]{0,40}):\s*(.{1,120})$")
MAX_HEADER_LINES = 30
MIN_BUCKET_COUNT = 2
MIN_VALUE_COUNT = 2
MIN_FILTERED_PEERS = 3


def _tokenize(s: str) -> list[str]:
    lower = s.lower()
    words = re.findall(r"[a-z0-9]+|[\(\)=#\[\]/\\@\+\-\.]", lower)
    compact = re.sub(r"\s+", "", lower)
    ngrams = []
    for n in range(2, 4):
        if len(compact) < n:
            continue
        for i in range(len(compact) - n + 1):
            ngrams.append(compact[i : i + n])
    return words + ngrams


def _parse_fields(text: str) -> dict[str, str]:
    out = {}
    for line in text.split("\n")[:MAX_HEADER_LINES]:
        m = FIELD_LINE_RE.match(line.strip())
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        if len(v) > 100 or k in out:
            continue
        out[k] = v
    return out


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


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class FieldFilteredCentroidBM25(MemorySystem):
    """Field-filtered + label-centroid BM25."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples = []
        self._doc_toks = []
        self._doc_tf = []
        self._doc_lens = []
        self._idf = {}
        self._avgdl = 0.0
        self._label_centroid_tf = {}
        self._label_centroid_len = {}
        self._avg_centroid_len = 0.0
        self._index_dirty = True
        self._disc_field = None
        self._field_buckets = {}

    def _detect_discriminative_field(self) -> str | None:
        field_values = defaultdict(lambda: defaultdict(int))
        for ex in self.examples:
            fields = _parse_fields(ex['input'])
            for k, v in fields.items():
                field_values[k][v] += 1
        candidates = []
        for field, val_counts in field_values.items():
            if len(val_counts) < MIN_BUCKET_COUNT:
                continue
            if any(cnt < MIN_VALUE_COUNT for cnt in val_counts.values()):
                continue
            candidates.append((field, len(val_counts)))
        return max(candidates, key=lambda x: x[1])[0] if candidates else None

    def _ensure_index(self):
        if not self._index_dirty:
            return
        n = len(self.examples)
        if n == 0:
            return
        self._doc_toks = [_tokenize(ex['input']) for ex in self.examples]
        self._doc_tf = [Counter(toks) for toks in self._doc_toks]
        self._doc_lens = [len(toks) for toks in self._doc_toks]
        self._avgdl = sum(self._doc_lens) / n if n > 0 else 0.0
        self._idf = _bm25_idf(self._doc_toks)
        label_toks = defaultdict(list)
        for ex, toks in zip(self.examples, self._doc_toks):
            label_toks[ex['target']].extend(toks)
        for label, toks in label_toks.items():
            self._label_centroid_tf[label] = Counter(toks)
            self._label_centroid_len[label] = len(toks)
        if self._label_centroid_len:
            self._avg_centroid_len = sum(self._label_centroid_len.values()) / len(self._label_centroid_len)
        self._disc_field = self._detect_discriminative_field()
        if self._disc_field:
            self._field_buckets = {}
            for idx, ex in enumerate(self.examples):
                fields = _parse_fields(ex['input'])
                val = fields.get(self._disc_field)
                if val:
                    if val not in self._field_buckets:
                        self._field_buckets[val] = []
                    self._field_buckets[val].append(idx)
        self._index_dirty = False

    def _select(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        query_fields = _parse_fields(query)
        pool_indices = list(range(len(self.examples)))
        if self._disc_field and self._disc_field in query_fields:
            val = query_fields[self._disc_field]
            if val in self._field_buckets and len(self._field_buckets[val]) >= MIN_FILTERED_PEERS:
                pool_indices = self._field_buckets[val]
        qtoks = _tokenize(query)
        scored = []
        for idx in pool_indices:
            inst_score = _bm25_score(qtoks, self._doc_tf[idx], self._doc_lens[idx], self._avgdl, self._idf)
            label = self.examples[idx]['target']
            centroid_tf = self._label_centroid_tf.get(label, Counter())
            centroid_len = self._label_centroid_len.get(label, 1)
            cent_score = _bm25_score(qtoks, centroid_tf, centroid_len, self._avg_centroid_len, self._idf) if centroid_tf else 0.0
            total = inst_score + CENTROID_ALPHA * cent_score
            scored.append((total, idx))
        scored.sort(reverse=True, key=lambda x: x[0])
        candidates = [idx for _, idx in scored[:CANDIDATE_POOL]]
        qtoks_set = set(qtoks)
        selected = []
        selected_sets = []
        for idx in candidates:
            if len(selected) >= TOP_K:
                break
            cand_toks = self._doc_toks[idx]
            cand_toks_set = set(cand_toks)
            relevance = _bm25_score(qtoks, self._doc_tf[idx], self._doc_lens[idx], self._avgdl, self._idf)
            max_sim = max((_jaccard(cand_toks_set, s) for s in selected_sets), default=0.0)
            mmr = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim * relevance
            if not selected or mmr > 0:
                selected.append(idx)
                selected_sets.append(cand_toks_set)
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
        return answer, {"full_response": response}

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
