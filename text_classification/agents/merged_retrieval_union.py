"""Merged retrieval union: combine multiple retrievers, single LLM call.

Builds on cross_retriever_vote (S2D 0.900) but replaces voting with merged
retrieval. Runs 3 retrievers (word-BM25, char-ngram BM25, label-centroid),
takes union of their top-k results, deduplicates, and makes single LLM call
on merged example set. Reduces voting noise while preserving complementary
retrieval signals.
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
TOP_K_PER_RETRIEVER = 10
CENTROID_ALPHA = 0.7


def _tokenize_word(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\(\)=#\[\]/\\@\+\-\.]", s.lower())


def _tokenize_charngram(s: str, n_min=2, n_max=3) -> list[str]:
    compact = re.sub(r"\s+", "", s.lower())
    ngrams = []
    for n in range(n_min, n_max + 1):
        if len(compact) < n:
            continue
        for i in range(len(compact) - n + 1):
            ngrams.append(compact[i : i + n])
    return ngrams


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


class MergedRetrievalUnion(MemorySystem):
    """Merged retrieval union from word/char/centroid BM25."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict] = []
        self._word_toks: list[list[str]] = []
        self._char_toks: list[list[str]] = []
        self._word_idf: dict[str, float] = {}
        self._char_idf: dict[str, float] = {}
        self._word_avgdl: float = 0.0
        self._char_avgdl: float = 0.0
        self._label_centroid: dict[str, Counter] = {}
        self._label_centroid_len: dict[str, int] = {}
        self._dirty = True

    def _ensure_index(self):
        if not self._dirty or not self.examples:
            return
        self._word_toks = [_tokenize_word(ex["input"]) for ex in self.examples]
        self._char_toks = [_tokenize_charngram(ex["input"]) for ex in self.examples]
        self._word_idf = _bm25_idf(self._word_toks)
        self._char_idf = _bm25_idf(self._char_toks)
        self._word_avgdl = sum(len(t) for t in self._word_toks) / len(self._word_toks) if self._word_toks else 1.0
        self._char_avgdl = sum(len(t) for t in self._char_toks) / len(self._char_toks) if self._char_toks else 1.0

        label_docs: defaultdict[str, list[list[str]]] = defaultdict(list)
        for ex, w_tok, c_tok in zip(self.examples, self._word_toks, self._char_toks):
            label_docs[ex["target"]].append(w_tok + c_tok)
        for label, docs in label_docs.items():
            all_toks = [t for doc in docs for t in doc]
            self._label_centroid[label] = Counter(all_toks)
            self._label_centroid_len[label] = len(all_toks)
        self._dirty = False

    def _retrieve_word(self, query: str, k: int) -> list[int]:
        q_tok = _tokenize_word(query)
        scores = []
        for i, doc_tok in enumerate(self._word_toks):
            tf = Counter(doc_tok)
            sc = _bm25_score(q_tok, tf, len(doc_tok), self._word_avgdl, self._word_idf)
            scores.append((sc, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]

    def _retrieve_char(self, query: str, k: int) -> list[int]:
        q_tok = _tokenize_charngram(query)
        scores = []
        for i, doc_tok in enumerate(self._char_toks):
            tf = Counter(doc_tok)
            sc = _bm25_score(q_tok, tf, len(doc_tok), self._char_avgdl, self._char_idf)
            scores.append((sc, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]

    def _retrieve_centroid(self, query: str, k: int) -> list[int]:
        q_tok = _tokenize_word(query) + _tokenize_charngram(query)
        q_idf = {**self._word_idf, **self._char_idf}
        q_avgdl = (self._word_avgdl + self._char_avgdl) / 2

        scores = []
        for i, ex in enumerate(self.examples):
            doc_tok = self._word_toks[i] + self._char_toks[i]
            tf = Counter(doc_tok)
            inst_score = _bm25_score(q_tok, tf, len(doc_tok), q_avgdl, q_idf)

            label = ex["target"]
            if label in self._label_centroid:
                centroid_score = _bm25_score(q_tok, self._label_centroid[label],
                                              self._label_centroid_len[label], q_avgdl, q_idf)
                total_score = inst_score + CENTROID_ALPHA * centroid_score
            else:
                total_score = inst_score
            scores.append((total_score, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]

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
        self._ensure_index()
        if not self.examples:
            prompt = PROMPT_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response}

        word_idxs = self._retrieve_word(input, TOP_K_PER_RETRIEVER)
        char_idxs = self._retrieve_char(input, TOP_K_PER_RETRIEVER)
        centroid_idxs = self._retrieve_centroid(input, TOP_K_PER_RETRIEVER)

        seen = set()
        merged = []
        sources = [centroid_idxs, word_idxs, char_idxs]
        max_len = max(len(s) for s in sources)
        for i in range(max_len):
            for source in sources:
                if i < len(source) and source[i] not in seen:
                    merged.append(source[i])
                    seen.add(source[i])

        examples_section = self._format_examples(merged)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")

        return answer, {"full_response": response, "merged_count": len(merged)}

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
