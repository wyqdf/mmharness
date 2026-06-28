"""Adaptive retriever fusion: confidence-weighted score merging.

Builds on cross_retriever_vote (S2D frontier 0.900) by replacing voting with
score-level fusion. Each retriever (word/char/centroid BM25) contributes scores
weighted by its confidence (top BM25 score). High-confidence retrievers dominate,
reducing noise. Single LLM call on fused results vs 3 parallel calls.
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
CENTROID_ALPHA = 0.7


def _tokenize_word(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\(\)=#\[\]/\\@\+\-\.]", s.lower())


def _tokenize_charngram(s: str) -> list[str]:
    compact = re.sub(r"\s+", "", s.lower())
    ngrams = []
    for n in range(2, 4):
        if len(compact) >= n:
            for i in range(len(compact) - n + 1):
                ngrams.append(compact[i : i + n])
    return ngrams


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


class AdaptiveRetrieverFusion(MemorySystem):
    """Score-weighted fusion of word/char/centroid retrievers."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict] = []
        self._word_toks: list[list[str]] = []
        self._char_toks: list[list[str]] = []
        self._word_idf: dict[str, float] = {}
        self._char_idf: dict[str, float] = {}
        self._word_avgdl: float = 0.0
        self._char_avgdl: float = 0.0
        self._label_centroid: defaultdict[str, Counter] = defaultdict(Counter)
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

        self._label_centroid.clear()
        for i, ex in enumerate(self.examples):
            label = ex["target"]
            self._label_centroid[label].update(self._word_toks[i])
            self._label_centroid[label].update(self._char_toks[i])

        self._dirty = False

    def _fused_retrieve(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []

        q_word = _tokenize_word(query)
        q_char = _tokenize_charngram(query)

        # Score with each retriever
        word_scores = []
        char_scores = []
        centroid_scores = []

        for i in range(len(self.examples)):
            w_tf = Counter(self._word_toks[i])
            w_score = _bm25_score(q_word, w_tf, len(self._word_toks[i]), self._word_avgdl, self._word_idf)
            word_scores.append(w_score)

            c_tf = Counter(self._char_toks[i])
            c_score = _bm25_score(q_char, c_tf, len(self._char_toks[i]), self._char_avgdl, self._char_idf)
            char_scores.append(c_score)

            label = self.examples[i]["target"]
            if label in self._label_centroid:
                centroid_tf = self._label_centroid[label]
                centroid_len = sum(centroid_tf.values())
                cent_score = _bm25_score(q_word + q_char, centroid_tf, centroid_len, self._word_avgdl, self._word_idf)
            else:
                cent_score = 0.0
            centroid_scores.append(cent_score)

        # Compute confidence weights
        word_conf = max(word_scores) if word_scores else 0.0
        char_conf = max(char_scores) if char_scores else 0.0
        cent_conf = max(centroid_scores) if centroid_scores else 0.0
        total_conf = word_conf + char_conf + cent_conf + 1e-6

        w_weight = word_conf / total_conf
        c_weight = char_conf / total_conf
        cent_weight = cent_conf / total_conf

        # Fuse scores
        fused = []
        for i in range(len(self.examples)):
            score = w_weight * word_scores[i] + c_weight * char_scores[i] + cent_weight * centroid_scores[i]
            fused.append((score, i))

        fused.sort(reverse=True)
        return [idx for _, idx in fused[:TOP_K]]

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
        idxs = self._fused_retrieve(input)
        examples_section = self._format_examples(idxs)
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
        self._dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._dirty = True
