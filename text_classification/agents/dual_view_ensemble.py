"""Dual-view ensemble vote: two LLM calls with different retrieval views, then vote.

Maintains two independent BM25 indices (word tokens, char-ngrams). At predict
time, retrieves two different top-K example sets, makes two parallel LLM calls,
and majority-votes on the answers. Tie-break prefers the retriever with higher
best score.
"""

import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
TOP_K = 14


def _tok_word(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\(\)=#\[\]/\\@\+\-\.]", s.lower())


def _tok_charngram(s: str) -> list[str]:
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


class _BM25Index:
    def __init__(self):
        self.docs_tokens: list[list[str]] = []
        self.doc_tfs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    def build(self, questions: list[str], tokenizer):
        self.docs_tokens = [tokenizer(q) for q in questions]
        self.doc_tfs = [Counter(t) for t in self.docs_tokens]
        self.doc_lens = [len(t) for t in self.docs_tokens]
        n = len(self.docs_tokens)
        self.avgdl = (sum(self.doc_lens) / n) if n else 0.0
        self.idf = _bm25_idf(self.docs_tokens)

    def scores(self, qtoks: list[str]) -> list[float]:
        return [
            _bm25_score(qtoks, self.doc_tfs[i], self.doc_lens[i], self.avgdl, self.idf)
            for i in range(len(self.docs_tokens))
        ]


class DualViewEnsemble(MemorySystem):
    """Two LLM calls per query, one per retrieval view (word/char). Vote."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._word_idx = _BM25Index()
        self._cn_idx = _BM25Index()
        self._index_dirty = True

    def _ensure_index(self):
        if not self._index_dirty:
            return
        questions = [ex.get("raw_question") or ex["input"] for ex in self.examples]
        self._word_idx.build(questions, _tok_word)
        self._cn_idx.build(questions, _tok_charngram)
        self._index_dirty = False

    def _topk_word(self, query: str, k: int) -> tuple[list[int], float]:
        if not self.examples:
            return [], 0.0
        s = self._word_idx.scores(_tok_word(query))
        idxs = sorted(range(len(s)), key=lambda i: -s[i])[:k]
        best = max(s) if s else 0.0
        return idxs, best

    def _topk_char(self, query: str, k: int) -> tuple[list[int], float]:
        if not self.examples:
            return [], 0.0
        s = self._cn_idx.scores(_tok_charngram(query))
        idxs = sorted(range(len(s)), key=lambda i: -s[i])[:k]
        best = max(s) if s else 0.0
        return idxs, best

    def _format_examples_block(self, idxs: list[int], budget: int) -> str:
        parts = []
        total = 0
        for i in idxs:
            ex = self.examples[i]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) > budget:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def _call_one(self, idxs: list[int], input: str) -> tuple[str, str]:
        examples_section = self._format_examples_block(idxs, MAX_CHARS)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer.strip(), response

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._ensure_index()
        if not self.examples:
            answer, response = self._call_one([], input)
            return answer, {"full_response": response, "num_examples": 0}

        idxs_word, best_w = self._topk_word(input, TOP_K)
        idxs_char, best_c = self._topk_char(input, TOP_K)

        if idxs_word == idxs_char:
            answer, response = self._call_one(idxs_word, input)
            return answer, {"full_response": response, "num_examples": len(self.examples)}

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda x: self._call_one(x[0], input), [(idxs_word,), (idxs_char,)]))
        ans_w, resp_w = results[0]
        ans_c, resp_c = results[1]

        if ans_w == ans_c:
            winner = ans_w
        else:
            winner = ans_w if best_w >= best_c else ans_c

        return winner, {"full_response": resp_w, "votes": [ans_w, ans_c], "num_examples": len(self.examples)}

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
