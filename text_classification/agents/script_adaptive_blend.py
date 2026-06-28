"""Script-adaptive retrieval blend: word/char-ngram BM25 weighted by ASCII ratio.

Builds on charngram_bm25_mmr by maintaining TWO separate BM25 indices (word-only,
char-ngram-only) and blending their normalized scores based on query's ASCII ratio.
This lets word-BM25 dominate on ASCII-heavy queries (USPTO, S2D) while char-ngram
handles CJK text (LawBench).
"""

import json
import math
import re
from collections import Counter
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


def _ascii_ratio(s: str) -> float:
    non_ws = [c for c in s if not c.isspace()]
    return sum(ord(c) < 128 for c in non_ws) / len(non_ws) if non_ws else 1.0


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
    return len(a & b) / len(a | b) if a and b else 0.0


class ScriptAdaptiveBlend(MemorySystem):
    """Script-adaptive BM25 blend with MMR reranking."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict] = []
        self._word_toks: list[list[str]] = []
        self._char_toks: list[list[str]] = []
        self._word_idf: dict[str, float] = {}
        self._char_idf: dict[str, float] = {}
        self._word_avgdl: float = 0.0
        self._char_avgdl: float = 0.0
        self._dirty = True

    def _ensure_index(self):
        if not self._dirty or not self.examples:
            return
        self._word_toks = [_tokenize_word(ex["input"]) for ex in self.examples]
        self._char_toks = [_tokenize_charngram(ex["input"]) for ex in self.examples]
        self._word_idf = _bm25_idf(self._word_toks)
        self._char_idf = _bm25_idf(self._char_toks)
        self._word_avgdl = sum(len(t) for t in self._word_toks) / len(self._word_toks)
        self._char_avgdl = sum(len(t) for t in self._char_toks) / len(self._char_toks)
        self._dirty = False

    def _retrieve(self, query: str, k: int) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []

        ar = _ascii_ratio(query)
        q_word = _tokenize_word(query)
        q_char = _tokenize_charngram(query)

        # Score with both indices
        word_scores = []
        char_scores = []
        for i in range(len(self.examples)):
            w_tf = Counter(self._word_toks[i])
            c_tf = Counter(self._char_toks[i])
            w_sc = _bm25_score(q_word, w_tf, len(self._word_toks[i]), self._word_avgdl, self._word_idf)
            c_sc = _bm25_score(q_char, c_tf, len(self._char_toks[i]), self._char_avgdl, self._char_idf)
            word_scores.append(w_sc)
            char_scores.append(c_sc)

        # Normalize and blend
        max_w = max(word_scores) if max(word_scores) > 0 else 1.0
        max_c = max(char_scores) if max(char_scores) > 0 else 1.0
        blended = [ar * (w / max_w) + (1 - ar) * (c / max_c)
                   for w, c in zip(word_scores, char_scores)]

        # Top candidates for MMR
        pool_k = min(CANDIDATE_POOL, len(self.examples))
        candidates = sorted(range(len(blended)), key=lambda i: blended[i], reverse=True)[:pool_k]

        # MMR rerank
        selected = []
        q_set = set(q_word + q_char)
        while len(selected) < k and candidates:
            best_idx = None
            best_mmr = -1e9
            for idx in candidates:
                rel = blended[idx]
                if not selected:
                    div = 0.0
                else:
                    div = max(_jaccard(set(self._word_toks[idx] + self._char_toks[idx]),
                                       set(self._word_toks[s] + self._char_toks[s]))
                              for s in selected)
                mmr = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * div
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            if best_idx is not None:
                selected.append(best_idx)
                candidates.remove(best_idx)
            else:
                break
        return selected

    def _format_examples(self, query: str) -> str:
        idxs = self._retrieve(query, TOP_K)
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
        self._dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._dirty = True


