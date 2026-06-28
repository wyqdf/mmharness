"""RRF triple fusion: word BM25 + char-ngram BM25 + SMILES n-gram similarity.

Uses Reciprocal Rank Fusion to combine three retrieval views:
1. Word-only BM25 for semantic term matching
2. Char-ngram BM25 for sub-word patterns (CJK, SMILES motifs)
3. SMILES n-gram Jaccard similarity for chemical structure matching

RRF reduces rank aggregation variance vs single retriever, helping USPTO.
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
RRF_K = 60
RRF_POOL = 200


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


def _extract_smiles(text: str) -> str:
    """Extract SMILES-like notation from input."""
    match = re.search(r'Input:\s*([^\n]+)', text)
    return match.group(1).strip() if match else text[:200]


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


def _rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion."""
    scores = Counter()
    for rank_list in rank_lists:
        for rank, idx in enumerate(rank_list[:RRF_POOL], start=1):
            scores[idx] += 1.0 / (k + rank)
    return [idx for idx, _ in scores.most_common()]


class RRFTripleFusion(MemorySystem):
    """RRF fusion of word, char-ngram, and SMILES retrievers."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._word_toks: list[list[str]] = []
        self._char_toks: list[list[str]] = []
        self._smiles_ngrams: list[set[str]] = []
        self._word_idf: dict[str, float] = {}
        self._char_idf: dict[str, float] = {}
        self._word_avgdl: float = 0.0
        self._char_avgdl: float = 0.0
        self._index_dirty = True

    def _ensure_index(self):
        if not self._index_dirty:
            return
        questions = [ex.get("raw_question") or ex["input"] for ex in self.examples]
        self._word_toks = [_tokenize_word(q) for q in questions]
        self._char_toks = [_tokenize_charngram(q) for q in questions]
        self._smiles_ngrams = [set(_tokenize_charngram(_extract_smiles(q))) for q in questions]

        n = len(self._word_toks)
        self._word_avgdl = sum(len(t) for t in self._word_toks) / n if n else 1.0
        self._char_avgdl = sum(len(t) for t in self._char_toks) / n if n else 1.0
        self._word_idf = _bm25_idf(self._word_toks)
        self._char_idf = _bm25_idf(self._char_toks)
        self._index_dirty = False

    def _select(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []

        # Retriever 1: Word BM25
        q_word = _tokenize_word(query)
        word_scores = []
        for i, doc_tok in enumerate(self._word_toks):
            tf = Counter(doc_tok)
            sc = _bm25_score(q_word, tf, len(doc_tok), self._word_avgdl, self._word_idf)
            word_scores.append((sc, i))
        word_scores.sort(reverse=True)
        word_ranks = [i for _, i in word_scores]

        # Retriever 2: Char-ngram BM25
        q_char = _tokenize_charngram(query)
        char_scores = []
        for i, doc_tok in enumerate(self._char_toks):
            tf = Counter(doc_tok)
            sc = _bm25_score(q_char, tf, len(doc_tok), self._char_avgdl, self._char_idf)
            char_scores.append((sc, i))
        char_scores.sort(reverse=True)
        char_ranks = [i for _, i in char_scores]

        # Retriever 3: SMILES n-gram Jaccard
        q_smiles_ng = set(_tokenize_charngram(_extract_smiles(query)))
        smiles_scores = []
        for i, doc_ng in enumerate(self._smiles_ngrams):
            sim = _jaccard(q_smiles_ng, doc_ng)
            smiles_scores.append((sim, i))
        smiles_scores.sort(reverse=True)
        smiles_ranks = [i for _, i in smiles_scores]

        # RRF fusion
        fused = _rrf_fuse([word_ranks, char_ranks, smiles_ranks])
        return fused[:TOP_K]

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
