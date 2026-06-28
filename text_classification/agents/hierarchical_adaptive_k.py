"""Hierarchical adaptive-k retrieval: cluster-then-diversify with query-adaptive k.

Builds on label_centroid_bm25 (LawBench 0.380) by adding:
1. Adaptive k based on query complexity (word count, clause count, long words)
2. Hierarchical retrieval: BM25 → cluster top-64 by label → MMR within clusters

Hypothesis: LawBench complex queries need more examples (k=18-24), simple need fewer (k=8-12).
Hierarchical ensures label diversity while MMR ensures input diversity.
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
CENTROID_ALPHA = 0.7
CANDIDATE_POOL = 64
MMR_LAMBDA = 0.7


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


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


class HierarchicalAdaptiveK(MemorySystem):
    """Adaptive k + hierarchical cluster-then-diversify retrieval."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict] = []
        self._doc_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._label_centroid: defaultdict[str, Counter] = defaultdict(Counter)
        self._dirty = True

    def _ensure_index(self):
        if not self._dirty or not self.examples:
            return
        self._doc_tokens = [_tokenize(ex["input"]) for ex in self.examples]
        self._idf = _bm25_idf(self._doc_tokens)
        self._avgdl = sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens) if self._doc_tokens else 1.0

        self._label_centroid.clear()
        for i, ex in enumerate(self.examples):
            label = ex["target"]
            self._label_centroid[label].update(self._doc_tokens[i])

        self._dirty = False

    def _compute_adaptive_k(self, query: str) -> int:
        words = len(query.split())
        clauses = len(re.findall(r'[,;]', query)) + 1
        long_words = sum(1 for w in query.split() if len(w) > 8)

        complexity = words * 0.3 + clauses * 2.0 + long_words * 1.5
        k = min(24, max(8, int(8 + complexity * 0.6)))
        return k

    def _retrieve_indices(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []

        k = self._compute_adaptive_k(query)
        q_tok = _tokenize(query)

        # Stage 1: BM25 + centroid scoring for top pool
        scores = []
        for i in range(len(self.examples)):
            doc_tok = self._doc_tokens[i]
            tf = Counter(doc_tok)
            inst_score = _bm25_score(q_tok, tf, len(doc_tok), self._avgdl, self._idf)

            label = self.examples[i]["target"]
            if label in self._label_centroid:
                centroid_tf = self._label_centroid[label]
                centroid_len = sum(centroid_tf.values())
                centroid_score = _bm25_score(q_tok, centroid_tf, centroid_len, self._avgdl, self._idf)
                total_score = inst_score + CENTROID_ALPHA * centroid_score
            else:
                total_score = inst_score

            scores.append((total_score, i))

        scores.sort(reverse=True)
        pool = [idx for _, idx in scores[:CANDIDATE_POOL]]

        # Stage 2: Cluster by label
        label_groups = defaultdict(list)
        for idx in pool:
            label = self.examples[idx]["target"]
            label_groups[label].append(idx)

        # Stage 3: MMR within each label cluster
        selected = []
        q_tok_set = set(q_tok)

        while len(selected) < k and label_groups:
            # Round-robin across labels
            for label in list(label_groups.keys()):
                if len(selected) >= k:
                    break
                candidates = label_groups[label]
                if not candidates:
                    del label_groups[label]
                    continue

                # MMR selection
                if not selected:
                    best_idx = candidates[0]
                else:
                    best_score = -float('inf')
                    best_idx = candidates[0]
                    for idx in candidates:
                        cand_tok = set(self._doc_tokens[idx])
                        relevance = len(q_tok_set & cand_tok) / max(1, len(q_tok_set))
                        max_sim = max(_jaccard(cand_tok, set(self._doc_tokens[s])) for s in selected)
                        mmr_score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim
                        if mmr_score > best_score:
                            best_score = mmr_score
                            best_idx = idx

                selected.append(best_idx)
                label_groups[label].remove(best_idx)
                if not label_groups[label]:
                    del label_groups[label]

            if not label_groups:
                break

        return selected[:k]

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
        idxs = self._retrieve_indices(input)
        examples_section = self._format_examples(idxs)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "k_used": len(idxs)}

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
