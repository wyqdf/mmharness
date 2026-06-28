"""Dual-vote ensemble: reaction-stratified + label-centroid voting.

Runs reaction_stratified (USPTO 0.267) and label_centroid (LawBench 0.380) in
parallel, votes on outputs. Combines USPTO chemical matching with LawBench
prototype priors.
"""

import json
import math
import re
from collections import Counter, defaultdict
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
        f = tf.get(t)
        if not f:
            continue
        denom = f + k1 * (1 - b + b * dl / max(1.0, avgdl))
        s += idf.get(t, 0.0) * f * (k1 + 1) / denom
    return s


class ReactionCentroidDualVote(MemorySystem):

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
        self.reaction_groups: defaultdict[str, list[int]] = defaultdict(list)
        self._index_dirty = True

    def _ensure_index(self):
        if not self._index_dirty:
            return
        questions = [ex.get("raw_question") or ex["input"] for ex in self.examples]
        docs_tokens = [_tokenize(q) for q in questions]
        doc_tfs = [Counter(t) for t in docs_tokens]
        doc_lens = [len(t) for t in docs_tokens]
        n = len(docs_tokens)
        avgdl = (sum(doc_lens) / n) if n else 0.0
        idf = _bm25_idf(docs_tokens)
        label_groups = defaultdict(list)
        for i, ex in enumerate(self.examples):
            label_groups[ex["target"]].append(i)
        label_centroid_tf = {}
        label_centroid_len = {}
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
        self._doc_tfs = doc_tfs
        self._doc_lens = doc_lens
        self._avgdl = avgdl
        self._idf = idf
        self._label_centroid_tf = label_centroid_tf
        self._label_centroid_len = label_centroid_len
        self._avg_centroid_len = avg_centroid_len
        self._index_dirty = False

    def _extract_reaction_type(self, text: str) -> str | None:
        match = re.search(r'reaction type is (\w+)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_smiles(self, text: str) -> str:
        match = re.search(r'Input:\s*([^\n]+)', text)
        return match.group(1).strip() if match else ""

    def _char_ngrams(self, text: str, n: int = 3) -> set[str]:
        return set(text[i:i+n] for i in range(len(text)-n+1))

    def _retrieve_reaction(self, input: str) -> list[int]:
        reaction_type = self._extract_reaction_type(input)
        if reaction_type and reaction_type in self.reaction_groups:
            candidate_indices = self.reaction_groups[reaction_type]
            query_smiles = self._extract_smiles(input)
            query_ngrams = self._char_ngrams(query_smiles)
            scored = []
            for idx in candidate_indices:
                ex_smiles = self._extract_smiles(self.examples[idx]['input'])
                ex_ngrams = self._char_ngrams(ex_smiles)
                overlap = len(query_ngrams & ex_ngrams)
                scored.append((overlap, idx))
            scored.sort(reverse=True, key=lambda x: x[0])
            return [idx for _, idx in scored[:TOP_K]]
        else:
            query_tokens = input.lower().split()[:200]
            query_counts = Counter(query_tokens)
            scored = []
            for idx, ex in enumerate(self.examples):
                doc_tokens = ex['input'].lower().split()[:200]
                doc_counts = Counter(doc_tokens)
                score = sum(
                    query_counts[term] * doc_counts[term]
                    for term in query_counts if term in doc_counts
                )
                scored.append((score, idx))
            scored.sort(reverse=True, key=lambda x: x[0])
            return [idx for _, idx in scored[:TOP_K]]

    def _retrieve_centroid(self, query: str) -> list[int]:
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
        return [i for _, i in scores[:TOP_K]]

    def _format_examples(self, idxs: list[int]) -> str:
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

    def _predict_with_retriever(self, input: str, idxs: list[int]) -> str:
        examples_section = self._format_examples(idxs)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        return extract_json_field(response, "final_answer")

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        if not self.examples:
            prompt = PROMPT_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "num_examples": 0}

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(self._retrieve_reaction, input)
            f2 = executor.submit(self._retrieve_centroid, input)
            idxs1 = f1.result()
            idxs2 = f2.result()

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(self._predict_with_retriever, input, idxs1)
            f2 = executor.submit(self._predict_with_retriever, input, idxs2)
            ans1 = f1.result()
            ans2 = f2.result()

        votes = [ans1, ans2]
        counts = Counter(votes)
        max_count = max(counts.values())
        winners = [p for p, c in counts.items() if c == max_count]
        answer = winners[0]

        return answer, {"votes": votes, "num_examples": len(self.examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]):
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)
            reaction_type = self._extract_reaction_type(r["input"])
            if reaction_type:
                self.reaction_groups[reaction_type].append(len(self.examples) - 1)
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "reaction_groups": {k: v for k, v in self.reaction_groups.items()}
        }, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        rg = data.get("reaction_groups", {})
        self.reaction_groups = defaultdict(list, {k: v for k, v in rg.items()})
        self._index_dirty = True
