"""LLM-distilled prototype: compress examples into pattern summaries per label.

Builds on label_centroid_bm25 but replaces raw example storage with LLM-generated
pattern summaries. After every 10 new examples per label, calls LLM to distill
patterns (e.g., "Protection reactions: add TMS/TBDMS to -OH; watch acid-sensitive").
At predict time, retrieves summaries via label-centroid BM25, then enriches prompt
with compressed knowledge instead of 16 raw examples.

Targets USPTO (+0.05-0.08) via precision gain from compressed SMILES patterns.
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below using these learned patterns:

{patterns_section}

**Problem:**
{input}

**Instructions:**
- Apply the patterns above to solve this problem
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

DISTILL_PROMPT = """Analyze these training examples and extract KEY PATTERNS:

{examples}

Identify:
1. Common structural motifs or transformation patterns
2. Distinguishing features in inputs or outputs
3. Domain-specific rules or constraints

Provide a concise summary (max 150 words) capturing the essential pattern.
Respond in JSON: {{"pattern": "[pattern summary]"}}"""

MAX_CHARS = 30000
DISTILL_THRESHOLD = 10
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
        f = tf.get(t, 0)
        if not f:
            continue
        denom = f + k1 * (1 - b + b * dl / max(1.0, avgdl))
        s += idf.get(t, 0.0) * f * (k1 + 1) / denom
    return s


class LlmDistilledPrototype(MemorySystem):
    """LLM-compressed pattern summaries per label."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.raw_examples: dict[str, list[dict]] = defaultdict(list)
        self.patterns: dict[str, str] = {}
        self._pattern_tokens: dict[str, list[str]] = {}
        self._label_centroid_tf: dict[str, Counter] = {}
        self._distilled_labels: set[str] = set()
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def _distill_label(self, label: str):
        examples = self.raw_examples[label][:20]
        ex_text = "\n\n".join([f"Input: {ex['input'][:150]}\nOutput: {ex['target'][:100]}"
                                for ex in examples[:10]])
        prompt = DISTILL_PROMPT.format(examples=ex_text)
        resp = self.call_llm(prompt)
        pattern = extract_json_field(resp, "pattern")
        self.patterns[label] = pattern
        self._pattern_tokens[label] = _tokenize(pattern)
        self._distilled_labels.add(label)

    def _ensure_index(self):
        for label, examples in self.raw_examples.items():
            if label not in self._distilled_labels and len(examples) >= DISTILL_THRESHOLD:
                self._distill_label(label)

        if not self.patterns:
            return

        all_tokens = list(self._pattern_tokens.values())
        self._idf = _bm25_idf(all_tokens)
        self._avgdl = sum(len(t) for t in all_tokens) / len(all_tokens) if all_tokens else 1.0

        for label, examples in self.raw_examples.items():
            all_toks = []
            for ex in examples:
                all_toks.extend(_tokenize(ex["input"]))
            self._label_centroid_tf[label] = Counter(all_toks)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._ensure_index()

        if not self.patterns:
            prompt = PROMPT_TEMPLATE.format(patterns_section="No patterns learned yet.", input=input)
            resp = self.call_llm(prompt)
            return extract_json_field(resp, "final_answer"), {"full_response": resp}

        q_toks = _tokenize(input)
        scores = []

        for label, pattern_toks in self._pattern_tokens.items():
            tf = Counter(pattern_toks)
            inst_score = _bm25_score(q_toks, tf, len(pattern_toks), self._avgdl, self._idf)

            if label in self._label_centroid_tf:
                centroid_tf = self._label_centroid_tf[label]
                cent_len = sum(centroid_tf.values())
                cent_score = _bm25_score(q_toks, centroid_tf, cent_len, self._avgdl, self._idf)
                total = inst_score + CENTROID_ALPHA * cent_score
            else:
                total = inst_score
            scores.append((total, label))

        scores.sort(reverse=True)
        top_labels = [label for _, label in scores[:5]]

        patterns_text = "\n\n".join([f"**Pattern {i+1} ({label}):**\n{self.patterns[label]}"
                                      for i, label in enumerate(top_labels)])

        if len(patterns_text) > MAX_CHARS:
            patterns_text = patterns_text[:MAX_CHARS]

        prompt = PROMPT_TEMPLATE.format(patterns_section=patterns_text, input=input)
        resp = self.call_llm(prompt)
        answer = extract_json_field(resp, "final_answer")

        return answer, {"full_response": resp, "top_labels": top_labels}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]):
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            self.raw_examples[r["ground_truth"]].append(ex)

    def get_state(self) -> str:
        return json.dumps({
            "raw_examples": {k: v for k, v in self.raw_examples.items()},
            "patterns": self.patterns,
            "distilled_labels": list(self._distilled_labels)
        }, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.raw_examples = defaultdict(list, data.get("raw_examples", {}))
        self.patterns = data.get("patterns", {})
        self._distilled_labels = set(data.get("distilled_labels", []))
        self._pattern_tokens = {label: _tokenize(pattern)
                                 for label, pattern in self.patterns.items()}
