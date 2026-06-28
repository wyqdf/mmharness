"""BM25-style hybrid retrieval with TF-IDF, label diversity, and adaptive k.

Builds on fewshot_all with semantic retrieval instead of random sampling.
Uses TF-IDF for term-based relevance, ensures label diversity, and adapts
retrieval count based on available context.
"""

import json
import math
import random
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


class BM25HybridRetrieval(MemorySystem):
    """Hybrid retrieval using TF-IDF + label diversity + adaptive k."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.df: Counter = Counter()
        self.corpus_tokens: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _tfidf_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        query_counts = Counter(query_tokens)
        doc_counts = Counter(doc_tokens)
        score = 0.0
        for term, qtf in query_counts.items():
            if term in doc_counts:
                idf = math.log((len(self.corpus_tokens) + 1) / (self.df[term] + 1))
                score += qtf * doc_counts[term] * idf
        return score

    def _retrieve_examples(self, input: str, max_k: int = 50) -> list[dict]:
        if not self.examples:
            return []

        query_tokens = self._tokenize(input[:1000])
        scores = [self._tfidf_score(query_tokens, doc_tokens) for doc_tokens in self.corpus_tokens]

        # Get top candidates
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Diversify by label
        selected = []
        seen_labels = set()
        label_counts = Counter()

        for idx in ranked_indices:
            if len(selected) >= max_k:
                break
            label = self.examples[idx]['target']
            # Allow 3 per label max
            if label_counts[label] < 3:
                selected.append(self.examples[idx])
                seen_labels.add(label)
                label_counts[label] += 1

        return selected

    def _format_examples_section(self, input: str) -> str:
        retrieved = self._retrieve_examples(input, max_k=50)
        parts = []
        total_chars = 0

        for i, ex in enumerate(retrieved, 1):
            question = ex.get("raw_question", ex["input"])
            part = f"Q: {question}\nA: {ex['target']}"
            if total_chars + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total_chars += len(part) + 2

        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples_section = self._format_examples_section(input)
        prompt = PROMPT_TEMPLATE.format(
            examples_section=examples_section,
            input=input,
        )

        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")

        return answer, {"full_response": response, "num_examples": len(self.examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

            # Update TF-IDF index
            tokens = self._tokenize(r["input"][:1000])
            self.corpus_tokens.append(tokens)
            self.df.update(set(tokens))

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        # Rebuild index
        self.corpus_tokens = []
        self.df = Counter()
        for ex in self.examples:
            tokens = self._tokenize(ex["input"][:1000])
            self.corpus_tokens.append(tokens)
            self.df.update(set(tokens))
