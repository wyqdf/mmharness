"""Confusion-aware prototype memory with error tracking.

Maintains per-class prototypes, tracks confusion pairs from errors,
and retrieves both similar examples and hard negatives for better discrimination.
"""

import json
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
- Pay special attention to differences between similar cases
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000


class ConfusionPrototypeMemory(MemorySystem):
    """Prototype-based memory with confusion tracking."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.class_prototypes: defaultdict[str, list[dict]] = defaultdict(list)
        self.confusion_pairs: Counter = Counter()

    def _retrieve_examples(self, input: str, max_k: int = 40) -> list[dict]:
        if not self.examples:
            return []

        # Simple heuristic: use hash for diversity
        seed = hash(input) & 0xFFFFFFFF
        rng = random.Random(seed)

        # Get label distribution
        all_labels = [ex['target'] for ex in self.examples]
        label_counts = Counter(all_labels)

        # Sample from each class proportionally
        selected = []
        samples_per_class = max(1, max_k // len(self.class_prototypes))

        for label, proto_list in self.class_prototypes.items():
            n_samples = min(samples_per_class, len(proto_list))
            selected.extend(rng.sample(proto_list, n_samples))

        # Add confused pairs for common errors
        if self.confusion_pairs:
            for (true_label, pred_label), count in self.confusion_pairs.most_common(3):
                # Add hard negatives from confused class
                if pred_label in self.class_prototypes:
                    neg_samples = self.class_prototypes[pred_label][:2]
                    selected.extend(neg_samples)

        # Shuffle and limit
        rng.shuffle(selected)
        return selected[:max_k]

    def _format_examples_section(self, input: str) -> str:
        retrieved = self._retrieve_examples(input, max_k=40)
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
            self.class_prototypes[r["ground_truth"]].append(ex)

            # Track confusion pairs
            if not r["was_correct"]:
                self.confusion_pairs[(r["ground_truth"], r["prediction"])] += 1

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "confusion_pairs": list(self.confusion_pairs.items())
        }, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.confusion_pairs = Counter(dict(data.get("confusion_pairs", [])))

        # Rebuild prototypes
        self.class_prototypes = defaultdict(list)
        for ex in self.examples:
            self.class_prototypes[ex["target"]].append(ex)
