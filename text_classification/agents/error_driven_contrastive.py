"""Error-driven contrastive retrieval with confusion tracking.

Tracks prediction errors during learning and retrieves contrastive examples
to prevent repeated mistakes. Combines correct examples of predicted label
with error cases and correct examples of frequently confused labels.
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
- Pay attention to differences between similar cases
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000


class ErrorDrivenContrastive(MemorySystem):
    """Error-driven retrieval with contrastive examples."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.correct_by_label: defaultdict[str, list[dict]] = defaultdict(list)
        self.error_patterns: list[dict] = []
        self.confusion_matrix: Counter = Counter()

    def _retrieve_examples(self, input: str, max_k: int = 40) -> list[dict]:
        if not self.correct_by_label:
            return []

        # Hash-based pseudo-prediction to simulate label prediction
        seed = hash(input) & 0xFFFFFFFF
        rng = random.Random(seed)
        all_labels = list(self.correct_by_label.keys())
        predicted_label = rng.choice(all_labels) if all_labels else None

        if not predicted_label:
            return []

        selected = []

        # 1. Correct examples of predicted label (k=15)
        correct_examples = self.correct_by_label[predicted_label]
        if correct_examples:
            k = min(15, len(correct_examples))
            selected.extend(rng.sample(correct_examples, k))

        # 2. Error patterns where predicted label was confused (k=5)
        confused_errors = [
            e for e in self.error_patterns
            if e["predicted"] == predicted_label
        ]
        if confused_errors:
            k = min(5, len(confused_errors))
            for err in rng.sample(confused_errors, k):
                selected.append({
                    "input": err["input"],
                    "target": err["actual"],
                    "note": f"[Commonly confused with {err['predicted']}]"
                })

        # 3. Correct examples of frequently confused labels (k=10)
        top_confused = self.confusion_matrix.most_common(5)
        for (pred, actual), count in top_confused:
            if pred == predicted_label and actual in self.correct_by_label:
                contrastive = self.correct_by_label[actual]
                if contrastive:
                    k = min(2, len(contrastive))
                    selected.extend(rng.sample(contrastive, k))
                    if len(selected) >= 30:
                        break

        # 4. Fill remaining with diverse correct examples (k=10)
        remaining = max_k - len(selected)
        if remaining > 0:
            other_labels = [l for l in all_labels if l != predicted_label]
            for label in rng.sample(other_labels, min(5, len(other_labels))):
                examples = self.correct_by_label[label]
                if examples:
                    k = min(2, len(examples), remaining)
                    selected.extend(rng.sample(examples, k))
                    remaining -= k
                    if remaining <= 0:
                        break

        return selected[:max_k]

    def _format_examples_section(self, examples: list[dict]) -> str:
        if not examples:
            return ""

        parts = []
        total_chars = 0
        for ex in examples:
            question = ex.get("raw_question", ex["input"])
            note = ex.get("note", "")
            part = f"Q: {question}\nA: {ex['target']}"
            if note:
                part += f"\n{note}"
            if total_chars + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total_chars += len(part) + 2

        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._retrieve_examples(input)
        examples_section = self._format_examples_section(examples)
        prompt = PROMPT_TEMPLATE.format(
            examples_section=examples_section,
            input=input,
        )

        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")

        return answer, {"full_response": response, "num_examples": len(examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]

            if r["was_correct"]:
                self.correct_by_label[r["ground_truth"]].append(ex)
            else:
                self.error_patterns.append({
                    "input": r["input"],
                    "predicted": r["prediction"],
                    "actual": r["ground_truth"],
                })
                self.confusion_matrix[(r["prediction"], r["ground_truth"])] += 1

    def get_state(self) -> str:
        return json.dumps({
            "correct_by_label": {k: list(v) for k, v in self.correct_by_label.items()},
            "error_patterns": self.error_patterns,
            "confusion_matrix": {f"{k[0]}->{k[1]}": v for k, v in self.confusion_matrix.items()}
        }, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.correct_by_label = defaultdict(list, data.get("correct_by_label", {}))
        self.error_patterns = data.get("error_patterns", [])
        cm_data = data.get("confusion_matrix", {})
        self.confusion_matrix = Counter({
            tuple(k.split("->")):v for k, v in cm_data.items()
        })
