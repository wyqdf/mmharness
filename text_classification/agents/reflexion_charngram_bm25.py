"""Reflexion-style error notes from leave-one-out self-test.

Builds on charngram_bm25_mmr by adding learned discriminative notes from
prediction errors. After training, runs leave-one-out self-test and groups
errors by confusion pairs. For each pair, asks LLM to write a note explaining
distinguishing features. At predict time, injects relevant notes above examples.
"""

import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below.

{notes_section}{examples_section}

**Problem:**
{input}

**Instructions:**
- The notes (if any) describe how to distinguish answers the model has previously confused.
- The examples are the most relevant prior cases.
- Use both to commit to one answer. Respond in JSON.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

SELFTEST_PROMPT = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- The examples above are the most relevant prior cases.
- Follow the patterns shown in the examples.
- Respond in JSON.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

NOTE_PROMPT = """You are reading a confusion case: in earlier self-testing, when given the QUERY below the model predicted "{predicted}" but the true answer was "{true}". You also see a few prior cases for each side.

QUERY:
{query}

Cases that share the TRUE answer ("{true}"):
{true_cases}

Cases that share the WRONG predicted answer ("{predicted}"):
{wrong_cases}

Write a SHORT discriminative note (<=60 words) that tells a future reader:
  - what surface features in the query point to "{true}" (and away from "{predicted}");
  - any keywords or phrasings that should trigger preferring "{true}" over "{predicted}".
Do not restate the answers. Be specific. Respond in JSON.

{{"note": "[the note]"}}"""

MAX_CHARS = 30000
TOP_K = 14
CANDIDATE_POOL = 64
MMR_LAMBDA = 0.7
SELF_TEST_N = 30
SELF_TEST_TOP_K = 12
NOTE_BUDGET = 5000
MAX_CASES_PER_NOTE = 3
SELFTEST_WORKERS = 8


def _tokenize(s: str) -> list[str]:
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


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ReflexionCharNgramBM25(MemorySystem):
    """BM25 over char-ngrams+words with reflexion error notes."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.notes: list[dict[str, str]] = []
        self._docs_tokens: list[list[str]] = []
        self._doc_token_sets: list[set[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._index_dirty = True
        self._notes_dirty = True
        self._index_lock = Lock()
        self._notes_lock = Lock()

    def _ensure_index(self):
        if not self._index_dirty:
            return
        with self._index_lock:
            if not self._index_dirty:
                return
            questions = [ex.get("raw_question") or ex["input"] for ex in self.examples]
            docs_tokens = [_tokenize(q) for q in questions]
            self._docs_tokens = docs_tokens
            self._doc_token_sets = [set(t) for t in docs_tokens]
            self._doc_tfs = [Counter(t) for t in docs_tokens]
            self._doc_lens = [len(t) for t in docs_tokens]
            n = len(docs_tokens)
            self._avgdl = (sum(self._doc_lens) / n) if n else 0.0
            self._idf = _bm25_idf(docs_tokens)
            self._index_dirty = False

    def _select(self, query: str, top_k: int = TOP_K) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        qtoks = _tokenize(query)
        n = len(self.examples)
        scores = [
            (_bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf), i)
            for i in range(n)
        ]
        scores.sort(key=lambda x: -x[0])
        pool = scores[:CANDIDATE_POOL]
        selected: list[int] = []
        selected_sets: list[set[str]] = []
        max_rel = max((s for s, _ in pool), default=1.0) or 1.0
        remaining = list(pool)
        while remaining and len(selected) < top_k:
            best_j, best_val = -1, -1e18
            for j, (rel, di) in enumerate(remaining):
                rel_norm = rel / max_rel
                if not selected_sets:
                    div = 0.0
                else:
                    div = max(_jaccard(self._doc_token_sets[di], s) for s in selected_sets)
                val = MMR_LAMBDA * rel_norm - (1 - MMR_LAMBDA) * div
                if val > best_val:
                    best_val = val
                    best_j = j
            rel, di = remaining.pop(best_j)
            selected.append(di)
            selected_sets.append(self._doc_token_sets[di])
        return selected

    def _run_self_test(self):
        if not self._notes_dirty or len(self.examples) < 10:
            return
        with self._notes_lock:
            if not self._notes_dirty or len(self.examples) < 10:
                return
            examples_snapshot = list(self.examples)
            n = min(SELF_TEST_N, len(examples_snapshot))
            rng = random.Random(42)
            test_idxs = rng.sample(range(len(examples_snapshot)), n)

            docs_tokens = [
                _tokenize(ex.get("raw_question") or ex["input"])
                for ex in examples_snapshot
            ]
            doc_tfs = [Counter(t) for t in docs_tokens]
            doc_lens = [len(t) for t in docs_tokens]
            doc_token_sets = [set(t) for t in docs_tokens]
            avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
            idf = _bm25_idf(docs_tokens)

        def select_from_snapshot(query: str, excluded: int) -> list[int]:
            qtoks = _tokenize(query)
            scores = [
                (_bm25_score(qtoks, doc_tfs[i], doc_lens[i], avgdl, idf), i)
                for i in range(len(examples_snapshot))
                if i != excluded
            ]
            scores.sort(key=lambda x: -x[0])
            pool = scores[:CANDIDATE_POOL]
            selected: list[int] = []
            selected_sets: list[set[str]] = []
            max_rel = max((s for s, _ in pool), default=1.0) or 1.0
            remaining = list(pool)
            while remaining and len(selected) < SELF_TEST_TOP_K:
                best_j, best_val = -1, -1e18
                for j, (rel, di) in enumerate(remaining):
                    rel_norm = rel / max_rel
                    div = (
                        max(_jaccard(doc_token_sets[di], s) for s in selected_sets)
                        if selected_sets
                        else 0.0
                    )
                    val = MMR_LAMBDA * rel_norm - (1 - MMR_LAMBDA) * div
                    if val > best_val:
                        best_val = val
                        best_j = j
                rel, di = remaining.pop(best_j)
                selected.append(di)
                selected_sets.append(doc_token_sets[di])
            return selected

        def test_one(i):
            ex = examples_snapshot[i]
            query = ex.get("raw_question", ex["input"])
            true = ex["target"]
            idxs = select_from_snapshot(query, i)
            parts = []
            for j in idxs:
                e = examples_snapshot[j]
                q = e.get("raw_question", e["input"])
                parts.append(f"Q: {q}\nA: {e['target']}")
            examples_section = "\n\n".join(parts)
            prompt = SELFTEST_PROMPT.format(examples_section=examples_section, input=query)
            try:
                response = self.call_llm(prompt)
                predicted = extract_json_field(response, "final_answer")
            except:
                predicted = ""
            return (i, query, predicted, true)

        with ThreadPoolExecutor(max_workers=SELFTEST_WORKERS) as executor:
            results = list(executor.map(test_one, test_idxs))

        errors = [r for r in results if r[2] != r[3] and r[2]]
        if not errors:
            self._notes_dirty = False
            return

        confusion_buckets: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        for i, query, predicted, true in errors:
            confusion_buckets[(predicted, true)].append((i, query))

        for (predicted, true), cases in confusion_buckets.items():
            if len(cases) == 0:
                continue
            first_i, first_query = cases[0]
            true_cases = [ex for ex in examples_snapshot if ex["target"] == true][:MAX_CASES_PER_NOTE]
            wrong_cases = [ex for ex in examples_snapshot if ex["target"] == predicted][:MAX_CASES_PER_NOTE]
            true_block = "\n".join([f"- {e.get('raw_question', e['input'])}" for e in true_cases])
            wrong_block = "\n".join([f"- {e.get('raw_question', e['input'])}" for e in wrong_cases])
            if not true_block:
                true_block = "(no prior cases)"
            if not wrong_block:
                wrong_block = "(no prior cases)"
            prompt = NOTE_PROMPT.format(query=first_query, predicted=predicted, true=true, true_cases=true_block, wrong_cases=wrong_block)
            try:
                response = self.call_llm(prompt)
                note_text = extract_json_field(response, "note")
                self.notes.append({"predicted": predicted, "true": true, "note": note_text})
            except:
                pass

        self._notes_dirty = False

    def _format_notes(self, retrieved_labels: list[str]) -> str:
        if not self.notes:
            return ""
        label_set = set(retrieved_labels)
        relevant = [n for n in self.notes if n["predicted"] in label_set]
        if not relevant:
            return ""
        parts = ["**Watch out for these confusions:**"]
        total = len(parts[0])
        for n in relevant:
            part = f"- Watch out: when you see features like those in the query, don't confuse '{n['predicted']}' with '{n['true']}'. {n['note']}"
            if total + len(part) > NOTE_BUDGET:
                break
            parts.append(part)
            total += len(part)
        return "\n".join(parts) + "\n\n"

    def _format_examples(self, query: str) -> tuple[str, list[str]]:
        idxs = self._select(query)
        if not idxs:
            return "", []
        parts = []
        total = 0
        labels = []
        for i in idxs:
            ex = self.examples[i]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) > MAX_CHARS:
                break
            parts.append(part)
            labels.append(ex["target"])
            total += len(part) + 2
        return "\n\n".join(parts), labels

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._run_self_test()
        examples_section, retrieved_labels = self._format_examples(input)
        notes_section = self._format_notes(retrieved_labels)
        prompt = PROMPT_TEMPLATE.format(notes_section=notes_section, examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_examples": len(self.examples), "num_notes": len(self.notes)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]):
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)
        self._index_dirty = True
        self._notes_dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples, "notes": self.notes}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.notes = data.get("notes", [])
        self._index_dirty = True
        self._notes_dirty = not bool(self.notes)
