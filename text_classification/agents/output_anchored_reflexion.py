"""Output-anchored retrieval with reflexion error notes.

Stage 1: predict via label-centroid BM25.
Stage 2 (gated): retrieve training examples whose TARGETS match the tentative answer,
then ask LLM to verify/revise. Adds reflexion-style error notes from leave-one-out self-test.
"""

import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
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

VERIFY_PROMPT = """You gave a tentative answer: "{tentative}".

Below are training cases whose TARGET matches your tentative:
{sibling_section}

**Problem:**
{input}

**Instructions:**
- If the sibling inputs resemble the problem, your tentative is likely correct.
- If the sibling inputs are dissimilar, reconsider.
- Commit to one answer. Respond in JSON.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

SELFTEST_PROMPT = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples.
- Respond in JSON.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

NOTE_PROMPT = """You are reading a confusion case: when given the QUERY below the model predicted "{predicted}" but the true answer was "{true}".

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
CENTROID_ALPHA = 0.7
GATE_OVERLAP = 0.3
SIBLING_MIN_OVERLAP = 0.4
MAX_SIBLINGS = 5
SELF_TEST_N = 30
NOTE_BUDGET = 5000
MAX_CASES_PER_NOTE = 3


def _tokenize(s: str) -> list[str]:
    lower = s.lower()
    words = re.findall(r"[a-z0-9]+|[\(\)=#\[\]/\\@\+\-\.]", lower)
    compact = re.sub(r"\s+", "", lower)
    ngrams = []
    for n in range(2, 4):
        if len(compact) < n:
            continue
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


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_overlap(text1: str, text2: str) -> float:
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


class OutputAnchoredReflexion(MemorySystem):
    """Output-anchored + reflexion."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples = []
        self.notes = []
        self._doc_toks = []
        self._doc_tf = []
        self._doc_lens = []
        self._idf = {}
        self._avgdl = 0.0
        self._label_centroid_tf = {}
        self._label_centroid_len = {}
        self._avg_centroid_len = 0.0
        self._target_toks = []
        self._target_tf = []
        self._target_lens = []
        self._target_idf = {}
        self._target_avgdl = 0.0
        self._index_dirty = True
        self._notes_dirty = True

    def _ensure_index(self):
        if not self._index_dirty:
            return
        n = len(self.examples)
        if n == 0:
            return
        self._doc_toks = [_tokenize(ex['input']) for ex in self.examples]
        self._doc_tf = [Counter(toks) for toks in self._doc_toks]
        self._doc_lens = [len(toks) for toks in self._doc_toks]
        self._avgdl = sum(self._doc_lens) / n if n > 0 else 0.0
        self._idf = _bm25_idf(self._doc_toks)
        label_toks = defaultdict(list)
        for ex, toks in zip(self.examples, self._doc_toks):
            label_toks[ex['target']].extend(toks)
        for label, toks in label_toks.items():
            self._label_centroid_tf[label] = Counter(toks)
            self._label_centroid_len[label] = len(toks)
        if self._label_centroid_len:
            self._avg_centroid_len = sum(self._label_centroid_len.values()) / len(self._label_centroid_len)
        self._target_toks = [_tokenize(ex['target']) for ex in self.examples]
        self._target_tf = [Counter(toks) for toks in self._target_toks]
        self._target_lens = [len(toks) for toks in self._target_toks]
        self._target_avgdl = sum(self._target_lens) / n if n > 0 else 0.0
        self._target_idf = _bm25_idf(self._target_toks)
        self._index_dirty = False
        self._notes_dirty = True

    def _select(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        qtoks = _tokenize(query)
        scored = []
        for idx in range(len(self.examples)):
            inst_score = _bm25_score(qtoks, self._doc_tf[idx], self._doc_lens[idx], self._avgdl, self._idf)
            label = self.examples[idx]['target']
            centroid_tf = self._label_centroid_tf.get(label, Counter())
            centroid_len = self._label_centroid_len.get(label, 1)
            cent_score = _bm25_score(qtoks, centroid_tf, centroid_len, self._avg_centroid_len, self._idf) if centroid_tf else 0.0
            total = inst_score + CENTROID_ALPHA * cent_score
            scored.append((total, idx))
        scored.sort(reverse=True, key=lambda x: x[0])
        candidates = [idx for _, idx in scored[:CANDIDATE_POOL]]
        qtoks_set = set(qtoks)
        selected = []
        selected_sets = []
        for idx in candidates:
            if len(selected) >= TOP_K:
                break
            cand_toks = self._doc_toks[idx]
            cand_toks_set = set(cand_toks)
            relevance = _bm25_score(qtoks, self._doc_tf[idx], self._doc_lens[idx], self._avgdl, self._idf)
            max_sim = max((_jaccard(cand_toks_set, s) for s in selected_sets), default=0.0)
            mmr = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim * relevance
            if not selected or mmr > 0:
                selected.append(idx)
                selected_sets.append(cand_toks_set)
        return selected

    def _find_siblings(self, tentative: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        tentative_toks = _tokenize(tentative)
        scored = []
        for idx in range(len(self.examples)):
            score = _bm25_score(tentative_toks, self._target_tf[idx], self._target_lens[idx], self._target_avgdl, self._target_idf)
            overlap = _token_overlap(tentative, self.examples[idx]['target'])
            if overlap >= SIBLING_MIN_OVERLAP:
                scored.append((score, idx))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [idx for _, idx in scored[:MAX_SIBLINGS]]

    def _format_notes(self) -> str:
        if not self.notes:
            return ""
        parts = []
        total = 0
        for note_data in self.notes:
            part = f"**Confusion Note ({note_data['predicted']} vs {note_data['true']}):** {note_data['note']}"
            if total + len(part) > NOTE_BUDGET:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts) + "\n\n" if parts else ""

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

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._generate_notes_if_needed()
        stage1_idxs = self._select(input)
        notes_section = self._format_notes()
        examples_section = self._format_examples(stage1_idxs)
        prompt = PROMPT_TEMPLATE.format(notes_section=notes_section, examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        tentative = extract_json_field(response, "final_answer")
        if not tentative or not self.examples:
            return tentative, {"full_response": response, "stage": 1}
        max_overlap = max((_token_overlap(tentative, ex['target']) for ex in self.examples), default=0.0)
        sibling_idxs = self._find_siblings(tentative)
        if max_overlap >= GATE_OVERLAP and len(sibling_idxs) >= 2:
            sibling_section = self._format_examples(sibling_idxs)
            verify_prompt = VERIFY_PROMPT.format(tentative=tentative, sibling_section=sibling_section, input=input)
            verify_response = self.call_llm(verify_prompt)
            final_answer = extract_json_field(verify_response, "final_answer")
            if final_answer:
                return final_answer, {"full_response": verify_response, "stage": 2, "tentative": tentative}
        return tentative, {"full_response": response, "stage": 1}

    def _generate_notes_if_needed(self):
        if not self._notes_dirty or len(self.examples) < 10:
            return
        self._ensure_index()
        sample_idxs = random.sample(range(len(self.examples)), min(SELF_TEST_N, len(self.examples)))
        confusion_pairs = Counter()
        test_results = []

        def self_test(idx):
            ex = self.examples[idx]
            others = [i for i in range(len(self.examples)) if i != idx]
            if not others:
                return None
            other_idxs = random.sample(others, min(SELF_TEST_N, len(others)))
            examples_section = self._format_examples(other_idxs[:12])
            prompt = SELFTEST_PROMPT.format(examples_section=examples_section, input=ex['input'])
            try:
                response = self.call_llm(prompt)
                predicted = extract_json_field(response, "final_answer")
                if predicted and predicted != ex['target']:
                    return (idx, predicted, ex['target'])
            except:
                pass
            return None

        with ThreadPoolExecutor(max_workers=4) as executor:
            for result in executor.map(self_test, sample_idxs):
                if result:
                    test_results.append(result)

        for idx, predicted, true in test_results:
            confusion_pairs[(predicted, true)] += 1

        for (predicted, true), count in confusion_pairs.most_common(5):
            if count < 2:
                break
            true_cases = [ex for ex in self.examples if ex['target'] == true][:MAX_CASES_PER_NOTE]
            wrong_cases = [ex for ex in self.examples if ex['target'] == predicted][:MAX_CASES_PER_NOTE]
            query = next((self.examples[idx]['input'] for idx, p, t in test_results if p == predicted and t == true), "")
            if not query or not true_cases or not wrong_cases:
                continue
            true_section = "\n".join(f"- {ex.get('raw_question', ex['input'])}" for ex in true_cases)
            wrong_section = "\n".join(f"- {ex.get('raw_question', ex['input'])}" for ex in wrong_cases)
            note_prompt = NOTE_PROMPT.format(predicted=predicted, true=true, query=query, true_cases=true_section, wrong_cases=wrong_section)
            try:
                note_response = self.call_llm(note_prompt)
                note_text = extract_json_field(note_response, "note")
                if note_text:
                    self.notes.append({"predicted": predicted, "true": true, "note": note_text})
            except:
                pass

        self._notes_dirty = False

    def learn_from_batch(self, batch_results: list[dict[str, Any]]):
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples, "notes": self.notes}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.notes = data.get("notes", [])
        self._index_dirty = True
        self._notes_dirty = len(self.notes) == 0
        for idx in range(len(self.examples)):
            inst_score = _bm25_score(qtoks, self._doc_tf[idx], self._doc_lens[idx], self._avgdl, self._idf)
            label = self.examples[idx]['target']
            centroid_tf = self._label_centroid_tf.get(label, Counter())
            centroid_len = self._label_centroid_len.get(label, 1)
            cent_score = _bm25_score(qtoks, centroid_tf, centroid_len, self._avg_centroid_len, self._idf) if centroid_tf else 0.0
            total = inst_score + CENTROID_ALPHA * cent_score
            scored.append((total, idx))
        scored.sort(reverse=True, key=lambda x: x[0])
        candidates = [idx for _, idx in scored[:CANDIDATE_POOL]]
        qtoks_set = set(qtoks)
        selected = []
        selected_sets = []
        for idx in candidates:
            if len(selected) >= TOP_K:
                break
            cand_toks = self._doc_toks[idx]
            cand_toks_set = set(cand_toks)
            relevance = _bm25_score(qtoks, self._doc_tf[idx], self._doc_lens[idx], self._avgdl, self._idf)
            max_sim = max((_jaccard(cand_toks_set, s) for s in selected_sets), default=0.0)
            mmr = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim * relevance
            if not selected or mmr > 0:
                selected.append(idx)
                selected_sets.append(cand_toks_set)
        return selected

    def _select_siblings(self, tentative: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        tent_toks = _tokenize(tentative)
        scored = []
        for idx in range(len(self.examples)):
            score = _bm25_score(tent_toks, self._target_tf[idx], self._target_lens[idx], self._target_avgdl, self._target_idf)
            overlap = _token_overlap(tentative, self.examples[idx]['target'])
            if overlap >= SIBLING_MIN_OVERLAP:
                scored.append((score, idx))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [idx for _, idx in scored[:MAX_SIBLINGS]]

    def _format_notes(self) -> str:
        if not self.notes:
            return ""
        parts = []
        total = 0
        for note_obj in self.notes:
            part = f"- {note_obj['note']}"
            if total + len(part) > NOTE_BUDGET:
                break
            parts.append(part)
            total += len(part) + 1
        return "**Confusion notes:**\n" + "\n".join(parts) + "\n\n" if parts else ""

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

    def _format_siblings(self, sib_idxs: list[int]) -> str:
        if not sib_idxs:
            return ""
        parts = []
        for i in sib_idxs:
            ex = self.examples[i]
            q = ex.get("raw_question", ex["input"])
            parts.append(f"- Input: {q}")
        return "\n".join(parts)

    def _gate_check(self, tentative: str) -> bool:
        if not self.examples:
            return False
        max_overlap = max((_token_overlap(tentative, ex['target']) for ex in self.examples), default=0.0)
        if max_overlap < GATE_OVERLAP:
            return False
        sib_count = sum(1 for ex in self.examples if _token_overlap(tentative, ex['target']) >= SIBLING_MIN_OVERLAP)
        return sib_count >= 2

    def _self_test(self):
        if not self._notes_dirty or len(self.examples) < 10:
            return
        rng = random.Random(42)
        test_idxs = rng.sample(range(len(self.examples)), min(SELF_TEST_N, len(self.examples)))
        confusion = []
        for idx in test_idxs:
            ex = self.examples[idx]
            leave_out = [i for i in range(len(self.examples)) if i != idx]
            leave_out_ex = [self.examples[i] for i in leave_out]
            formatted = "\n\n".join(f"Q: {e.get('raw_question', e['input'])}\nA: {e['target']}" for e in leave_out_ex[:20])
            prompt = SELFTEST_PROMPT.format(examples_section=formatted, input=ex['input'])
            response = self.call_llm(prompt)
            pred = extract_json_field(response, "final_answer")
            if pred != ex['target']:
                confusion.append({'idx': idx, 'predicted': pred, 'true': ex['target']})
        if not confusion:
            self._notes_dirty = False
            return
        for conf in confusion[:10]:
            idx = conf['idx']
            pred = conf['predicted']
            true = conf['true']
            true_cases = [ex for ex in self.examples if ex['target'] == true][:MAX_CASES_PER_NOTE]
            pred_cases = [ex for ex in self.examples if ex['target'] == pred][:MAX_CASES_PER_NOTE]
            if not true_cases or not pred_cases:
                continue
            true_fmt = "\n".join(f"- {e.get('raw_question', e['input'])}" for e in true_cases)
            pred_fmt = "\n".join(f"- {e.get('raw_question', e['input'])}" for e in pred_cases)
            note_prompt = NOTE_PROMPT.format(
                query=self.examples[idx]['input'],
                predicted=pred,
                true=true,
                true_cases=true_fmt,
                wrong_cases=pred_fmt,
            )
            note_response = self.call_llm(note_prompt)
            note_text = extract_json_field(note_response, "note")
            self.notes.append({'note': note_text, 'pair': (pred, true)})
        self._notes_dirty = False

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._ensure_index()
        if self._notes_dirty:
            self._self_test()
        notes_section = self._format_notes()
        examples_section = self._format_examples(input)
        prompt = PROMPT_TEMPLATE.format(notes_section=notes_section, examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        tentative = extract_json_field(response, "final_answer")
        if not self._gate_check(tentative):
            return tentative, {"full_response": response, "stage": 1}
        sib_idxs = self._select_siblings(tentative)
        if not sib_idxs:
            return tentative, {"full_response": response, "stage": 1}
        sib_section = self._format_siblings(sib_idxs)
        verify_prompt = VERIFY_PROMPT.format(tentative=tentative, sibling_section=sib_section, input=input)
        verify_response = self.call_llm(verify_prompt)
        final_answer = extract_json_field(verify_response, "final_answer")
        if not final_answer:
            final_answer = tentative
        return final_answer, {"full_response": verify_response, "stage": 2, "tentative": tentative}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]):
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples, "notes": self.notes}, indent=2)

    def set_state(self, state: str):
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.notes = data.get("notes", [])
        self._index_dirty = True
        self._notes_dirty = False
