from __future__ import annotations

import json

from text_classification.node_memory import (
    format_meta_meta_context,
    generate_memory,
    root_memory,
    write_edge_trace,
)


def test_root_memory_mentions_scores() -> None:
    mem = root_memory("base", [0.1, 0.2], ["A", "B"])
    assert "base" in mem["summary"]
    assert "A=0.100" in mem["summary"]
    assert mem["refs"] == []


def test_generate_memory_falls_back_and_adds_ref() -> None:
    mem = generate_memory(
        {"summary": "Started.", "refs": []},
        "diff",
        [0.1, -0.2],
        3,
        ["A", "B"],
        llm=lambda prompt: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert "Started." in mem["summary"]
    assert mem["refs"] == [{"edge_id": 3, "trace_path": "traces/edge_3.jsonl"}]


def test_write_edge_trace_includes_diff(tmp_path) -> None:
    rel = write_edge_trace(tmp_path, 1, "p", "c", "--- p\n+++ c\n", [0.1], ["A"])
    data = json.loads((tmp_path / rel).read_text())

    assert data["diff"].startswith("--- p")
    assert data["delta_r"] == [0.1]


def test_format_context_includes_memory_and_diff() -> None:
    text = format_meta_meta_context(
        parent_node={"name": "p", "r_vec": [0.1], "memory": {"summary": "parent story"}, "code": "class P: pass"},
        frontier_nodes=[{"name": "p", "r_vec": [0.1], "memory": {"summary": "parent story"}}],
        recent_edges=[{"edge_id": 1, "parent": "p", "child": "c", "delta_r": [0.2], "diff": "+new"}],
        dims=["A"],
        show_memory=True,
        show_edges=True,
    )

    assert "Meta-Meta evolution state" in text
    assert "parent story" in text
    assert "+new" in text
    assert "A=+0.200" in text

