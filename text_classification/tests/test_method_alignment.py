from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from text_classification.lineage import LineageGraph
from text_classification.node_memory import generate_memory, write_edge_trace

TEXT_CLASSIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEXT_CLASSIFICATION_DIR))

import meta_harness


def test_s2_shows_edges_without_memory() -> None:
    cfg = yaml.safe_load((TEXT_CLASSIFICATION_DIR / "config_s2.yaml").read_text())

    assert cfg["meta_meta"]["enabled"] is True
    assert cfg["meta_meta"]["vector_reward"] is True
    assert cfg["meta_meta"]["show_edges"] is True
    assert cfg["meta_meta"]["show_memory"] is False


def test_s3_enables_calibration() -> None:
    cfg = yaml.safe_load((TEXT_CLASSIFICATION_DIR / "config_s3.yaml").read_text())

    assert cfg["meta_meta"]["calibration"] is True


def test_prompt_mentions_active_config_path(tmp_path) -> None:
    old_config = meta_harness.CONFIG_PATH
    try:
        meta_harness.CONFIG_PATH = tmp_path / "config_s3.yaml"
        text = meta_harness.render_task_prompt(1, 3)
    finally:
        meta_harness.CONFIG_PATH = old_config

    assert "Active config:" in text
    assert "config_s3.yaml" in text


def test_calibration_row_written(tmp_path) -> None:
    old_logs = meta_harness.LOGS_DIR
    try:
        meta_harness.LOGS_DIR = tmp_path
        node = {"delta_r": [0.1, -0.2, 0.0]}
        cand = {"name": "child", "predicted_delta_r": {"A": 0.1, "B": -0.1, "C": 0.0}}
        meta_harness._record_calibration(2, cand, node, ["A", "B", "C"])
    finally:
        meta_harness.LOGS_DIR = old_logs

    row = json.loads((tmp_path / "calibration.jsonl").read_text())
    assert row["system"] == "child"
    assert row["predicted_delta_r"] == [0.1, -0.1, 0.0]
    assert row["actual_delta_r"] == [0.1, -0.2, 0.0]
    assert row["pearson"] is not None


def test_memory_summary_is_capped() -> None:
    text = " ".join(f"w{i}" for i in range(250))
    mem = generate_memory({"summary": text, "refs": []}, "diff", [0.1], 1, ["A"], llm=None)

    assert len(mem["summary"].split()) <= 201


def test_edges_are_derived_not_stored_in_node(tmp_path) -> None:
    graph = LineageGraph(tmp_path, ["A"])
    graph.add_node("p", "old\n", {"A": 0.1}, 0.1, 1, 0)
    child = graph.add_node("c", "new\n", {"A": 0.2}, 0.2, 1, 1, parent_name="p")

    assert "diff" not in child
    edge = graph.recent_edges(1)[0]
    assert "--- p.py" in edge["diff"]
    assert "+++ c.py" in edge["diff"]


def test_edge_trace_has_raw_refs(tmp_path) -> None:
    rel = write_edge_trace(
        tmp_path,
        4,
        "p",
        "c",
        "--- p\n+++ c\n",
        [0.1],
        ["A"],
        proposer_trace_path="claude_sessions/iter4.jsonl",
        eval_trace_paths=["logs/A/c/model/log.jsonl"],
    )
    payload = json.loads((tmp_path / rel).read_text())

    assert payload["raw_refs"]["proposer_trace_path"] == "claude_sessions/iter4.jsonl"
    assert payload["raw_refs"]["eval_trace_paths"] == ["logs/A/c/model/log.jsonl"]


def test_warm_start_imports_nodes(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source = LineageGraph(source_dir, ["A"])
    source.add_node("root", "code", {"A": 0.3}, 0.3, 1, 0, memory={"summary": "m", "refs": []})

    target = LineageGraph(tmp_path / "target", ["A"])
    meta_harness._load_warm_graph(target, source_dir)

    node = target.get("root")
    assert node is not None
    assert node["memory"]["summary"] == "m"
