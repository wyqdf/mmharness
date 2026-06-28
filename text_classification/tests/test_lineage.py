from __future__ import annotations

import json

from text_classification.lineage import (
    LineageGraph,
    dominates,
    reward_vector,
    write_frontier_vec_from_results,
)


def test_reward_vector_uses_dataset_order() -> None:
    assert reward_vector({"B": 0.2, "A": 0.1}, ["A", "B", "C"]) == [0.1, 0.2, 0.0]


def test_dominates_requires_no_worse_and_one_better() -> None:
    assert dominates([0.2, 0.5], [0.2, 0.4])
    assert not dominates([0.2, 0.4], [0.2, 0.5])
    assert not dominates([0.2, 0.5], [0.2, 0.5])


def test_lineage_graph_stores_code_and_parent(tmp_path) -> None:
    graph = LineageGraph(tmp_path, ["A", "B"])
    root = graph.add_node("root", "print('root')", {"A": 0.1, "B": 0.2}, 0.15, 10, 0)
    child = graph.add_node("child", "print('child')", {"A": 0.2, "B": 0.1}, 0.15, 12, 1, parent_name="root")

    assert root["code"] == "print('root')"
    assert child["parent_id"] == root["id"]
    assert child["delta_r"] == [0.1, -0.1]
    rows = [json.loads(line) for line in (tmp_path / "nodes.jsonl").read_text().splitlines()]
    assert rows[1]["code"] == "print('child')"


def test_lineage_writes_k_dim_frontier(tmp_path) -> None:
    graph = LineageGraph(tmp_path, ["A", "B"])
    graph.add_node("a", "a", {"A": 0.3, "B": 0.2}, 0.25, 1, 0)
    graph.add_node("b", "b", {"A": 0.2, "B": 0.4}, 0.30, 1, 0)
    graph.add_node("c", "c", {"A": 0.1, "B": 0.1}, 0.10, 1, 0)

    path = graph.write_frontier_vec()
    data = json.loads(path.read_text())
    assert {row["name"] for row in data["frontier"]} == {"a", "b"}


def test_write_frontier_vec_from_results(tmp_path) -> None:
    results = {
        ("m", "A", "sys1"): {"accuracy": 0.2, "memory_context_chars": 10},
        ("m", "B", "sys1"): {"accuracy": 0.2, "memory_context_chars": 10},
        ("m", "A", "sys2"): {"accuracy": 0.3, "memory_context_chars": 20},
        ("m", "B", "sys2"): {"accuracy": 0.1, "memory_context_chars": 20},
    }
    path = write_frontier_vec_from_results(results, tmp_path / "frontier_vec.json", ["A", "B"], "m")
    data = json.loads(path.read_text())

    assert data["dimensions"] == ["A", "B"]
    assert {row["system"] for row in data["frontier"]} == {"sys1", "sys2"}

