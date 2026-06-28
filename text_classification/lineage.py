"""Vector reward and lineage state for S2/S3 harness evolution.

Protocol boundaries:
- S1: scalar average reward, no explicit lineage, no edge data, no memory.
- S2: K-dimensional ``r_vec``, explicit parent lineage, recent edge data,
  no per-node memory, no warm-start, no calibration.
- S3: K-dimensional ``r_vec``, explicit lineage, recent edge data,
  per-node memory, optional warm-start, optional calibration.
"""

from __future__ import annotations

import json
import random
import difflib
from pathlib import Path
from typing import Any, Mapping, Sequence


NODES_FILENAME = "nodes.jsonl"
FRONTIER_VEC_FILENAME = "frontier_vec.json"


def meta_meta_config(config: Mapping[str, Any]) -> dict[str, Any]:
    block = dict(config.get("meta_meta", {}) or {})
    return {
        "enabled": bool(block.get("enabled", False)),
        "vector_reward": bool(block.get("vector_reward", False)),
        "show_memory": bool(block.get("show_memory", False)),
        "show_edges": bool(block.get("show_edges", False)),
        "recent_edges": int(block.get("recent_edges", 10)),
        "calibration": bool(block.get("calibration", False)),
        "memory_llm": dict(block.get("memory_llm", {}) or {}),
    }


def meta_meta_enabled(config: Mapping[str, Any]) -> bool:
    return meta_meta_config(config)["enabled"]


def vector_reward_enabled(config: Mapping[str, Any]) -> bool:
    cfg = meta_meta_config(config)
    return cfg["enabled"] and cfg["vector_reward"]


def dataset_order(config_or_datasets: Mapping[str, Any] | Sequence[str]) -> list[str]:
    if isinstance(config_or_datasets, Mapping):
        return [str(name) for name in config_or_datasets.get("datasets", [])]
    return [str(name) for name in config_or_datasets]


def reward_vector(per_dataset: Mapping[str, float], order: Sequence[str]) -> list[float]:
    return [round(float(per_dataset.get(name, 0.0) or 0.0), 6) for name in order]


def delta_vector(child: Sequence[float], parent: Sequence[float] | None) -> list[float] | None:
    if parent is None:
        return None
    return [round(float(c) - float(p), 6) for c, p in zip(child, parent)]


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    if len(a) != len(b) or not a:
        return False
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_frontier(nodes: list[dict]) -> list[dict]:
    return [
        n
        for n in nodes
        if not any(
            dominates(n2["r_vec"], n["r_vec"])
            for n2 in nodes
            if n2["id"] != n["id"]
        )
    ]


def build_r_vec_results(
    results: Mapping[tuple[str, str, str], Mapping[str, Any]],
    datasets: Sequence[str],
    model_filter: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate val/test results into one K-dim row per memory system."""
    grouped: dict[str, dict[str, Any]] = {}
    for (model, dataset, memory), data in results.items():
        if model_filter and model != model_filter:
            continue
        row = grouped.setdefault(
            memory,
            {
                "system": memory,
                "model": model,
                "per_dataset": {},
                "ctx_lens": [],
            },
        )
        row["per_dataset"][dataset] = float(data.get("accuracy") or 0.0)
        row["ctx_lens"].append(int(data.get("memory_context_chars", 0) or 0))

    output: dict[str, dict[str, Any]] = {}
    for memory, row in grouped.items():
        r_vec = reward_vector(row["per_dataset"], datasets)
        non_zero = [v for v in row["ctx_lens"] if v > 0]
        output[memory] = {
            "system": memory,
            "model": row["model"],
            "dimensions": list(datasets),
            "r_vec": r_vec,
            "avg": round(sum(r_vec) / len(datasets), 6) if datasets else 0.0,
            "ctx_len": int(sum(non_zero) / len(non_zero)) if non_zero else 0,
        }
    return output


def pareto_frontier_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for cand in rows:
        cand_vec = list(cand.get("r_vec", []))
        if not any(
            other is not cand and dominates(list(other.get("r_vec", [])), cand_vec)
            for other in rows
        ):
            frontier.append(dict(cand))
    return sorted(frontier, key=lambda r: (-float(r.get("avg", 0.0)), int(r.get("ctx_len", 0)), str(r.get("system", ""))))


def _frontier_node_view(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": int(node["id"]),
        "name": str(node["name"]),
        "r_vec": list(node.get("r_vec", [])),
        "avg_val": float(node.get("avg_val", 0.0) or 0.0),
        "ctx_len": int(node.get("ctx_len", 0) or 0),
    }


def write_frontier_vec_from_results(
    results: Mapping[tuple[str, str, str], Mapping[str, Any]],
    output_path: str | Path,
    datasets: Sequence[str],
    model_filter: str | None = None,
) -> Path:
    rows = build_r_vec_results(results, datasets, model_filter=model_filter)
    payload = {
        "dimensions": list(datasets),
        "objective": "maximize each dimension of r_vec (Pareto non-dominated)",
        "systems": list(rows.values()),
        "frontier": pareto_frontier_rows(list(rows.values())),
    }
    out = Path(output_path)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


class LineageGraph:
    """Append-only jsonl graph for evaluated systems."""

    def __init__(self, run_dir: str | Path, dimensions: Sequence[str]):
        self.run_dir = Path(run_dir)
        self.dimensions = list(dimensions)
        self._nodes: list[dict[str, Any]] = []
        self._by_name: dict[str, dict[str, Any]] = {}
        self._next_id = 0
        self._load_existing()

    @property
    def path(self) -> Path:
        return self.run_dir / NODES_FILENAME

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            node = json.loads(line)
            self._nodes.append(node)
            self._by_name[str(node["name"])] = node
            self._next_id = max(self._next_id, int(node["id"]) + 1)

    def add_node(
        self,
        name: str,
        code: str,
        per_dataset: Mapping[str, float],
        avg_val: float,
        ctx_len: int,
        iteration: int,
        parent_name: str | None = None,
        memory: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        parent = self._by_name.get(parent_name or "")
        r_vec = reward_vector(per_dataset, self.dimensions)
        node = {
            "id": self._next_id,
            "name": name,
            "parent_id": parent["id"] if parent else None,
            "parent_name": parent["name"] if parent else None,
            "code": code,
            "r_vec": r_vec,
            "delta_r": delta_vector(r_vec, parent["r_vec"] if parent else None),
            "avg_val": round(float(avg_val), 6),
            "ctx_len": int(ctx_len),
            "iter": int(iteration),
            "memory": dict(memory) if memory is not None else None,
        }
        self._next_id += 1
        self._nodes.append(node)
        self._by_name[name] = node
        self._append(node)
        return node

    def nodes(self) -> list[dict[str, Any]]:
        return list(self._nodes)

    def get(self, name: str | None) -> dict[str, Any] | None:
        return self._by_name.get(str(name)) if name else None

    def pareto_frontier(self) -> list[dict[str, Any]]:
        return sorted(
            pareto_frontier(list(self._by_name.values())),
            key=lambda r: (
                -float(r.get("avg_val", 0.0)),
                int(r.get("ctx_len", 0) or 0),
                str(r.get("name", "")),
            ),
        )

    def choose_parent(self, seed: int | None = None) -> dict[str, Any] | None:
        frontier = self.pareto_frontier()
        if not frontier:
            return None
        return random.Random(seed).choice(frontier)

    def recent_edges(self, limit: int = 10) -> list[dict[str, Any]]:
        edges = []
        for node in self._nodes:
            if node.get("parent_id") is None:
                continue
            parent = self._node_by_id(int(node["parent_id"]))
            if not parent:
                continue
            edges.append(
                {
                    "edge_id": node["id"],
                    "parent_id": parent["id"],
                    "child_id": node["id"],
                    "parent": parent["name"],
                    "child": node["name"],
                    "diff": unified_diff(
                        str(parent.get("code", "")),
                        str(node.get("code", "")),
                        str(parent.get("name", "parent")),
                        str(node.get("name", "child")),
                    ),
                    "delta_r": node.get("delta_r"),
                    "dimensions": self.dimensions,
                }
            )
        return edges[-int(limit):] if limit and limit > 0 else edges

    def write_frontier_vec(self) -> Path:
        payload = [_frontier_node_view(node) for node in self.pareto_frontier()]
        out = self.run_dir / FRONTIER_VEC_FILENAME
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    def update_node(self, node_id: int, **fields: Any) -> None:
        for node in self._nodes:
            if int(node["id"]) == int(node_id):
                node.update(fields)
                self._rewrite()
                self._by_name[str(node["name"])] = node
                return
        raise KeyError(f"Unknown node id: {node_id}")

    def _node_by_id(self, node_id: int) -> dict[str, Any] | None:
        for node in self._nodes:
            if int(node["id"]) == node_id:
                return node
        return None

    def _append(self, node: Mapping[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(node), ensure_ascii=False) + "\n")

    def _rewrite(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for node in self._nodes:
                handle.write(json.dumps(node, ensure_ascii=False) + "\n")


def unified_diff(parent_code: str, child_code: str, parent_name: str, child_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            parent_code.splitlines(keepends=True),
            child_code.splitlines(keepends=True),
            fromfile=f"{parent_name}.py",
            tofile=f"{child_name}.py",
        )
    )
