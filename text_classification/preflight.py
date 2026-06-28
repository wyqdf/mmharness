"""Pareto diagnostic for METHOD.md section 5.1.

The full paper diagnostic uses the public TBench2 artifact. This local helper
implements the same go/no-go checks for any run directory that already contains
K-dimensional `nodes.jsonl` or benchmark `val.json` files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmark import load_results
from lineage import build_r_vec_results, dominates


def _cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denom == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def _avg_pairwise_cosine(rows: list[dict]) -> float | None:
    if len(rows) < 2:
        return None
    vals = []
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            vals.append(_cosine(left["r_vec"], right["r_vec"]))
    return sum(vals) / len(vals) if vals else None


def _frontier(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if not any(other is not row and dominates(other["r_vec"], row["r_vec"]) for other in rows)
    ]


def _rows_from_nodes(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        node = json.loads(line)
        if node.get("r_vec"):
            rows.append({"system": node.get("name"), "r_vec": node["r_vec"], "avg": node.get("avg_val", 0.0)})
    return rows


def _rows_from_results(run_dir: Path, datasets: list[str], model: str | None) -> list[dict]:
    results = load_results(run_dir, "val.json")
    return list(build_r_vec_results(results, datasets, model_filter=model).values())


def build_diagnostic(run_dir: Path, datasets: list[str], model: str | None = None) -> dict:
    nodes_path = run_dir / "nodes.jsonl"
    rows = _rows_from_nodes(nodes_path) if nodes_path.exists() else _rows_from_results(run_dir, datasets, model)
    frontier = _frontier(rows)
    scalar_top = max(rows, key=lambda r: r.get("avg", sum(r["r_vec"]) / len(r["r_vec"])), default=None)
    dim_tops = {
        dim: max(rows, key=lambda r, i=i: r["r_vec"][i], default=None)
        for i, dim in enumerate(datasets)
    }
    scalar_top_differs = bool(
        scalar_top
        and any(top and top.get("system") != scalar_top.get("system") for top in dim_tops.values())
    )
    avg_cos = _avg_pairwise_cosine(frontier)
    passed = len(frontier) >= 3 and (avg_cos is not None and avg_cos < 0.95) and scalar_top_differs
    return {
        "run_dir": str(run_dir),
        "dimensions": datasets,
        "num_rows": len(rows),
        "frontier_size": len(frontier),
        "frontier_avg_pairwise_cosine": avg_cos,
        "scalar_top": scalar_top.get("system") if scalar_top else None,
        "dimension_tops": {dim: (row.get("system") if row else None) for dim, row in dim_tops.items()},
        "go": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--datasets", nargs="+", default=["USPTO", "Symptom2Disease", "LawBench"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_diagnostic(Path(args.run), args.datasets, args.model)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"run: {payload['run_dir']}")
        print(f"rows: {payload['num_rows']}")
        print(f"frontier_size: {payload['frontier_size']}")
        print(f"frontier_avg_pairwise_cosine: {payload['frontier_avg_pairwise_cosine']}")
        print(f"scalar_top: {payload['scalar_top']}")
        print(f"dimension_tops: {payload['dimension_tops']}")
        print(f"go: {payload['go']}")
    return 0 if payload["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
