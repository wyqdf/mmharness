"""Small reporting helper for mmharness runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import load_results
from lineage import build_r_vec_results


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_report(run_dir: str | Path, datasets: list[str], model: str | None = None) -> dict:
    run_dir = Path(run_dir)
    results = load_results(run_dir, "val.json")
    systems = build_r_vec_results(results, datasets, model_filter=model)
    return {
        "run_dir": str(run_dir),
        "datasets": datasets,
        "systems": sorted(systems.values(), key=lambda r: (-r["avg"], r["system"])),
        "frontier_val": _read_json(run_dir / "frontier_val.json"),
        "frontier_vec": _read_json(run_dir / "frontier_vec.json"),
        "nodes": _read_jsonl(run_dir / "nodes.jsonl"),
        "calibration": _read_jsonl(run_dir / "calibration.jsonl"),
    }


def format_report(payload: dict) -> str:
    lines = [f"# mmharness report: {payload['run_dir']}", ""]
    dims = payload["datasets"]
    lines.append("## Validation systems")
    if not payload["systems"]:
        lines.append("No validation results found.")
    else:
        header = ["system", *dims, "avg", "ctx"]
        rows = []
        for row in payload["systems"]:
            rows.append(
                [
                    row["system"],
                    *[f"{v:.3f}" for v in row["r_vec"]],
                    f"{row['avg']:.3f}",
                    str(row["ctx_len"]),
                ]
            )
        lines.extend(_table(header, rows))
    frontier_vec = payload.get("frontier_vec") or {}
    frontier_rows = frontier_vec if isinstance(frontier_vec, list) else frontier_vec.get("frontier", [])
    if frontier_rows:
        lines.extend(["", "## K-dimensional Pareto frontier"])
        rows = []
        for row in frontier_rows:
            rows.append(
                [
                    row.get("system") or row.get("name"),
                    *[f"{v:.3f}" for v in row.get("r_vec", [])],
                    f"{float(row.get('avg', row.get('avg_val', 0.0))):.3f}",
                ]
            )
        lines.extend(_table(["system", *dims, "avg"], rows))
    nodes = payload.get("nodes") or []
    if nodes:
        lines.extend(["", f"## Lineage nodes: {len(nodes)}"])
        for node in nodes[-10:]:
            summary = ((node.get("memory") or {}).get("summary") or "").strip()
            lines.append(
                f"- {node['id']} {node['name']} parent={node.get('parent_name')} "
                f"r_vec={node.get('r_vec')} {summary[:160]}"
            )
    calibration = payload.get("calibration") or []
    if calibration:
        vals = [row.get("pearson") for row in calibration if row.get("pearson") is not None]
        avg = sum(vals) / len(vals) if vals else None
        lines.extend(["", "## Calibration"])
        lines.append(f"rows={len(calibration)} avg_pearson={avg if avg is not None else 'n/a'}")
    return "\n".join(lines)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(col) for col in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = ["  ".join(str(c).ljust(widths[i]) for i, c in enumerate(header))]
    out.append("  ".join("-" * width for width in widths))
    for row in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run logs directory")
    parser.add_argument("--datasets", nargs="+", default=["USPTO", "Symptom2Disease", "LawBench"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_report(args.run, args.datasets, args.model)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
