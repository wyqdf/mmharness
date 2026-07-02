"""Small reporting helper for mmharness runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import load_results
from lineage import build_r_vec_results


DEFAULT_DATASETS = ["USPTO", "Symptom2Disease", "LawBench"]


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


def _system_avg_by_iteration(run_dir: Path, datasets: list[str], model: str | None = None) -> dict[int, float]:
    nodes = _read_jsonl(run_dir / "nodes.jsonl")
    values: dict[int, float] = {}
    for node in nodes:
        if node.get("warm_start"):
            continue
        iteration = int(node.get("iter", 0) or 0)
        r_vec = [float(v) for v in node.get("r_vec", [])]
        if len(r_vec) != len(datasets):
            continue
        avg = sum(r_vec) / len(datasets)
        values[iteration] = max(avg, values.get(iteration, 0.0))
    if values:
        best = 0.0
        for iteration in sorted(values):
            best = max(best, values[iteration])
            values[iteration] = best
    return values


def calibration_trend(run_dir: str | Path, window: int = 15) -> dict:
    rows = [
        row
        for row in _read_jsonl(Path(run_dir) / "calibration.jsonl")
        if row.get("pearson") is not None
    ]
    rows.sort(key=lambda row: int(row.get("iteration", 0) or 0))
    windows = []
    for start in range(0, len(rows), max(1, window)):
        chunk = rows[start : start + max(1, window)]
        if not chunk:
            continue
        vals = [float(row["pearson"]) for row in chunk]
        windows.append(
            {
                "start_iteration": int(chunk[0].get("iteration", 0) or 0),
                "end_iteration": int(chunk[-1].get("iteration", 0) or 0),
                "n": len(chunk),
                "avg_pearson": sum(vals) / len(vals),
            }
        )
    monotonic_non_decreasing = all(
        windows[i]["avg_pearson"] <= windows[i + 1]["avg_pearson"]
        for i in range(len(windows) - 1)
    )
    return {
        "run_dir": str(run_dir),
        "window": max(1, window),
        "rows": len(rows),
        "windows": windows,
        "monotonic_non_decreasing": monotonic_non_decreasing,
    }


def warm_start_comparison(
    cold_run: str | Path,
    warm_run: str | Path,
    datasets: list[str],
    model: str | None = None,
) -> dict:
    cold = _system_avg_by_iteration(Path(cold_run), datasets, model=model)
    warm = _system_avg_by_iteration(Path(warm_run), datasets, model=model)
    final_cold = max(cold.values(), default=0.0)
    final_warm = max(warm.values(), default=0.0)

    def first_reaches(values: dict[int, float], target: float) -> int | None:
        for iteration in sorted(values):
            if values[iteration] >= target:
                return iteration
        return None

    cold_target_iter = first_reaches(cold, final_cold)
    warm_to_cold_iter = first_reaches(warm, final_cold)
    return {
        "cold_run": str(cold_run),
        "warm_run": str(warm_run),
        "datasets": datasets,
        "final_cold_avg": final_cold,
        "final_warm_avg": final_warm,
        "cold_reaches_final_cold_at": cold_target_iter,
        "warm_reaches_final_cold_at": warm_to_cold_iter,
        "warm_m_only_advantage": (
            warm_to_cold_iter is not None
            and cold_target_iter is not None
            and warm_to_cold_iter < cold_target_iter
        ),
    }


def build_report(run_dir: str | Path, datasets: list[str], model: str | None = None) -> dict:
    run_dir = Path(run_dir)
    calibration = _read_jsonl(run_dir / "calibration.jsonl")
    results = load_results(run_dir, "val.json")
    systems = build_r_vec_results(results, datasets, model_filter=model)
    return {
        "run_dir": str(run_dir),
        "datasets": datasets,
        "systems": sorted(systems.values(), key=lambda r: (-r["avg"], r["system"])),
        "frontier_val": _read_json(run_dir / "frontier_val.json"),
        "frontier_vec": _read_json(run_dir / "frontier_vec.json"),
        "nodes": _read_jsonl(run_dir / "nodes.jsonl"),
        "calibration": calibration,
        "calibration_trend": calibration_trend(run_dir) if calibration else None,
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
        trend = payload.get("calibration_trend") or {}
        if trend.get("windows"):
            lines.append(f"window={trend.get('window')} monotonic_non_decreasing={trend.get('monotonic_non_decreasing')}")
            for row in trend["windows"]:
                lines.append(
                    f"- iter {row['start_iteration']}..{row['end_iteration']}: "
                    f"n={row['n']} avg_pearson={row['avg_pearson']:.3f}"
                )
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
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--calibration-trend", action="store_true")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--warm-run", default=None, help="Warm-start run for C2 comparison; --run is treated as cold.")
    args = parser.parse_args()
    if args.calibration_trend:
        payload = calibration_trend(args.run, window=args.window)
        print(json.dumps(payload, indent=2, ensure_ascii=False) if args.json else _format_calibration_trend(payload))
        return 0
    if args.warm_run:
        payload = warm_start_comparison(args.run, args.warm_run, args.datasets, args.model)
        print(json.dumps(payload, indent=2, ensure_ascii=False) if args.json else _format_warm_start(payload))
        return 0
    payload = build_report(args.run, args.datasets, args.model)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_report(payload))
    return 0


def _format_calibration_trend(payload: dict) -> str:
    lines = [
        f"run: {payload['run_dir']}",
        f"rows: {payload['rows']}",
        f"window: {payload['window']}",
        f"monotonic_non_decreasing: {payload['monotonic_non_decreasing']}",
    ]
    for row in payload["windows"]:
        lines.append(
            f"iter {row['start_iteration']}..{row['end_iteration']}: "
            f"n={row['n']} avg_pearson={row['avg_pearson']:.3f}"
        )
    return "\n".join(lines)


def _format_warm_start(payload: dict) -> str:
    return "\n".join(
        [
            f"cold_run: {payload['cold_run']}",
            f"warm_run: {payload['warm_run']}",
            f"final_cold_avg: {payload['final_cold_avg']:.3f}",
            f"final_warm_avg: {payload['final_warm_avg']:.3f}",
            f"cold_reaches_final_cold_at: {payload['cold_reaches_final_cold_at']}",
            f"warm_reaches_final_cold_at: {payload['warm_reaches_final_cold_at']}",
            f"warm_m_only_advantage: {payload['warm_m_only_advantage']}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
