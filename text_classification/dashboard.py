"""Live dashboard for text_classification Meta-Harness runs."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
DEFAULT_DATASETS = ["USPTO", "Symptom2Disease", "LawBench"]
DATASET_VAL_SIZES = {"USPTO": 30, "Symptom2Disease": 50, "LawBench": 50}


def _read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-limit:] if limit else data


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _latest_run() -> Path | None:
    if not LOGS_DIR.exists():
        return None
    runs = [
        p
        for p in LOGS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / ".launcher").exists()
    ]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _safe_run_path(name_or_path: str | None) -> Path | None:
    if not name_or_path:
        return _latest_run()
    p = Path(name_or_path)
    if not p.is_absolute():
        p = LOGS_DIR / name_or_path
    try:
        p = p.resolve()
        logs_root = LOGS_DIR.resolve()
        if p != logs_root and logs_root not in p.parents:
            return None
    except Exception:
        return None
    return p if p.exists() else None


def _pid_status(run: Path) -> dict:
    pid_file = LOGS_DIR / f"{run.name}.pid"
    status = {"pid": None, "alive": False, "elapsed": None, "stat": None, "cmd": None}
    if not pid_file.exists():
        return status
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return status
    status["pid"] = pid
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "etime=,stat=,cmd="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return status
    if out:
        parts = out.split(None, 2)
        status.update(
            {
                "alive": True,
                "elapsed": parts[0] if len(parts) > 0 else None,
                "stat": parts[1] if len(parts) > 1 else None,
                "cmd": parts[2] if len(parts) > 2 else None,
            }
        )
        status["children"] = _child_processes(pid)
    return status


def _child_processes(pid: int) -> list[dict]:
    try:
        child_pids = subprocess.check_output(
            ["pgrep", "-P", str(pid)], text=True, stderr=subprocess.DEVNULL
        ).split()
    except Exception:
        return []
    children = []
    for child in child_pids:
        try:
            out = subprocess.check_output(
                ["ps", "-p", child, "-o", "pid=,etime=,stat=,cmd="],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            continue
        if not out:
            continue
        parts = out.split(None, 3)
        children.append(
            {
                "pid": int(parts[0]) if parts else None,
                "elapsed": parts[1] if len(parts) > 1 else None,
                "stat": parts[2] if len(parts) > 2 else None,
                "cmd": parts[3] if len(parts) > 3 else "",
            }
        )
    return children


def _launcher_log(run: Path) -> Path:
    return LOGS_DIR / ".launcher" / f"{run.name}.out"


def _parse_phase(log_text: str) -> dict:
    phase = {
        "current": "unknown",
        "iteration": None,
        "total_iterations": None,
        "frontier": None,
        "frontier_score": None,
        "message": "",
    }
    iter_matches = list(
        re.finditer(
            r"Iteration\s+(\d+)\s+\((\d+)/(\d+)\)\s+frontier=([^\s]+)\s+@\s+([0-9.]+)%",
            log_text,
        )
    )
    if iter_matches:
        m = iter_matches[-1]
        phase.update(
            {
                "current": "proposing" if "proposing new candidates" in log_text[m.start() :] else "iteration",
                "iteration": int(m.group(1)),
                "total_iterations": int(m.group(3)),
                "frontier": m.group(4),
                "frontier_score": float(m.group(5)),
            }
        )
    elif "Phase 0: Baselines" in log_text:
        phase["current"] = "baselines"
    if "benchmarking" in log_text.splitlines()[-20:]:
        phase["current"] = "benchmarking"
    if "Evolution complete." in log_text:
        phase["current"] = "complete"
    tail_lines = [line for line in log_text.splitlines()[-12:] if line.strip()]
    phase["message"] = tail_lines[-1] if tail_lines else ""
    return phase


def _val_rows(run: Path) -> list[dict]:
    rows = []
    for val_file in run.rglob("val.json"):
        rel = val_file.relative_to(run).parts
        if len(rel) < 4:
            continue
        dataset, system, model = rel[:3]
        data = _read_json(val_file) or {}
        log_data = _last_done(val_file.parent / "log.jsonl")
        rows.append(
            {
                "dataset": dataset,
                "system": system,
                "model": model,
                "accuracy": data.get("accuracy"),
                "ctx_len": data.get("avg_context_len") or data.get("ctx_len"),
                "runtime_seconds": log_data.get("runtime_seconds"),
                "llm_calls": log_data.get("llm_calls"),
                "llm_input_tokens": log_data.get("llm_input_tokens"),
                "llm_output_tokens": log_data.get("llm_output_tokens"),
                "updated": val_file.stat().st_mtime,
            }
        )
    return sorted(rows, key=lambda r: (r["system"], r["dataset"]))


def _last_done(log_path: Path) -> dict:
    done = {}
    if not log_path.exists():
        return done
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"type": "done"' not in line:
                continue
            try:
                done = json.loads(line)
            except Exception:
                continue
    except Exception:
        return {}
    return done


def _speed(rows: list[dict]) -> dict:
    complete = [r for r in rows if r.get("runtime_seconds")]
    runtime = sum(float(r.get("runtime_seconds") or 0) for r in complete)
    calls = sum(int(r.get("llm_calls") or 0) for r in complete)
    unique_questions = sum(DATASET_VAL_SIZES.get(r["dataset"], 0) for r in complete)
    by_system = {}
    for row in complete:
        sys = row["system"]
        item = by_system.setdefault(sys, {"jobs": 0, "runtime": 0.0, "calls": 0, "questions": 0})
        item["jobs"] += 1
        item["runtime"] += float(row.get("runtime_seconds") or 0)
        item["calls"] += int(row.get("llm_calls") or 0)
        item["questions"] += DATASET_VAL_SIZES.get(row["dataset"], 0)
    for item in by_system.values():
        item["sec_per_question"] = item["runtime"] / item["questions"] if item["questions"] else None
        item["sec_per_call"] = item["runtime"] / item["calls"] if item["calls"] else None
    return {
        "jobs": len(complete),
        "runtime": runtime,
        "calls": calls,
        "questions": unique_questions,
        "sec_per_question": runtime / unique_questions if unique_questions else None,
        "sec_per_call": runtime / calls if calls else None,
        "by_system": by_system,
    }


def _leaderboard(nodes: list[dict], frontier: dict) -> list[dict]:
    by_name: dict[str, dict] = {}
    for node in nodes:
        name = node.get("name")
        if not name:
            continue
        by_name[name] = {
            "system": name,
            "iteration": node.get("iter"),
            "parent": node.get("parent_name"),
            "avg_val": float(node.get("avg_val") or 0.0) * 100,
            "r_vec": node.get("r_vec") or [],
            "ctx_len": node.get("ctx_len"),
            "delta_r": node.get("delta_r"),
            "node_id": node.get("id"),
        }
    for row in frontier.get("_pareto", []) if isinstance(frontier, dict) else []:
        name = row.get("system")
        if not name:
            continue
        current = by_name.setdefault(name, {"system": name})
        current.setdefault("iteration", None)
        current.setdefault("parent", None)
        current["avg_val"] = float(row.get("val_accuracy") or current.get("avg_val") or 0.0)
        current["r_vec"] = row.get("r_vec") or current.get("r_vec") or []
        current["ctx_len"] = row.get("ctx_len") or current.get("ctx_len")
    return sorted(by_name.values(), key=lambda r: (-float(r.get("avg_val") or 0.0), str(r.get("system"))))


def _loop_summary(nodes: list[dict], evolution: list[dict]) -> list[dict]:
    evo_by_key: dict[tuple[int, str], dict] = {}
    for row in evolution:
        try:
            key = (int(row.get("iteration") or 0), str(row.get("system") or ""))
        except Exception:
            continue
        evo_by_key[key] = row

    loops: dict[int, list[dict]] = {}
    for node in nodes:
        iteration = int(node.get("iter") or 0)
        if iteration <= 0:
            continue
        name = str(node.get("name") or "")
        evo = evo_by_key.get((iteration, name), {})
        avg_pct = float(node.get("avg_val") or 0.0) * 100
        parent_vec = None
        delta_r = node.get("delta_r")
        if delta_r and node.get("r_vec"):
            parent_vec = [
                round(float(v) - float(d), 6)
                for v, d in zip(node.get("r_vec") or [], delta_r or [])
            ]
        loops.setdefault(iteration, []).append(
            {
                "iteration": iteration,
                "system": name,
                "node_id": node.get("id"),
                "parent": node.get("parent_name"),
                "avg_val": avg_pct,
                "r_vec": node.get("r_vec") or [],
                "delta_r": delta_r,
                "parent_r_vec": parent_vec,
                "ctx_len": node.get("ctx_len"),
                "axis": evo.get("axis"),
                "hypothesis": evo.get("hypothesis"),
                "outcome": evo.get("outcome"),
                "timing_s": evo.get("timing_s") or {},
                "components": evo.get("components") or [],
            }
        )

    out = []
    running_best = 0.0
    for iteration in sorted(loops):
        candidates = sorted(loops[iteration], key=lambda r: -float(r.get("avg_val") or 0.0))
        loop_best = candidates[0] if candidates else None
        improved = bool(loop_best and float(loop_best.get("avg_val") or 0.0) > running_best)
        if loop_best:
            running_best = max(running_best, float(loop_best.get("avg_val") or 0.0))
        out.append(
            {
                "iteration": iteration,
                "best_system": loop_best.get("system") if loop_best else None,
                "best_avg_val": loop_best.get("avg_val") if loop_best else None,
                "improved_global_best": improved,
                "candidates": candidates,
            }
        )
    return out


def _latest_launcher_files(run: Path) -> list[dict]:
    launcher = run / ".launcher"
    if not launcher.exists():
        return []
    rows = []
    for path in launcher.glob("*.log"):
        text = _read_text(path, limit=4000)
        exit_match = re.search(r"exit=(\d+)", text)
        rows.append(
            {
                "name": path.name,
                "mtime": path.stat().st_mtime,
                "size": path.stat().st_size,
                "exit": int(exit_match.group(1)) if exit_match else None,
                "tail": "\n".join(text.splitlines()[-8:]),
            }
        )
    return sorted(rows, key=lambda r: r["mtime"], reverse=True)[:16]


def _events(log_text: str) -> list[str]:
    interesting = []
    needles = (
        "Phase 0:",
        "Iteration ",
        "proposed ",
        "valid out of",
        "benchmarking ",
        "avg_val=",
        "NEW BEST",
        "FAIL",
        "proposer failed",
        "timing:",
        "Evolution complete.",
    )
    for line in log_text.splitlines():
        if any(n in line for n in needles):
            interesting.append(line)
    return interesting[-80:]


def build_status(run: Path | None) -> dict:
    if run is None:
        return {"error": "No run found", "runs": list_runs()}
    log_path = _launcher_log(run)
    log_text = _read_text(log_path)
    rows = _val_rows(run)
    frontier = _read_json(run / "frontier_val.json") or {}
    nodes = _read_jsonl(run / "nodes.jsonl")
    evolution = _read_jsonl(run / "evolution_summary.jsonl")
    calibration = _read_jsonl(run / "calibration.jsonl")
    pending = _read_json(run / "pending_eval.json")
    leaderboard = _leaderboard(nodes, frontier)
    loops = _loop_summary(nodes, evolution)
    return {
        "now": time.time(),
        "run": run.name,
        "run_dir": str(run),
        "pid": _pid_status(run),
        "phase": _parse_phase(log_text),
        "counts": {
            "val_json": len(list(run.rglob("val.json"))),
            "test_json": len(list(run.rglob("test.json"))),
            "nodes": len(nodes),
            "evolution_rows": len(evolution),
            "calibration": len(calibration),
            "pending_eval": pending is not None,
        },
        "frontier": frontier,
        "best_system": leaderboard[0] if leaderboard else None,
        "leaderboard": leaderboard,
        "loops": loops,
        "nodes": nodes[-20:],
        "evolution": evolution[-30:],
        "calibration": calibration[-30:],
        "pending_eval": pending,
        "val_rows": rows,
        "speed": _speed(rows),
        "launcher_files": _latest_launcher_files(run),
        "events": _events(log_text),
        "log_tail": "\n".join(log_text.splitlines()[-160:]),
        "runs": list_runs(),
    }


def list_runs() -> list[dict]:
    if not LOGS_DIR.exists():
        return []
    rows = []
    for p in LOGS_DIR.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            rows.append(
                {
                    "name": p.name,
                    "mtime": p.stat().st_mtime,
                    "val_json": len(list(p.rglob("val.json"))),
                    "nodes": len(_read_jsonl(p / "nodes.jsonl")),
                }
            )
    return sorted(rows, key=lambda r: r["mtime"], reverse=True)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MMHarness Live Progress</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6fa;
      --panel: #ffffff;
      --panel-2: #eef3f8;
      --text: #172033;
      --muted: #627084;
      --line: #d9e0e8;
      --accent: #0b7285;
      --accent-2: #4f46e5;
      --accent-3: #d97706;
      --accent-4: #be3455;
      --usp: #0b7285;
      --s2d: #6f42c1;
      --law: #d9480f;
      --good: #0b7a3b;
      --bad: #b42318;
      --warn: #946200;
      --shadow: 0 1px 2px rgba(20, 28, 38, 0.08);
    }
    * { box-sizing: border-box; }
    html, body { overflow-x: hidden; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(246,247,249,.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }
    .wrap { max-width: 1500px; margin: 0 auto; padding: 16px 20px; }
    .top { display: flex; align-items: center; gap: 16px; justify-content: space-between; }
    h1 { font-size: 20px; margin: 0; font-weight: 700; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 2px; }
    .controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    select, button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      height: 34px;
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    main { max-width: 1500px; margin: 0 auto; padding: 18px 20px 40px; }
    .grid { display: grid; gap: 14px; min-width: 0; }
    .grid > * { min-width: 0; }
    .kpis { grid-template-columns: repeat(6, minmax(0, 1fr)); }
    .cols { grid-template-columns: minmax(0, 1.2fr) minmax(460px, .8fr); align-items: start; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
      min-width: 0;
    }
    .panel h2 {
      margin: 0;
      padding: 12px 14px;
      font-size: 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .body { padding: 14px; }
    .kpi { padding: 14px; min-height: 86px; border-top: 4px solid var(--accent); }
    .kpi:nth-child(2) { border-top-color: var(--good); }
    .kpi:nth-child(3) { border-top-color: var(--accent-2); }
    .kpi:nth-child(4) { border-top-color: var(--accent-3); }
    .kpi:nth-child(5) { border-top-color: var(--accent-4); }
    .kpi:nth-child(6) { border-top-color: var(--muted); }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .value { font-size: 24px; font-weight: 750; margin-top: 4px; white-space: nowrap; }
    .note { color: var(--muted); margin-top: 4px; font-size: 12px; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th { color: var(--muted); font-size: 12px; font-weight: 650; background: #fbfcfd; position: sticky; top: 0; }
    tr:last-child td { border-bottom: 0; }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      white-space: nowrap;
    }
    .dot { width: 8px; height: 8px; border-radius: 99px; background: var(--muted); }
    .dot.live { background: var(--good); }
    .dot.dead { background: var(--bad); }
    .bars { display: grid; gap: 8px; }
    .bar-row { display: grid; grid-template-columns: 160px 1fr 64px; gap: 10px; align-items: center; }
    .track { height: 10px; background: var(--panel-2); border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; background: var(--accent); }
    .fill.usp { background: var(--usp); }
    .fill.s2d { background: var(--s2d); }
    .fill.law { background: var(--law); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 520px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.4;
      background: #101820;
      color: #dce7ee;
      padding: 12px;
      border-radius: 6px;
      max-width: 100%;
    }
    .timeline { display: grid; gap: 7px; }
    .event { border-left: 3px solid var(--line); padding-left: 9px; color: #2d3642; }
    .event.best { border-color: var(--good); font-weight: 650; }
        .event.fail { border-color: var(--bad); color: var(--bad); }
    .scroll { max-height: 380px; overflow: auto; }
    .wide-scroll { max-height: 640px; overflow: auto; }
    .small { font-size: 12px; color: var(--muted); }
    .best-card {
      display: grid;
      grid-template-columns: minmax(240px, .8fr) minmax(0, 1.2fr);
      gap: 14px;
      align-items: start;
    }
    .score {
      font-size: 34px;
      font-weight: 800;
      margin: 6px 0;
      color: var(--accent);
    }
    .dim-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .dim {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfd;
      border-top-width: 4px;
    }
    .dim.usp { border-top-color: var(--usp); }
    .dim.s2d { border-top-color: var(--s2d); }
    .dim.law { border-top-color: var(--law); }
    .delta-pos { color: var(--good); font-weight: 650; }
    .delta-neg { color: var(--bad); font-weight: 650; }
    .loop-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 12px;
      background: #fff;
    }
    .loop-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      background: #fbfcfd;
      border-bottom: 1px solid var(--line);
    }
    .loop-title { font-weight: 750; }
    .candidate-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      padding: 12px;
    }
    .candidate-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
      min-width: 0;
      border-left: 5px solid var(--accent);
    }
    .candidate-card.best { border-left-color: var(--good); }
    .candidate-card.bad { border-left-color: var(--bad); }
    .candidate-title {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }
    .candidate-name {
      font-weight: 750;
      font-size: 15px;
    }
    .candidate-score {
      font-weight: 800;
      font-size: 22px;
      white-space: nowrap;
      color: var(--accent);
    }
    .metric-row {
      display: grid;
      grid-template-columns: 96px 1fr 76px 70px;
      gap: 8px;
      align-items: center;
      margin: 7px 0;
    }
    .metric-label { color: var(--muted); font-size: 12px; }
    .metric-track { height: 9px; background: var(--panel-2); border-radius: 999px; overflow: hidden; }
    .metric-fill { height: 100%; }
    .metric-fill.usp { background: var(--usp); }
    .metric-fill.s2d { background: var(--s2d); }
    .metric-fill.law { background: var(--law); }
    .hypothesis {
      color: var(--muted);
      font-size: 12px;
      margin-top: 10px;
      max-height: 4.4em;
      overflow: auto;
    }
    .process-cmd {
      max-height: 160px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      padding: 8px;
    }
    @media (max-width: 1100px) {
      .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .cols { grid-template-columns: 1fr; }
      .best-card { grid-template-columns: 1fr; }
      .dim-grid { grid-template-columns: 1fr; }
      .candidate-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap top">
      <div>
        <h1>MMHarness Live Progress</h1>
        <div class="sub" id="subtitle">Loading...</div>
      </div>
      <div class="controls">
        <select id="runSelect"></select>
        <span class="pill"><span id="liveDot" class="dot"></span><span id="pidText">pid</span></span>
        <button id="pauseBtn">Pause</button>
        <button id="refreshBtn" class="primary">Refresh</button>
      </div>
    </div>
  </header>
  <main class="grid">
    <section class="grid kpis">
      <div class="panel kpi"><div class="label">Phase</div><div id="phase" class="value">-</div><div id="phaseNote" class="note">-</div></div>
      <div class="panel kpi"><div class="label">Best Avg Val</div><div id="best" class="value">-</div><div id="bestNote" class="note">-</div></div>
      <div class="panel kpi"><div class="label">Iteration</div><div id="iteration" class="value">-</div><div id="iterNote" class="note">-</div></div>
      <div class="panel kpi"><div class="label">Val Jobs</div><div id="valJobs" class="value">-</div><div id="valNote" class="note">-</div></div>
      <div class="panel kpi"><div class="label">Throughput</div><div id="throughput" class="value">-</div><div id="speedNote" class="note">-</div></div>
      <div class="panel kpi"><div class="label">Last Update</div><div id="updated" class="value">-</div><div id="updatedNote" class="note">auto refresh 2s</div></div>
    </section>

    <section class="grid cols">
      <div class="grid">
        <div class="panel">
          <h2>Global Best Harness</h2>
          <div class="body" id="bestHarness"></div>
        </div>
        <div class="panel">
          <h2>Loop Candidate Performance</h2>
          <div class="body wide-scroll" id="loopPerformance"></div>
        </div>
        <div class="panel">
          <h2>Validation Systems</h2>
          <div class="body scroll"><table id="systemsTable"></table></div>
        </div>
        <div class="panel">
          <h2>Per Dataset Best</h2>
          <div class="body" id="datasetBest"></div>
        </div>
        <div class="panel">
          <h2>Evolution Timeline</h2>
          <div class="body timeline" id="timeline"></div>
        </div>
        <div class="panel">
          <h2>Recent Launcher Jobs</h2>
          <div class="body scroll"><table id="launcherTable"></table></div>
        </div>
      </div>
      <div class="grid">
        <div class="panel">
          <h2>Current Process</h2>
          <div class="body" id="processBox"></div>
        </div>
        <div class="panel">
          <h2>Speed By System</h2>
          <div class="body scroll"><table id="speedTable"></table></div>
        </div>
        <div class="panel">
          <h2>Lineage Nodes</h2>
          <div class="body scroll"><table id="nodesTable"></table></div>
        </div>
        <div class="panel">
          <h2>Log Tail</h2>
          <div class="body"><pre id="logTail"></pre></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const datasets = ["USPTO", "Symptom2Disease", "LawBench"];
    const dimClasses = ["usp", "s2d", "law"];
    let paused = false;
    let currentRun = new URLSearchParams(location.search).get("run") || "";

    const $ = (id) => document.getElementById(id);
    const fmtPct = (v) => Number.isFinite(v) ? `${v.toFixed(1)}%` : "-";
    const fmtSec = (v) => {
      if (!Number.isFinite(v)) return "-";
      if (v < 60) return `${v.toFixed(1)}s`;
      const m = Math.floor(v / 60), s = Math.round(v % 60);
      return `${m}m ${s}s`;
    };
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#039;'}[ch]));

    async function load() {
      const url = `/api/status${currentRun ? `?run=${encodeURIComponent(currentRun)}` : ""}`;
      const res = await fetch(url, {cache: "no-store"});
      const data = await res.json();
      render(data);
    }

    function render(data) {
      if (data.error) {
        $("subtitle").textContent = data.error;
        return;
      }
      currentRun = data.run;
      renderRuns(data.runs || []);
      const pid = data.pid || {};
      $("subtitle").textContent = `${data.run_dir}`;
      $("liveDot").className = `dot ${pid.alive ? "live" : "dead"}`;
      $("pidText").textContent = pid.alive ? `pid ${pid.pid} · ${pid.elapsed}` : "not running";

      const phase = data.phase || {};
      $("phase").textContent = phase.current || "-";
      $("phaseNote").textContent = phase.message || "-";
      $("iteration").textContent = phase.iteration ? `${phase.iteration}/${phase.total_iterations || "?"}` : "-";
      $("iterNote").textContent = phase.frontier ? `frontier ${phase.frontier}` : "-";

      const best = bestSystem(data);
      $("best").textContent = best ? fmtPct(best.val_accuracy) : "-";
      $("bestNote").textContent = best ? `${best.system} · ctx ${best.ctx_len ?? "-"}` : "-";
      $("valJobs").textContent = data.counts?.val_json ?? "-";
      $("valNote").textContent = `${data.counts?.nodes ?? 0} nodes · ${data.counts?.calibration ?? 0} calibration`;
      const spq = data.speed?.sec_per_question;
      $("throughput").textContent = Number.isFinite(spq) ? `${spq.toFixed(2)}s/q` : "-";
      $("speedNote").textContent = `${data.speed?.calls ?? 0} solver calls · ${fmtSec(data.speed?.runtime)}`;
      $("updated").textContent = new Date((data.now || Date.now()/1000) * 1000).toLocaleTimeString();

      renderSystems(data);
      renderBestHarness(data);
      renderLoopPerformance(data.loops || []);
      renderDatasetBest(data);
      renderTimeline(data.events || []);
      renderLauncher(data.launcher_files || []);
      renderProcess(data.pid || {});
      renderSpeed(data.speed?.by_system || {});
      renderNodes(data.nodes || []);
      $("logTail").textContent = data.log_tail || "";
    }

    function renderRuns(runs) {
      const sel = $("runSelect");
      const known = Array.from(sel.options).map(o => o.value).join("|");
      const next = runs.map(r => r.name).join("|");
      if (known === next) return;
      sel.innerHTML = runs.map(r => `<option value="${esc(r.name)}">${esc(r.name)} (${r.val_json} vals)</option>`).join("");
      sel.value = currentRun;
    }

    function bestSystem(data) {
      const p = data.frontier?._pareto || [];
      if (!p.length) return null;
      return [...p].sort((a,b) => (b.val_accuracy || 0) - (a.val_accuracy || 0))[0];
    }

    function renderSystems(data) {
      const rows = data.leaderboard || [];
      const head = `<tr><th>Rank</th><th>Harness</th><th>Iter</th><th>Parent</th>${datasets.map(d=>`<th>${d}</th>`).join("")}<th>Avg</th><th>Ctx</th></tr>`;
      const body = rows
        .slice()
        .sort((a,b)=>(b.avg_val||0)-(a.avg_val||0))
        .map((r, i) => `<tr><td>${i+1}</td><td class="mono">${esc(r.system)}</td><td>${r.iteration ?? "-"}</td><td class="mono">${esc(r.parent || "-")}</td>${(r.r_vec||[]).map(v=>`<td>${fmtPct(v*100)}</td>`).join("")}<td><b>${fmtPct(r.avg_val)}</b></td><td>${esc(r.ctx_len ?? "-")}</td></tr>`)
        .join("");
      $("systemsTable").innerHTML = head + body;
    }

    function renderBestHarness(data) {
      const b = data.best_system;
      if (!b) {
        $("bestHarness").innerHTML = `<div class="small">No evaluated harness yet.</div>`;
        return;
      }
      const dims = datasets.map((d, i) => {
        const v = (b.r_vec || [])[i];
        return `<div class="dim ${dimClasses[i]}"><div class="label">${esc(d)}</div><div class="value">${fmtPct((v || 0) * 100)}</div></div>`;
      }).join("");
      $("bestHarness").innerHTML = `
        <div class="best-card">
          <div>
            <div class="label">Best Harness</div>
            <div class="score">${fmtPct(b.avg_val)}</div>
            <div class="mono"><b>${esc(b.system)}</b></div>
            <div class="note">iteration ${b.iteration ?? "-"} · parent ${esc(b.parent || "-")} · ctx ${esc(b.ctx_len ?? "-")}</div>
          </div>
          <div class="dim-grid">${dims}</div>
        </div>
      `;
    }

    function deltaCell(v) {
      if (!Number.isFinite(v)) return "-";
      const pct = v * 100;
      const cls = pct >= 0 ? "delta-pos" : "delta-neg";
      return `<span class="${cls}">${pct >= 0 ? "+" : ""}${pct.toFixed(1)}</span>`;
    }

    function renderLoopPerformance(loops) {
      if (!loops.length) {
        $("loopPerformance").innerHTML = `<div class="small">No evolution loop candidates evaluated yet.</div>`;
        return;
      }
      $("loopPerformance").innerHTML = loops.slice().reverse().map(loop => {
        const maxScore = Math.max(1, ...loop.candidates.flatMap(c => (c.r_vec || []).map(v => v * 100)));
        const cards = loop.candidates.map((c, idx) => {
          const isBest = c.system === loop.best_system;
          const avgClass = c.avg_val >= 45 ? "best" : (c.avg_val < 30 ? "bad" : "");
          const metrics = datasets.map((d, i) => {
            const val = ((c.r_vec || [])[i] || 0) * 100;
            const delta = (c.delta_r || [])[i];
            return `
              <div class="metric-row">
                <div class="metric-label mono">${esc(d)}</div>
                <div class="metric-track"><div class="metric-fill ${dimClasses[i]}" style="width:${Math.max(0, Math.min(100, val / maxScore * 100))}%"></div></div>
                <div>${fmtPct(val)}</div>
                <div>${deltaCell(Number(delta))}</div>
              </div>
            `;
          }).join("");
          return `
            <div class="candidate-card ${isBest ? "best" : avgClass}">
              <div class="candidate-title">
                <div>
                  <div class="candidate-name mono">${idx + 1}. ${esc(c.system)}</div>
                  <div class="small">parent: <span class="mono">${esc(c.parent || "-")}</span> · axis: ${esc(c.axis || "-")} · ctx ${esc(c.ctx_len ?? "-")}</div>
                </div>
                <div class="candidate-score">${fmtPct(c.avg_val)}</div>
              </div>
              ${metrics}
              <div class="small">benchmark ${fmtSec(c.timing_s?.bench)} · wall ${fmtSec(c.timing_s?.wall)} · components ${(c.components || []).map(esc).join(", ") || "-"}</div>
              <div class="hypothesis">${esc(c.hypothesis || c.outcome || "")}</div>
            </div>
          `;
        }).join("");
        return `
          <div class="loop-card">
            <div class="loop-head">
              <div class="loop-title">Iteration ${loop.iteration}</div>
              <div>${loop.improved_global_best ? '<span class="pill"><span class="dot live"></span>new best</span>' : '<span class="pill">no new best</span>'}</div>
              <div class="mono">best: ${esc(loop.best_system || "-")} · ${fmtPct(loop.best_avg_val)}</div>
            </div>
            <div class="candidate-grid">${cards}</div>
          </div>
        `;
      }).join("");
    }

    function renderDatasetBest(data) {
      const f = data.frontier || {};
      const maxVal = Math.max(1, ...datasets.map(d => f[d]?.accuracy || 0));
      $("datasetBest").innerHTML = `<div class="bars">` + datasets.map(d => {
        const idx = datasets.indexOf(d);
        const row = f[d] || {};
        const acc = row.accuracy || 0;
        return `<div class="bar-row"><div class="mono">${esc(d)}</div><div class="track"><div class="fill ${dimClasses[idx]}" style="width:${Math.max(0, acc / maxVal * 100)}%"></div></div><div>${fmtPct(acc)}</div><div class="small" style="grid-column:2/4">${esc(row.best_system || "-")}</div></div>`;
      }).join("") + `</div>`;
    }

    function renderTimeline(events) {
      $("timeline").innerHTML = events.map(e => {
        const cls = e.includes("NEW BEST") ? "event best" : (e.includes("FAIL") || e.includes("failed") ? "event fail" : "event");
        return `<div class="${cls} mono">${esc(e)}</div>`;
      }).join("");
    }

    function renderLauncher(rows) {
      $("launcherTable").innerHTML = `<tr><th>Job</th><th>Exit</th><th>Updated</th><th>Tail</th></tr>` +
        rows.map(r => `<tr><td class="mono">${esc(r.name)}</td><td>${r.exit ?? "..."}</td><td>${new Date(r.mtime*1000).toLocaleTimeString()}</td><td><pre>${esc(r.tail)}</pre></td></tr>`).join("");
    }

    function renderProcess(pid) {
      const children = pid.children || [];
      $("processBox").innerHTML = `
        <div class="pill"><span class="dot ${pid.alive ? "live" : "dead"}"></span>${pid.alive ? "running" : "stopped"}</div>
        <p class="small mono">pid=${esc(pid.pid)} elapsed=${esc(pid.elapsed)} stat=${esc(pid.stat)}</p>
        <div class="small mono process-cmd">${esc(pid.cmd || "")}</div>
        ${children.length ? `<h3>Children</h3>${children.map(c => `<p class="small mono">pid=${esc(c.pid)} ${esc(c.elapsed)} ${esc(c.stat)} ${esc(c.cmd).slice(0, 220)}</p>`).join("")}` : ""}
      `;
    }

    function renderSpeed(bySystem) {
      const rows = Object.entries(bySystem).sort((a,b)=>(b[1].runtime||0)-(a[1].runtime||0));
      $("speedTable").innerHTML = `<tr><th>System</th><th>Jobs</th><th>Runtime</th><th>s/q</th><th>s/call</th></tr>` +
        rows.map(([name, r]) => `<tr><td class="mono">${esc(name)}</td><td>${r.jobs}</td><td>${fmtSec(r.runtime)}</td><td>${Number.isFinite(r.sec_per_question) ? r.sec_per_question.toFixed(2) : "-"}</td><td>${Number.isFinite(r.sec_per_call) ? r.sec_per_call.toFixed(2) : "-"}</td></tr>`).join("");
    }

    function renderNodes(nodes) {
      $("nodesTable").innerHTML = `<tr><th>ID</th><th>Name</th><th>Parent</th><th>Avg</th><th>r_vec</th></tr>` +
        nodes.slice().reverse().map(n => `<tr><td>${n.id}</td><td class="mono">${esc(n.name)}</td><td class="mono">${esc(n.parent_name)}</td><td>${fmtPct((n.avg_val || 0)*100)}</td><td class="mono">${esc(JSON.stringify(n.r_vec || []))}</td></tr>`).join("");
    }

    $("refreshBtn").onclick = () => load();
    $("pauseBtn").onclick = () => {
      paused = !paused;
      $("pauseBtn").textContent = paused ? "Resume" : "Pause";
    };
    $("runSelect").onchange = () => {
      currentRun = $("runSelect").value;
      history.replaceState(null, "", `?run=${encodeURIComponent(currentRun)}`);
      load();
    };

    load();
    setInterval(() => { if (!paused) load().catch(console.error); }, 2000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    run_name: str | None = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            query = parse_qs(parsed.query)
            run = _safe_run_path(query.get("run", [self.run_name])[0])
            self._json(build_status(run))
            return
        if parsed.path == "/api/runs":
            self._json({"runs": list_runs()})
            return
        if parsed.path in {"/", "/index.html"}:
            self._html(INDEX_HTML)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        return

    def _json(self, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self, text: str):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live dashboard for Meta-Harness runs")
    parser.add_argument("--run", default=None, help="Run name or run directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    Handler.run_name = args.run
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    run = _safe_run_path(args.run)
    run_label = run.name if run else "latest"
    print(f"Dashboard: http://{args.host}:{args.port}/?run={html.escape(run_label)}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
