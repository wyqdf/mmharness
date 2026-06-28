"""Autonomous evolution loop for memory systems.

Val-only during evolution (test never exposed).
Uses claude_wrapper + meta-harness skill to propose new memory systems.

    uv run python meta_harness.py --iterations 20 --fresh
    uv run python meta_harness.py --iterations 10 --run-name my-run
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

import claude_wrapper
from benchmark import get_model_short_name, load_results
from lineage import LineageGraph, dataset_order, meta_meta_config, reward_vector, unified_diff, vector_reward_enabled
from node_memory import (
    build_memory_llm,
    format_meta_meta_context,
    generate_memory,
    root_memory,
    write_edge_trace,
)

EVOLVE_DIR = Path(__file__).parent
CONFIG_PATH = Path(os.environ.get("TEXT_CLASSIFICATION_CONFIG", EVOLVE_DIR / "config.yaml"))
AGENTS_DIR = EVOLVE_DIR / "agents"
BASELINE_FILES = {"__init__.py", "no_memory.py", "fewshot_memory.py", "fewshot_all.py"}

# These are updated per-run if --run-name is set
LOGS_DIR = EVOLVE_DIR / "logs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"

PROPOSER_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Agent",
    "Write",
    "Edit",
    "Bash",
]

_interrupted = False

# ── ANSI colors ──────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _pct(val):
    s = f"{val:.1f}%"
    if val >= 60:
        return _green(s)
    elif val >= 40:
        return _yellow(s)
    return _red(s)


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


def run_cmd(cmd, timeout=7200, cwd=None):
    """Wraps subprocess.run; returns CompletedProcess with returncode=124 on timeout."""
    try:
        return subprocess.run(
            cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd, returncode=124, stdout="", stderr=f"Timed out after {timeout}s"
        )


def run_benchmark(args):
    return run_cmd(
        [
            "uv",
            "run",
            "python",
            "benchmark.py",
            "--config",
            str(CONFIG_PATH),
            "--logs-dir",
            str(LOGS_DIR),
        ]
        + args,
        cwd=str(EVOLVE_DIR),
    )


def render_task_prompt(
    iteration,
    num_datasets,
    cfg=None,
    lineage=None,
    parent_node=None,
):
    """Build the prompt for the proposer Claude session."""
    base = (
        f"Run iteration {iteration} of the evolution loop. There are {num_datasets} datasets.\n\n"
        f"## Run directories\n"
        f"All logs and results for this run are under `{LOGS_DIR}/`.\n"
        f"- `{EVOLUTION_SUMMARY}` — past results\n"
        f"- `{FRONTIER_VAL}` — frontier\n"
        f"- `{LOGS_DIR / 'reports'}/` — post-eval reports\n"
        f"- Active config: `{CONFIG_PATH}`\n"
        f"- Write pending_eval.json to: `{PENDING_EVAL}`"
    )
    if not cfg or not lineage:
        return base
    mm_cfg = meta_meta_config(cfg)
    if not (mm_cfg["show_memory"] or mm_cfg["show_edges"]):
        return base
    block = format_meta_meta_context(
        parent_node=parent_node,
        frontier_nodes=lineage.pareto_frontier(),
        recent_edges=lineage.recent_edges(mm_cfg["recent_edges"]),
        dims=lineage.dimensions,
        show_memory=mm_cfg["show_memory"],
        show_edges=mm_cfg["show_edges"],
    )
    return base + "\n\n" + block if block else base


def count_iterations_from_summary():
    """Highest iteration number in evolution_summary.jsonl (for resume)."""
    if not EVOLUTION_SUMMARY.exists():
        return 0
    max_iter = 0
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            max_iter = max(max_iter, json.loads(line).get("iteration", 0))
        except json.JSONDecodeError:
            continue
    return max_iter


def _apply_proposer_env(cfg):
    proposer_cfg = dict((cfg.get("mmharness", {}) or {}).get("proposer", {}) or {})
    base_url_env = str(proposer_cfg.get("base_url_env", "ANTHROPIC_BASE_URL"))
    auth_token_env = str(proposer_cfg.get("auth_token_env", "ANTHROPIC_AUTH_TOKEN"))
    base_url = proposer_cfg.get("base_url") or proposer_cfg.get("default_base_url", "https://api.pioneer.ai")
    updates = {}
    if base_url:
        updates[base_url_env] = str(base_url)
    token = proposer_cfg.get("auth_token") or os.environ.get(auth_token_env)
    if token:
        updates[auth_token_env] = str(token)
        updates.setdefault("ANTHROPIC_API_KEY", str(token))
    return updates


def _proposer_skill_dir(cfg=None):
    mm_cfg = meta_meta_config(cfg or {})
    skill_name = "meta-harness-mm" if mm_cfg["enabled"] else "meta-harness"
    return EVOLVE_DIR / ".claude/skills" / skill_name


def propose_claude(task_prompt, iteration, cfg=None, timeout=2400):
    """Returns True if candidates were produced (pending_eval.json exists)."""
    old_env = os.environ.copy()
    os.environ.pop("CLAUDECODE", None)
    if cfg:
        os.environ.update(_apply_proposer_env(cfg))
    proposer_model = (
        ((cfg or {}).get("mmharness", {}) or {}).get("proposer", {}) or {}
    ).get("model", "claude-opus-4-6")
    try:
        result = claude_wrapper.run(
            prompt=task_prompt,
            model=proposer_model,
            allowed_tools=PROPOSER_ALLOWED_TOOLS,
            skills=[str(_proposer_skill_dir(cfg))],
            cwd=str(EVOLVE_DIR),
            log_dir=str(LOGS_DIR / "claude_sessions"),
            name=f"iter{iteration}",
            timeout_seconds=timeout,
            effort="max",
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    if result.exit_code != 0:
        print(f"  {_red('proposer failed')} exit={result.exit_code}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
        return False
    result.show()
    return PENDING_EVAL.exists()


def validate_candidates(candidates):
    """Import-check each candidate. Returns list of valid candidates."""
    valid = []
    for c in candidates:
        name = c["name"]
        result = run_cmd(
            [
                "uv",
                "run",
                "python",
                "-c",
                f"from text_classification.agents.{name} import *; print('OK')",
            ],
            cwd=str(EVOLVE_DIR.parent),
            timeout=30,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            print(f"    {_green('OK')} {name}")
            valid.append(c)
        else:
            print(f"    {_red('FAIL')} {name}")
            if result.stderr:
                print(f"      {_dim(result.stderr[:200])}")
    return valid


def update_evolution_summary(
    iteration,
    candidates,
    val_scores,
    propose_time=None,
    bench_time=None,
    wall_time=None,
):
    """Append one JSONL row per candidate to evolution_summary.jsonl."""
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    pareto = frontier.get("_pareto", [])
    best_val = pareto[0].get("val_accuracy", 0) if pareto else 0

    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            avg_val = val_scores.get(name, 0)
            row = {
                "iteration": iteration,
                "system": name,
                "avg_val": round(avg_val, 1),
                "axis": c.get("axis", "?"),
                "hypothesis": c.get("hypothesis", ""),
                "delta": round(avg_val - best_val, 1) if best_val else None,
                "outcome": f"{avg_val:.1f}% ({avg_val - best_val:+.1f})"
                if avg_val > 0
                else "failed",
            }
            if "components" in c:
                row["components"] = c["components"]
            if i == 0 and wall_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1),
                    "wall": round(wall_time, 1),
                }
            f.write(json.dumps(row) + "\n")


def fresh_start():
    """Clear proposed memory systems and reset logs for a fresh run."""
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    if AGENTS_DIR.exists():
        files = [f for f in AGENTS_DIR.glob("*.py") if f.name not in BASELINE_FILES]
        for f in files:
            f.unlink()
        if files:
            print(f"  Cleared {len(files)} candidate file(s) from agents/")

    for f in [
        EVOLUTION_SUMMARY,
        FRONTIER_VAL,
        LOGS_DIR / "frontier.json",
        LOGS_DIR / "frontier_vec.json",
        LOGS_DIR / "nodes.jsonl",
        LOGS_DIR / "calibration.jsonl",
        PENDING_EVAL,
    ]:
        if f.exists():
            f.unlink()
    traces_dir = LOGS_DIR / "traces"
    if traces_dir.exists():
        for f in traces_dir.glob("edge_*.jsonl"):
            f.unlink()

    if LOGS_DIR.exists():
        val_files = list(LOGS_DIR.rglob("val.json"))
        for f in val_files:
            f.unlink()
        if val_files:
            print(f"  Cleared {len(val_files)} val result files")
        launcher_logs = list((LOGS_DIR / ".launcher").glob("*.log"))
        for f in launcher_logs:
            f.unlink()
        if launcher_logs:
            print(f"  Cleared {len(launcher_logs)} launcher log file(s)")

    print(f"  {_green('Fresh start')}: cleared generated agents and log files")


def _load_warm_graph(lineage, warm_from):
    if not warm_from:
        return
    source = Path(warm_from)
    if source.is_dir():
        source = source / "nodes.jsonl"
    if not source.exists():
        print(f"  {_yellow('warm-start skipped')}: {source} not found")
        return
    imported = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        node = json.loads(line)
        if lineage.get(node.get("name")):
            continue
        lineage.add_node(
            name=str(node["name"]),
            code=str(node.get("code", "")),
            per_dataset={
                dim: float(value)
                for dim, value in zip(lineage.dimensions, node.get("r_vec", []))
            },
            avg_val=float(node.get("avg_val", 0.0)),
            ctx_len=int(node.get("ctx_len", 0) or 0),
            iteration=int(node.get("iter", 0) or 0),
            parent_name=node.get("parent_name"),
            memory=node.get("memory"),
        )
        imported += 1
    if imported:
        lineage.write_frontier_vec()
        print(f"  {_green('warm-start')}: imported {imported} lineage node(s) from {source}")


def run_evolve(args):
    global LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datasets = cfg["datasets"]
    mm_cfg = meta_meta_config(cfg)

    config_model_ids = [m["model"] for m in cfg.get("models", [])]
    if args.model not in config_model_ids:
        print(f"ERROR: --model {args.model} not in config.yaml: {config_model_ids}")
        sys.exit(1)

    model_short = get_model_short_name(args.model)

    # Isolate run outputs under run-name subdirs
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR = EVOLVE_DIR / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        fresh_start()

    lineage = LineageGraph(LOGS_DIR, dataset_order(cfg)) if vector_reward_enabled(cfg) else None
    if lineage is not None:
        _load_warm_graph(lineage, args.warm_from)
    memory_llm = build_memory_llm(cfg) if mm_cfg["show_memory"] else None

    print(
        f"{_ts()} {_bold('Evolution (memory systems)')}  "
        f"run={_cyan(run_name)}  model={_cyan(args.model)}  "
        f"iters={args.iterations}  datasets={datasets}"
    )

    # ── Phase 0: Baselines ─────────────────────────────────────
    baselines = cfg["memory_systems"]["baselines"]
    if not args.skip_baseline:
        print(f"\n{_ts()} {_bold('Phase 0: Baselines')}  systems={baselines}")
        for bl in baselines:
            if _interrupted:
                break
            print(f"  {_ts()} benchmarking {_bold(bl)}...", flush=True)
            t0 = time.time()
            result = run_benchmark(["--memory", bl])
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"    {_red('FAIL')} {bl}: {result.stderr[:200]}")
                if result.stdout:
                    print(f"      {_dim(result.stdout[-800:])}")
            else:
                print(f"    {_green('OK')} ({_elapsed(elapsed)})")
                if lineage is not None:
                    _record_lineage_node(
                        lineage=lineage,
                        system_name=bl,
                        model_short=model_short,
                        datasets=datasets,
                        iteration=0,
                        parent_node=None,
                        cfg=cfg,
                        memory_llm=memory_llm,
                    )

        run_benchmark(["--frontier", "--model", model_short])
        if lineage is not None:
            lineage.write_frontier_vec()

        # Show baseline results
        results = load_results(LOGS_DIR, "val.json")
        for bl in baselines:
            accs = [
                results[k]["accuracy"] * 100
                for ds in datasets
                for k in [(model_short, ds, bl)]
                if k in results and results[k].get("accuracy") is not None
            ]
            if accs:
                avg = sum(accs) / len(accs)
                print(f"    {_bold(bl)}: avg_val={_pct(avg)}")

    # ── Phase 1..N: Evolution ──────────────────────────────────
    start_iteration = count_iterations_from_summary() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i
        iter_start = time.time()

        # Show frontier status
        frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        pareto = frontier.get("_pareto", [])
        best_val = pareto[0].get("val_accuracy", 0) if pareto else 0
        best_sys = pareto[0].get("system", "none") if pareto else "none"

        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  "
            f"frontier={best_sys} @ {_pct(best_val * 100 if best_val <= 1 else best_val)}"
        )
        print(f"{'─' * 60}")

        parent_node = None
        if lineage is not None:
            parent_node = lineage.choose_parent(seed=int(cfg["inner_loop"].get("seed", 42)) + iteration)
        task_prompt = render_task_prompt(iteration, len(datasets), cfg, lineage, parent_node)

        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

        # Propose
        propose_start = time.time()
        print(f"  {_ts()} {_cyan('proposing')} new candidates...", flush=True)
        ok = propose_claude(task_prompt, iteration, cfg=cfg, timeout=args.propose_timeout)
        propose_time = time.time() - propose_start

        if not ok:
            print(
                f"  {_red('FAIL')} proposer returned no candidates after {_elapsed(propose_time)}"
            )
            continue

        candidates = json.loads(PENDING_EVAL.read_text()).get("candidates", [])
        print(
            f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}"
        )
        for ci, c in enumerate(candidates):
            hyp = c.get("hypothesis", "")
            print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")

        # Validate
        print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
        valid_candidates = validate_candidates(candidates)

        if not valid_candidates:
            print(
                f"  {_red('0 valid')} out of {len(candidates)} candidates, skipping iteration"
            )
            update_evolution_summary(
                iteration, candidates, {}, propose_time=propose_time
            )
            continue
        print(
            f"  {_green(f'{len(valid_candidates)} valid')} out of {len(candidates)} candidates"
        )

        # Benchmark
        bench_start = time.time()
        print(
            f"  {_ts()} {_cyan('benchmarking')} {len(valid_candidates)} system(s) x {len(datasets)} datasets"
        )
        for ci, c in enumerate(valid_candidates):
            if _interrupted:
                break
            name = c["name"]
            print(
                f"    [{ci + 1}/{len(valid_candidates)}] {_bold(name)}...", flush=True
            )
            t0 = time.time()
            result = run_benchmark(["--memory", name])
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"      {_red('FAIL')} benchmark crashed ({_elapsed(elapsed)})")
            else:
                print(f"      {_green('OK')} ({_elapsed(elapsed)})")
                if lineage is not None:
                    declared_parent = lineage.get(c.get("base_system")) if c.get("base_system") else None
                    node = _record_lineage_node(
                        lineage=lineage,
                        system_name=name,
                        model_short=model_short,
                        datasets=datasets,
                        iteration=iteration,
                        parent_node=declared_parent or parent_node,
                        cfg=cfg,
                        memory_llm=memory_llm,
                    )
                    if node is not None and mm_cfg["calibration"]:
                        _record_calibration(iteration, c, node, lineage.dimensions)
        bench_time = time.time() - bench_start

        run_benchmark(["--frontier", "--model", model_short])
        if lineage is not None:
            lineage.write_frontier_vec()

        # Compute scores and show results
        val_scores = {}
        results = load_results(LOGS_DIR, "val.json")
        for c in valid_candidates:
            name = c["name"]
            accs = []
            missing = []
            for ds in datasets:
                k = (model_short, ds, name)
                if k in results and results[k].get("accuracy") is not None:
                    accs.append(results[k]["accuracy"] * 100)
                else:
                    missing.append(ds)
            if missing:
                val_scores[name] = 0
                print(
                    f"    {_bold(name)}: incomplete ({len(accs)}/{len(datasets)} datasets; "
                    f"missing {', '.join(missing)})"
                )
                continue
            val_scores[name] = sum(accs) / len(datasets)
            delta = val_scores[name] - (best_val * 100 if best_val <= 1 else best_val)
            delta_str = f"{delta:+.1f}"
            delta_colored = (
                _green(delta_str)
                if delta > 0
                else (_red(delta_str) if delta < 0 else _dim(delta_str))
            )
            print(
                f"    {_bold(name)}: avg_val={_pct(val_scores[name])}  delta={delta_colored}"
            )

        wall_time = time.time() - iter_start
        update_evolution_summary(
            iteration,
            valid_candidates,
            val_scores,
            propose_time=propose_time,
            bench_time=bench_time,
            wall_time=wall_time,
        )

        # Show iteration summary
        improved = any(
            v > (best_val * 100 if best_val <= 1 else best_val)
            for v in val_scores.values()
        )
        status = _green("NEW BEST") if improved else _dim("no improvement")
        print(f"  {_ts()} {status}")
        print(
            f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}"
        )

    # ── Phase Final: Test eval ─────────────────────────────────
    if _interrupted:
        return

    print(f"\n{_ts()} {_bold('Phase Final: Test evaluation')}")

    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    pareto = frontier.get("_pareto", [])

    test_systems = set(baselines)
    for entry in pareto:
        test_systems.add(entry["system"])
    for key, val in frontier.items():
        if not key.startswith("_") and isinstance(val, dict) and "best_system" in val:
            test_systems.add(val["best_system"])

    for name in sorted(test_systems):
        print(f"  {_ts()} test eval: {_bold(name)}", flush=True)
        result = run_benchmark(["--memory", name, "--test"])
        if result.returncode != 0:
            print(f"    {_red('FAIL')} {name} test eval failed")

    run_benchmark(["--frontier", "--test", "--model", model_short])

    result = run_benchmark(["--results"])
    if result.stdout:
        print(result.stdout)

    print(f"\n{_ts()} {_bold('Evolution complete.')}")


def _system_result(system_name, model_short, datasets):
    results = load_results(LOGS_DIR, "val.json")
    per_dataset = {}
    ctx_lens = []
    for ds in datasets:
        data = results.get((model_short, ds, system_name))
        if not data:
            return None
        per_dataset[ds] = float(data.get("accuracy") or 0.0)
        ctx_lens.append(int(data.get("memory_context_chars", 0) or 0))
    if len(per_dataset) != len(datasets):
        return None
    avg = sum(per_dataset.get(ds, 0.0) for ds in datasets) / len(datasets)
    non_zero = [value for value in ctx_lens if value > 0]
    ctx_len = int(sum(non_zero) / len(non_zero)) if non_zero else 0
    return per_dataset, avg, ctx_len


def _agent_code(system_name):
    path = AGENTS_DIR / f"{system_name}.py"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _unified_diff(parent_code, child_code, parent_name, child_name):
    return unified_diff(parent_code, child_code, parent_name, child_name)


def _coerce_delta_vec(value, dims):
    if isinstance(value, dict):
        return [float(value.get(dim, 0.0) or 0.0) for dim in dims]
    if isinstance(value, list):
        out = []
        for item in value[: len(dims)]:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(0.0)
        while len(out) < len(dims):
            out.append(0.0)
        return out
    return None


def _pearson(a, b):
    if len(a) != len(b) or len(a) < 2:
        return None
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    da = [x - mean_a for x in a]
    db = [y - mean_b for y in b]
    denom_a = sum(x * x for x in da) ** 0.5
    denom_b = sum(y * y for y in db) ** 0.5
    if denom_a == 0 or denom_b == 0:
        return None
    return sum(x * y for x, y in zip(da, db)) / (denom_a * denom_b)


def _record_calibration(iteration, candidate, node, dims):
    predicted = _coerce_delta_vec(candidate.get("predicted_delta_r"), dims)
    actual = node.get("delta_r")
    if predicted is None or actual is None:
        return
    actual_vec = [float(x) for x in actual]
    row = {
        "iteration": int(iteration),
        "system": candidate.get("name"),
        "dimensions": list(dims),
        "predicted_delta_r": predicted,
        "actual_delta_r": actual_vec,
        "pearson": _pearson(predicted, actual_vec),
    }
    with (LOGS_DIR / "calibration.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _record_lineage_node(
    lineage,
    system_name,
    model_short,
    datasets,
    iteration,
    parent_node,
    cfg,
    memory_llm=None,
):
    if lineage.get(system_name):
        return lineage.get(system_name)
    result = _system_result(system_name, model_short, datasets)
    if result is None:
        return None
    per_dataset, avg, ctx_len = result
    code = _agent_code(system_name)
    mm_cfg = meta_meta_config(cfg)
    memory = None
    parent_name = parent_node.get("name") if parent_node else None
    if parent_node is None and mm_cfg["show_memory"]:
        memory = root_memory(system_name, reward_vector(per_dataset, lineage.dimensions), lineage.dimensions)
    node = lineage.add_node(
        name=system_name,
        code=code,
        per_dataset=per_dataset,
        avg_val=avg,
        ctx_len=ctx_len,
        iteration=iteration,
        parent_name=parent_name,
        memory=memory,
    )
    if parent_node is not None and (mm_cfg["show_memory"] or mm_cfg["show_edges"]):
        diff = _unified_diff(parent_node.get("code", ""), code, str(parent_name), system_name)
        trace_path = write_edge_trace(
            LOGS_DIR,
            int(node["id"]),
            str(parent_name),
            system_name,
            diff,
            node.get("delta_r"),
            lineage.dimensions,
            parent_memory=parent_node.get("memory"),
        )
        memory = None
        if mm_cfg["show_memory"]:
            memory = generate_memory(
                parent_node.get("memory"),
                diff,
                node.get("delta_r"),
                int(node["id"]),
                lineage.dimensions,
                llm=memory_llm,
                trace_path=trace_path,
            )
        if memory is not None:
            lineage.update_node(int(node["id"]), memory=memory)
            write_edge_trace(
                LOGS_DIR,
                int(node["id"]),
                str(parent_name),
                system_name,
                diff,
                node.get("delta_r"),
                lineage.dimensions,
                parent_memory=parent_node.get("memory"),
                child_memory=memory,
            )
    return node


def main():
    global CONFIG_PATH
    parser = argparse.ArgumentParser(description="Evolution loop for memory systems")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Config YAML path")
    parser.add_argument("--iterations", type=int, default=20)
    prelim, _ = parser.parse_known_args()
    CONFIG_PATH = Path(prelim.config).resolve()
    os.environ["TEXT_CLASSIFICATION_CONFIG"] = str(CONFIG_PATH)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    _default_model = _cfg["models"][0]["model"] if _cfg.get("models") else None
    parser.add_argument(
        "--model",
        default=_default_model,
        help=f"Solver model (default: {_default_model})",
    )
    parser.add_argument(
        "--propose-timeout",
        type=int,
        default=2400,
        help="Timeout per propose step (default: 2400s)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for isolated output dirs. Auto-generated if not set.",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Clear proposed systems and reset logs"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Phase 0 baseline eval"
    )
    parser.add_argument(
        "--warm-from",
        type=str,
        default=None,
        help="Preload nodes.jsonl from a previous S3 run for memory-only warm-start experiments.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_evolve(args)


if __name__ == "__main__":
    main()
