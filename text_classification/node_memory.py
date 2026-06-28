"""Per-node memory and edge-context formatting for Meta-Meta-Harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


LLMCallable = Callable[[str], str]
MAX_DIFF_CHARS = 1800


def build_memory_llm(config: Mapping[str, Any]) -> LLMCallable | None:
    meta = dict(config.get("meta_meta", {}) or {})
    if not (meta.get("enabled") and meta.get("show_memory")):
        return None
    mem_cfg = dict(meta.get("memory_llm", {}) or {})
    model = str(mem_cfg.get("model") or "claude-haiku-4-5")
    base_url = os.environ.get(str(mem_cfg.get("base_url_env") or "")) or str(
        mem_cfg.get("base_url") or mem_cfg.get("default_base_url") or ""
    )
    auth_token = os.environ.get(str(mem_cfg.get("auth_token_env") or "")) or str(
        mem_cfg.get("auth_token") or ""
    )
    if not base_url or not auth_token:
        return None

    def _call(prompt: str) -> str:
        import httpx

        response = httpx.post(
            base_url.rstrip("/") + "/v1/messages",
            headers={
                "x-api-key": auth_token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": int(mem_cfg.get("max_tokens", 512)),
                "temperature": float(mem_cfg.get("temperature", 0.0)),
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=float(mem_cfg.get("timeout_seconds", 120.0)),
        )
        response.raise_for_status()
        data = response.json()
        parts = data.get("content", [])
        return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))

    return _call


def root_memory(name: str, r_vec: Sequence[float], dims: Sequence[str]) -> dict[str, Any]:
    scores = ", ".join(f"{dim}={value:.3f}" for dim, value in zip(dims, r_vec))
    return {"summary": f"Root baseline {name}. Current validation vector: {scores}.", "refs": []}


def generate_memory(
    parent_memory: Mapping[str, Any] | None,
    diff: str,
    delta_r: Sequence[float] | None,
    edge_id: int,
    dims: Sequence[str],
    llm: LLMCallable | None = None,
    trace_path: str | None = None,
    max_words: int = 200,
) -> dict[str, Any]:
    parent_summary = str((parent_memory or {}).get("summary", "")).strip()
    refs = list((parent_memory or {}).get("refs", []))
    refs.append({"edge_id": int(edge_id), "trace_path": trace_path or f"traces/edge_{edge_id}.jsonl"})
    summary = ""
    if llm is not None:
        try:
            summary = str(llm(_memory_prompt(parent_summary, diff, delta_r, dims, max_words))).strip()
        except Exception:
            summary = ""
    if not summary:
        summary = _fallback_summary(parent_summary, delta_r, dims, edge_id)
    summary = _limit_words(summary, max_words)
    return {"summary": summary, "refs": refs}


def write_edge_trace(
    run_dir: str | Path,
    edge_id: int,
    parent: str,
    child: str,
    diff: str,
    delta_r: Sequence[float] | None,
    dims: Sequence[str],
    parent_memory: Mapping[str, Any] | None = None,
    child_memory: Mapping[str, Any] | None = None,
    proposer_trace_path: str | None = None,
    eval_trace_paths: Sequence[str] | None = None,
) -> str:
    traces_dir = Path(run_dir) / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"traces/edge_{edge_id}.jsonl"
    path = Path(run_dir) / rel_path
    payload = {
        "edge_id": int(edge_id),
        "parent": parent,
        "child": child,
        "diff": diff,
        "delta_r": list(delta_r) if delta_r is not None else None,
        "dimensions": list(dims),
        "parent_memory": dict(parent_memory) if parent_memory else None,
        "child_memory": dict(child_memory) if child_memory else None,
        "raw_refs": {
            "proposer_trace_path": proposer_trace_path,
            "eval_trace_paths": list(eval_trace_paths or []),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return rel_path


def format_meta_meta_context(
    parent_node: Mapping[str, Any] | None,
    frontier_nodes: Sequence[Mapping[str, Any]],
    recent_edges: Sequence[Mapping[str, Any]],
    dims: Sequence[str],
    show_memory: bool,
    show_edges: bool,
) -> str:
    if not (parent_node or show_memory or show_edges):
        return ""
    sections = [
        "# Meta-Meta evolution state",
        "Use this graph state to reason at the strategy level: which code changes tend to produce which multi-objective effects.",
    ]
    if parent_node:
        sections.extend(
            [
                "## Selected parent node",
                f"Name: {parent_node.get('name')}",
                f"r_vec: {_format_vec(parent_node.get('r_vec', []), dims)}",
                f"memory: {((parent_node.get('memory') or {}).get('summary') or '(none)')}",
                "Parent code follows; proposed candidates should build from this mechanism unless you have a strong reason not to.",
                "```python",
                str(parent_node.get("code", ""))[:12000],
                "```",
            ]
        )
    if show_memory and frontier_nodes:
        lines = ["## Pareto frontier node memories"]
        for node in frontier_nodes:
            summary = ((node.get("memory") or {}).get("summary") or "(no memory)").strip()
            lines.append(f"- {node.get('name')}: {_format_vec(node.get('r_vec', []), dims)}; {summary}")
        sections.append("\n".join(lines))
    if show_edges and recent_edges:
        lines = ["## Recent causal observations (diff -> delta_r)"]
        for edge in recent_edges:
            diff = str(edge.get("diff") or "")
            if len(diff) > MAX_DIFF_CHARS:
                diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated] ..."
            lines.append(
                "\n".join(
                    [
                        f"### edge {edge.get('edge_id')}: {edge.get('parent')} -> {edge.get('child')}",
                        f"delta_r: {_format_vec(edge.get('delta_r') or [], dims, signed=True)}",
                        "```diff",
                        diff or "(no diff recorded)",
                        "```",
                    ]
                )
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections).strip() + "\n"


def _memory_prompt(
    parent_summary: str,
    diff: str,
    delta_r: Sequence[float] | None,
    dims: Sequence[str],
    max_words: int,
) -> str:
    diff_excerpt = diff[:MAX_DIFF_CHARS]
    return (
        "Update a concise evolution story for a self-improving text-classification harness.\n\n"
        f"Story so far:\n{parent_summary or '(root)'}\n\n"
        f"New diff:\n{diff_excerpt}\n\n"
        f"Observed effect: {_format_vec(delta_r or [], dims, signed=True)}\n\n"
        f"Return one paragraph under {max_words} words. End with current strengths and weaknesses per dimension."
    )


def _fallback_summary(
    parent_summary: str,
    delta_r: Sequence[float] | None,
    dims: Sequence[str],
    edge_id: int,
) -> str:
    step = f"Edge {edge_id} produced {_format_vec(delta_r or [], dims, signed=True)}."
    if not parent_summary:
        return step
    return f"Previous path: {parent_summary} New step: {step}".strip()


def _format_vec(values: Sequence[float], dims: Sequence[str], signed: bool = False) -> str:
    if not values:
        return "[]"
    if signed:
        return "[" + ", ".join(f"{dim}={value:+.3f}" for dim, value in zip(dims, values)) + "]"
    return "[" + ", ".join(f"{dim}={value:.3f}" for dim, value in zip(dims, values)) + "]"


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip() + " ..."
