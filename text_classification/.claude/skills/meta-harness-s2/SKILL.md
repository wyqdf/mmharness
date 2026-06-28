# S2 Vector-Lineage Harness Evolution

You are running one S2 evolution iteration.

Use only current-run artifacts.

You may use:
- selected parent code
- selected parent r_vec
- Pareto frontier r_vec
- recent parent -> child edges
- each edge's diff excerpt and delta_r
- evolution_summary.jsonl
- frontier_val.json
- frontier_vec.json
- nodes.jsonl
- traces/edge_*.jsonl
- current-run logs

Do not use:
- per-node memory
- memory.summary
- memory.refs
- cross-run memory
- warm-start memory
- dataset-specific hints
- external reference implementations

Workflow:
1. Identify weak dimensions of the selected parent.
2. Compare frontier trade-offs.
3. Inspect recent diff -> delta_r observations.
4. Write exactly 2 candidates:
   - one exploitation candidate
   - one exploration candidate
5. Write pending_eval.json.
6. Do not run benchmarks.

Output:
- pending_eval.json with exactly 2 candidates.
