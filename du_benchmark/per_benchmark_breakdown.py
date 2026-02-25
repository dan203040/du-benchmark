# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
#!/usr/bin/env python3
"""
Per-benchmark breakdown of DU benchmark scores.

Splits Table 5 scores by benchmark source (KramaBench / AgentBench / DACode)
and computes per-system, per-benchmark, per-dimension averages.

Usage:
    python3 evaluation/du_benchmark/per_benchmark_breakdown.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

RESULTS_FILE = Path("evaluation/results/du_benchmark/analysis_decontaminated/du_benchmark_decontaminated.json")
OUTPUT_DIR = Path("evaluation/results/du_benchmark/analysis_decontaminated")

BENCHMARK_PREFIXES = {
    "KramaBench": [
        "legal", "wildfire", "biomedical", "environment",
        "archeology", "astronomy",
    ],
    "AgentBench": ["dbbench"],
    "DACode": ["dm-", "ml-", "plot-", "di-", "data-"],
}


def classify_task(task_id: str) -> str:
    for benchmark, prefixes in BENCHMARK_PREFIXES.items():
        for prefix in prefixes:
            if task_id.startswith(prefix):
                return benchmark
    return "Unknown"


def compute_breakdown(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-system, per-benchmark, per-dimension breakdown."""
    breakdown: Dict[str, Any] = {}
    dimensions = ["D1", "D2", "D3", "D4", "D5"]

    for sys_name, data in results.items():
        task_scores = data.get("task_scores", [])
        if not task_scores:
            continue

        # Group by benchmark
        by_benchmark: Dict[str, List[Dict]] = {}
        for ts in task_scores:
            bench = classify_task(ts["task_id"])
            by_benchmark.setdefault(bench, []).append(ts)

        sys_breakdown = {}
        for bench, scores in sorted(by_benchmark.items()):
            per_dim = {}
            for d in dimensions:
                dim_scores = [
                    ts["dimensions"].get(d, {}).get("score", 0.0)
                    for ts in scores
                ]
                per_dim[d] = round(sum(dim_scores) / len(dim_scores), 3) if dim_scores else 0.0

            quality_scores = [ts["du_quality_score"] for ts in scores]
            avg_quality = round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0

            sys_breakdown[bench] = {
                "n_tasks": len(scores),
                "per_dimension": per_dim,
                "du_quality_score": avg_quality,
            }

        breakdown[sys_name] = sys_breakdown

    return breakdown


def print_table(breakdown: Dict[str, Any]):
    """Print formatted per-benchmark breakdown table."""
    benchmarks = ["KramaBench", "AgentBench", "DACode"]
    dimensions = ["D1", "D2", "D3", "D4", "D5"]

    # Sort systems by overall quality (from the results)
    system_order = sorted(
        breakdown.keys(),
        key=lambda s: max(
            breakdown[s].get(b, {}).get("du_quality_score", 0)
            for b in benchmarks
        ),
        reverse=True,
    )

    for bench in benchmarks:
        print(f"\n{'=' * 80}")
        print(f"  {bench}")
        print(f"{'=' * 80}")
        header = f"{'System':<22} {'N':>4} {'D1':>6} {'D2':>6} {'D3':>6} {'D4':>6} {'D5':>6} {'Quality':>8}"
        print(header)
        print("-" * 80)

        for sys_name in system_order:
            bench_data = breakdown[sys_name].get(bench)
            if not bench_data:
                continue
            dims = bench_data["per_dimension"]
            print(
                f"{sys_name:<22} "
                f"{bench_data['n_tasks']:>4} "
                f"{dims.get('D1', 0):>6.3f} "
                f"{dims.get('D2', 0):>6.3f} "
                f"{dims.get('D3', 0):>6.3f} "
                f"{dims.get('D4', 0):>6.3f} "
                f"{dims.get('D5', 0):>6.3f} "
                f"{bench_data['du_quality_score']:>8.3f}"
            )


def generate_latex(breakdown: Dict[str, Any]) -> str:
    """Generate LaTeX table for per-benchmark breakdown."""
    benchmarks = ["KramaBench", "AgentBench", "DACode"]
    dimensions = ["D1", "D2", "D3", "D4", "D5"]

    # Sort by overall quality desc
    system_order = sorted(
        breakdown.keys(),
        key=lambda s: max(
            breakdown[s].get(b, {}).get("du_quality_score", 0)
            for b in benchmarks
        ),
        reverse=True,
    )

    lines = [
        r"\begin{table}[t]",
        r"\caption{Per-benchmark DU quality breakdown. Scores for each system split by benchmark source (KramaBench 105 / AgentBench 100 / DACode 52 tasks).}",
        r"\label{tab:du_per_benchmark}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{ll" + "c" * 6 + "}",
        r"\toprule",
        r"\textbf{System} & \textbf{Bench.} & \textbf{D1} & \textbf{D2} & \textbf{D3} & \textbf{D4} & \textbf{D5} & \textbf{Quality} \\",
        r"\midrule",
    ]

    for i, sys_name in enumerate(system_order):
        display_name = sys_name.replace("-", " ").replace("_", " ")
        first = True
        for bench in benchmarks:
            bench_data = breakdown[sys_name].get(bench)
            if not bench_data:
                continue
            dims = bench_data["per_dimension"]
            bench_abbrev = {"KramaBench": "KB", "AgentBench": "AB", "DACode": "DAC"}[bench]
            name_col = display_name if first else ""
            first = False

            # Find best D values across systems for this benchmark for bolding
            lines.append(
                f"  {name_col} & {bench_abbrev} & "
                f"{dims.get('D1', 0):.3f} & "
                f"{dims.get('D2', 0):.3f} & "
                f"{dims.get('D3', 0):.3f} & "
                f"{dims.get('D4', 0):.3f} & "
                f"{dims.get('D5', 0):.3f} & "
                f"{bench_data['du_quality_score']:.3f} \\\\"
            )
        if i < len(system_order) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found: {RESULTS_FILE}", file=sys.stderr)
        sys.exit(1)

    results = json.loads(RESULTS_FILE.read_text())
    breakdown = compute_breakdown(results)

    # Print table
    print_table(breakdown)

    # Save JSON
    out_json = OUTPUT_DIR / "per_benchmark_breakdown.json"
    out_json.write_text(json.dumps(breakdown, indent=2))
    print(f"\nJSON saved to {out_json}")

    # Save LaTeX
    latex = generate_latex(breakdown)
    out_tex = OUTPUT_DIR / "per_benchmark_breakdown.tex"
    out_tex.write_text(latex)
    print(f"LaTeX saved to {out_tex}")


if __name__ == "__main__":
    main()
