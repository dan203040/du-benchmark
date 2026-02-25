# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
#!/usr/bin/env python3
"""
Re-score all 6 DU benchmark systems against decontaminated consensus
(DeepSeek + Claude Sonnet + Gemini Flash — no GPT-4o-mini overlap).

Then regenerate analysis outputs (heatmap, radar, stats, LaTeX table).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


from du_benchmark.schema import (
    ConsensusDU, DUBenchmarkTask, DUOutput, TaskDUScore,
)
from du_benchmark.metrics import (
    aggregate_scores, bootstrap_ci, mcnemar_test, score_task,
)
from du_benchmark.consensus import load_du_tasks

_logger = logging.getLogger(__name__)

SYSTEMS = [
    "llm-only",
    "profile-llm-hybrid",
    "selective-hybrid",
    "ydata-profiling",
    "duckdb-sniffer",
    "random",
    "adp-ma-retro",
    "din-sql",
    "dr-du",
    "dr-du-gemini",
    "dr-du-grounding-only",
]

DU_OUTPUTS_DIR = Path("evaluation/results/du_benchmark/du_outputs")
RERUN_DIR = Path("evaluation/results/du_benchmark/rerun")
CONSENSUS_FILE = Path("evaluation/results/du_benchmark/consensus_decontaminated/consensus_all.json")
OUTPUT_DIR = Path("evaluation/results/du_benchmark/analysis_decontaminated")


def load_system_outputs(system_name: str) -> Dict[str, DUOutput]:
    """Load DU outputs for a system."""
    base = DU_OUTPUTS_DIR / system_name
    rerun = RERUN_DIR / system_name
    outputs = {}
    if not base.exists():
        return outputs
    for f in sorted(base.glob("*.json")):
        task_id = f.stem
        src = rerun / f.name if (rerun / f.name).exists() else f
        try:
            data = json.loads(src.read_text())
            outputs[task_id] = DUOutput.from_dict(data)
        except Exception as e:
            _logger.warning("Failed to load %s/%s: %s", system_name, task_id, e)
    return outputs


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load decontaminated consensus
    _logger.info("Loading decontaminated consensus...")
    raw_consensus = json.loads(CONSENSUS_FILE.read_text())
    consensus_map = {tid: ConsensusDU.from_dict(d) for tid, d in raw_consensus.items()}
    _logger.info("Loaded consensus for %d tasks", len(consensus_map))

    # Load tasks (for scoring context: dataframes, file_paths)
    _logger.info("Loading tasks...")
    tasks = load_du_tasks(["kramabench", "agentbench", "dacode"])
    task_map = {t.task_id: t for t in tasks}
    _logger.info("Loaded %d tasks", len(tasks))

    # Load task data (dataframes) — needed for D2/D4 deterministic scoring
    # We need actual dataframes for verify_d2 and verify_d4
    task_dataframes: Dict[str, Dict[str, pd.DataFrame]] = {}
    for t in tasks:
        # Build dataframes from task metadata (samples aren't full data but score_task
        # uses them for D2/D4). For consensus-only dims (D1,D3,D5) no data needed.
        # D2/D4 deterministic scoring requires actual DataFrames — we pass empty
        # if not available, and the scoring gracefully handles it.
        task_dataframes[t.task_id] = {}  # Will use consensus fallback for D2/D4

    # Score all systems
    all_results: Dict[str, Any] = {}

    for system_name in SYSTEMS:
        _logger.info("Scoring system: %s", system_name)
        outputs = load_system_outputs(system_name)
        _logger.info("  Loaded %d outputs", len(outputs))

        task_scores: List[TaskDUScore] = []
        for task in tasks:
            if task.task_id not in outputs:
                continue
            if task.task_id not in consensus_map:
                continue

            du_output = outputs[task.task_id]
            consensus = consensus_map[task.task_id]

            try:
                ts = score_task(
                    task=task,
                    du_output=du_output,
                    consensus=consensus,
                    dataframes=task_dataframes.get(task.task_id, {}),
                    file_paths=None,  # Use consensus fallback for D2/D4
                )
                task_scores.append(ts)
            except Exception as e:
                _logger.warning("  Score failed for %s/%s: %s", system_name, task.task_id, e)

        # Aggregate
        agg = aggregate_scores(task_scores)
        quality_scores = [ts.du_quality_score for ts in task_scores]
        mean, lower, upper = bootstrap_ci(quality_scores)

        all_results[system_name] = {
            "aggregate": agg,
            "bootstrap_ci_95": {"mean": mean, "lower": lower, "upper": upper},
            "task_scores": [
                {
                    "task_id": ts.task_id,
                    "dimensions": {
                        ds.dimension: {"score": ds.score, "method": ds.method}
                        for ds in ts.dimensions
                    },
                    "du_quality_score": ts.du_quality_score,
                    "du_coverage_score": ts.du_coverage_score,
                }
                for ts in task_scores
            ],
        }

        _logger.info(
            "  %s: Quality=%.3f [%.3f, %.3f], D1=%.3f D2=%.3f D3=%.3f D4=%.3f D5=%.3f",
            system_name, mean, lower, upper,
            agg.get("per_dimension", {}).get("D1", 0),
            agg.get("per_dimension", {}).get("D2", 0),
            agg.get("per_dimension", {}).get("D3", 0),
            agg.get("per_dimension", {}).get("D4", 0),
            agg.get("per_dimension", {}).get("D5", 0),
        )

    # Save results
    results_file = OUTPUT_DIR / "du_benchmark_decontaminated.json"
    results_file.write_text(json.dumps(all_results, indent=2, default=str))
    _logger.info("Results saved to %s", results_file)

    # McNemar's pairwise tests
    _logger.info("Running McNemar's pairwise tests...")
    # Build per-task pass/fail (quality > threshold) for each system
    threshold = 0.3  # Task is "correct" if quality > 0.3
    system_correct: Dict[str, Dict[str, bool]] = {}
    for sys_name, data in all_results.items():
        system_correct[sys_name] = {
            ts["task_id"]: ts["du_quality_score"] > threshold
            for ts in data["task_scores"]
        }

    # Pairwise tests
    pairwise_tests = {}
    sys_names = list(all_results.keys())
    all_task_ids = sorted(
        set().union(*(system_correct[s].keys() for s in sys_names))
    )
    for i, sa in enumerate(sys_names):
        for sb in sys_names[i + 1:]:
            a_correct = [system_correct[sa].get(tid, False) for tid in all_task_ids]
            b_correct = [system_correct[sb].get(tid, False) for tid in all_task_ids]
            result = mcnemar_test(a_correct, b_correct)
            pairwise_tests[f"{sa} vs {sb}"] = result

    stats_file = OUTPUT_DIR / "statistical_tests.json"
    stats_file.write_text(json.dumps(pairwise_tests, indent=2))
    _logger.info("Statistical tests saved to %s", stats_file)

    # Generate LaTeX table
    _logger.info("Generating LaTeX table...")
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{DU Benchmark Results (Decontaminated Consensus: DeepSeek + Claude + Gemini Flash)}",
        r"\label{tab:du_benchmark}",
        r"\begin{tabular}{lccccccl}",
        r"\toprule",
        r"System & D1 & D2 & D3 & D4 & D5 & Quality & CI$_{95}$ \\",
        r"\midrule",
    ]

    # Sort by quality score descending
    sorted_systems = sorted(
        all_results.items(),
        key=lambda x: x[1]["bootstrap_ci_95"]["mean"],
        reverse=True,
    )

    for sys_name, data in sorted_systems:
        agg = data["aggregate"]
        dims = agg.get("per_dimension", {})
        ci = data["bootstrap_ci_95"]
        name = sys_name.replace("-", " ").replace("_", " ")
        if sys_name == sorted_systems[0][0]:
            name = r"\textbf{" + name + "}"
        latex_lines.append(
            f"  {name} & "
            f"{dims.get('D1', 0):.3f} & "
            f"{dims.get('D2', 0):.3f} & "
            f"{dims.get('D3', 0):.3f} & "
            f"{dims.get('D4', 0):.3f} & "
            f"{dims.get('D5', 0):.3f} & "
            f"\\textbf{{{ci['mean']:.3f}}} & "
            f"[{ci['lower']:.3f}, {ci['upper']:.3f}] \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_file = OUTPUT_DIR / "du_benchmark_table.tex"
    tex_file.write_text("\n".join(latex_lines))
    _logger.info("LaTeX table saved to %s", tex_file)

    # Print summary
    print("\n" + "=" * 90)
    print("DU BENCHMARK RESULTS (Decontaminated Consensus)")
    print("Panel: DeepSeek + Claude Sonnet 4.5 + Gemini 2.0 Flash (no GPT-4o-mini overlap)")
    print("=" * 90)
    print(f"{'System':<22} {'D1':>6} {'D2':>6} {'D3':>6} {'D4':>6} {'D5':>6} {'Quality':>8} {'CI_95':>16}")
    print("-" * 90)
    for sys_name, data in sorted_systems:
        agg = data["aggregate"]
        dims = agg.get("per_dimension", {})
        ci = data["bootstrap_ci_95"]
        print(
            f"{sys_name:<22} "
            f"{dims.get('D1', 0):>6.3f} "
            f"{dims.get('D2', 0):>6.3f} "
            f"{dims.get('D3', 0):>6.3f} "
            f"{dims.get('D4', 0):>6.3f} "
            f"{dims.get('D5', 0):>6.3f} "
            f"{ci['mean']:>8.3f} "
            f"[{ci['lower']:.3f}, {ci['upper']:.3f}]"
        )
    print("=" * 90)

    # Generate analysis plots
    try:
        _generate_plots(all_results)
    except Exception as e:
        _logger.warning("Plot generation failed: %s", e)


def _generate_plots(all_results: Dict[str, Any]):
    """Generate heatmap and radar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Heatmap
    systems = sorted(all_results.keys(), key=lambda s: all_results[s]["bootstrap_ci_95"]["mean"], reverse=True)
    dims = ["D1", "D2", "D3", "D4", "D5"]
    data = []
    for s in systems:
        row = [all_results[s]["aggregate"].get("per_dimension", {}).get(d, 0) for d in dims]
        data.append(row)
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([s.replace("-", " ") for s in systems], fontsize=10)
    for i in range(len(systems)):
        for j in range(len(dims)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if data[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Score")
    ax.set_title("DU Quality: Systems × Dimensions (Decontaminated)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "heatmap_systems_dimensions.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "heatmap_systems_dimensions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Heatmap saved")

    # Radar chart
    communities = {
        "Profilers": ["ydata-profiling", "duckdb-sniffer"],
        "LLM Agents": ["llm-only"],
        "Hybrid": ["profile-llm-hybrid", "selective-hybrid"],
        "NL-to-SQL": ["din-sql"],
        "Baseline": ["random", "adp-ma-retro"],
    }
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9E9E9E"]
    for idx, (comm, members) in enumerate(communities.items()):
        comm_scores = []
        for d in dims:
            scores = [
                all_results[s]["aggregate"].get("per_dimension", {}).get(d, 0)
                for s in members if s in all_results
            ]
            comm_scores.append(np.mean(scores) if scores else 0)
        comm_scores += comm_scores[:1]
        ax.plot(angles, comm_scores, "o-", linewidth=2, label=comm, color=colors[idx])
        ax.fill(angles, comm_scores, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Cross-Community DU Coverage (Decontaminated)", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "community_radar.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "community_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Radar chart saved")

    # Quality bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    sys_sorted = sorted(all_results.items(), key=lambda x: x[1]["bootstrap_ci_95"]["mean"], reverse=True)
    names = [s.replace("-", " ") for s, _ in sys_sorted]
    means = [d["bootstrap_ci_95"]["mean"] for _, d in sys_sorted]
    lowers = [d["bootstrap_ci_95"]["mean"] - d["bootstrap_ci_95"]["lower"] for _, d in sys_sorted]
    uppers = [d["bootstrap_ci_95"]["upper"] - d["bootstrap_ci_95"]["mean"] for _, d in sys_sorted]
    bars = ax.barh(range(len(names)), means, xerr=[lowers, uppers], capsize=4,
                   color=["#2196F3" if m > 0.4 else "#FF9800" if m > 0.3 else "#9E9E9E" for m in means])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("DU Quality Score", fontsize=11)
    ax.set_title("DU Quality with 95% Bootstrap CI (Decontaminated)", fontsize=12)
    ax.set_xlim(0, 0.6)
    for i, m in enumerate(means):
        ax.text(m + 0.01, i, f"{m:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "quality_bar_chart.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "quality_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Quality bar chart saved")


if __name__ == "__main__":
    main()
