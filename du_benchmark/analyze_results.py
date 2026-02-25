# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Analysis and visualization for DU benchmark results.

Generates:
  1. Empirical heatmap (systems x D1-D5)
  2. Deterministic vs consensus agreement validation
  3. DU-Pipeline ablation bar chart
  4. Per-dimension importance chart
  5. Cross-community radar chart
  6. Bootstrap CIs and McNemar's pairwise tests

Usage:
    python3 evaluation/du_benchmark/analyze_results.py \\
        --results-dir evaluation/results/du_benchmark/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


from du_benchmark.metrics import (
    bootstrap_ci, compute_correlation, mcnemar_test,
)

_logger = logging.getLogger(__name__)

# System type classification for radar chart
SYSTEM_TYPES = {
    "ydata-profiling": "Profiler",
    "dataprofiler": "Profiler",
    "duckdb-sniffer": "Profiler",
    "adp-ma": "LLM Agent",
    "adp-ma-retro": "LLM Agent",
    "smolagents": "LLM Agent",
    "pandasai": "LLM Agent",
    "langchain": "LLM Agent",
    "chess": "Text-to-SQL",
    "din-sql": "Text-to-SQL",
    "profile-llm-hybrid": "Hybrid",
    "llm-only": "Baseline",
    "random": "Baseline",
}


# ── Data Loading ────────────────────────────────────────────────────

def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load the most recent benchmark results."""
    result_files = sorted(results_dir.glob("du_benchmark_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No result files in {results_dir}")
    latest = result_files[-1]
    _logger.info("Loading results from %s", latest)
    return json.loads(latest.read_text())


def load_ablation(results_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ablation results if available."""
    ablation_file = results_dir / "ablation" / "ablation_results.json"
    if ablation_file.exists():
        return json.loads(ablation_file.read_text())
    return None


# ── 1. Empirical Heatmap ───────────────────────────────────────────

def generate_heatmap(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Generate systems x D1-D5 heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _logger.error("matplotlib not available, generating text table only")
        _print_heatmap_text(results)
        return

    per_system = results.get("per_system", {})
    if not per_system:
        return

    systems = list(per_system.keys())
    dims = ["D1", "D2", "D3", "D4", "D5"]

    # Build matrix
    matrix = np.zeros((len(systems), len(dims)))
    for i, sys_name in enumerate(systems):
        dim_scores = per_system[sys_name].get("per_dimension", {})
        for j, dim in enumerate(dims):
            matrix[i, j] = dim_scores.get(dim, 0.0)

    fig, ax = plt.subplots(figsize=(10, max(6, len(systems) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems, fontsize=10)

    # Add text annotations
    for i in range(len(systems)):
        for j in range(len(dims)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_xlabel("DU Dimension", fontsize=12)
    ax.set_title("Data Understanding Quality: Systems x Dimensions", fontsize=13)
    plt.colorbar(im, ax=ax, label="Score (0-1)", shrink=0.8)

    plt.tight_layout()
    outf = output_dir / "heatmap_systems_dimensions.pdf"
    plt.savefig(outf, dpi=150, bbox_inches="tight")
    plt.close()
    _logger.info("Saved heatmap to %s", outf)

    # Also save as CSV for LaTeX
    _save_heatmap_csv(systems, dims, matrix, output_dir)


def _print_heatmap_text(results: Dict[str, Any]):
    """Fallback text table when matplotlib unavailable."""
    print("\n=== DU Quality Heatmap ===")
    print(f"{'System':<25} {'D1':>6} {'D2':>6} {'D3':>6} {'D4':>6} {'D5':>6} {'Quality':>8}")
    print("-" * 70)
    for name, data in results.get("per_system", {}).items():
        dims = data.get("per_dimension", {})
        print(
            f"{name:<25} "
            f"{dims.get('D1', 0):>6.3f} "
            f"{dims.get('D2', 0):>6.3f} "
            f"{dims.get('D3', 0):>6.3f} "
            f"{dims.get('D4', 0):>6.3f} "
            f"{dims.get('D5', 0):>6.3f} "
            f"{data.get('du_quality_score', 0):>8.3f}"
        )


def _save_heatmap_csv(
    systems: List[str], dims: List[str],
    matrix: np.ndarray, output_dir: Path,
):
    """Save heatmap data as CSV for LaTeX tables."""
    import csv
    outf = output_dir / "heatmap_data.csv"
    with open(outf, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["System"] + dims + ["Quality"])
        for i, sys_name in enumerate(systems):
            row = [sys_name] + [f"{matrix[i, j]:.3f}" for j in range(len(dims))]
            quality = sum(matrix[i, :]) / len(dims)
            row.append(f"{quality:.3f}")
            writer.writerow(row)
    _logger.info("Saved heatmap CSV to %s", outf)


# ── 2. Consensus Validation ────────────────────────────────────────

def validate_consensus(
    results: Dict[str, Any],
    output_dir: Path,
):
    """
    Compare consensus scores against deterministic ground truth on D2/D4.
    Reports agreement rate.
    """
    agreements = {"D2": [], "D4": []}

    for sys_name, sys_data in results.get("per_system", {}).items():
        for task_data in sys_data.get("per_task", []):
            for dim_data in task_data.get("dimensions", []):
                dim = dim_data.get("dimension")
                method = dim_data.get("method")
                if dim in ("D2", "D4") and method == "deterministic":
                    # Compare against consensus score
                    score = dim_data.get("score", 0)
                    agreements[dim].append(score)

    report = {}
    for dim, scores in agreements.items():
        if scores:
            above_50 = sum(1 for s in scores if s > 0.5)
            report[dim] = {
                "n_tasks": len(scores),
                "mean_score": round(np.mean(scores), 3),
                "above_50_pct": round(above_50 / len(scores) * 100, 1),
                "std": round(np.std(scores), 3),
            }

    outf = output_dir / "consensus_validation.json"
    outf.write_text(json.dumps(report, indent=2))
    _logger.info("Consensus validation: %s", report)
    return report


# ── 3. Ablation Bar Chart ──────────────────────────────────────────

def plot_ablation(
    ablation_data: Dict[str, Any],
    output_dir: Path,
):
    """Generate bar chart of ablation conditions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _print_ablation_text(ablation_data)
        return

    conditions = []
    pass_rates = []
    colors = []

    # Order: normal, consensus, no_du, scrambled, then degradations
    order = [
        "normal", "consensus_du", "no_du", "scrambled_du",
        "degrade_d1", "degrade_d2", "degrade_d3", "degrade_d4", "degrade_d5",
    ]
    color_map = {
        "normal": "#2196F3",
        "consensus_du": "#4CAF50",
        "no_du": "#F44336",
        "scrambled_du": "#FF9800",
    }

    for cond in order:
        if cond not in ablation_data:
            continue
        data = ablation_data[cond]
        conditions.append(cond.replace("_", "\n"))
        pass_rates.append(data["pass_rate"] * 100)
        colors.append(color_map.get(cond, "#9E9E9E"))

    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(conditions)), pass_rates, color=colors)

    # Add value labels
    for bar, rate in zip(bars, pass_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_title("DU Ablation: Impact on Pipeline Success", fontsize=13)
    ax.set_ylim(0, max(pass_rates) + 10)
    ax.axhline(
        y=pass_rates[0] if pass_rates else 0,
        color="blue", linestyle="--", alpha=0.3, label="Baseline",
    )

    plt.tight_layout()
    outf = output_dir / "ablation_bar_chart.pdf"
    plt.savefig(outf, dpi=150, bbox_inches="tight")
    plt.close()
    _logger.info("Saved ablation chart to %s", outf)


def _print_ablation_text(ablation_data: Dict[str, Any]):
    """Fallback text ablation summary."""
    print("\n=== Ablation Results ===")
    for cond, data in ablation_data.items():
        print(
            f"  {cond:<25} {data['passed']}/{data['total']} "
            f"({100 * data['pass_rate']:.1f}%)"
        )


# ── 4. Per-Dimension Importance ────────────────────────────────────

def plot_dimension_importance(
    ablation_data: Dict[str, Any],
    output_dir: Path,
):
    """Plot which dimension's corruption hurts most."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    baseline_rate = ablation_data.get("normal", {}).get("pass_rate", 0)
    dims = ["D1", "D2", "D3", "D4", "D5"]
    drops = []

    for dim in dims:
        cond = f"degrade_{dim.lower()}"
        if cond in ablation_data:
            degraded_rate = ablation_data[cond]["pass_rate"]
            drops.append((baseline_rate - degraded_rate) * 100)
        else:
            drops.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E53935" if d > 0 else "#43A047" for d in drops]
    bars = ax.bar(dims, drops, color=colors)

    for bar, drop in zip(bars, drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{drop:.1f}pp", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Pass Rate Drop (pp)", fontsize=12)
    ax.set_title("Per-Dimension Importance: Impact of DU Corruption", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    outf = output_dir / "dimension_importance.pdf"
    plt.savefig(outf, dpi=150, bbox_inches="tight")
    plt.close()
    _logger.info("Saved dimension importance chart to %s", outf)


# ── 5. Cross-Community Radar Chart ─────────────────────────────────

def plot_community_radar(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Radar chart comparing Profiler vs LLM Agent vs Text-to-SQL vs Hybrid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        _print_community_text(results)
        return

    # Aggregate per community
    communities: Dict[str, Dict[str, List[float]]] = {}
    for sys_name, sys_data in results.get("per_system", {}).items():
        comm = SYSTEM_TYPES.get(sys_name, "Other")
        if comm not in communities:
            communities[comm] = {f"D{i}": [] for i in range(1, 6)}
        dims = sys_data.get("per_dimension", {})
        for d in range(1, 6):
            key = f"D{d}"
            communities[comm][key].append(dims.get(key, 0))

    # Average per community
    comm_avgs: Dict[str, List[float]] = {}
    for comm, dim_lists in communities.items():
        comm_avgs[comm] = [
            np.mean(dim_lists[f"D{i}"]) if dim_lists[f"D{i}"] else 0
            for i in range(1, 6)
        ]

    # Radar chart
    dims = ["D1\nQuery", "D2\nSchema", "D3\nSemantic", "D4\nFormat", "D5\nConstraints"]
    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = {
        "Profiler": "#2196F3",
        "LLM Agent": "#4CAF50",
        "Text-to-SQL": "#FF9800",
        "Hybrid": "#9C27B0",
        "Baseline": "#9E9E9E",
    }

    for comm, avgs in comm_avgs.items():
        values = avgs + avgs[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                label=comm, color=colors.get(comm, "#333"))
        ax.fill(angles, values, alpha=0.1, color=colors.get(comm, "#333"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("DU Coverage by System Community", fontsize=13, pad=20)

    plt.tight_layout()
    outf = output_dir / "community_radar.pdf"
    plt.savefig(outf, dpi=150, bbox_inches="tight")
    plt.close()
    _logger.info("Saved community radar to %s", outf)


def _print_community_text(results: Dict[str, Any]):
    """Fallback text community summary."""
    communities: Dict[str, List[str]] = {}
    for sys_name in results.get("per_system", {}).keys():
        comm = SYSTEM_TYPES.get(sys_name, "Other")
        communities.setdefault(comm, []).append(sys_name)
    print("\n=== Community Breakdown ===")
    for comm, systems in communities.items():
        print(f"  {comm}: {', '.join(systems)}")


# ── 6. Statistical Tests ───────────────────────────────────────────

def run_statistical_tests(
    results: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run bootstrap CIs and McNemar's pairwise tests."""
    per_system = results.get("per_system", {})
    systems = list(per_system.keys())

    stats_report: Dict[str, Any] = {
        "bootstrap_ci": {},
        "mcnemar_tests": {},
    }

    # Bootstrap CIs per system
    for sys_name, sys_data in per_system.items():
        per_task = sys_data.get("per_task", [])
        quality_scores = [t.get("du_quality_score", 0) for t in per_task]
        mean, lower, upper = bootstrap_ci(quality_scores)
        stats_report["bootstrap_ci"][sys_name] = {
            "mean": mean, "lower": lower, "upper": upper,
        }

    # McNemar's pairwise
    for i, sys_a in enumerate(systems):
        for sys_b in systems[i + 1:]:
            a_tasks = per_system[sys_a].get("per_task", [])
            b_tasks = per_system[sys_b].get("per_task", [])

            # Match by task_id
            a_by_id = {t["task_id"]: t for t in a_tasks}
            b_by_id = {t["task_id"]: t for t in b_tasks}
            common = set(a_by_id) & set(b_by_id)

            if len(common) < 5:
                continue

            a_correct = [a_by_id[tid].get("du_quality_score", 0) > 0.5 for tid in common]
            b_correct = [b_by_id[tid].get("du_quality_score", 0) > 0.5 for tid in common]

            test = mcnemar_test(a_correct, b_correct)
            key = f"{sys_a}_vs_{sys_b}"
            stats_report["mcnemar_tests"][key] = test

    outf = output_dir / "statistical_tests.json"
    outf.write_text(json.dumps(stats_report, indent=2))
    _logger.info("Saved statistical tests to %s", outf)
    return stats_report


# ── 7. DU-Pipeline Correlation ─────────────────────────────────────

def analyze_du_pipeline_correlation(
    results: Dict[str, Any],
    pipeline_results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compute Pearson r between DU quality and pipeline pass rate.
    Requires existing pipeline results (e.g., KramaBench runs).
    """
    if pipeline_results_dir is None:
        pipeline_results_dir = Path("evaluation/results/kramabench/")

    # Load pipeline results
    pipeline_files = sorted(pipeline_results_dir.glob("kramabench_all_*.json"))
    if not pipeline_files:
        _logger.warning("No pipeline results found")
        return {}

    try:
        pipeline_data = json.loads(pipeline_files[-1].read_text())
    except Exception as e:
        _logger.error("Failed to load pipeline results: %s", e)
        return {}

    # Build task-level pass map
    task_pass = {}
    for task in pipeline_data.get("tasks", []):
        task_pass[task["task_id"]] = 1.0 if task.get("passed") else 0.0

    # Match with DU quality scores
    correlation_data: Dict[str, Dict[str, float]] = {}
    for sys_name, sys_data in results.get("per_system", {}).items():
        du_scores = []
        pass_scores = []
        for task_data in sys_data.get("per_task", []):
            tid = task_data.get("task_id")
            if tid in task_pass:
                du_scores.append(task_data.get("du_quality_score", 0))
                pass_scores.append(task_pass[tid])

        if du_scores:
            corr = compute_correlation(du_scores, pass_scores)
            correlation_data[sys_name] = corr

    if output_dir:
        outf = output_dir / "du_pipeline_correlation.json"
        outf.write_text(json.dumps(correlation_data, indent=2))

    return correlation_data


# ── Main Analysis ──────────────────────────────────────────────────

def run_analysis(results_dir: Path):
    """Run all analysis and generate all outputs."""
    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    results = load_results(results_dir)
    ablation = load_ablation(results_dir)

    # Generate all outputs
    _logger.info("Generating heatmap...")
    generate_heatmap(results, output_dir)

    _logger.info("Validating consensus...")
    validate_consensus(results, output_dir)

    if ablation:
        _logger.info("Generating ablation chart...")
        plot_ablation(ablation, output_dir)

        _logger.info("Generating dimension importance chart...")
        plot_dimension_importance(ablation, output_dir)

    _logger.info("Generating community radar...")
    plot_community_radar(results, output_dir)

    _logger.info("Running statistical tests...")
    run_statistical_tests(results, output_dir)

    _logger.info("Analyzing DU-pipeline correlation...")
    analyze_du_pipeline_correlation(results, output_dir=output_dir)

    _logger.info("Analysis complete. Outputs in %s", output_dir)

    # Print summary
    _print_heatmap_text(results)
    if ablation:
        _print_ablation_text(ablation)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze DU benchmark results",
    )
    parser.add_argument(
        "--results-dir",
        default="evaluation/results/du_benchmark/",
        help="Directory containing benchmark results",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_analysis(Path(args.results_dir))


if __name__ == "__main__":
    main()
