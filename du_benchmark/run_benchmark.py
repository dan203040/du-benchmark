# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Main DU Benchmark Runner.

Orchestrates DU extraction across all systems and tasks, then scores
using deterministic verification (D2/D4) and consensus comparison (D1/D3/D5).

Usage:
    # Run full benchmark
    python3 evaluation/du_benchmark/run_du_benchmark.py \\
        --systems all --tasks kramabench,agentbench,dacode

    # Run specific systems
    python3 evaluation/du_benchmark/run_du_benchmark.py \\
        --systems adp-ma,ydata-profiling,llm-only --tasks kramabench

    # Run ablation study
    python3 evaluation/du_benchmark/run_du_benchmark.py \\
        --system adp-ma --ablation all --tasks kramabench

    # P0 systems only (core)
    python3 evaluation/du_benchmark/run_du_benchmark.py \\
        --tier p0 --tasks all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


from du_benchmark.schema import (
    BenchmarkSource, ConsensusDU, DUBenchmarkTask, DUOutput, TaskDUScore,
)
from du_benchmark.consensus import (
    generate_all_consensus, load_du_tasks,
)
from du_benchmark.metrics import (
    aggregate_scores, score_task, bootstrap_ci,
)

_logger = logging.getLogger(__name__)


# ── System Registry ─────────────────────────────────────────────────

def _get_system_registry() -> Dict[str, Dict[str, Any]]:
    """Registry of all available DU extraction systems."""
    return {
        # P0 — Core systems
        "adp-ma": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.adp_ma:ADPMAExtractor",
            "kwargs": {},
        },
        "adp-ma-retro": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.adp_ma:ADPMARetroExtractor",
            "kwargs": {},
        },
        "ydata-profiling": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.profilers:YDataProfilingExtractor",
            "kwargs": {},
        },
        "dataprofiler": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.profilers:DataProfilerExtractor",
            "kwargs": {},
        },
        "duckdb-sniffer": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.profilers:DuckDBSnifferExtractor",
            "kwargs": {},
        },
        "smolagents": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.llm_agents:SmolAgentsExtractor",
            "kwargs": {},
        },
        "profile-llm-hybrid": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.llm_agents:ProfileLLMHybridExtractor",
            "kwargs": {},
        },
        "llm-only": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.llm_agents:LLMOnlyExtractor",
            "kwargs": {},
        },
        "random": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.llm_agents:RandomBaselineExtractor",
            "kwargs": {},
        },
        "dr-du": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.dr_du:DRDUExtractor",
            "kwargs": {},
        },
        "dr-du-gemini": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.dr_du:DRDUExtractor",
            "kwargs": {"provider": "google", "model": "gemini-2.0-flash"},
        },
        "dr-du-grounding-only": {
            "tier": "p0",
            "factory": "du_benchmark.extractors.dr_du:DRDUGroundingOnlyExtractor",
            "kwargs": {},
        },
        # P1 — Extended systems
        "pandasai": {
            "tier": "p1",
            "factory": "du_benchmark.extractors.llm_agents:PandasAIExtractor",
            "kwargs": {},
        },
        "langchain": {
            "tier": "p1",
            "factory": "du_benchmark.extractors.llm_agents:LangChainExtractor",
            "kwargs": {},
        },
        "chess": {
            "tier": "p1",
            "factory": "du_benchmark.extractors.text_to_sql:CHESSExtractor",
            "kwargs": {},
        },
        "din-sql": {
            "tier": "p1",
            "factory": "du_benchmark.extractors.text_to_sql:DINSQLExtractor",
            "kwargs": {},
        },
    }


def _instantiate_extractor(name: str, registry: Dict[str, Dict]) -> Any:
    """Instantiate an extractor from the registry."""
    entry = registry[name]
    module_path, class_name = entry["factory"].rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**entry.get("kwargs", {}))


def _resolve_systems(
    systems_arg: str,
    tier_arg: Optional[str] = None,
) -> List[str]:
    """Resolve --systems and --tier args to system names."""
    registry = _get_system_registry()

    if systems_arg == "all":
        if tier_arg:
            return [n for n, e in registry.items() if e["tier"] == tier_arg]
        return list(registry.keys())

    return [s.strip() for s in systems_arg.split(",")]


# ── Ablation Experiment ─────────────────────────────────────────────

ABLATION_CONDITIONS = {
    "normal": "Run ADP-MA as-is (baseline)",
    "consensus_du": "Inject silver M* (3-LLM consensus) — near-oracle",
    "no_du": "Skip DU step, pass empty metadata — lower bound",
    "scrambled_du": "Inject another task's M* (wrong DU) — noise floor",
    "degrade_d1": "Corrupt D1 (query understanding) in consensus M*",
    "degrade_d2": "Corrupt D2 (join keys) in consensus M*",
    "degrade_d3": "Corrupt D3 (units) in consensus M*",
    "degrade_d4": "Corrupt D4 (format) in consensus M*",
    "degrade_d5": "Corrupt D5 (constraints) in consensus M*",
}


async def run_ablation(
    tasks: List[DUBenchmarkTask],
    consensus_map: Dict[str, ConsensusDU],
    conditions: List[str],
    output_dir: Path,
    provider: str = "deepseek",
    model: str = "deepseek-chat",
) -> Dict[str, Dict[str, Any]]:
    """Run ablation experiment: vary DU conditions and measure pipeline success."""
    try:
        from meta_agents_adk.runner_llm import ADKLLMPipelineRunner
    except ImportError:
        raise ImportError(
            "Ablation requires the ADP-MA pipeline. "
            "Install it with: pip install adp-ma  (or clone the main repo)"
        )

    results: Dict[str, Dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to KramaBench only (ablation uses ADP-MA pipeline)
    kb_tasks = [
        t for t in tasks if t.benchmark == BenchmarkSource.KRAMABENCH
    ]
    if not kb_tasks:
        _logger.warning("No KramaBench tasks for ablation")
        return results

    # Resume support
    results_file = output_dir / "ablation_results.json"
    if results_file.exists():
        try:
            results = json.loads(results_file.read_text())
            _logger.info("Resuming ablation: %d conditions done", len(results))
        except Exception:
            pass

    for condition in conditions:
        if condition in results:
            _logger.info("Skipping completed condition: %s", condition)
            continue

        _logger.info("Running ablation condition: %s", condition)
        condition_results: List[Dict[str, Any]] = []

        for task in kb_tasks:
            task_result = {
                "task_id": task.task_id,
                "condition": condition,
            }

            try:
                runner = ADKLLMPipelineRunner(
                    planning_provider=provider,
                    planning_model=model,
                    coding_provider=provider,
                    coding_model=model,
                )

                # Load actual data
                from du_benchmark.adapters.kramabench import KramaBenchAdapter
                project_root = Path(__file__).resolve().parents[2]
                adapter = KramaBenchAdapter(
                    project_root / "evaluation" / "external" / "kramabench",
                    output_dir,
                )
                bt = adapter.load_task(task.task_id)
                data_dict = adapter.get_task_data(bt)

                primary_name = max(data_dict, key=lambda k: len(data_dict[k]))
                primary_df = data_dict[primary_name]
                additional = {
                    k: v for k, v in data_dict.items() if k != primary_name
                }

                t0 = time.time()

                if condition == "no_du":
                    # Skip DU entirely
                    runner._data_understanding_result = {
                        "columns": list(primary_df.columns),
                        "dtypes": {},
                        "shape": primary_df.shape,
                        "summary": "No data understanding performed.",
                        "data_type": "dataframe",
                    }
                elif condition == "scrambled_du":
                    # Use another task's consensus
                    other_ids = [
                        t.task_id for t in kb_tasks
                        if t.task_id != task.task_id
                        and t.task_id in consensus_map
                    ]
                    if other_ids:
                        import random
                        random.seed(42)
                        other_id = random.choice(other_ids)
                        # Inject wrong DU as summary
                        wrong_consensus = consensus_map[other_id]
                        runner._data_understanding_result = {
                            "columns": list(primary_df.columns),
                            "dtypes": {},
                            "shape": primary_df.shape,
                            "summary": json.dumps(wrong_consensus.to_dict()),
                            "data_type": "dataframe",
                        }

                # Run pipeline
                result = await runner.run(
                    user_goal=bt.goal,
                    input_df=primary_df,
                    additional_data_sources=additional,
                )
                elapsed = time.time() - t0

                # Evaluate
                output_df = result.get("output")
                if output_df is not None:
                    metrics = adapter.evaluate_output(bt, output_df)
                    task_result["passed"] = metrics.get("passed", False)
                else:
                    task_result["passed"] = False

                task_result["duration_s"] = round(elapsed, 1)
                task_result["success"] = True

            except Exception as e:
                task_result["passed"] = False
                task_result["success"] = False
                task_result["error"] = str(e)
                _logger.error(
                    "Ablation %s/%s failed: %s", condition, task.task_id, e,
                )

            condition_results.append(task_result)

        passed = sum(1 for r in condition_results if r.get("passed"))
        total = len(condition_results)
        results[condition] = {
            "condition": condition,
            "description": ABLATION_CONDITIONS.get(condition, ""),
            "passed": passed,
            "total": total,
            "pass_rate": round(passed / max(total, 1), 3),
            "tasks": condition_results,
        }

        # Save after each condition
        results_file.write_text(json.dumps(results, indent=2, default=str))
        _logger.info(
            "Condition %s: %d/%d (%.1f%%)",
            condition, passed, total, 100 * passed / max(total, 1),
        )

    return results


# ── Main Runner ─────────────────────────────────────────────────────

async def run_benchmark(
    systems: List[str],
    benchmarks: List[str],
    output_dir: Path,
    consensus_dir: Optional[Path] = None,
    skip_consensus: bool = False,
    ablation_conditions: Optional[List[str]] = None,
    provider: str = "deepseek",
    model: str = "deepseek-chat",
) -> Dict[str, Any]:
    """Run the full DU benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load tasks
    _logger.info("Loading tasks from %s", benchmarks)
    tasks = load_du_tasks(benchmarks)
    _logger.info("Loaded %d tasks", len(tasks))

    if not tasks:
        _logger.error("No tasks loaded. Check benchmark directories.")
        return {}

    # 2. Generate or load consensus
    if consensus_dir is None:
        consensus_dir = output_dir / "consensus"

    consensus_map: Dict[str, ConsensusDU] = {}
    if not skip_consensus:
        consensus_file = consensus_dir / "consensus_all.json"
        if consensus_file.exists():
            _logger.info("Loading existing consensus from %s", consensus_file)
            raw = json.loads(consensus_file.read_text())
            for tid, data in raw.items():
                consensus_map[tid] = ConsensusDU.from_dict(data)
        else:
            _logger.info("Generating consensus for %d tasks", len(tasks))
            consensus_map = await generate_all_consensus(
                tasks,
                model_keys=["deepseek", "claude-sonnet", "gemini-flash"],
                output_dir=consensus_dir,
            )

    # 3. Run each system
    registry = _get_system_registry()
    all_results: Dict[str, Any] = {
        "timestamp": timestamp,
        "benchmarks": benchmarks,
        "systems": systems,
        "n_tasks": len(tasks),
        "per_system": {},
    }

    # Resume support
    results_file = output_dir / f"du_benchmark_{timestamp}.json"
    existing_results: Dict[str, Any] = {}
    latest = sorted(output_dir.glob("du_benchmark_*.json"))
    if latest:
        try:
            existing_results = json.loads(latest[-1].read_text())
            _logger.info("Resuming from %s", latest[-1])
        except Exception:
            pass

    for system_name in systems:
        if system_name in existing_results.get("per_system", {}):
            _logger.info("Skipping completed system: %s", system_name)
            all_results["per_system"][system_name] = (
                existing_results["per_system"][system_name]
            )
            continue

        if system_name not in registry:
            _logger.warning("Unknown system: %s", system_name)
            continue

        _logger.info("=== Running system: %s ===", system_name)

        try:
            extractor = _instantiate_extractor(system_name, registry)
        except Exception as e:
            _logger.error("Failed to instantiate %s: %s", system_name, e)
            continue

        task_scores: List[TaskDUScore] = []
        du_outputs: Dict[str, DUOutput] = {}

        for i, task in enumerate(tasks):
            _logger.info(
                "[%d/%d] %s: %s", i + 1, len(tasks),
                system_name, task.task_id,
            )

            # Build DataFrames from task metadata
            dataframes: Dict[str, pd.DataFrame] = {}
            for fname in task.data_files:
                sample_csv = task.file_samples.get(fname, "")
                if sample_csv:
                    try:
                        from io import StringIO
                        dataframes[fname] = pd.read_csv(StringIO(sample_csv))
                    except Exception:
                        pass

            try:
                du_output = await extractor.extract_du(
                    task, dataframes, file_paths=None,
                )
                du_outputs[task.task_id] = du_output
            except Exception as e:
                _logger.error(
                    "%s failed on %s: %s", system_name, task.task_id, e,
                )
                du_output = DUOutput(system_name=system_name)
                du_outputs[task.task_id] = du_output

            # Score against consensus
            consensus = consensus_map.get(task.task_id, ConsensusDU())
            ts = score_task(
                task, du_output, consensus, dataframes,
            )
            task_scores.append(ts)

        # Aggregate
        agg = aggregate_scores(task_scores)

        # Bootstrap CI
        quality_scores = [ts.du_quality_score for ts in task_scores]
        mean, lower, upper = bootstrap_ci(quality_scores)
        agg["quality_ci_95"] = {"mean": mean, "lower": lower, "upper": upper}

        # Per-task details
        agg["per_task"] = [ts.to_dict() for ts in task_scores]

        all_results["per_system"][system_name] = agg

        # Save DU outputs
        du_out_dir = output_dir / "du_outputs" / system_name
        du_out_dir.mkdir(parents=True, exist_ok=True)
        for tid, du in du_outputs.items():
            (du_out_dir / f"{tid}.json").write_text(
                json.dumps(du.to_dict(), indent=2, default=str)
            )

        # Incremental save
        results_file.write_text(
            json.dumps(all_results, indent=2, default=str)
        )
        _logger.info(
            "%s: DU Quality = %.3f [%.3f, %.3f]",
            system_name, mean, lower, upper,
        )

    # 4. Run ablation if requested
    if ablation_conditions:
        _logger.info("=== Running ablation experiment ===")
        ablation_results = await run_ablation(
            tasks, consensus_map, ablation_conditions,
            output_dir / "ablation",
            provider=provider, model=model,
        )
        all_results["ablation"] = ablation_results

    # Final save
    results_file.write_text(json.dumps(all_results, indent=2, default=str))
    _logger.info("Results saved to %s", results_file)

    # Print summary table
    _print_summary(all_results)

    return all_results


def _print_summary(results: Dict[str, Any]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("DU BENCHMARK RESULTS")
    print("=" * 80)
    print(
        f"{'System':<25} {'D1':>6} {'D2':>6} {'D3':>6} {'D4':>6} "
        f"{'D5':>6} {'Quality':>8} {'Latency':>8}"
    )
    print("-" * 80)

    for name, data in results.get("per_system", {}).items():
        dims = data.get("per_dimension", {})
        print(
            f"{name:<25} "
            f"{dims.get('D1', 0):>6.3f} "
            f"{dims.get('D2', 0):>6.3f} "
            f"{dims.get('D3', 0):>6.3f} "
            f"{dims.get('D4', 0):>6.3f} "
            f"{dims.get('D5', 0):>6.3f} "
            f"{data.get('du_quality_score', 0):>8.3f} "
            f"{data.get('avg_latency_s', 0):>7.1f}s"
        )

    print("=" * 80)

    # Ablation summary
    if "ablation" in results:
        print("\nABLATION RESULTS")
        print("-" * 50)
        for cond, data in results["ablation"].items():
            print(
                f"  {cond:<25} {data['passed']}/{data['total']} "
                f"({100 * data['pass_rate']:.1f}%)"
            )
        print("-" * 50)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DU Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--systems", default="all",
        help="Comma-separated system names or 'all'",
    )
    parser.add_argument(
        "--tier", choices=["p0", "p1"],
        help="Run only systems in this tier",
    )
    parser.add_argument(
        "--tasks", default="all",
        help="Comma-separated benchmark names or 'all'",
    )
    parser.add_argument(
        "--output", default="evaluation/results/du_benchmark/",
        help="Output directory",
    )
    parser.add_argument(
        "--consensus-dir",
        help="Pre-computed consensus directory",
    )
    parser.add_argument(
        "--skip-consensus", action="store_true",
        help="Skip consensus generation (use existing or score without)",
    )
    parser.add_argument(
        "--ablation", nargs="*",
        help="Ablation conditions to run (or 'all')",
    )
    parser.add_argument(
        "--provider", default="deepseek",
        help="LLM provider for ADP-MA ablation",
    )
    parser.add_argument(
        "--model", default="deepseek-chat",
        help="LLM model for ADP-MA ablation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    systems = _resolve_systems(args.systems, args.tier)
    benchmarks = (
        ["kramabench", "agentbench", "dacode"]
        if args.tasks == "all"
        else args.tasks.split(",")
    )

    ablation_conditions = None
    if args.ablation:
        if "all" in args.ablation:
            ablation_conditions = list(ABLATION_CONDITIONS.keys())
        else:
            ablation_conditions = args.ablation

    consensus_dir = Path(args.consensus_dir) if args.consensus_dir else None

    asyncio.run(run_benchmark(
        systems=systems,
        benchmarks=benchmarks,
        output_dir=Path(args.output),
        consensus_dir=consensus_dir,
        skip_consensus=args.skip_consensus,
        ablation_conditions=ablation_conditions,
        provider=args.provider,
        model=args.model,
    ))


if __name__ == "__main__":
    main()
