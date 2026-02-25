# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
#!/usr/bin/env python3
"""
DU Ablation Experiment: Inject different DU quality levels into ADP-MA pipeline.

Tests the causal impact of data understanding on downstream pipeline success
by replacing ADP-MA's native DU with external DU outputs from the DU benchmark.

Conditions:
  1. consensus    — 3-LLM silver M* (near-oracle upper bound)
  2. llm-only     — GPT-4o-mini single-prompt M*
  3. profiler     — ydata-profiling M* (D2/D4 only, no semantic)
  4. no-du        — empty metadata (lower bound)
  5. scrambled    — wrong task's M* (noise floor)

Usage:
    DEEPSEEK_API_KEY="sk-..." python3 evaluation/run_du_ablation.py \
        --conditions all --provider deepseek --model deepseek-chat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import traceback
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Data directories — override with DU_BENCHMARK_DATA env var
_DATA_ROOT = Path(os.environ.get("DU_BENCHMARK_DATA", str(PROJECT_DIR / "data")))
KRAMABENCH_DIR = _DATA_ROOT / "kramabench"
RESULTS_DIR = PROJECT_DIR / "results"

from du_benchmark.adapters.base import BenchmarkTask
from du_benchmark.adapters.kramabench import KramaBenchAdapter
from du_benchmark.schema import (
    ConsensusDU, ConsensusField, DUOutput, JoinKey,
)

_logger = logging.getLogger(__name__)

# ── All 5 ablation conditions ──────────────────────────────────────

ABLATION_CONDITIONS = [
    "consensus",     # 3-LLM silver M* → upper bound
    "llm-only",      # GPT-4o-mini single-prompt
    "profiler",      # ydata-profiling (D2/D4 only)
    "no-du",         # empty metadata
    "scrambled",     # wrong task's M*
]


def build_adapter() -> KramaBenchAdapter:
    """Create a KramaBenchAdapter instance."""
    return KramaBenchAdapter(
        benchmark_dir=KRAMABENCH_DIR,
        output_dir=RESULTS_DIR,
    )


# ── M* → ADP-MA DU Adapter ────────────────────────────────────────

def _du_output_to_adp_format(
    du: DUOutput,
    input_df: pd.DataFrame,
    additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Convert DU benchmark M* output to ADP-MA's internal DU dict format.

    ADP-MA's pipeline expects a dict with keys: columns, dtypes, shape,
    summary, data_type, join_keys, additional_sources, data_already_loaded.
    """
    # Infer data_type from M_Q output cardinality or default
    data_type = "dataframe"

    # Build summary text from M* fields
    summary_parts = []
    if du.m_q.target_metric:
        summary_parts.append(f"Target metric: {du.m_q.target_metric}")
    if du.m_q.filters:
        summary_parts.append(f"Filters: {', '.join(du.m_q.filters)}")
    if du.m_q.grouping_variables:
        summary_parts.append(f"Grouping: {', '.join(du.m_q.grouping_variables)}")
    if du.m_u.unit_annotations:
        summary_parts.append(
            f"Units: {', '.join(f'{k}={v}' for k, v in du.m_u.unit_annotations.items())}"
        )
    if du.m_c.constraints:
        summary_parts.append(f"Constraints: {'; '.join(du.m_c.constraints)}")
    summary = "\n".join(summary_parts) if summary_parts else "No data understanding available."

    # Build join_keys in ADP-MA format: {source_name: [{"column": col_name}]}
    join_keys = {}
    for jk in du.m_s.join_keys:
        if jk.right_source and jk.right_column:
            src_name = jk.right_source.replace(".csv", "").replace(".xlsx", "")
            if src_name not in join_keys:
                join_keys[src_name] = []
            join_keys[src_name].append({"column": jk.right_column})
        if jk.left_source and jk.left_column:
            src_name = jk.left_source.replace(".csv", "").replace(".xlsx", "")
            if src_name not in join_keys:
                join_keys[src_name] = []
            join_keys[src_name].append({"column": jk.left_column})

    # If no explicit join keys but additional DFs exist, try column overlap
    if not join_keys and additional_dfs:
        primary_cols = set(c.lower() for c in input_df.columns)
        for src_name, adf in additional_dfs.items():
            overlap = primary_cols & set(c.lower() for c in adf.columns)
            if overlap:
                join_keys[src_name] = [
                    {"column": c} for c in adf.columns if c.lower() in overlap
                ]

    # Build additional_sources metadata
    additional_sources = {}
    if additional_dfs:
        for src_name, adf in additional_dfs.items():
            additional_sources[src_name] = {
                "columns": list(adf.columns),
                "dtypes": {col: str(dtype) for col, dtype in adf.dtypes.items()},
                "shape": adf.shape,
            }

    return {
        "columns": list(input_df.columns),
        "dtypes": {col: str(dtype) for col, dtype in input_df.dtypes.items()},
        "shape": input_df.shape,
        "summary": summary,
        "format_info": {
            "has_header": du.m_f.has_header,
            "delimiter": du.m_f.delimiter,
            "encoding": du.m_f.encoding,
            "file_format": du.m_f.file_format,
        },
        "data_already_loaded": True,
        "data_type": data_type,
        "join_keys": join_keys,
        "additional_sources": additional_sources,
    }


def _consensus_to_du_output(consensus: ConsensusDU) -> DUOutput:
    """Convert a ConsensusDU silver standard to a DUOutput for injection."""
    du = DUOutput(system_name="consensus")

    # M_Q
    du.m_q.target_metric = (
        consensus.m_q.get("target_metric", ConsensusField()).value or ""
    )
    du.m_q.filters = consensus.m_q.get("filters", ConsensusField(value=[])).value or []
    du.m_q.grouping_variables = (
        consensus.m_q.get("grouping_variables", ConsensusField(value=[])).value or []
    )
    du.m_q.output_cardinality = (
        consensus.m_q.get("output_cardinality", ConsensusField()).value or ""
    )

    # M_S
    du.m_s.sources = (
        consensus.m_s.get("sources", ConsensusField(value=[])).value or []
    )
    raw_jks = consensus.m_s.get("join_keys", ConsensusField(value=[])).value or []
    for jk_raw in raw_jks:
        if isinstance(jk_raw, dict):
            du.m_s.join_keys.append(JoinKey(
                left_source=jk_raw.get("left_source", ""),
                right_source=jk_raw.get("right_source", ""),
                left_column=jk_raw.get("left_column", ""),
                right_column=jk_raw.get("right_column", ""),
                join_type=jk_raw.get("join_type", "inner"),
            ))

    # M_U
    du.m_u.unit_annotations = (
        consensus.m_u.get("unit_annotations", ConsensusField(value={})).value or {}
    )

    # M_F
    du.m_f.has_header = (
        consensus.m_f.get("has_header", ConsensusField(value=True)).value
    )
    du.m_f.delimiter = (
        consensus.m_f.get("delimiter", ConsensusField(value=",")).value or ","
    )
    du.m_f.encoding = (
        consensus.m_f.get("encoding", ConsensusField(value="utf-8")).value or "utf-8"
    )
    du.m_f.file_format = (
        consensus.m_f.get("file_format", ConsensusField(value="csv")).value or "csv"
    )

    # M_C
    du.m_c.constraints = (
        consensus.m_c.get("constraints", ConsensusField(value=[])).value or []
    )
    du.m_c.derived_filters = (
        consensus.m_c.get("derived_filters", ConsensusField(value=[])).value or []
    )

    return du


def _build_empty_du() -> DUOutput:
    """Build an empty DU output for the no-DU condition."""
    return DUOutput(system_name="no-du")


# ── Load Pre-Computed DU Outputs ──────────────────────────────────

def load_system_du_outputs(system_name: str) -> Dict[str, DUOutput]:
    """Load saved DU outputs for a system from the benchmark results."""
    base = Path("evaluation/results/du_benchmark/du_outputs") / system_name
    rerun = Path("evaluation/results/du_benchmark/rerun") / system_name

    outputs = {}
    if not base.exists():
        _logger.warning("No DU outputs found for system %s at %s", system_name, base)
        return outputs

    for f in sorted(base.glob("*.json")):
        task_id = f.stem
        # Use rerun version if available (e.g., profile-llm-hybrid)
        src = rerun / f.name if (rerun / f.name).exists() else f
        try:
            data = json.loads(src.read_text())
            outputs[task_id] = DUOutput.from_dict(data)
        except Exception as e:
            _logger.warning("Failed to load DU for %s/%s: %s", system_name, task_id, e)
    return outputs


def load_consensus_outputs() -> Dict[str, ConsensusDU]:
    """Load 3-model consensus from the benchmark results."""
    consensus_file = Path(
        "evaluation/results/du_benchmark/consensus_3model/consensus_all.json"
    )
    if not consensus_file.exists():
        raise FileNotFoundError(f"Consensus file not found: {consensus_file}")

    raw = json.loads(consensus_file.read_text())
    return {tid: ConsensusDU.from_dict(cdata) for tid, cdata in raw.items()}


# ── Pipeline Runner with DU Injection ─────────────────────────────

def _answer_type_to_output_type(answer_type: str) -> str:
    """Map KramaBench answer_type to pipeline output_type."""
    if answer_type.startswith("numeric"):
        return "scalar"
    elif answer_type.startswith("string"):
        return "scalar"
    elif answer_type.startswith("list"):
        return "list"
    return "dataframe"


def _find_output_df(case_dir: str):
    """Try to load the output DataFrame from a case directory."""
    if not case_dir or not os.path.isdir(case_dir):
        return None
    for fname in os.listdir(case_dir):
        if fname.endswith("_output.csv") or fname == "output.csv":
            try:
                return pd.read_csv(os.path.join(case_dir, fname))
            except Exception:
                continue
    return None


async def run_task_with_injected_du(
    adapter: KramaBenchAdapter,
    task: BenchmarkTask,
    du_output: DUOutput,
    provider: str,
    model: str,
    max_refinements: int = 3,
) -> Dict[str, Any]:
    """Run a single KramaBench task with injected DU, bypassing ADP-MA's native DU."""
    try:
        from meta_agents_adk.runner_llm import ADKLLMPipelineRunner
        from meta_agents_adk.execution.pipeline_monitor import MonitorConfig
    except ImportError:
        raise ImportError(
            "Ablation requires the ADP-MA pipeline. "
            "Install it with: pip install adp-ma  (or clone the main repo)"
        )

    result = {
        "task_id": task.task_id,
        "category": task.category,
        "difficulty": task.difficulty,
        "passed": False,
        "status": "CRASHED",
        "duration_seconds": 0,
        "total_llm_calls": 0,
    }

    # Load data files via adapter
    try:
        data_dict = adapter.get_task_data(task)
        if not data_dict:
            result["error"] = f"No loadable data files for task {task.task_id}"
            return result

        # Primary data source: first available file
        primary_name = next(iter(data_dict))
        primary_df = data_dict[primary_name]
        primary_path = task.data_files[0] if task.data_files else ""

        # Additional data sources (everything beyond the first)
        additional_data = {}
        additional_paths = {}
        for i, (name, df) in enumerate(list(data_dict.items())[1:], start=1):
            stem = Path(name).stem
            additional_data[stem] = df
            if i < len(task.data_files):
                additional_paths[stem] = task.data_files[i]
    except Exception as e:
        _logger.error("Failed to load data for %s: %s", task.task_id, e)
        result["error"] = str(e)
        return result

    # Convert M* to ADP-MA format
    injected_du = _du_output_to_adp_format(
        du_output, primary_df, additional_data if additional_data else None
    )

    # Create runner (same config as run_kramabench.py)
    monitor_config = MonitorConfig(budget_usd=20.0, batch_mode=True)
    runner = ADKLLMPipelineRunner(
        planning_provider=provider,
        planning_model=model,
        coding_provider=provider,
        coding_model=model,
        max_refinements=max_refinements,
        start_sample_level="XS",
        monitor_config=monitor_config,
    )

    # Monkey-patch _run_data_understanding to return our injected DU
    async def _injected_du(df: pd.DataFrame, data_source: str = ""):
        # Still need to set up additional data sources on the runner
        if additional_data:
            runner._additional_data_sources = additional_data
            if additional_paths:
                runner._additional_data_source_paths = additional_paths
        return injected_du, df

    runner._run_data_understanding = _injected_du

    # Build expected_output and task_parameters (same as run_kramabench.py)
    expected = task.expected_output
    answer_type = expected.get("answer_type", "string_exact")
    pipeline_expected = {
        "answer_type": answer_type,
        "output_type": _answer_type_to_output_type(answer_type),
    }
    if answer_type.startswith("list"):
        pipeline_expected["columns"] = ["answer"]
        pipeline_expected["row_description"] = "One row per list item with answer column"
    elif answer_type.startswith("numeric") or answer_type.startswith("string"):
        pipeline_expected["columns"] = ["answer"]
        pipeline_expected["row_description"] = "Single row with answer column"

    task_params = {
        "task_category": task.category,
        "answer_type": answer_type,
    }

    start = datetime.now()
    try:
        run_result = await runner.run(
            user_goal=task.goal,
            data_source=primary_path,
            input_df=primary_df,
            task_description=task.description,
            expected_output=pipeline_expected,
            task_parameters=task_params,
            additional_data_sources=additional_data if additional_data else None,
            additional_data_source_paths=additional_paths if additional_paths else None,
        )

        duration = (datetime.now() - start).total_seconds()
        result["duration_seconds"] = round(duration, 2)
        result["success"] = run_result.get("success", False)
        result["total_llm_calls"] = run_result.get("total_llm_calls", 0)
        result["case_id"] = run_result.get("case_id", "")

        # Evaluate output against KramaBench ground truth
        if result["success"]:
            output_df = run_result.get("output_df")
            if output_df is None:
                output_df = _find_output_df(run_result.get("case_dir", ""))

            if output_df is not None:
                result["output_shape"] = list(output_df.shape)
                evaluation = adapter.evaluate_output(
                    task, output_df,
                    input_columns=list(primary_df.columns),
                )
                result["passed"] = evaluation.get("passed", False)
                result["status"] = "PASSED" if result["passed"] else "RUN_OK"
                result["evaluation"] = {
                    k: v for k, v in evaluation.items()
                    if k not in ("kramabench_metrics",)  # keep it small
                }
            else:
                result["status"] = "RUN_OK"
        else:
            result["status"] = "FAILED"

    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        result["duration_seconds"] = round(duration, 2)
        result["error"] = str(e)[:500]
        _logger.error("Task %s crashed: %s", task.task_id, str(e)[:200])
        traceback.print_exc()

    return result


# ── Main Ablation Orchestrator ────────────────────────────────────

async def run_ablation(
    conditions: List[str],
    provider: str,
    model: str,
    max_refinements: int = 3,
    task_filter: Optional[str] = None,
    resume: bool = True,
):
    """Run the full ablation experiment."""
    output_dir = Path("evaluation/results/du_benchmark/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks via adapter
    adapter = build_adapter()
    all_task_ids = adapter.list_tasks()

    # Filter to KramaBench tasks only (DU benchmark covers KB + AB + DAC)
    if task_filter and task_filter != "all":
        filter_ids = set(task_filter.split(","))
        all_task_ids = [tid for tid in all_task_ids if tid in filter_ids]

    _logger.info("Loaded %d KramaBench tasks", len(all_task_ids))

    # Load DU outputs for each condition
    du_sources: Dict[str, Dict[str, DUOutput]] = {}

    if "consensus" in conditions:
        consensus_map = load_consensus_outputs()
        du_sources["consensus"] = {
            tid: _consensus_to_du_output(c) for tid, c in consensus_map.items()
        }
        _logger.info("Loaded consensus DU for %d tasks", len(du_sources["consensus"]))

    if "llm-only" in conditions:
        du_sources["llm-only"] = load_system_du_outputs("llm-only")
        _logger.info("Loaded llm-only DU for %d tasks", len(du_sources["llm-only"]))

    if "profiler" in conditions:
        du_sources["profiler"] = load_system_du_outputs("ydata-profiling")
        _logger.info("Loaded profiler DU for %d tasks", len(du_sources["profiler"]))

    if "no-du" in conditions:
        du_sources["no-du"] = {tid: _build_empty_du() for tid in all_task_ids}

    if "scrambled" in conditions:
        # Use llm-only outputs but shuffled (each task gets another task's DU)
        llm_outputs = load_system_du_outputs("llm-only")
        task_ids_with_du = sorted(llm_outputs.keys())
        rng = random.Random(42)
        shuffled_ids = task_ids_with_du.copy()
        rng.shuffle(shuffled_ids)
        du_sources["scrambled"] = {}
        for orig_id, scrambled_id in zip(task_ids_with_du, shuffled_ids):
            if orig_id != scrambled_id:
                du_sources["scrambled"][orig_id] = llm_outputs[scrambled_id]
            elif len(task_ids_with_du) > 1:
                alt = shuffled_ids[(shuffled_ids.index(scrambled_id) + 1) % len(shuffled_ids)]
                du_sources["scrambled"][orig_id] = llm_outputs[alt]
        _logger.info("Loaded scrambled DU for %d tasks", len(du_sources["scrambled"]))

    # Run each condition
    all_results: Dict[str, Dict[str, Any]] = {}

    for condition in conditions:
        if condition not in du_sources:
            _logger.warning("No DU source for condition %s, skipping", condition)
            continue

        results_file = output_dir / f"ablation_{condition}.json"

        # Resume logic
        existing_results = {}
        if resume and results_file.exists():
            try:
                existing_data = json.loads(results_file.read_text())
                existing_results = {
                    r["task_id"]: r
                    for r in existing_data.get("tasks", [])
                    # Skip crashed/API-limit failures (< 20s, not passed)
                    if r.get("passed") or r.get("duration_seconds", 0) >= 20
                }
                _logger.info(
                    "Resuming %s: %d tasks already done", condition, len(existing_results)
                )
            except Exception:
                pass

        condition_results = []
        du_map = du_sources[condition]

        for i, task_id in enumerate(all_task_ids):
            # Skip if already done
            if task_id in existing_results:
                condition_results.append(existing_results[task_id])
                continue

            # Skip if no DU available for this task
            if task_id not in du_map:
                _logger.warning(
                    "[%s] No DU for task %s, skipping", condition, task_id
                )
                continue

            du_output = du_map[task_id]

            # Load task via adapter
            task = adapter.load_task(task_id)

            _logger.info(
                "[%s] Running task %d/%d: %s",
                condition, i + 1, len(all_task_ids), task_id,
            )

            result = await run_task_with_injected_du(
                adapter, task, du_output, provider, model, max_refinements
            )
            result["condition"] = condition
            condition_results.append(result)

            # Incremental save
            _save_condition_results(results_file, condition, condition_results)

            # Log progress
            passed = sum(1 for r in condition_results if r.get("passed"))
            total = len(condition_results)
            _logger.info(
                "[%s] Progress: %d/%d done, %d/%d passed (%.1f%%)",
                condition, total, len(all_task_ids), passed, total,
                100 * passed / max(total, 1),
            )

        # Final save
        _save_condition_results(results_file, condition, condition_results)
        all_results[condition] = _summarize_condition(condition, condition_results)

    # Save combined summary
    summary_file = output_dir / "ablation_summary.json"
    summary_file.write_text(json.dumps(all_results, indent=2, default=str))
    _logger.info("Ablation complete. Summary saved to %s", summary_file)

    # Print summary table
    _print_summary(all_results)

    return all_results


def _save_condition_results(
    path: Path, condition: str, results: List[Dict[str, Any]]
):
    """Save condition results with incremental update."""
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    data = {
        "condition": condition,
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / max(total, 1), 4),
        "tasks": results,
    }
    path.write_text(json.dumps(data, indent=2, default=str))


def _summarize_condition(
    condition: str, results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Summarize results for one ablation condition."""
    passed = sum(1 for r in results if r.get("passed"))
    crashed = sum(1 for r in results if r.get("status") == "CRASHED")
    total = len(results)
    durations = [r.get("duration_seconds", 0) for r in results]
    llm_calls = [r.get("total_llm_calls", 0) for r in results]

    # Per-difficulty breakdown
    easy = [r for r in results if r.get("difficulty") == "easy"]
    hard = [r for r in results if r.get("difficulty") == "hard"]

    return {
        "condition": condition,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / max(total, 1), 4),
        "crashed": crashed,
        "crash_rate": round(crashed / max(total, 1), 4),
        "avg_duration_s": round(sum(durations) / max(total, 1), 1),
        "avg_llm_calls": round(sum(llm_calls) / max(total, 1), 1),
        "easy_passed": sum(1 for r in easy if r.get("passed")),
        "easy_total": len(easy),
        "hard_passed": sum(1 for r in hard if r.get("passed")),
        "hard_total": len(hard),
    }


def _print_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print a summary table of ablation results."""
    print("\n" + "=" * 85)
    print("DU ABLATION RESULTS")
    print("=" * 85)
    print(
        f"{'Condition':<15} {'Passed':>8} {'Total':>6} {'Rate':>8} "
        f"{'Crash':>6} {'Easy':>12} {'Hard':>12} {'Avg Time':>10}"
    )
    print("-" * 85)

    # Sort by pass rate descending
    sorted_conditions = sorted(
        all_results.items(), key=lambda x: x[1].get("pass_rate", 0), reverse=True
    )

    for condition, data in sorted_conditions:
        easy_str = f"{data['easy_passed']}/{data['easy_total']}"
        hard_str = f"{data['hard_passed']}/{data['hard_total']}"
        print(
            f"{condition:<15} {data['passed']:>8} {data['total']:>6} "
            f"{data['pass_rate']:>7.1%} {data['crashed']:>6} "
            f"{easy_str:>12} {hard_str:>12} {data['avg_duration_s']:>9.1f}s"
        )
    print("=" * 85)


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DU Ablation: inject different DU quality into ADP-MA pipeline"
    )
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated conditions or 'all'",
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        help="LLM provider for code generation",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="LLM model for code generation",
    )
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=3,
        help="Max code refinement iterations",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task IDs or 'all'",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous results",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    conditions = (
        ABLATION_CONDITIONS
        if args.conditions == "all"
        else args.conditions.split(",")
    )

    asyncio.run(
        run_ablation(
            conditions=conditions,
            provider=args.provider,
            model=args.model,
            max_refinements=args.max_refinements,
            task_filter=args.tasks if args.tasks != "all" else None,
            resume=not args.no_resume,
        )
    )


if __name__ == "__main__":
    main()
