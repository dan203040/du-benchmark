# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
#!/usr/bin/env python3
"""
Run DIN-SQL extractor on AgentBench tasks and score against consensus.

Usage:
    DEEPSEEK_API_KEY="sk-xxx" python3 evaluation/du_benchmark/run_dinsql.py [--resume]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


from du_benchmark.extractors.text_to_sql import DINSQLExtractor
from du_benchmark.schema import (
    BenchmarkSource, ConsensusDU, DUBenchmarkTask, DUOutput, TaskDUScore,
)
from du_benchmark.metrics import aggregate_scores, bootstrap_ci, score_task
from du_benchmark.consensus import load_du_tasks

_logger = logging.getLogger(__name__)

DU_OUTPUTS_DIR = Path("evaluation/results/du_benchmark/du_outputs/din-sql")
CONSENSUS_FILE = Path("evaluation/results/du_benchmark/consensus_decontaminated/consensus_all.json")
RESULTS_DIR = Path("evaluation/results/du_benchmark/analysis_decontaminated")


async def run_dinsql(resume: bool = False):
    """Run DIN-SQL extraction on all tasks and score."""
    DU_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load consensus
    _logger.info("Loading consensus...")
    raw_consensus = json.loads(CONSENSUS_FILE.read_text())
    consensus_map = {tid: ConsensusDU.from_dict(d) for tid, d in raw_consensus.items()}

    # Load tasks
    _logger.info("Loading tasks...")
    tasks = load_du_tasks(["kramabench", "agentbench", "dacode"])
    _logger.info("Loaded %d tasks", len(tasks))

    # Initialize extractor
    extractor = DINSQLExtractor()

    # Check for existing outputs (resume)
    existing = set()
    if resume:
        existing = {f.stem for f in DU_OUTPUTS_DIR.glob("*.json")}
        _logger.info("Resuming: %d tasks already completed", len(existing))

    # Run extraction
    du_outputs: Dict[str, DUOutput] = {}
    total = len(tasks)
    skipped_non_agent = 0

    for i, task in enumerate(tasks):
        if task.task_id in existing:
            # Load existing output
            try:
                data = json.loads((DU_OUTPUTS_DIR / f"{task.task_id}.json").read_text())
                du_outputs[task.task_id] = DUOutput.from_dict(data)
            except Exception:
                pass
            continue

        _logger.info("[%d/%d] Processing %s (%s)", i + 1, total, task.task_id, task.benchmark.value)

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
            du_output = await extractor.extract_du(task, dataframes)
        except Exception as e:
            _logger.error("Failed for %s: %s", task.task_id, e)
            du_output = DUOutput(system_name="din-sql")

        if task.benchmark != BenchmarkSource.AGENTBENCH:
            skipped_non_agent += 1

        du_outputs[task.task_id] = du_output

        # Save per-task output
        out_file = DU_OUTPUTS_DIR / f"{task.task_id}.json"
        out_file.write_text(json.dumps(du_output.to_dict(), indent=2, default=str))

    _logger.info("Extraction complete: %d outputs, %d skipped (non-AgentBench)", len(du_outputs), skipped_non_agent)

    # Score against consensus
    _logger.info("Scoring against consensus...")
    task_scores: List[TaskDUScore] = []

    for task in tasks:
        if task.task_id not in du_outputs:
            continue
        if task.task_id not in consensus_map:
            continue

        try:
            ts = score_task(
                task=task,
                du_output=du_outputs[task.task_id],
                consensus=consensus_map[task.task_id],
                dataframes={},
                file_paths=None,
            )
            task_scores.append(ts)
        except Exception as e:
            _logger.warning("Score failed for %s: %s", task.task_id, e)

    # Aggregate
    agg = aggregate_scores(task_scores)
    quality_scores = [ts.du_quality_score for ts in task_scores]
    mean, lower, upper = bootstrap_ci(quality_scores)

    result = {
        "system_name": "din-sql",
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

    # Save
    out_file = RESULTS_DIR / "din_sql_scores.json"
    out_file.write_text(json.dumps(result, indent=2, default=str))

    # Print summary
    dims = agg.get("per_dimension", {})
    print(f"\n{'=' * 70}")
    print(f"DIN-SQL DU Benchmark Results")
    print(f"{'=' * 70}")
    print(f"Tasks: {agg.get('n_tasks', 0)}")
    print(f"D1={dims.get('D1',0):.3f} D2={dims.get('D2',0):.3f} D3={dims.get('D3',0):.3f} D4={dims.get('D4',0):.3f} D5={dims.get('D5',0):.3f}")
    print(f"Quality: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
    print(f"{'=' * 70}")
    print(f"\nResults saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Run DIN-SQL DU extraction")
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    asyncio.run(run_dinsql(resume=args.resume))


if __name__ == "__main__":
    main()
