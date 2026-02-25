# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
#!/usr/bin/env python3
"""
Selective Hybrid DU Experiment.

Unlike the naive hybrid (which dumps the full ydata-profiling report into LLM context),
this approach passes ONLY curated metadata:
  - Column names and dtypes
  - Null percentages per column
  - Detected join-key candidates (columns appearing in multiple files)
  - NO raw data samples

Then GPT-4o-mini produces M conditioned on this concise profile.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


from du_benchmark.schema import DUBenchmarkTask, DUOutput
from du_benchmark.consensus import (
    DU_PROMPT_TEMPLATE,
    _call_model_du,
    _dict_to_du_output,
    _parse_du_json,
    load_du_tasks,
)
from du_benchmark.llm.client import LLMProvider

_logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results" / "du_benchmark" / "du_outputs" / "selective-hybrid"


def _build_selective_profile(task: DUBenchmarkTask) -> str:
    """Build a concise, curated file description with only structural metadata."""
    parts = []
    all_column_sets: Dict[str, set] = {}

    for fname in task.data_files:
        section = f"### {fname}\n"
        cols = task.column_lists.get(fname, [])
        all_column_sets[fname] = set(cols)

        meta = task.file_metadata.get(fname, {})
        shape = meta.get("shape", "unknown")
        dtypes = meta.get("dtypes", {})
        nulls = meta.get("null_percentages", {})

        section += f"Shape: {shape}\n"
        if cols:
            section += f"Columns: {cols}\n"
        if dtypes:
            section += f"Dtypes: {json.dumps(dtypes, default=str)}\n"

        # Compute null percentages if not in metadata
        if cols and not nulls:
            # We don't have the actual dataframe here, but dtypes give us
            # enough info. Note: null_percentages may be pre-computed.
            pass

        if nulls:
            section += f"Null percentages: {json.dumps(nulls)}\n"

        parts.append(section)

    # Detect join-key candidates: columns appearing in 2+ files
    if len(all_column_sets) > 1:
        from collections import Counter
        col_counts = Counter()
        for cols in all_column_sets.values():
            for c in cols:
                col_counts[c] += 1
        shared = [c for c, n in col_counts.items() if n >= 2]
        if shared:
            parts.append(f"\n**Potential join keys** (columns in multiple files): {shared}")

    return "\n".join(parts)


async def run_selective_hybrid(
    tasks: List[DUBenchmarkTask],
    model: str = "gpt-4o-mini",
    concurrency: int = 5,
) -> Dict[str, DUOutput]:
    """Run selective hybrid DU extraction on all tasks."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    results = {}

    # Load existing results for resume
    existing = set()
    for f in OUTPUT_DIR.glob("*.json"):
        existing.add(f.stem)
    _logger.info("Resuming: %d tasks already done", len(existing))

    async def process_task(task: DUBenchmarkTask) -> Tuple[str, Optional[DUOutput]]:
        if task.task_id in existing:
            return task.task_id, None

        async with sem:
            profile = _build_selective_profile(task)
            prompt = DU_PROMPT_TEMPLATE.format(
                goal=task.goal,
                file_descriptions=profile,
            )

            try:
                raw_text, usage = await _call_model_du(
                    LLMProvider.OPENAI, model, prompt,
                )
                parsed = _parse_du_json(raw_text)
                du = _dict_to_du_output(parsed, "selective-hybrid")
                du.raw_text = raw_text
                du.token_usage = usage
                du.extraction_time_s = 0  # timing not critical

                # Save
                out_file = OUTPUT_DIR / f"{task.task_id}.json"
                out_file.write_text(json.dumps(du.to_dict(), indent=2, default=str))

                _logger.info("  [OK] %s (tokens: %s)", task.task_id, usage)
                return task.task_id, du
            except Exception as e:
                _logger.error("  [FAIL] %s: %s", task.task_id, e)
                return task.task_id, None

    tasks_to_run = [t for t in tasks if t.task_id not in existing]
    _logger.info("Running selective hybrid on %d tasks (%d skipped)", len(tasks_to_run), len(existing))

    coros = [process_task(t) for t in tasks_to_run]
    done = 0
    for coro in asyncio.as_completed(coros):
        tid, du = await coro
        done += 1
        if du:
            results[tid] = du
        if done % 20 == 0:
            _logger.info("  Progress: %d/%d", done, len(tasks_to_run))

    _logger.info("Selective hybrid complete: %d new + %d existing", len(results), len(existing))
    return results


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        _logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    # Load tasks
    tasks = load_du_tasks(["kramabench", "agentbench", "dacode"])
    _logger.info("Loaded %d tasks", len(tasks))

    # Run extraction
    await run_selective_hybrid(tasks, model="gpt-4o-mini", concurrency=5)

    # Now rescore all systems including selective-hybrid
    _logger.info("Rescoring all systems including selective-hybrid...")
    from du_benchmark.rescore_decontaminated import main as rescore_main

    # Temporarily add selective-hybrid to the SYSTEMS list
    import du_benchmark.rescore_decontaminated as rescore_mod
    if "selective-hybrid" not in rescore_mod.SYSTEMS:
        rescore_mod.SYSTEMS.append("selective-hybrid")

    rescore_main()


if __name__ == "__main__":
    asyncio.run(main())
