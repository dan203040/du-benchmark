# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Multi-LLM Consensus Generator for DU Silver Standard.

Runs 3 independent LLMs (DeepSeek-chat, Claude Sonnet 4.5, Gemini 2.0 Flash)
on identical DU prompts and resolves their outputs via majority vote.

Usage:
    python3 evaluation/du_benchmark/consensus.py \\
        --tasks all \\
        --models deepseek,claude-sonnet,gemini-flash \\
        --output evaluation/results/du_benchmark/consensus/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path

from du_benchmark.schema import (
    ConsensusDU, ConsensusField, DUBenchmarkTask, DUOutput,
    JoinKey, MC, MF, MQ, MS, MU, BenchmarkSource,
)
from du_benchmark.llm.client import (
    LLMProvider, create_llm_client, retry_on_transient,
)

_logger = logging.getLogger(__name__)

# ── Consensus Models ────────────────────────────────────────────────

CONSENSUS_MODELS = {
    "deepseek": (LLMProvider.DEEPSEEK, "deepseek-chat"),
    "gpt4o-mini": (LLMProvider.OPENAI, "gpt-4o-mini"),
    "gemini-flash": (LLMProvider.GOOGLE, "gemini-2.0-flash"),
    "claude-sonnet": (LLMProvider.ANTHROPIC, "claude-sonnet-4-5-20250929"),
}

# ── DU Extraction Prompt ────────────────────────────────────────────

DU_PROMPT_TEMPLATE = """\
You are a data understanding expert. Given data files and an analytical question,
produce a structured JSON describing the data understanding needed.

## Analytical Question
{goal}

## Data Files
{file_descriptions}

## Instructions

Produce a JSON object with exactly these five keys (no markdown, just raw JSON):

{{
  "M_Q": {{
    "target_metric": "<what the question asks to compute>",
    "filters": ["<filter1>", ...],
    "grouping_variables": ["<var1>", ...],
    "output_cardinality": "<scalar|list|table|plot>",
    "sub_questions": ["<sub-question1>", ...]
  }},
  "M_S": {{
    "sources": ["<file1>", ...],
    "join_keys": [
      {{"left_source": "<file>", "right_source": "<file>",
        "left_column": "<col>", "right_column": "<col>", "join_type": "inner"}}
    ],
    "schema_conflicts": ["<conflict1>", ...],
    "column_mappings": {{"<old_name>": "<new_name>"}}
  }},
  "M_U": {{
    "unit_annotations": {{"<column>": "<unit>"}},
    "scale_issues": ["<issue1>", ...],
    "cross_source_conflicts": ["<conflict1>", ...]
  }},
  "M_F": {{
    "has_header": true,
    "delimiter": ",",
    "encoding": "utf-8",
    "sentinel_values": ["NA", "N/A", ""],
    "expected_columns": 10,
    "file_format": "csv",
    "format_notes": ["<note1>", ...]
  }},
  "M_C": {{
    "constraints": ["<constraint1>", ...],
    "derived_filters": ["<filter1>", ...],
    "statistical_tests": ["<test1>", ...],
    "output_format_requirements": ["<req1>", ...]
  }}
}}

Be precise and specific. Reference actual column names from the data.
Output ONLY valid JSON, no explanation text."""


def _build_file_descriptions(task: DUBenchmarkTask) -> str:
    """Build file description block from task metadata."""
    parts = []
    for fname in task.data_files:
        section = f"### {fname}\n"
        cols = task.column_lists.get(fname, [])
        if cols:
            section += f"Columns: {cols}\n"
        meta = task.file_metadata.get(fname, {})
        if meta:
            shape = meta.get("shape", "unknown")
            dtypes = meta.get("dtypes", {})
            section += f"Shape: {shape}\n"
            if dtypes:
                section += f"Dtypes: {json.dumps(dtypes, default=str)}\n"
        sample = task.file_samples.get(fname, "")
        if sample:
            # Truncate large samples to avoid exceeding LLM context limits
            if len(sample) > 5000:
                sample = sample[:5000] + "\n... (truncated)"
            section += f"Sample (first 20 rows):\n```\n{sample}\n```\n"
        parts.append(section)
    return "\n".join(parts)


# ── Single-Model DU Call ────────────────────────────────────────────

@retry_on_transient(max_retries=3, initial_delay=2.0, backoff_factor=2.0)
async def _call_model_du(
    provider: LLMProvider,
    model: str,
    prompt: str,
) -> Tuple[str, Dict[str, int]]:
    """Call a single LLM for DU extraction. Returns (response_text, usage)."""
    client = create_llm_client(provider, model=model)
    response = await client.generate(
        prompt=prompt,
        system="You are a data understanding expert. Output only valid JSON.",
        seed=42,
    )
    if not response.success:
        raise RuntimeError(f"LLM call failed: {response.error}")
    return response.content, response.usage


def _parse_du_json(raw: str) -> Dict[str, Any]:
    """Parse DU JSON from LLM response, handling markdown code blocks."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    _logger.warning("Failed to parse DU JSON from response: %s...", text[:200])
    return {}


def _as_dict(val: Any) -> Dict[str, Any]:
    """Safely coerce a value to dict; return {} if it's a string or other non-dict."""
    return val if isinstance(val, dict) else {}


def _as_list(val: Any) -> list:
    """Safely coerce a value to list; return [] if it's a string or other non-list."""
    return val if isinstance(val, list) else []


def _dict_to_du_output(d: Dict[str, Any], system_name: str) -> DUOutput:
    """Convert a parsed JSON dict into a DUOutput dataclass."""
    du = DUOutput(system_name=system_name)

    mq = _as_dict(d.get("M_Q", {}))
    du.m_q = MQ(
        target_metric=str(mq.get("target_metric", "")),
        filters=_as_list(mq.get("filters", [])),
        grouping_variables=_as_list(mq.get("grouping_variables", [])),
        output_cardinality=str(mq.get("output_cardinality", "")),
        sub_questions=_as_list(mq.get("sub_questions", [])),
    )

    ms = _as_dict(d.get("M_S", {}))
    raw_jks = _as_list(ms.get("join_keys", []))
    du.m_s = MS(
        sources=_as_list(ms.get("sources", [])),
        join_keys=[
            JoinKey(
                left_source=jk.get("left_source", ""),
                right_source=jk.get("right_source", ""),
                left_column=jk.get("left_column", ""),
                right_column=jk.get("right_column", ""),
                join_type=jk.get("join_type", "inner"),
            )
            for jk in raw_jks if isinstance(jk, dict)
        ],
        schema_conflicts=_as_list(ms.get("schema_conflicts", [])),
        column_mappings=_as_dict(ms.get("column_mappings", {})),
    )

    mu = _as_dict(d.get("M_U", {}))
    du.m_u = MU(
        unit_annotations=_as_dict(mu.get("unit_annotations", {})),
        scale_issues=_as_list(mu.get("scale_issues", [])),
        cross_source_conflicts=_as_list(mu.get("cross_source_conflicts", [])),
    )

    mf = _as_dict(d.get("M_F", {}))
    du.m_f = MF(
        has_header=mf.get("has_header", True),
        delimiter=str(mf.get("delimiter", ",")),
        encoding=str(mf.get("encoding", "utf-8")),
        sentinel_values=_as_list(mf.get("sentinel_values", [])),
        expected_columns=mf.get("expected_columns", 0),
        expected_rows=mf.get("expected_rows", 0),
        file_format=str(mf.get("file_format", "csv")),
        format_notes=_as_list(mf.get("format_notes", [])),
    )

    mc = _as_dict(d.get("M_C", {}))
    du.m_c = MC(
        constraints=_as_list(mc.get("constraints", [])),
        derived_filters=_as_list(mc.get("derived_filters", [])),
        statistical_tests=_as_list(mc.get("statistical_tests", [])),
        output_format_requirements=_as_list(mc.get("output_format_requirements", [])),
    )

    return du


# ── Consensus Resolution ───────────────────────────────────────────

def _majority_vote_categorical(values: List[str]) -> ConsensusField:
    """Majority vote for categorical fields."""
    if not values:
        return ConsensusField(value="", confidence=0.0, votes=values)
    counter = Counter(v.lower().strip() for v in values if v)
    if not counter:
        return ConsensusField(value="", confidence=0.0, votes=values)
    winner, count = counter.most_common(1)[0]
    confidence = count / len(values)
    # Return the original-case version of the winner
    for v in values:
        if v.lower().strip() == winner:
            return ConsensusField(value=v, confidence=confidence, votes=values)
    return ConsensusField(value=winner, confidence=confidence, votes=values)


def _union_with_confidence(value_lists: List[List[str]]) -> ConsensusField:
    """Union for set-valued fields with confidence = agreement fraction."""
    if not value_lists:
        return ConsensusField(value=[], confidence=0.0, votes=value_lists)
    all_items: Dict[str, int] = {}
    n_models = len(value_lists)
    for vlist in value_lists:
        for item in vlist:
            if item is None:
                continue
            key = str(item).lower().strip()
            all_items[key] = all_items.get(key, 0) + 1
    # Include items mentioned by at least 1 model, confidence = agreement fraction
    result_items = []
    confidences = []
    for key, count in all_items.items():
        # Find original-case version
        for vlist in value_lists:
            for item in vlist:
                if item.lower().strip() == key:
                    result_items.append(item)
                    confidences.append(count / n_models)
                    break
            else:
                continue
            break
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return ConsensusField(
        value=result_items, confidence=avg_conf, votes=value_lists,
    )


def _majority_vote_bool(values: List[bool]) -> ConsensusField:
    """Majority vote for boolean fields."""
    true_count = sum(1 for v in values if v)
    winner = true_count > len(values) / 2
    confidence = max(true_count, len(values) - true_count) / len(values)
    return ConsensusField(value=winner, confidence=confidence, votes=values)


def _safe_float(val) -> float:
    """Safely convert a value to float, extracting first number from strings."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        import re
        m = re.search(r'[\d.]+', val)
        return float(m.group()) if m else 0.0
    return 0.0


def _median_numeric(values: List[float]) -> ConsensusField:
    """Median for numeric fields."""
    valid = [v for v in values if v is not None and v > 0]
    if not valid:
        return ConsensusField(value=0, confidence=0.0, votes=values)
    valid.sort()
    median = valid[len(valid) // 2]
    # Confidence: how close are all values to the median?
    if len(valid) == 1:
        confidence = 0.33
    else:
        spread = max(valid) - min(valid)
        confidence = 1.0 if spread == 0 else max(0.33, 1.0 - spread / (median + 1))
    return ConsensusField(value=median, confidence=confidence, votes=values)


def _consensus_dict(dicts: List[Dict[str, str]]) -> ConsensusField:
    """Consensus for dict-valued fields (e.g. unit_annotations)."""
    if not dicts:
        return ConsensusField(value={}, confidence=0.0, votes=dicts)
    all_keys: Dict[str, List[str]] = {}
    for d in dicts:
        for k, v in d.items():
            all_keys.setdefault(k.lower().strip(), []).append(v)
    result = {}
    confidences = []
    for key, vals in all_keys.items():
        cf = _majority_vote_categorical(vals)
        result[key] = cf.value
        confidences.append(len(vals) / len(dicts))
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return ConsensusField(value=result, confidence=avg_conf, votes=dicts)


def resolve_consensus(du_outputs: List[DUOutput]) -> ConsensusDU:
    """Resolve multiple DU outputs into a consensus with confidence scores."""
    consensus = ConsensusDU(
        models_used=[du.system_name for du in du_outputs],
    )

    # D1 (M_Q)
    consensus.m_q = {
        "target_metric": _majority_vote_categorical(
            [du.m_q.target_metric for du in du_outputs]
        ),
        "filters": _union_with_confidence(
            [du.m_q.filters for du in du_outputs]
        ),
        "grouping_variables": _union_with_confidence(
            [du.m_q.grouping_variables for du in du_outputs]
        ),
        "output_cardinality": _majority_vote_categorical(
            [du.m_q.output_cardinality for du in du_outputs]
        ),
        "sub_questions": _union_with_confidence(
            [du.m_q.sub_questions for du in du_outputs]
        ),
    }

    # D2 (M_S)
    consensus.m_s = {
        "sources": _union_with_confidence(
            [du.m_s.sources for du in du_outputs]
        ),
        "schema_conflicts": _union_with_confidence(
            [du.m_s.schema_conflicts for du in du_outputs]
        ),
    }

    # D3 (M_U)
    consensus.m_u = {
        "unit_annotations": _consensus_dict(
            [du.m_u.unit_annotations for du in du_outputs]
        ),
        "scale_issues": _union_with_confidence(
            [du.m_u.scale_issues for du in du_outputs]
        ),
        "cross_source_conflicts": _union_with_confidence(
            [du.m_u.cross_source_conflicts for du in du_outputs]
        ),
    }

    # D4 (M_F)
    consensus.m_f = {
        "has_header": _majority_vote_bool(
            [du.m_f.has_header for du in du_outputs]
        ),
        "delimiter": _majority_vote_categorical(
            [du.m_f.delimiter for du in du_outputs]
        ),
        "encoding": _majority_vote_categorical(
            [du.m_f.encoding for du in du_outputs]
        ),
        "sentinel_values": _union_with_confidence(
            [du.m_f.sentinel_values for du in du_outputs]
        ),
        "expected_columns": _median_numeric(
            [_safe_float(du.m_f.expected_columns) for du in du_outputs]
        ),
        "file_format": _majority_vote_categorical(
            [du.m_f.file_format for du in du_outputs]
        ),
    }

    # D5 (M_C)
    consensus.m_c = {
        "constraints": _union_with_confidence(
            [du.m_c.constraints for du in du_outputs]
        ),
        "derived_filters": _union_with_confidence(
            [du.m_c.derived_filters for du in du_outputs]
        ),
        "statistical_tests": _union_with_confidence(
            [du.m_c.statistical_tests for du in du_outputs]
        ),
        "output_format_requirements": _union_with_confidence(
            [du.m_c.output_format_requirements for du in du_outputs]
        ),
    }

    # Overall confidence = mean of all dimension field confidences
    all_confs = []
    for dim in (consensus.m_q, consensus.m_s, consensus.m_u,
                consensus.m_f, consensus.m_c):
        for cf in dim.values():
            all_confs.append(cf.confidence)
    consensus.overall_confidence = (
        sum(all_confs) / len(all_confs) if all_confs else 0.0
    )

    return consensus


# ── Task Loading ────────────────────────────────────────────────────

def load_du_tasks(
    benchmarks: List[str],
    project_root: Optional[Path] = None,
) -> List[DUBenchmarkTask]:
    """Load tasks from specified benchmarks, returning DUBenchmarkTask descriptors."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    tasks: List[DUBenchmarkTask] = []

    if "kramabench" in benchmarks:
        tasks.extend(_load_kramabench_tasks(project_root))
    if "agentbench" in benchmarks:
        tasks.extend(_load_agentbench_tasks(project_root))
    if "dacode" in benchmarks:
        tasks.extend(_load_dacode_tasks(project_root))

    return tasks


def _load_kramabench_tasks(project_root: Path) -> List[DUBenchmarkTask]:
    """Load KramaBench tasks as DUBenchmarkTask descriptors."""
    try:
        from du_benchmark.adapters.kramabench import KramaBenchAdapter
    except ImportError:
        _logger.warning("KramaBench adapter not available")
        return []

    benchmark_dir = project_root / "evaluation" / "external" / "kramabench"
    output_dir = project_root / "evaluation" / "results" / "du_benchmark"
    if not benchmark_dir.exists():
        _logger.warning("KramaBench directory not found: %s", benchmark_dir)
        return []

    adapter = KramaBenchAdapter(benchmark_dir, output_dir)
    task_ids = adapter.list_tasks()
    tasks = []

    for tid in task_ids:
        try:
            bt = adapter.load_task(tid)
            data_dict = adapter.get_task_data(bt)

            file_samples = {}
            column_lists = {}
            file_metadata = {}
            for fname, df in data_dict.items():
                column_lists[fname] = list(df.columns)
                file_metadata[fname] = {
                    "shape": list(df.shape),
                    "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
                }
                file_samples[fname] = df.head(20).to_csv(index=False)

            tasks.append(DUBenchmarkTask(
                task_id=tid,
                benchmark=BenchmarkSource.KRAMABENCH,
                goal=bt.goal,
                data_files=list(data_dict.keys()),
                category=bt.category,
                difficulty=bt.difficulty,
                primary_dimensions=["D1", "D2", "D3", "D4", "D5"],
                file_samples=file_samples,
                column_lists=column_lists,
                file_metadata=file_metadata,
            ))
        except Exception as e:
            _logger.warning("Failed to load KramaBench task %s: %s", tid, e)

    _logger.info("Loaded %d KramaBench tasks", len(tasks))
    return tasks


def _load_agentbench_tasks(project_root: Path) -> List[DUBenchmarkTask]:
    """Load AgentBench DBBench tasks."""
    try:
        from du_benchmark.adapters.agentbench import AgentBenchAdapter
    except ImportError:
        _logger.warning("AgentBench adapter not available")
        return []

    benchmark_dir = project_root / "evaluation" / "external" / "agentbench"
    output_dir = project_root / "evaluation" / "results" / "du_benchmark"
    if not benchmark_dir.exists():
        _logger.warning("AgentBench directory not found: %s", benchmark_dir)
        return []

    adapter = AgentBenchAdapter(benchmark_dir, output_dir)
    task_ids = adapter.list_tasks()
    tasks = []

    for tid in task_ids:
        try:
            bt = adapter.load_task(tid)
            data_dict = adapter.get_task_data(bt)

            file_samples = {}
            column_lists = {}
            file_metadata = {}
            for fname, df in data_dict.items():
                column_lists[fname] = list(df.columns)
                file_metadata[fname] = {
                    "shape": list(df.shape),
                    "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
                }
                file_samples[fname] = df.head(20).to_csv(index=False)

            tasks.append(DUBenchmarkTask(
                task_id=tid,
                benchmark=BenchmarkSource.AGENTBENCH,
                goal=bt.goal,
                data_files=list(data_dict.keys()),
                category=bt.category,
                difficulty=bt.difficulty,
                primary_dimensions=["D1", "D2"],
                file_samples=file_samples,
                column_lists=column_lists,
                file_metadata=file_metadata,
            ))
        except Exception as e:
            _logger.warning("Failed to load AgentBench task %s: %s", tid, e)

    _logger.info("Loaded %d AgentBench tasks", len(tasks))
    return tasks


def _load_dacode_tasks(project_root: Path) -> List[DUBenchmarkTask]:
    """Load DACode tasks."""
    try:
        from du_benchmark.adapters.da_code import DACodeAdapter
    except ImportError:
        _logger.warning("DACode adapter not available")
        return []

    benchmark_dir = project_root / "evaluation" / "external" / "da_code"
    output_dir = project_root / "evaluation" / "results" / "du_benchmark"
    if not benchmark_dir.exists():
        _logger.warning("DACode directory not found: %s", benchmark_dir)
        return []

    adapter = DACodeAdapter(benchmark_dir, output_dir)
    task_ids = adapter.list_tasks()
    tasks = []

    for tid in task_ids:
        try:
            bt = adapter.load_task(tid)
            data_dict = adapter.get_task_data(bt)

            file_samples = {}
            column_lists = {}
            file_metadata = {}
            for fname, df in data_dict.items():
                column_lists[fname] = list(df.columns)
                file_metadata[fname] = {
                    "shape": list(df.shape),
                    "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
                }
                file_samples[fname] = df.head(20).to_csv(index=False)

            tasks.append(DUBenchmarkTask(
                task_id=tid,
                benchmark=BenchmarkSource.DACODE,
                goal=bt.goal,
                data_files=list(data_dict.keys()),
                category=bt.category,
                difficulty=bt.difficulty,
                primary_dimensions=["D1", "D5"],
                file_samples=file_samples,
                column_lists=column_lists,
                file_metadata=file_metadata,
            ))
        except Exception as e:
            _logger.warning("Failed to load DACode task %s: %s", tid, e)

    _logger.info("Loaded %d DACode tasks", len(tasks))
    return tasks


# ── Main: Generate Consensus for All Tasks ──────────────────────────

async def generate_consensus_for_task(
    task: DUBenchmarkTask,
    model_keys: List[str],
) -> Tuple[ConsensusDU, List[DUOutput]]:
    """Run multiple LLMs on a single task and resolve consensus."""
    prompt = DU_PROMPT_TEMPLATE.format(
        goal=task.goal,
        file_descriptions=_build_file_descriptions(task),
    )

    du_outputs: List[DUOutput] = []

    for model_key in model_keys:
        provider, model = CONSENSUS_MODELS[model_key]
        t0 = time.time()
        try:
            raw_text, usage = await _call_model_du(provider, model, prompt)
            elapsed = time.time() - t0
            parsed = _parse_du_json(raw_text)
            du = _dict_to_du_output(parsed, system_name=model_key)
            du.raw_text = raw_text
            du.extraction_time_s = elapsed
            du.token_usage = usage
            du_outputs.append(du)
            _logger.info(
                "  [%s] %s: %.1fs, %d tokens",
                task.task_id, model_key, elapsed,
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            )
        except Exception as e:
            _logger.error("  [%s] %s failed: %s", task.task_id, model_key, e)
            du_outputs.append(DUOutput(system_name=model_key))

    consensus = resolve_consensus(du_outputs)
    return consensus, du_outputs


async def generate_all_consensus(
    tasks: List[DUBenchmarkTask],
    model_keys: List[str],
    output_dir: Path,
    max_concurrent: int = 5,
) -> Dict[str, ConsensusDU]:
    """Generate consensus for all tasks with concurrency control."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, ConsensusDU] = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    # Check for existing results (resume logic)
    consensus_file = output_dir / "consensus_all.json"
    existing: Dict[str, Any] = {}
    if consensus_file.exists():
        try:
            existing = json.loads(consensus_file.read_text())
            _logger.info("Resuming: %d tasks already completed", len(existing))
        except Exception:
            pass

    async def _process_task(task: DUBenchmarkTask):
        if task.task_id in existing:
            results[task.task_id] = ConsensusDU.from_dict(existing[task.task_id])
            return
        async with semaphore:
            _logger.info("Processing task %s (%s)", task.task_id, task.benchmark.value)
            try:
                consensus, du_outputs = await generate_consensus_for_task(task, model_keys)
            except Exception as e:
                _logger.error("Consensus FAILED for %s: %s — using empty consensus", task.task_id, e)
                consensus = ConsensusDU(models_used=model_keys, overall_confidence=0.0)
                du_outputs = []
            results[task.task_id] = consensus

            # Save per-task raw outputs
            task_dir = output_dir / "per_task" / task.task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            for du in du_outputs:
                outf = task_dir / f"{du.system_name}.json"
                outf.write_text(json.dumps(du.to_dict(), indent=2, default=str))
            (task_dir / "consensus.json").write_text(
                json.dumps(consensus.to_dict(), indent=2, default=str)
            )

            # Incremental save
            all_data = {**existing}
            for tid, c in results.items():
                all_data[tid] = c.to_dict()
            consensus_file.write_text(
                json.dumps(all_data, indent=2, default=str)
            )

    await asyncio.gather(*[_process_task(t) for t in tasks])

    _logger.info(
        "Consensus complete: %d tasks, avg confidence %.2f",
        len(results),
        sum(c.overall_confidence for c in results.values()) / max(len(results), 1),
    )
    return results


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate multi-LLM DU consensus")
    parser.add_argument(
        "--tasks", default="all",
        help="Comma-separated benchmark names: kramabench,agentbench,dacode or 'all'",
    )
    parser.add_argument(
        "--models", default="deepseek,claude-sonnet,gemini-flash",
        help="Comma-separated model keys",
    )
    parser.add_argument(
        "--output", default="evaluation/results/du_benchmark/consensus/",
        help="Output directory",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="Max concurrent LLM calls",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    benchmarks = (
        ["kramabench", "agentbench", "dacode"]
        if args.tasks == "all"
        else args.tasks.split(",")
    )
    model_keys = args.models.split(",")
    output_dir = Path(args.output)

    tasks = load_du_tasks(benchmarks)
    _logger.info("Loaded %d tasks from %s", len(tasks), benchmarks)

    asyncio.run(generate_all_consensus(tasks, model_keys, output_dir, args.max_concurrent))


if __name__ == "__main__":
    main()
