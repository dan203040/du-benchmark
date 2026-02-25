# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
ADP-MA DU extractor.

Extracts DU output directly from ADP-MA's _run_data_understanding() step
by running the pipeline up to (and including) the DU phase, then stopping.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from du_benchmark.extractors.base import BaseDUExtractor
from du_benchmark.schema import (
    DUBenchmarkTask, DUOutput, JoinKey, MC, MF, MQ, MS, MU,
)

_logger = logging.getLogger(__name__)


class ADPMAExtractor(BaseDUExtractor):
    """Extract DU from ADP-MA's data understanding step."""

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-chat",
    ):
        self._provider = provider
        self._model = model

    @property
    def system_name(self) -> str:
        return "adp-ma"

    @property
    def system_type(self) -> str:
        return "llm_agent"

    @property
    def requires_api(self) -> bool:
        return True

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        """Run ADP-MA's _run_data_understanding and convert to DUOutput."""
        try:
            from meta_agents_adk.runner_llm import ADKLLMPipelineRunner
        except ImportError:
            raise ImportError(
                "ADPMAExtractor requires the ADP-MA pipeline. "
                "Install it with: pip install adp-ma  (or clone the main repo)"
            )

        t0 = time.time()
        runner = ADKLLMPipelineRunner(
            planning_provider=self._provider,
            planning_model=self._model,
            coding_provider=self._provider,
            coding_model=self._model,
        )

        # Identify primary DataFrame (largest) and additional sources
        primary_name = max(dataframes, key=lambda k: len(dataframes[k]))
        primary_df = dataframes[primary_name]
        additional = {
            k: v for k, v in dataframes.items() if k != primary_name
        }
        data_source = ""
        if file_paths and primary_name in file_paths:
            data_source = file_paths[primary_name]

        try:
            du_result = await runner._run_data_understanding(
                df=primary_df,
                data_source=data_source,
                _additional_data_sources=additional,
            )
            # Also set the goal for context
            runner._user_goal = task.goal
        except Exception as e:
            _logger.error("ADP-MA DU failed for %s: %s", task.task_id, e)
            return DUOutput(
                system_name=self.system_name,
                extraction_time_s=time.time() - t0,
            )

        elapsed = time.time() - t0
        return self._convert_du_result(du_result, elapsed)

    def _convert_du_result(
        self, du_result: Dict[str, Any], elapsed: float,
    ) -> DUOutput:
        """Convert ADP-MA's DU dict to our DUOutput schema."""
        du = DUOutput(system_name=self.system_name, extraction_time_s=elapsed)
        du.raw_text = du_result.get("summary", "")

        # M_F: Format info
        fmt = du_result.get("format_info", {})
        du.m_f = MF(
            has_header=fmt.get("has_header", True),
            delimiter=fmt.get("delimiter", ","),
            encoding=fmt.get("encoding", "utf-8"),
            expected_columns=du_result.get("shape", (0, 0))[1] if du_result.get("shape") else 0,
            file_format=fmt.get("file_format", "csv"),
        )

        # M_S: Schema / join info
        columns = du_result.get("columns", [])
        join_keys_raw = du_result.get("join_keys", {})
        join_keys = []
        if isinstance(join_keys_raw, dict):
            for left_col, right_col in join_keys_raw.items():
                join_keys.append(JoinKey(
                    left_column=left_col,
                    right_column=right_col if isinstance(right_col, str) else str(right_col),
                ))
        du.m_s = MS(
            sources=list(du_result.get("additional_sources", {}).keys()) or [],
            join_keys=join_keys,
        )

        # M_Q: Extract from summary text (basic heuristic parsing)
        summary = du_result.get("summary", "")
        qa = du_result.get("query_analysis")
        if qa:
            du.m_q = MQ(
                target_metric="",
                filters=[],
                grouping_variables=[],
                output_cardinality=getattr(qa, "output_type", "table"),
                sub_questions=[],
            )
            du.m_q.filters = []

        # M_U: unit annotations from DU (D3)
        du.m_u = MU(
            unit_annotations=du_result.get("unit_annotations", {}),
            scale_issues=du_result.get("scale_issues", []),
            cross_source_conflicts=du_result.get("cross_source_conflicts", []),
        )

        # M_C: analytical constraints from DU (D5)
        du.m_c = MC(
            constraints=du_result.get("constraints", []),
            derived_filters=du_result.get("derived_filters", []),
            output_format_requirements=du_result.get("output_format_reqs", []),
        )

        return du


class ADPMARetroExtractor(BaseDUExtractor):
    """
    Extract DU from existing ADP-MA case logs (no new API calls).

    Reads case_log.json from previous benchmark runs.
    """

    def __init__(self, results_dir: str = "evaluation/results/kramabench/"):
        self._results_dir = results_dir

    @property
    def system_name(self) -> str:
        return "adp-ma-retro"

    @property
    def system_type(self) -> str:
        return "llm_agent"

    @property
    def requires_api(self) -> bool:
        return False

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        """Extract DU from existing case logs."""
        import json
        from pathlib import Path

        results_dir = Path(self._results_dir)
        # Look for case directory matching task_id
        case_dirs = sorted(results_dir.glob(f"case_*_{task.task_id.replace('-', '_')}*"))
        if not case_dirs:
            case_dirs = sorted(results_dir.glob(f"case_*"))

        for case_dir in reversed(case_dirs):
            log_file = case_dir / "case_log.json"
            if not log_file.exists():
                continue
            try:
                log_data = json.loads(log_file.read_text())
                activities = log_data.get("meta_agent_activities", [])
                for act in activities:
                    if act.get("activity_type") == "DATA_UNDERSTANDING":
                        output = act.get("output_summary", {})
                        du = DUOutput(system_name=self.system_name)
                        du.raw_text = json.dumps(output)
                        du.extraction_time_s = act.get("duration_ms", 0) / 1000.0
                        # Parse what we can from the output summary
                        if "columns" in output:
                            du.m_f.expected_columns = len(output["columns"])
                        return du
            except Exception as e:
                _logger.debug("Failed to parse case log %s: %s", log_file, e)

        return DUOutput(system_name=self.system_name)
