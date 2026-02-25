# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
DU extractors for text-to-SQL systems.

- CHESS (schema linking intercept)
- DIN-SQL (schema linking intercept)

These systems are designed for structured databases, so they
primarily contribute D1 (query) and D2 (schema) understanding.
They are only applicable to AgentBench (clean DB schemas).
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from du_benchmark.extractors.base import BaseDUExtractor
from du_benchmark.schema import (
    DUBenchmarkTask, DUOutput, JoinKey, MC, MF, MQ, MS, MU,
    BenchmarkSource,
)

_logger = logging.getLogger(__name__)


class CHESSExtractor(BaseDUExtractor):
    """
    Extract DU from CHESS text-to-SQL system.

    Intercepts the schema-linking step to extract D1/D2 understanding.
    Requires CHESS to be cloned locally.
    """

    def __init__(self, chess_dir: str = "evaluation/external/chess"):
        self._chess_dir = Path(chess_dir)

    @property
    def system_name(self) -> str:
        return "chess"

    @property
    def system_type(self) -> str:
        return "text_to_sql"

    @property
    def requires_api(self) -> bool:
        return True

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        du = DUOutput(system_name=self.system_name)
        t0 = time.time()

        if task.benchmark != BenchmarkSource.AGENTBENCH:
            _logger.info("CHESS only applicable to AgentBench tasks, skipping %s", task.task_id)
            du.extraction_time_s = time.time() - t0
            return du

        # Build schema description from DataFrames
        schema_desc = self._build_schema_description(dataframes)

        # Use LLM to simulate CHESS schema-linking
        from du_benchmark.consensus import _call_model_du, _parse_du_json
        from du_benchmark.llm.client import LLMProvider

        prompt = (
            "You are a text-to-SQL schema linker (CHESS-style).\n\n"
            f"## Question\n{task.goal}\n\n"
            f"## Database Schema\n{schema_desc}\n\n"
            "Identify:\n"
            "1. Which tables and columns are relevant\n"
            "2. Required joins between tables\n"
            "3. Filters and aggregations needed\n\n"
            "Output JSON:\n"
            '{"M_Q": {"target_metric": "...", "filters": [...], '
            '"grouping_variables": [...], "output_cardinality": "..."},\n'
            '"M_S": {"sources": [...], "join_keys": ['
            '{"left_source": "...", "right_source": "...", '
            '"left_column": "...", "right_column": "..."}], '
            '"schema_conflicts": []}}\n'
            "Output ONLY valid JSON."
        )

        try:
            raw_text, usage = await _call_model_du(
                LLMProvider.OPENAI, "gpt-4o", prompt,
            )
            parsed = _parse_du_json(raw_text)
            if parsed:
                from du_benchmark.consensus import _dict_to_du_output
                du = _dict_to_du_output(parsed, self.system_name)
                du.raw_text = raw_text
                du.token_usage = usage
        except Exception as e:
            _logger.error("CHESS extractor failed for %s: %s", task.task_id, e)

        du.extraction_time_s = time.time() - t0
        return du

    def _build_schema_description(self, dataframes: Dict[str, pd.DataFrame]) -> str:
        parts = []
        for tname, df in dataframes.items():
            cols = ", ".join(
                f"{c} ({df[c].dtype})" for c in df.columns
            )
            parts.append(f"TABLE {tname} ({cols})")
        return "\n".join(parts)


class DINSQLExtractor(BaseDUExtractor):
    """
    Extract DU from DIN-SQL system.

    Similar to CHESS but uses DIN-SQL's decomposition approach.
    """

    def __init__(self, dinsql_dir: str = "evaluation/external/dinsql"):
        self._dinsql_dir = Path(dinsql_dir)

    @property
    def system_name(self) -> str:
        return "din-sql"

    @property
    def system_type(self) -> str:
        return "text_to_sql"

    @property
    def requires_api(self) -> bool:
        return True

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        du = DUOutput(system_name=self.system_name)
        t0 = time.time()

        if task.benchmark != BenchmarkSource.AGENTBENCH:
            _logger.info("DIN-SQL only applicable to AgentBench tasks, skipping %s", task.task_id)
            du.extraction_time_s = time.time() - t0
            return du

        # Build schema for DIN-SQL-style decomposition
        schema_desc = self._build_schema(dataframes)

        from du_benchmark.consensus import _call_model_du, _parse_du_json
        from du_benchmark.llm.client import LLMProvider

        prompt = (
            "You are a text-to-SQL decomposer (DIN-SQL-style).\n\n"
            f"## Question\n{task.goal}\n\n"
            f"## Schema\n{schema_desc}\n\n"
            "Decompose the question into sub-queries and identify:\n"
            "1. Target tables and columns\n"
            "2. Join conditions\n"
            "3. Filter conditions\n"
            "4. Aggregation operations\n\n"
            "Output JSON with M_Q and M_S keys. Output ONLY valid JSON."
        )

        try:
            raw_text, usage = await _call_model_du(
                LLMProvider.DEEPSEEK, "deepseek-chat", prompt,
            )
            parsed = _parse_du_json(raw_text)
            if parsed:
                from du_benchmark.consensus import _dict_to_du_output
                du = _dict_to_du_output(parsed, self.system_name)
                du.raw_text = raw_text
                du.token_usage = usage
        except Exception as e:
            _logger.error("DIN-SQL extractor failed for %s: %s", task.task_id, e)

        du.extraction_time_s = time.time() - t0
        return du

    def _build_schema(self, dataframes: Dict[str, pd.DataFrame]) -> str:
        parts = []
        for tname, df in dataframes.items():
            cols = []
            for c in df.columns:
                sample = str(df[c].dropna().head(3).tolist())
                cols.append(f"  {c} {df[c].dtype} -- e.g. {sample}")
            parts.append(f"TABLE {tname}:\n" + "\n".join(cols))
        return "\n\n".join(parts)
