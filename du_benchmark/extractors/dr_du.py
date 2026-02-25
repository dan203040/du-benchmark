# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
DR-DU (Dimension-Routed Data Understanding) Extractor.

Routes each DU dimension to its strongest extraction method:
  - D2 (Schema/Joins): Deterministic probes (name+dtype matching)
  - D4 (Format):       DuckDB CSV sniffer + pandas inspection
  - D1, D3, D5:        Focused LLM call with verified D2/D4 as context

This eliminates redundant schema re-discovery by the LLM, yielding
higher D2 accuracy and ~40% fewer tokens.
"""

from __future__ import annotations

import json
import logging
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from du_benchmark.extractors.base import BaseDUExtractor
from du_benchmark.schema import (
    DUBenchmarkTask, DUOutput, JoinKey, MC, MF, MQ, MS, MU,
)

_logger = logging.getLogger(__name__)


# ── Focused LLM Prompt (D1 + D3 + D5 only) ────────────────────────

_DRDU_LLM_PROMPT = """\
You are a data understanding expert. The schema and format of the data files
have ALREADY been verified by deterministic probes. Your job is to produce
ONLY the semantic understanding that requires reasoning.

## Analytical Question
{goal}

## Verified Schema (ground truth — do NOT re-derive)
{schema_block}

## Data Samples
{file_descriptions}

## Instructions

Using the verified schema above, produce a JSON object with exactly THREE keys
(no markdown fences, just raw JSON):

{{
  "M_Q": {{
    "target_metric": "<what the question asks to compute>",
    "filters": ["<filter1>", ...],
    "grouping_variables": ["<var1>", ...],
    "output_cardinality": "<scalar|list|table|plot>",
    "sub_questions": ["<sub-question1>", ...]
  }},
  "M_U": {{
    "unit_annotations": {{"<column>": "<unit>"}},
    "scale_issues": ["<issue1>", ...],
    "cross_source_conflicts": ["<conflict1>", ...]
  }},
  "M_C": {{
    "constraints": ["<constraint1>", ...],
    "derived_filters": ["<filter1>", ...],
    "statistical_tests": ["<test1>", ...],
    "output_format_requirements": ["<req1>", ...]
  }}
}}

Be precise. Reference actual column names. Output ONLY valid JSON."""


class DRDUExtractor(BaseDUExtractor):
    """Dimension-Routed Data Understanding: deterministic D2/D4, LLM D1/D3/D5."""

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-chat",
    ):
        self._provider = provider
        self._model = model

    @property
    def system_name(self) -> str:
        return "dr-du"

    @property
    def system_type(self) -> str:
        return "hybrid"

    @property
    def requires_api(self) -> bool:
        return True

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        t0 = time.time()
        du = DUOutput(system_name=self.system_name)

        # ── Phase A: Deterministic probes for D2 + D4 ──────────────
        m_s = self._probe_d2(task, dataframes)
        m_f = self._probe_d4(task, dataframes, file_paths)
        du.m_s = m_s
        du.m_f = m_f

        # ── Phase B: Focused LLM call for D1 + D3 + D5 ────────────
        schema_block = self._build_schema_block(task, dataframes, m_s, m_f)
        file_desc = self._build_file_descriptions(task)

        prompt = _DRDU_LLM_PROMPT.format(
            goal=task.goal,
            schema_block=schema_block,
            file_descriptions=file_desc,
        )

        try:
            from du_benchmark.consensus import (
                _call_model_du, _parse_du_json,
            )
            from du_benchmark.llm.client import LLMProvider

            provider_enum = LLMProvider(self._provider.lower())
            raw_text, usage = await _call_model_du(
                provider_enum, self._model, prompt,
            )
            du.raw_text = raw_text
            du.token_usage = usage

            parsed = _parse_du_json(raw_text)
            self._merge_llm_results(du, parsed)

        except Exception as e:
            _logger.error("DR-DU LLM phase failed for %s: %s", task.task_id, e)
            # D1/D3/D5 stay as defaults; D2/D4 are already populated

        du.extraction_time_s = time.time() - t0
        return du

    # ── Phase A helpers ────────────────────────────────────────────

    def _probe_d2(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
    ) -> MS:
        """Deterministic D2: extract sources, columns, and join keys."""
        sources = list(task.data_files) or list(dataframes.keys())
        join_keys: List[JoinKey] = []
        column_mappings: Dict[str, str] = {}
        schema_conflicts: List[str] = []

        df_list = list(dataframes.items())
        if len(df_list) >= 2:
            # Pairwise join key detection (name + dtype matching)
            for i in range(len(df_list)):
                for j in range(i + 1, len(df_list)):
                    name_a, df_a = df_list[i]
                    name_b, df_b = df_list[j]
                    keys = self._detect_join_keys_pair(
                        name_a, df_a, name_b, df_b,
                    )
                    join_keys.extend(keys)

                    # Detect schema conflicts (same column, different dtype)
                    for col in set(df_a.columns) & set(df_b.columns):
                        ga = _dtype_group(str(df_a[col].dtype))
                        gb = _dtype_group(str(df_b[col].dtype))
                        if ga != gb:
                            schema_conflicts.append(
                                f"Column '{col}' has {df_a[col].dtype} in "
                                f"{name_a} but {df_b[col].dtype} in {name_b}"
                            )

        return MS(
            sources=sources,
            join_keys=join_keys,
            schema_conflicts=schema_conflicts,
            column_mappings=column_mappings,
        )

    @staticmethod
    def _detect_join_keys_pair(
        name_a: str, df_a: pd.DataFrame,
        name_b: str, df_b: pd.DataFrame,
    ) -> List[JoinKey]:
        """Detect join keys between two DataFrames via name+dtype matching."""
        keys: List[JoinKey] = []
        common_cols = set(df_a.columns) & set(df_b.columns)
        for col in sorted(common_cols):
            ga = _dtype_group(str(df_a[col].dtype))
            gb = _dtype_group(str(df_b[col].dtype))
            if ga == gb:
                keys.append(JoinKey(
                    left_source=name_a,
                    right_source=name_b,
                    left_column=col,
                    right_column=col,
                    join_type="inner",
                ))
        return keys

    def _probe_d4(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> MF:
        """Deterministic D4: detect format via DuckDB sniffer + pandas."""
        # Try DuckDB sniffer first if file paths are available
        if file_paths:
            for fname, fpath in file_paths.items():
                sniffed = self._duckdb_sniff(fpath)
                if sniffed:
                    df = dataframes.get(fname)
                    return MF(
                        has_header=sniffed.get("has_header", True),
                        delimiter=sniffed.get("delimiter", ","),
                        encoding="utf-8",
                        expected_columns=sniffed.get("columns", 0) or (
                            len(df.columns) if df is not None else 0
                        ),
                        expected_rows=len(df) if df is not None else 0,
                        file_format=self._detect_format(fpath),
                    )

        # Fallback: infer from task metadata and DataFrames
        first_file = task.data_files[0] if task.data_files else ""
        first_df = next(iter(dataframes.values()), None) if dataframes else None

        # Detect delimiter from file sample
        delimiter = ","
        has_header = True
        sample = task.file_samples.get(first_file, "")
        if sample:
            first_line = sample.split("\n")[0] if sample else ""
            if "\t" in first_line:
                delimiter = "\t"
            elif "|" in first_line:
                delimiter = "|"
            elif ";" in first_line:
                delimiter = ";"

        return MF(
            has_header=has_header,
            delimiter=delimiter,
            encoding="utf-8",
            expected_columns=len(first_df.columns) if first_df is not None else 0,
            expected_rows=len(first_df) if first_df is not None else 0,
            file_format=self._detect_format(first_file),
        )

    @staticmethod
    def _duckdb_sniff(file_path: str) -> Optional[Dict[str, Any]]:
        """Use DuckDB CSV sniffer for deterministic format detection."""
        from du_benchmark.deterministic_verify import duckdb_sniff_d4
        result = duckdb_sniff_d4(file_path)
        if "error" in result:
            return None
        return result

    @staticmethod
    def _detect_format(file_path: str) -> str:
        """Detect file format from extension."""
        suffix = Path(file_path).suffix.lower()
        fmt_map = {
            ".csv": "csv", ".tsv": "tsv", ".txt": "csv",
            ".json": "json", ".jsonl": "jsonl",
            ".xlsx": "excel", ".xls": "excel",
            ".parquet": "parquet", ".pq": "parquet",
        }
        return fmt_map.get(suffix, "csv")

    # ── Phase B helpers ────────────────────────────────────────────

    @staticmethod
    def _build_schema_block(
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        m_s: MS,
        m_f: MF,
    ) -> str:
        """Build verified schema block for the LLM prompt."""
        parts = []

        # Sources and columns
        parts.append("### Sources")
        for fname in m_s.sources:
            cols = task.column_lists.get(fname, [])
            meta = task.file_metadata.get(fname, {})
            dtypes = meta.get("dtypes", {})
            shape = meta.get("shape", "unknown")
            parts.append(f"- {fname}: {len(cols)} columns, shape={shape}")
            if cols:
                col_info = []
                for c in cols:
                    dt = dtypes.get(c, "")
                    col_info.append(f"  {c} ({dt})" if dt else f"  {c}")
                parts.append("  Columns:\n" + "\n".join(col_info))

        # Join keys
        if m_s.join_keys:
            parts.append("\n### Verified Join Keys")
            for jk in m_s.join_keys:
                parts.append(
                    f"- {jk.left_source}.{jk.left_column} = "
                    f"{jk.right_source}.{jk.right_column} ({jk.join_type})"
                )

        # Schema conflicts
        if m_s.schema_conflicts:
            parts.append("\n### Schema Conflicts")
            for conflict in m_s.schema_conflicts:
                parts.append(f"- {conflict}")

        # Format
        parts.append(f"\n### File Format")
        parts.append(f"- Format: {m_f.file_format}")
        parts.append(f"- Delimiter: '{m_f.delimiter}'")
        parts.append(f"- Header: {m_f.has_header}")
        parts.append(f"- Encoding: {m_f.encoding}")
        if m_f.expected_columns:
            parts.append(f"- Expected columns: {m_f.expected_columns}")

        return "\n".join(parts)

    @staticmethod
    def _build_file_descriptions(task: DUBenchmarkTask) -> str:
        """Build file sample block (truncated for token savings)."""
        parts = []
        for fname in task.data_files:
            sample = task.file_samples.get(fname, "")
            if sample:
                if len(sample) > 5000:
                    sample = sample[:5000] + "\n... (truncated)"
                parts.append(f"### {fname}\n```\n{sample}\n```")
        return "\n".join(parts)

    @staticmethod
    def _merge_llm_results(du: DUOutput, parsed: Dict[str, Any]) -> None:
        """Merge LLM D1/D3/D5 results into DUOutput (preserving D2/D4)."""
        if not parsed:
            return

        # Helper for safe coercion
        def _as_dict(val: Any) -> Dict[str, Any]:
            return val if isinstance(val, dict) else {}

        def _as_list(val: Any) -> list:
            return val if isinstance(val, list) else []

        # D1 (M_Q)
        mq = _as_dict(parsed.get("M_Q", {}))
        if mq:
            du.m_q = MQ(
                target_metric=str(mq.get("target_metric", "")),
                filters=_as_list(mq.get("filters", [])),
                grouping_variables=_as_list(mq.get("grouping_variables", [])),
                output_cardinality=str(mq.get("output_cardinality", "")),
                sub_questions=_as_list(mq.get("sub_questions", [])),
            )

        # D3 (M_U)
        mu = _as_dict(parsed.get("M_U", {}))
        if mu:
            du.m_u = MU(
                unit_annotations=_as_dict(mu.get("unit_annotations", {})),
                scale_issues=_as_list(mu.get("scale_issues", [])),
                cross_source_conflicts=_as_list(mu.get("cross_source_conflicts", [])),
            )

        # D5 (M_C)
        mc = _as_dict(parsed.get("M_C", {}))
        if mc:
            du.m_c = MC(
                constraints=_as_list(mc.get("constraints", [])),
                derived_filters=_as_list(mc.get("derived_filters", [])),
                statistical_tests=_as_list(mc.get("statistical_tests", [])),
                output_format_requirements=_as_list(mc.get("output_format_requirements", [])),
            )


# ── Grounding-only prompt (all 5 dims, schema as context) ─────────

_GROUNDING_ONLY_PROMPT = """\
You are a data understanding expert. Given data files and an analytical question,
produce a structured JSON describing the data understanding needed.

## Analytical Question
{goal}

## Verified Schema Context (from deterministic probes)
The following schema information was extracted deterministically from the data
files. Use it as reference context for your analysis, but produce ALL five
metadata artifacts yourself.

{schema_block}

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


class DRDUGroundingOnlyExtractor(DRDUExtractor):
    """Grounding-only variant: inject probe schema as context, LLM produces all 5 dims.

    This isolates the grounding effect from dimension routing: the probe's
    schema is injected as reference context, but the LLM produces ALL five
    M artifacts (including M_S and M_F). The final output uses the LLM's
    D2/D4, not the probe's.
    """

    @property
    def system_name(self) -> str:
        return "dr-du-grounding-only"

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        import time as _time
        t0 = _time.time()

        # Phase A: run probes to generate context (but NOT for final output)
        m_s = self._probe_d2(task, dataframes)
        m_f = self._probe_d4(task, dataframes, file_paths)

        # Build schema context block + file descriptions
        schema_block = self._build_schema_block(task, dataframes, m_s, m_f)
        file_desc = self._build_file_descriptions(task)

        prompt = _GROUNDING_ONLY_PROMPT.format(
            goal=task.goal,
            schema_block=schema_block,
            file_descriptions=file_desc,
        )

        try:
            from du_benchmark.consensus import (
                _call_model_du, _dict_to_du_output, _parse_du_json,
            )
            from du_benchmark.llm.client import LLMProvider

            provider_enum = LLMProvider(self._provider.lower())
            raw_text, usage = await _call_model_du(
                provider_enum, self._model, prompt,
            )

            parsed = _parse_du_json(raw_text)
            du = _dict_to_du_output(parsed, self.system_name)
            du.raw_text = raw_text
            du.token_usage = usage

        except Exception as e:
            _logger.error(
                "DR-DU grounding-only failed for %s: %s", task.task_id, e
            )
            du = DUOutput(system_name=self.system_name)

        du.extraction_time_s = _time.time() - t0
        return du


# ── Module-level helpers ───────────────────────────────────────────

def _dtype_group(dtype_str: str) -> str:
    """Classify a dtype string into a broad group for join compatibility."""
    d = dtype_str.lower()
    if "int" in d or "float" in d:
        return "numeric"
    if "datetime" in d or "timestamp" in d:
        return "datetime"
    if "bool" in d:
        return "bool"
    return "string"
