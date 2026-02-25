# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
DU extractors for LLM-agent-based systems.

- SmolAgents (GPT-4o-mini)
- PandasAI
- LangChain Pandas Agent
- LLM-only baseline (direct prompt)
- Profile+LLM hybrid
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from du_benchmark.extractors.base import BaseDUExtractor
from du_benchmark.schema import (
    DUBenchmarkTask, DUOutput, JoinKey, MC, MF, MQ, MS, MU,
)

_logger = logging.getLogger(__name__)


class SmolAgentsExtractor(BaseDUExtractor):
    """Extract DU from HuggingFace SmolAgents via DU-first prompt (Tier C)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model

    @property
    def system_name(self) -> str:
        return "smolagents"

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
        try:
            from smolagents import CodeAgent, LiteLLMModel
        except ImportError:
            _logger.error("smolagents not installed. pip install smolagents")
            return DUOutput(system_name=self.system_name)

        t0 = time.time()
        du = DUOutput(system_name=self.system_name)

        # Build a DU-extraction prompt for SmolAgents
        file_desc = []
        for fname, df in dataframes.items():
            file_desc.append(
                f"File: {fname}\nColumns: {list(df.columns)}\n"
                f"Shape: {df.shape}\nSample:\n{df.head(5).to_string()}"
            )
        file_block = "\n\n".join(file_desc)

        prompt = (
            f"Analyze these datasets for the question: {task.goal}\n\n"
            f"{file_block}\n\n"
            "Produce a JSON with keys: target_metric, filters, join_keys, "
            "data_format, constraints. Be precise."
        )

        try:
            model = LiteLLMModel(model_id=self._model)
            agent = CodeAgent(tools=[], model=model)
            result = agent.run(prompt)
            du.raw_text = str(result)

            # Parse structured output if JSON
            parsed = _try_parse_json(str(result))
            if parsed:
                du = _json_to_du_output(parsed, self.system_name)
                du.raw_text = str(result)
        except Exception as e:
            _logger.error("SmolAgents failed for %s: %s", task.task_id, e)

        du.extraction_time_s = time.time() - t0
        return du


class PandasAIExtractor(BaseDUExtractor):
    """Extract DU from PandasAI's schema step."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model

    @property
    def system_name(self) -> str:
        return "pandasai"

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
        try:
            from pandasai import SmartDataframe
        except ImportError:
            _logger.error("pandasai not installed. pip install pandasai")
            return DUOutput(system_name=self.system_name)

        t0 = time.time()
        du = DUOutput(system_name=self.system_name)

        try:
            primary_df = next(iter(dataframes.values()))
            sdf = SmartDataframe(primary_df, config={"llm_model": self._model})
            # Get schema info
            schema = sdf.dataframe.dtypes.to_dict()
            du.m_f = MF(
                expected_columns=len(primary_df.columns),
                expected_rows=len(primary_df),
            )
            du.m_s = MS(sources=list(dataframes.keys()))
            du.m_u = MU(
                unit_annotations={str(k): str(v) for k, v in schema.items()},
            )
        except Exception as e:
            _logger.error("PandasAI failed for %s: %s", task.task_id, e)

        du.extraction_time_s = time.time() - t0
        return du


class LangChainExtractor(BaseDUExtractor):
    """Extract DU from LangChain Pandas Agent's trace."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model

    @property
    def system_name(self) -> str:
        return "langchain"

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
        try:
            from langchain_experimental.agents import create_pandas_dataframe_agent
            from langchain_openai import ChatOpenAI
        except ImportError:
            _logger.error(
                "langchain not installed. "
                "pip install langchain-experimental langchain-openai"
            )
            return DUOutput(system_name=self.system_name)

        t0 = time.time()
        du = DUOutput(system_name=self.system_name)

        try:
            llm = ChatOpenAI(model=self._model, temperature=0)
            primary_df = next(iter(dataframes.values()))
            agent = create_pandas_dataframe_agent(
                llm, primary_df, verbose=False,
                allow_dangerous_code=True,
            )

            du_prompt = (
                f"Analyze this dataset for: {task.goal}\n"
                "Return JSON with: target_metric, filters, grouping_variables, "
                "join_keys, data_types, constraints"
            )
            result = agent.invoke(du_prompt)
            output_text = result.get("output", str(result))
            du.raw_text = output_text

            parsed = _try_parse_json(output_text)
            if parsed:
                du = _json_to_du_output(parsed, self.system_name)
                du.raw_text = output_text
        except Exception as e:
            _logger.error("LangChain agent failed for %s: %s", task.task_id, e)

        du.extraction_time_s = time.time() - t0
        return du


class LLMOnlyExtractor(BaseDUExtractor):
    """LLM-only baseline: GPT-4o-mini produces M from (D,Q) directly."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
    ):
        self._provider = provider
        self._model = model

    @property
    def system_name(self) -> str:
        return "llm-only"

    @property
    def system_type(self) -> str:
        return "baseline"

    @property
    def requires_api(self) -> bool:
        return True

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        from du_benchmark.consensus import (
            DU_PROMPT_TEMPLATE, _build_file_descriptions,
            _call_model_du, _dict_to_du_output, _parse_du_json,
        )
        from du_benchmark.llm.client import LLMProvider

        t0 = time.time()
        prompt = DU_PROMPT_TEMPLATE.format(
            goal=task.goal,
            file_descriptions=_build_file_descriptions(task),
        )

        provider_enum = LLMProvider(self._provider.lower())
        try:
            raw_text, usage = await _call_model_du(
                provider_enum, self._model, prompt,
            )
            parsed = _parse_du_json(raw_text)
            du = _dict_to_du_output(parsed, self.system_name)
            du.raw_text = raw_text
            du.token_usage = usage
        except Exception as e:
            _logger.error("LLM-only baseline failed for %s: %s", task.task_id, e)
            du = DUOutput(system_name=self.system_name)

        du.extraction_time_s = time.time() - t0
        return du


class ProfileLLMHybridExtractor(BaseDUExtractor):
    """
    Profile+LLM hybrid: ydata-profiling → GPT-4o-mini.

    First profiles data with ydata, then feeds profile to LLM
    for task-conditioned DU extraction.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model

    @property
    def system_name(self) -> str:
        return "profile-llm-hybrid"

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
        from du_benchmark.llm.client import LLMProvider, create_llm_client

        t0 = time.time()
        du = DUOutput(system_name=self.system_name)

        # Step 1: Profile with ydata
        profile_summary = ""
        try:
            from ydata_profiling import ProfileReport
            for fname, df in dataframes.items():
                profile = ProfileReport(df, minimal=True, progress_bar=False)
                desc = profile.get_description()
                table = desc.get("table", {}) if isinstance(desc, dict) else {}
                profile_summary += (
                    f"File: {fname}\n"
                    f"  Rows: {table.get('n', len(df))}, "
                    f"Cols: {table.get('n_var', len(df.columns))}\n"
                    f"  Columns: {list(df.columns)}\n"
                    f"  Types: {df.dtypes.to_dict()}\n\n"
                )
        except ImportError:
            # Fallback: basic pandas profiling
            for fname, df in dataframes.items():
                profile_summary += (
                    f"File: {fname}\n"
                    f"  Shape: {df.shape}\n"
                    f"  Columns: {list(df.columns)}\n"
                    f"  Types: {df.dtypes.to_dict()}\n"
                    f"  Nulls: {df.isnull().sum().to_dict()}\n\n"
                )

        # Step 2: Feed profile + task goal to LLM
        from du_benchmark.consensus import (
            DU_PROMPT_TEMPLATE, _build_file_descriptions,
        )
        file_desc = _build_file_descriptions(task)
        prompt = DU_PROMPT_TEMPLATE.format(
            goal=task.goal,
            file_descriptions=file_desc,
        ) + f"\n\n## Additional Data Profile (from ydata-profiling)\n{profile_summary}"

        try:
            from du_benchmark.consensus import (
                _call_model_du, _dict_to_du_output, _parse_du_json,
            )
            raw_text, usage = await _call_model_du(
                LLMProvider.OPENAI, self._model, prompt,
            )
            parsed = _parse_du_json(raw_text)
            du = _dict_to_du_output(parsed, self.system_name)
            du.raw_text = raw_text
            du.token_usage = usage
        except Exception as e:
            _logger.error("Profile+LLM hybrid failed for %s: %s", task.task_id, e)

        du.extraction_time_s = time.time() - t0
        return du


class RandomBaselineExtractor(BaseDUExtractor):
    """Random baseline: random M from plausible vocabulary → floor."""

    @property
    def system_name(self) -> str:
        return "random"

    @property
    def system_type(self) -> str:
        return "baseline"

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        import random

        du = DUOutput(system_name=self.system_name)
        t0 = time.time()

        # Random target metric from available columns
        all_columns = []
        for cols in task.column_lists.values():
            all_columns.extend(cols)
        if all_columns:
            du.m_q = MQ(
                target_metric=random.choice(all_columns),
                filters=[random.choice(all_columns)] if len(all_columns) > 1 else [],
                grouping_variables=(
                    [random.choice(all_columns)] if len(all_columns) > 2 else []
                ),
                output_cardinality=random.choice(
                    ["scalar", "list", "table", "plot"]
                ),
            )

        # Random format params
        du.m_f = MF(
            has_header=random.choice([True, False]),
            delimiter=random.choice([",", "\t", "|", ";"]),
            encoding=random.choice(["utf-8", "latin-1", "ascii"]),
        )

        du.m_s = MS(sources=list(dataframes.keys()))
        du.m_u = MU()
        du.m_c = MC()
        du.extraction_time_s = time.time() - t0

        return du


# ── Helpers ─────────────────────────────────────────────────────────

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from text, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _json_to_du_output(d: Dict[str, Any], system_name: str) -> DUOutput:
    """Convert a generic JSON dict to DUOutput (flexible key matching)."""
    from du_benchmark.consensus import _dict_to_du_output

    # Try standard keys first
    if any(k in d for k in ("M_Q", "M_S", "M_U", "M_F", "M_C")):
        return _dict_to_du_output(d, system_name)

    # Flexible key mapping for non-standard outputs
    du = DUOutput(system_name=system_name)
    du.m_q = MQ(
        target_metric=d.get("target_metric", ""),
        filters=d.get("filters", []),
        grouping_variables=d.get("grouping_variables", d.get("group_by", [])),
        output_cardinality=d.get("output_cardinality", d.get("output_type", "")),
    )
    du.m_s = MS(
        sources=d.get("sources", d.get("data_sources", [])),
        join_keys=[
            JoinKey(**jk) if isinstance(jk, dict) else JoinKey()
            for jk in d.get("join_keys", d.get("joins", []))
        ],
    )
    du.m_f = MF(
        has_header=d.get("has_header", True),
        delimiter=d.get("delimiter", ","),
        file_format=d.get("data_format", d.get("format", "csv")),
    )
    du.m_c = MC(
        constraints=d.get("constraints", []),
    )
    return du
