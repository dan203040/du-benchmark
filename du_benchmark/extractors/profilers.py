# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
DU extractors for profiler-based systems.

- ydata-profiling (formerly pandas-profiling)
- DataProfiler (Capital One)
- DuckDB sniffer (D4-only baseline)
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


class YDataProfilingExtractor(BaseDUExtractor):
    """Extract DU from ydata-profiling ProfileReport."""

    @property
    def system_name(self) -> str:
        return "ydata-profiling"

    @property
    def system_type(self) -> str:
        return "profiler"

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        try:
            from ydata_profiling import ProfileReport
        except ImportError:
            _logger.error("ydata-profiling not installed. pip install ydata-profiling")
            return DUOutput(system_name=self.system_name)

        du = DUOutput(system_name=self.system_name)
        t0 = time.time()

        all_sources = []
        join_keys = []

        for fname, df in dataframes.items():
            all_sources.append(fname)
            try:
                profile = ProfileReport(
                    df, minimal=True, progress_bar=False,
                )
                report = profile.to_json()
                import json
                report_dict = json.loads(report)

                # D4: Format info from profile
                table_info = report_dict.get("table", {})
                du.m_f = MF(
                    expected_columns=table_info.get("n_var", len(df.columns)),
                    expected_rows=table_info.get("n", len(df)),
                    has_header=True,  # profiler assumes headers
                )

                # D2: Column types and potential join keys
                variables = report_dict.get("variables", {})
                for col_name, col_info in variables.items():
                    col_type = col_info.get("type", "")
                    # Detect potential join keys: unique or near-unique categorical/int cols
                    n_distinct = col_info.get("n_distinct", 0)
                    n_total = col_info.get("count", 1)
                    if n_distinct > 0 and n_distinct / max(n_total, 1) > 0.5:
                        # High cardinality column — potential join key
                        pass

                # D3: Unit detection is not available from profilers
                du.m_u = MU()

            except Exception as e:
                _logger.warning("ydata-profiling failed on %s: %s", fname, e)

        du.m_s = MS(sources=all_sources, join_keys=join_keys)
        # Profilers don't do D1 (query) or D5 (constraints)
        du.m_q = MQ()
        du.m_c = MC()
        du.extraction_time_s = time.time() - t0

        return du


class DataProfilerExtractor(BaseDUExtractor):
    """Extract DU from Capital One's DataProfiler."""

    @property
    def system_name(self) -> str:
        return "dataprofiler"

    @property
    def system_type(self) -> str:
        return "profiler"

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        try:
            import dataprofiler as dp
        except ImportError:
            _logger.error("DataProfiler not installed. pip install DataProfiler")
            return DUOutput(system_name=self.system_name)

        du = DUOutput(system_name=self.system_name)
        t0 = time.time()

        all_sources = []
        for fname, df in dataframes.items():
            all_sources.append(fname)
            try:
                profiler = dp.Profiler(df)
                report = profiler.report()

                # D4: Format info
                global_stats = report.get("global_stats", {})
                du.m_f = MF(
                    expected_columns=global_stats.get("column_count", len(df.columns)),
                    expected_rows=global_stats.get("row_count", len(df)),
                    has_header=global_stats.get("has_header", True),
                    file_format=global_stats.get("file_type", "csv"),
                    encoding=global_stats.get("encoding", "utf-8"),
                )

                # D2: Schema info
                data_stats = report.get("data_stats", [])
                for col_stat in data_stats:
                    col_name = col_stat.get("column_name", "")
                    data_type = col_stat.get("data_type", "")
                    # DataProfiler provides rich type detection
                    if data_type:
                        du.m_u.unit_annotations[col_name] = data_type

            except Exception as e:
                _logger.warning("DataProfiler failed on %s: %s", fname, e)

        du.m_s = MS(sources=all_sources)
        du.m_q = MQ()
        du.m_c = MC()
        du.extraction_time_s = time.time() - t0

        return du


class DuckDBSnifferExtractor(BaseDUExtractor):
    """DuckDB CSV sniffer baseline — D4 only."""

    @property
    def system_name(self) -> str:
        return "duckdb-sniffer"

    @property
    def system_type(self) -> str:
        return "baseline"

    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        from du_benchmark.deterministic_verify import duckdb_sniff_d4

        du = DUOutput(system_name=self.system_name)
        t0 = time.time()

        if file_paths:
            for fname, fpath in file_paths.items():
                sniffed = duckdb_sniff_d4(fpath)
                if "error" not in sniffed:
                    du.m_f = MF(
                        has_header=sniffed.get("has_header", True),
                        delimiter=sniffed.get("delimiter", ","),
                        expected_columns=sniffed.get("columns", 0),
                    )
                    break

        du.m_s = MS(sources=list(dataframes.keys()))
        du.m_q = MQ()
        du.m_u = MU()
        du.m_c = MC()
        du.extraction_time_s = time.time() - t0

        return du
