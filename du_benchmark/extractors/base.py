# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Base class for DU extractors.

Each extractor adapts a specific system's DU output into the
standardized DUOutput schema (M_Q, M_S, M_U, M_F, M_C).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from du_benchmark.schema import DUBenchmarkTask, DUOutput


class BaseDUExtractor(ABC):
    """Base class for per-system DU extraction."""

    @property
    @abstractmethod
    def system_name(self) -> str:
        """Unique identifier for this system."""
        ...

    @property
    def system_type(self) -> str:
        """System type: 'profiler', 'llm_agent', 'text_to_sql', 'hybrid', 'baseline'."""
        return "unknown"

    @property
    def requires_api(self) -> bool:
        """Whether this extractor requires API calls (costs money)."""
        return False

    @abstractmethod
    async def extract_du(
        self,
        task: DUBenchmarkTask,
        dataframes: Dict[str, pd.DataFrame],
        file_paths: Optional[Dict[str, str]] = None,
    ) -> DUOutput:
        """
        Extract DU output from this system for the given task.

        Args:
            task: The DU benchmark task descriptor
            dataframes: Loaded DataFrames keyed by filename
            file_paths: Absolute file paths keyed by filename

        Returns:
            Standardized DUOutput
        """
        ...
