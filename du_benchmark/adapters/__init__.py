# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Benchmark task adapters for DU Benchmark.

Provides adapters to load tasks from external benchmarks:
- KramaBench: Data science pipeline benchmark (105 tasks)
- AgentBench: General agent benchmark — DBBench SELECT tasks (100 tasks)
- DACode: Data science code generation (52 tasks)
"""

from .base import ExternalBenchmarkAdapter, BenchmarkTask
from .kramabench import KramaBenchAdapter
from .da_code import DACodeAdapter
from .agentbench import AgentBenchAdapter

__all__ = [
    "ExternalBenchmarkAdapter",
    "BenchmarkTask",
    "KramaBenchAdapter",
    "DACodeAdapter",
    "AgentBenchAdapter",
]
