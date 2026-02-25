# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Base class for external benchmark adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


@dataclass
class BenchmarkTask:
    """Standardized representation of a benchmark task."""
    task_id: str
    name: str
    description: str
    goal: str
    difficulty: str
    category: str
    data_files: List[str]
    expected_output: Dict[str, Any]
    source_benchmark: str
    original_config: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Result from running a benchmark task."""
    task_id: str
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = None
    execution_time: Optional[float] = None


class ExternalBenchmarkAdapter(ABC):
    """
    Base class for adapting external benchmarks to ADP-MA.

    Subclasses should implement methods to:
    1. Load tasks from the external benchmark format
    2. Convert tasks to ADP-MA compatible format
    3. Evaluate ADP-MA outputs against benchmark ground truth
    """

    def __init__(self, benchmark_dir: Path, output_dir: Path):
        """
        Initialize adapter.

        Args:
            benchmark_dir: Path to the cloned benchmark repository
            output_dir: Path to save evaluation results
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Name of the benchmark."""
        pass

    @abstractmethod
    def list_tasks(self) -> List[str]:
        """
        List all available task IDs in the benchmark.

        Returns:
            List of task identifiers
        """
        pass

    @abstractmethod
    def load_task(self, task_id: str) -> BenchmarkTask:
        """
        Load a task from the benchmark.

        Args:
            task_id: Task identifier

        Returns:
            BenchmarkTask with standardized format
        """
        pass

    @abstractmethod
    def get_task_data(self, task: BenchmarkTask) -> Dict[str, pd.DataFrame]:
        """
        Load input data for a task.

        Args:
            task: The benchmark task

        Returns:
            Dict mapping data file names to DataFrames
        """
        pass

    @abstractmethod
    def evaluate_output(
        self,
        task: BenchmarkTask,
        output: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate ADP-MA output against benchmark ground truth.

        Args:
            task: The benchmark task
            output: ADP-MA output DataFrame

        Returns:
            Dict with evaluation metrics
        """
        pass

    def convert_to_adp_state(self, task: BenchmarkTask) -> Dict[str, Any]:
        """
        Convert task to ADP-MA state format.

        Args:
            task: The benchmark task

        Returns:
            Dict compatible with ADPState initialization
        """
        return {
            "user_goal": task.goal,
            "data_source_paths": task.data_files,
            "metadata": {
                "benchmark": self.benchmark_name,
                "task_id": task.task_id,
                "category": task.category,
                "difficulty": task.difficulty,
            }
        }

    def run_task(
        self,
        task_id: str,
        runner_callback
    ) -> BenchmarkResult:
        """
        Run a single benchmark task through ADP-MA.

        Args:
            task_id: Task identifier
            runner_callback: Callback function that takes (state_dict, data_dict)
                           and returns output DataFrame

        Returns:
            BenchmarkResult with evaluation
        """
        import time

        try:
            # Load task
            task = self.load_task(task_id)
            data = self.get_task_data(task)

            # Convert to ADP-MA format
            state_dict = self.convert_to_adp_state(task)

            # Run through ADP-MA
            start_time = time.time()
            output = runner_callback(state_dict, data)
            execution_time = time.time() - start_time

            # Evaluate
            metrics = self.evaluate_output(task, output)

            # Save output
            output_path = self.output_dir / f"{task_id}_output.csv"
            output.to_csv(output_path, index=False)

            return BenchmarkResult(
                task_id=task_id,
                success=metrics.get("passed", False),
                output_path=str(output_path),
                metrics=metrics,
                execution_time=execution_time,
            )

        except Exception as e:
            return BenchmarkResult(
                task_id=task_id,
                success=False,
                error=str(e),
            )

    def run_all(
        self,
        runner_callback,
        task_filter: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """
        Run all tasks in the benchmark.

        Args:
            runner_callback: Callback for running tasks
            task_filter: Optional list of task IDs to run

        Returns:
            List of BenchmarkResults
        """
        task_ids = self.list_tasks()

        if task_filter:
            task_ids = [t for t in task_ids if t in task_filter]

        results = []
        for task_id in task_ids:
            print(f"Running {self.benchmark_name} task: {task_id}")
            result = self.run_task(task_id, runner_callback)
            results.append(result)
            status = "PASS" if result.success else "FAIL"
            print(f"  {status}: {result.metrics if result.metrics else result.error}")

        return results
