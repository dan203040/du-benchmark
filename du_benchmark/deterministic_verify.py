# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Deterministic Verification for D2 (joins) and D4 (format diagnosis).

Ground truth that doesn't need any LLM judge — either the join works or it doesn't,
either the file loads correctly or it doesn't.

Also includes DuckDB sniffer baseline for D4.

Usage:
    python3 evaluation/du_benchmark/deterministic_verify.py \\
        --tasks kramabench \\
        --output evaluation/results/du_benchmark/deterministic/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


from du_benchmark.schema import (
    DUBenchmarkTask, DUOutput, DimensionScore,
)

_logger = logging.getLogger(__name__)


# ── D4: Format Diagnosis Verification ──────────────────────────────

def verify_d4(
    du_output: DUOutput,
    file_path: str,
    actual_df: Optional[pd.DataFrame] = None,
) -> DimensionScore:
    """
    Try to load the file using the system's predicted format params.

    Scoring:
      0.4 — correct number of columns (or within 10%)
      0.3 — low NaN rate (< 50% of cells)
      0.3 — no unnamed columns

    If actual_df is provided, compare loaded shape against it.
    """
    mf = du_output.m_f
    details: Dict[str, Any] = {"file": file_path}
    score = 0.0

    try:
        read_kwargs: Dict[str, Any] = {}
        if mf.delimiter:
            read_kwargs["sep"] = mf.delimiter
        if not mf.has_header:
            read_kwargs["header"] = None
        if mf.encoding and mf.encoding != "utf-8":
            read_kwargs["encoding"] = mf.encoding
        if mf.sentinel_values:
            read_kwargs["na_values"] = mf.sentinel_values

        fp = Path(file_path)
        if not fp.exists():
            details["error"] = "file not found"
            return DimensionScore(
                dimension="D4", score=0.0, method="deterministic",
                details=details,
            )

        fmt = mf.file_format.lower() if mf.file_format else "csv"
        if fmt in ("csv", "tsv", "txt"):
            df = pd.read_csv(fp, **read_kwargs)
        elif fmt in ("json", "jsonl"):
            df = pd.read_json(fp)
        elif fmt in ("excel", "xlsx", "xls"):
            df = pd.read_excel(fp)
        elif fmt == "parquet":
            df = pd.read_parquet(fp)
        else:
            df = pd.read_csv(fp, **read_kwargs)

        details["loaded_shape"] = list(df.shape)
        details["loaded_columns"] = len(df.columns)
        details["predicted_columns"] = mf.expected_columns

        # Sub-score 1: Column count accuracy (0.4)
        if mf.expected_columns > 0:
            col_ratio = min(len(df.columns), mf.expected_columns) / max(
                len(df.columns), mf.expected_columns, 1
            )
            s1 = 0.4 * col_ratio
        elif actual_df is not None:
            col_ratio = min(len(df.columns), len(actual_df.columns)) / max(
                len(df.columns), len(actual_df.columns), 1
            )
            s1 = 0.4 * col_ratio
        else:
            # No reference — give partial credit if loading succeeded
            s1 = 0.2

        # Sub-score 2: Low NaN rate (0.3)
        total_cells = df.shape[0] * df.shape[1]
        if total_cells > 0:
            nan_rate = df.isnull().sum().sum() / total_cells
            s2 = 0.3 * max(0.0, 1.0 - 2 * nan_rate)  # 0% NaN → 0.3, 50%→0
        else:
            s2 = 0.0

        # Sub-score 3: No unnamed columns (0.3)
        unnamed_count = sum(
            1 for c in df.columns if str(c).startswith("Unnamed")
        )
        s3 = 0.3 * max(0.0, 1.0 - unnamed_count / max(len(df.columns), 1))

        score = s1 + s2 + s3
        details["sub_scores"] = {"column_accuracy": s1, "nan_rate": s2, "no_unnamed": s3}

    except Exception as e:
        details["error"] = str(e)
        score = 0.0

    return DimensionScore(
        dimension="D4", score=round(score, 3), method="deterministic",
        details=details,
    )


# ── D2: Join Key Verification ──────────────────────────────────────

def verify_d2(
    du_output: DUOutput,
    dataframes: Dict[str, pd.DataFrame],
) -> DimensionScore:
    """
    Try to execute proposed joins and check for data quality.

    Scoring per join:
      0.6 — match rate (fraction of left rows that matched)
      0.4 — no cartesian explosion (merged < 3x max input)

    Overall = mean across all proposed joins.
    """
    join_keys = du_output.m_s.join_keys
    details: Dict[str, Any] = {"n_joins": len(join_keys)}

    if not join_keys:
        # If no joins proposed and task has only 1 source, that's correct
        if len(dataframes) <= 1:
            return DimensionScore(
                dimension="D2", score=1.0, method="deterministic",
                details={"reason": "single source, no joins needed"},
            )
        else:
            return DimensionScore(
                dimension="D2", score=0.0, method="deterministic",
                details={"reason": "multi-source task but no joins proposed"},
            )

    join_scores = []
    join_details = []

    for jk in join_keys:
        jd: Dict[str, Any] = {
            "left_source": jk.left_source,
            "right_source": jk.right_source,
            "left_column": jk.left_column,
            "right_column": jk.right_column,
        }

        # Find DataFrames by source name (fuzzy match on filename)
        left_df = _find_df(jk.left_source, dataframes)
        right_df = _find_df(jk.right_source, dataframes)

        if left_df is None:
            jd["error"] = f"left source not found: {jk.left_source}"
            join_scores.append(0.0)
            join_details.append(jd)
            continue
        if right_df is None:
            jd["error"] = f"right source not found: {jk.right_source}"
            join_scores.append(0.0)
            join_details.append(jd)
            continue

        # Check if columns exist
        left_col = _find_column(jk.left_column, left_df)
        right_col = _find_column(jk.right_column, right_df)

        if left_col is None:
            jd["error"] = f"left column not found: {jk.left_column}"
            join_scores.append(0.0)
            join_details.append(jd)
            continue
        if right_col is None:
            jd["error"] = f"right column not found: {jk.right_column}"
            join_scores.append(0.0)
            join_details.append(jd)
            continue

        try:
            merged = left_df.merge(right_df, left_on=left_col, right_on=right_col)
            match_rate = len(merged) / max(len(left_df), 1)
            no_explosion = len(merged) < 3 * max(len(left_df), len(right_df))

            js = 0.6 * min(match_rate, 1.0) + 0.4 * float(no_explosion)
            jd["merged_rows"] = len(merged)
            jd["left_rows"] = len(left_df)
            jd["right_rows"] = len(right_df)
            jd["match_rate"] = round(match_rate, 3)
            jd["no_explosion"] = no_explosion
            jd["score"] = round(js, 3)
            join_scores.append(js)
        except Exception as e:
            jd["error"] = str(e)
            join_scores.append(0.0)

        join_details.append(jd)

    overall = sum(join_scores) / len(join_scores) if join_scores else 0.0
    details["joins"] = join_details
    details["mean_join_score"] = round(overall, 3)

    return DimensionScore(
        dimension="D2", score=round(overall, 3), method="deterministic",
        details=details,
    )


def _find_df(
    source_name: str, dataframes: Dict[str, pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Find a DataFrame by source name with fuzzy matching."""
    source_lower = source_name.lower().strip()
    # Exact match
    if source_name in dataframes:
        return dataframes[source_name]
    # Case-insensitive match
    for k, v in dataframes.items():
        if k.lower().strip() == source_lower:
            return v
    # Stem match (filename without path or extension)
    source_stem = Path(source_lower).stem
    for k, v in dataframes.items():
        if Path(k).stem.lower() == source_stem:
            return v
    # Substring match
    for k, v in dataframes.items():
        if source_stem in k.lower() or k.lower() in source_lower:
            return v
    return None


def _find_column(col_name, df: pd.DataFrame) -> Optional[str]:
    """Find a column by name with case-insensitive matching."""
    # Handle non-string inputs (e.g. LLM returned a list)
    if isinstance(col_name, list):
        col_name = col_name[0] if col_name else ""
    col_name = str(col_name)
    if not col_name:
        return None
    if col_name in df.columns:
        return col_name
    col_lower = col_name.lower().strip()
    for c in df.columns:
        if c.lower().strip() == col_lower:
            return c
    # Underscore/space normalization
    normalized = col_lower.replace(" ", "_").replace("-", "_")
    for c in df.columns:
        cn = c.lower().strip().replace(" ", "_").replace("-", "_")
        if cn == normalized:
            return c
    return None


# ── DuckDB Sniffer Baseline (D4) ───────────────────────────────────

def duckdb_sniff_d4(file_path: str) -> Dict[str, Any]:
    """
    Use DuckDB's CSV sniffer as a deterministic D4 baseline.
    Returns sniffed parameters.
    """
    try:
        import duckdb
    except ImportError:
        _logger.warning("duckdb not installed, skipping sniffer baseline")
        return {"error": "duckdb not installed"}

    fp = Path(file_path)
    if not fp.exists() or fp.suffix.lower() not in (".csv", ".tsv", ".txt"):
        return {"error": "not a CSV-like file"}

    try:
        conn = duckdb.connect(":memory:")
        result = conn.execute(
            f"SELECT * FROM sniff_csv('{fp}')"
        ).fetchone()
        conn.close()

        if result:
            return {
                "delimiter": result[0] if result[0] else ",",
                "has_header": bool(result[2]) if len(result) > 2 else True,
                "columns": result[4] if len(result) > 4 else 0,
            }
    except Exception as e:
        return {"error": str(e)}

    return {}


def verify_d4_duckdb(
    file_path: str,
    actual_df: Optional[pd.DataFrame] = None,
) -> DimensionScore:
    """
    DuckDB sniffer baseline for D4 — deterministic ceiling.
    Any system scoring below this is worse than a rule-based approach.
    """
    sniffed = duckdb_sniff_d4(file_path)
    if "error" in sniffed:
        return DimensionScore(
            dimension="D4", score=0.0, method="deterministic",
            details={"baseline": "duckdb_sniffer", "error": sniffed["error"]},
        )

    # Build a DUOutput from sniffed params
    from du_benchmark.schema import MF
    du = DUOutput(system_name="duckdb_sniffer")
    du.m_f = MF(
        has_header=sniffed.get("has_header", True),
        delimiter=sniffed.get("delimiter", ","),
        expected_columns=sniffed.get("columns", 0),
    )
    result = verify_d4(du, file_path, actual_df)
    result.details["baseline"] = "duckdb_sniffer"
    return result


# ── Batch Verification ──────────────────────────────────────────────

def verify_task_deterministic(
    task: DUBenchmarkTask,
    du_output: DUOutput,
    dataframes: Dict[str, pd.DataFrame],
    file_paths: Optional[Dict[str, str]] = None,
) -> List[DimensionScore]:
    """Run all deterministic verifications for a task."""
    scores = []

    # D2: Join verification
    if len(dataframes) > 1 or du_output.m_s.join_keys:
        scores.append(verify_d2(du_output, dataframes))
    else:
        scores.append(DimensionScore(
            dimension="D2", score=1.0, method="deterministic",
            details={"reason": "single source, no joins needed"},
        ))

    # D4: Format verification for each file
    d4_scores = []
    if file_paths:
        for fname, fpath in file_paths.items():
            actual_df = dataframes.get(fname)
            d4_score = verify_d4(du_output, fpath, actual_df)
            d4_scores.append(d4_score.score)
    if d4_scores:
        avg_d4 = sum(d4_scores) / len(d4_scores)
        scores.append(DimensionScore(
            dimension="D4", score=round(avg_d4, 3), method="deterministic",
            details={"n_files": len(d4_scores), "per_file_scores": d4_scores},
        ))
    else:
        scores.append(DimensionScore(
            dimension="D4", score=0.5, method="deterministic",
            details={"reason": "no file paths available for verification"},
        ))

    return scores


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run deterministic D2/D4 verification",
    )
    parser.add_argument(
        "--tasks", default="kramabench",
        help="Comma-separated benchmark names",
    )
    parser.add_argument(
        "--du-results", required=True,
        help="Path to DU extraction results JSON",
    )
    parser.add_argument(
        "--output", default="evaluation/results/du_benchmark/deterministic/",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from du_benchmark.consensus import load_du_tasks

    benchmarks = args.tasks.split(",")
    tasks = load_du_tasks(benchmarks)

    du_results = json.loads(Path(args.du_results).read_text())

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    for task in tasks:
        if task.task_id not in du_results:
            continue
        du_data = du_results[task.task_id]
        du_output = DUOutput.from_dict(du_data)

        # Build DataFrames from task metadata
        dataframes = {}
        for fname in task.data_files:
            sample_csv = task.file_samples.get(fname, "")
            if sample_csv:
                try:
                    from io import StringIO
                    dataframes[fname] = pd.read_csv(StringIO(sample_csv))
                except Exception:
                    pass

        scores = verify_task_deterministic(task, du_output, dataframes)
        all_scores[task.task_id] = [s.__dict__ for s in scores]

    outf = output_dir / "deterministic_scores.json"
    outf.write_text(json.dumps(all_scores, indent=2, default=str))
    _logger.info("Saved deterministic scores for %d tasks to %s", len(all_scores), outf)


if __name__ == "__main__":
    main()
