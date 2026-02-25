# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
#!/usr/bin/env python3
"""
Cross-model consensus stability analysis.

Analyzes pairwise inter-model agreement for the 3-model consensus panel
(DeepSeek, Claude Sonnet 4.5, Gemini 2.0 Flash) across all 5 DU dimensions.

Usage:
    python3 evaluation/du_benchmark/consensus_stability.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONSENSUS_DIR = Path("evaluation/results/du_benchmark/consensus_decontaminated/per_task")
OUTPUT_DIR = Path("evaluation/results/du_benchmark/analysis_decontaminated")

MODELS = ["deepseek", "claude-sonnet", "gemini-flash"]
MODEL_PAIRS = [
    ("deepseek", "claude-sonnet"),
    ("deepseek", "gemini-flash"),
    ("claude-sonnet", "gemini-flash"),
]

DIMENSION_FIELDS = {
    "D1": {"M_Q": ["target_metric", "filters", "grouping_variables", "output_cardinality", "sub_questions"]},
    "D2": {"M_S": ["sources", "join_keys", "schema_conflicts"]},
    "D3": {"M_U": ["unit_annotations", "scale_issues", "cross_source_conflicts"]},
    "D4": {"M_F": ["has_header", "delimiter", "encoding", "sentinel_values", "expected_columns", "file_format"]},
    "D5": {"M_C": ["constraints", "derived_filters", "statistical_tests", "output_format_requirements"]},
}


def _normalize(val: Any) -> str:
    """Normalize a value for comparison."""
    if isinstance(val, list):
        return json.dumps(sorted(str(v).lower().strip() for v in val))
    if isinstance(val, dict):
        return json.dumps({k.lower().strip(): str(v).lower().strip() for k, v in sorted(val.items())})
    if isinstance(val, bool):
        return str(val).lower()
    return str(val).lower().strip()


def _get_field_value(du_data: Dict, dim_key: str, field: str) -> Any:
    """Extract a field value from a DU output dict."""
    dim_map = {"M_Q": "m_q", "M_S": "m_s", "M_U": "m_u", "M_F": "m_f", "M_C": "m_c"}
    dim_data = du_data.get(dim_map.get(dim_key, dim_key), {})
    if isinstance(dim_data, dict):
        return dim_data.get(field, "")
    return ""


def compute_pairwise_agreement(
    task_dirs: List[Path],
) -> Dict[str, Any]:
    """Compute pairwise inter-model agreement across all tasks."""

    # Per-dimension, per-pair agreement counts
    pair_agree: Dict[str, Dict[Tuple[str, str], List[bool]]] = defaultdict(lambda: defaultdict(list))
    # Per-dimension, per-field, per-pair
    field_agree: Dict[str, Dict[str, Dict[Tuple[str, str], List[bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    # Per-model quality indicators
    model_non_empty: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    n_tasks = 0

    for task_dir in sorted(task_dirs):
        # Load per-model outputs
        model_outputs = {}
        for model in MODELS:
            model_file = task_dir / f"{model}.json"
            if model_file.exists():
                try:
                    model_outputs[model] = json.loads(model_file.read_text())
                except Exception:
                    pass

        if len(model_outputs) < 2:
            continue

        n_tasks += 1

        # Compare each pair on each dimension/field
        for dim, dim_fields in DIMENSION_FIELDS.items():
            for dim_key, fields in dim_fields.items():
                for field in fields:
                    for m1, m2 in MODEL_PAIRS:
                        if m1 not in model_outputs or m2 not in model_outputs:
                            continue
                        v1 = _get_field_value(model_outputs[m1], dim_key, field)
                        v2 = _get_field_value(model_outputs[m2], dim_key, field)
                        agrees = _normalize(v1) == _normalize(v2)
                        pair_agree[dim][(m1, m2)].append(agrees)
                        field_agree[dim][field][(m1, m2)].append(agrees)

                    # Track non-empty outputs per model
                    for model in MODELS:
                        if model not in model_outputs:
                            continue
                        val = _get_field_value(model_outputs[model], dim_key, field)
                        if val and val != "" and val != [] and val != {}:
                            model_non_empty[model][dim] = model_non_empty[model].get(dim, 0) + 1

    return {
        "n_tasks": n_tasks,
        "pair_agree": pair_agree,
        "field_agree": field_agree,
        "model_non_empty": model_non_empty,
    }


def format_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format results into serializable summary."""
    n_tasks = results["n_tasks"]

    # Pairwise agreement per dimension
    dim_agreement = {}
    for dim in ["D1", "D2", "D3", "D4", "D5"]:
        pair_data = results["pair_agree"].get(dim, {})
        dim_pairs = {}
        for (m1, m2), agrees in pair_data.items():
            rate = sum(agrees) / len(agrees) if agrees else 0.0
            dim_pairs[f"{m1} vs {m2}"] = {
                "agreement_rate": round(rate, 3),
                "n_comparisons": len(agrees),
                "n_agree": sum(agrees),
            }
        if dim_pairs:
            all_rates = [v["agreement_rate"] for v in dim_pairs.values()]
            dim_agreement[dim] = {
                "pairs": dim_pairs,
                "avg_agreement": round(sum(all_rates) / len(all_rates), 3),
            }

    # Per-field agreement
    field_details = {}
    for dim in ["D1", "D2", "D3", "D4", "D5"]:
        field_data = results["field_agree"].get(dim, {})
        for field, pair_data in field_data.items():
            rates = []
            for (m1, m2), agrees in pair_data.items():
                if agrees:
                    rates.append(sum(agrees) / len(agrees))
            if rates:
                field_details[f"{dim}.{field}"] = round(sum(rates) / len(rates), 3)

    # Model completeness (how often each model produces non-empty output)
    model_completeness = {}
    for model in MODELS:
        model_completeness[model] = {
            dim: results["model_non_empty"].get(model, {}).get(dim, 0)
            for dim in ["D1", "D2", "D3", "D4", "D5"]
        }

    return {
        "n_tasks": n_tasks,
        "pairwise_agreement_by_dimension": dim_agreement,
        "per_field_agreement": field_details,
        "model_completeness": model_completeness,
    }


def print_summary(summary: Dict[str, Any]):
    """Print formatted summary."""
    print(f"\n{'=' * 80}")
    print(f"CROSS-MODEL CONSENSUS STABILITY ANALYSIS")
    print(f"Panel: DeepSeek, Claude Sonnet 4.5, Gemini 2.0 Flash")
    print(f"Tasks: {summary['n_tasks']}")
    print(f"{'=' * 80}")

    print(f"\n--- Pairwise Agreement by Dimension ---")
    print(f"{'Dim':<5} {'DS vs Claude':>14} {'DS vs Gemini':>14} {'Claude vs Gem':>14} {'Average':>10}")
    print("-" * 60)

    for dim in ["D1", "D2", "D3", "D4", "D5"]:
        data = summary["pairwise_agreement_by_dimension"].get(dim, {})
        pairs = data.get("pairs", {})
        ds_cl = pairs.get("deepseek vs claude-sonnet", {}).get("agreement_rate", 0)
        ds_gm = pairs.get("deepseek vs gemini-flash", {}).get("agreement_rate", 0)
        cl_gm = pairs.get("claude-sonnet vs gemini-flash", {}).get("agreement_rate", 0)
        avg = data.get("avg_agreement", 0)
        print(f"{dim:<5} {ds_cl:>14.3f} {ds_gm:>14.3f} {cl_gm:>14.3f} {avg:>10.3f}")

    print(f"\n--- Per-Field Agreement (avg across pairs) ---")
    for field, rate in sorted(summary["per_field_agreement"].items()):
        print(f"  {field:<30} {rate:.3f}")

    print(f"\n--- Model Completeness (non-empty field counts across {summary['n_tasks']} tasks) ---")
    for model, dims in summary["model_completeness"].items():
        total = sum(dims.values())
        print(f"  {model:<20} D1={dims.get('D1',0):>4} D2={dims.get('D2',0):>4} D3={dims.get('D3',0):>4} D4={dims.get('D4',0):>4} D5={dims.get('D5',0):>4} Total={total}")


def main():
    if not CONSENSUS_DIR.exists():
        print(f"Error: Consensus directory not found: {CONSENSUS_DIR}", file=sys.stderr)
        sys.exit(1)

    task_dirs = sorted([d for d in CONSENSUS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(task_dirs)} task directories")

    results = compute_pairwise_agreement(task_dirs)
    summary = format_results(results)

    # Print
    print_summary(summary)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "consensus_stability.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
