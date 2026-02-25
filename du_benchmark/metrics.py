# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Per-dimension scoring for DU benchmark.

Combines consensus-based scoring (D1, D3, D5) with deterministic
verification (D2, D4) to produce per-system, per-task scores.

Aggregate metrics:
  - DU Coverage Score: unweighted mean of D1-D5
  - DU Quality Score: weighted (.25*D1 + .25*D2 + .15*D3 + .15*D4 + .20*D5)
  - DU-to-Pipeline Correlation: Pearson r between quality and pass rate
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from du_benchmark.schema import (
    ConsensusDU, ConsensusField, DUBenchmarkTask, DUOutput,
    DimensionScore, TaskDUScore,
)
from du_benchmark.deterministic_verify import (
    verify_d2, verify_d4, verify_task_deterministic,
)

_logger = logging.getLogger(__name__)

# Quality score weights
DIM_WEIGHTS = {"D1": 0.25, "D2": 0.25, "D3": 0.15, "D4": 0.15, "D5": 0.20}


# ── D1: Query Understanding Scoring (Consensus) ───────────────────

def score_d1(
    du_output: DUOutput,
    consensus: ConsensusDU,
) -> DimensionScore:
    """
    Score D1 (query understanding) against consensus silver standard.

    Sub-scores:
      0.30 — target_metric match
      0.25 — filter overlap (Jaccard)
      0.25 — grouping_variables overlap (Jaccard)
      0.20 — output_cardinality match
    """
    details: Dict[str, Any] = {}
    sub_scores = []

    # Target metric
    cons_tm = consensus.m_q.get("target_metric", ConsensusField())
    s_tm = _string_similarity(du_output.m_q.target_metric, cons_tm.value or "")
    sub_scores.append(("target_metric", 0.30, s_tm, cons_tm.confidence))
    details["target_metric"] = {
        "predicted": du_output.m_q.target_metric,
        "consensus": cons_tm.value,
        "similarity": s_tm,
    }

    # Filters
    cons_f = consensus.m_q.get("filters", ConsensusField(value=[]))
    s_f = _set_jaccard(du_output.m_q.filters, cons_f.value or [])
    sub_scores.append(("filters", 0.25, s_f, cons_f.confidence))
    details["filters"] = {
        "predicted": du_output.m_q.filters,
        "consensus": cons_f.value,
        "jaccard": s_f,
    }

    # Grouping variables
    cons_g = consensus.m_q.get("grouping_variables", ConsensusField(value=[]))
    s_g = _set_jaccard(du_output.m_q.grouping_variables, cons_g.value or [])
    sub_scores.append(("grouping_variables", 0.25, s_g, cons_g.confidence))

    # Output cardinality
    cons_oc = consensus.m_q.get("output_cardinality", ConsensusField())
    s_oc = 1.0 if (
        du_output.m_q.output_cardinality.lower().strip()
        == (cons_oc.value or "").lower().strip()
    ) else 0.0
    sub_scores.append(("output_cardinality", 0.20, s_oc, cons_oc.confidence))

    # Weighted score (confidence-adjusted)
    score = 0.0
    total_weight = 0.0
    for name, weight, s, conf in sub_scores:
        effective_weight = weight * max(conf, 0.33)  # minimum weight for low-conf
        score += effective_weight * s
        total_weight += effective_weight
    final = score / total_weight if total_weight > 0 else 0.0

    details["sub_scores"] = {
        name: {"raw": s, "weight": w, "confidence": c}
        for name, w, s, c in sub_scores
    }

    return DimensionScore(
        dimension="D1", score=round(final, 3), method="consensus",
        confidence=_avg_confidence(consensus.m_q),
        details=details,
    )


# ── D3: Semantic/Unit Understanding Scoring (Consensus) ────────────

def score_d3(
    du_output: DUOutput,
    consensus: ConsensusDU,
) -> DimensionScore:
    """
    Score D3 (semantic/unit understanding) against consensus.

    Sub-scores:
      0.50 — unit_annotations overlap (dict key match + value match)
      0.25 — scale_issues mentioned
      0.25 — cross_source_conflicts mentioned
    """
    details: Dict[str, Any] = {}

    # Unit annotations
    cons_ua = consensus.m_u.get("unit_annotations", ConsensusField(value={}))
    s_ua = _dict_overlap(du_output.m_u.unit_annotations, cons_ua.value or {})
    details["unit_annotations_overlap"] = s_ua

    # Scale issues
    cons_si = consensus.m_u.get("scale_issues", ConsensusField(value=[]))
    s_si = _set_jaccard(du_output.m_u.scale_issues, cons_si.value or [])

    # Cross-source conflicts
    cons_cc = consensus.m_u.get("cross_source_conflicts", ConsensusField(value=[]))
    s_cc = _set_jaccard(
        du_output.m_u.cross_source_conflicts, cons_cc.value or [],
    )

    # Coverage bonus: did the system mention units at all?
    coverage_bonus = 0.0
    if du_output.m_u.unit_annotations:
        coverage_bonus = 0.1  # Partial credit for attempting D3

    score = 0.50 * s_ua + 0.25 * s_si + 0.25 * s_cc + coverage_bonus
    score = min(score, 1.0)

    return DimensionScore(
        dimension="D3", score=round(score, 3), method="consensus",
        confidence=_avg_confidence(consensus.m_u),
        details=details,
    )


# ── D5: Analytical Constraints Scoring (Consensus + End-to-End) ────

def score_d5(
    du_output: DUOutput,
    consensus: ConsensusDU,
) -> DimensionScore:
    """
    Score D5 (analytical constraints) against consensus.

    Sub-scores:
      0.40 — constraints overlap
      0.25 — derived_filters overlap
      0.20 — statistical_tests overlap
      0.15 — output_format_requirements overlap
    """
    cons_c = consensus.m_c.get("constraints", ConsensusField(value=[]))
    s_c = _set_jaccard(du_output.m_c.constraints, cons_c.value or [])

    cons_df = consensus.m_c.get("derived_filters", ConsensusField(value=[]))
    s_df = _set_jaccard(du_output.m_c.derived_filters, cons_df.value or [])

    cons_st = consensus.m_c.get("statistical_tests", ConsensusField(value=[]))
    s_st = _set_jaccard(du_output.m_c.statistical_tests, cons_st.value or [])

    cons_ofr = consensus.m_c.get(
        "output_format_requirements", ConsensusField(value=[]),
    )
    s_ofr = _set_jaccard(
        du_output.m_c.output_format_requirements, cons_ofr.value or [],
    )

    score = 0.40 * s_c + 0.25 * s_df + 0.20 * s_st + 0.15 * s_ofr

    return DimensionScore(
        dimension="D5", score=round(score, 3), method="consensus",
        confidence=_avg_confidence(consensus.m_c),
        details={
            "constraints_jaccard": s_c,
            "derived_filters_jaccard": s_df,
            "statistical_tests_jaccard": s_st,
            "output_format_jaccard": s_ofr,
        },
    )


# ── Cross-Cutting: Structured Output Score ──────────────────────────

def score_structured_output(du_output: DUOutput) -> float:
    """
    Score whether the system produces structured output.
    1.0 = typed JSON/dict, 0.5 = semi-structured, 0.0 = free text only.
    """
    # Check if M fields are populated (not default empty)
    has_mq = bool(du_output.m_q.target_metric or du_output.m_q.filters)
    has_ms = bool(du_output.m_s.sources or du_output.m_s.join_keys)
    has_mf = bool(du_output.m_f.expected_columns > 0)
    has_mc = bool(du_output.m_c.constraints)

    populated = sum([has_mq, has_ms, has_mf, has_mc])
    if populated >= 3:
        return 1.0
    elif populated >= 1:
        return 0.5
    return 0.0


# ── Full Task Scoring ──────────────────────────────────────────────

def score_task(
    task: DUBenchmarkTask,
    du_output: DUOutput,
    consensus: ConsensusDU,
    dataframes: Dict[str, pd.DataFrame],
    file_paths: Optional[Dict[str, str]] = None,
) -> TaskDUScore:
    """
    Compute full DU score for one system on one task.

    Uses deterministic verification for D2/D4, consensus for D1/D3/D5.
    """
    dimensions = []

    # D1: Consensus-based
    d1 = score_d1(du_output, consensus)
    dimensions.append(d1)

    # D2: Deterministic join verification
    d2 = verify_d2(du_output, dataframes)
    dimensions.append(d2)

    # D3: Consensus-based
    d3 = score_d3(du_output, consensus)
    dimensions.append(d3)

    # D4: Deterministic format verification
    if file_paths:
        d4_scores = []
        for fname, fpath in file_paths.items():
            d4 = verify_d4(du_output, fpath, dataframes.get(fname))
            d4_scores.append(d4.score)
        avg_d4 = sum(d4_scores) / len(d4_scores) if d4_scores else 0.0
        dimensions.append(DimensionScore(
            dimension="D4", score=round(avg_d4, 3), method="deterministic",
            details={"per_file_scores": d4_scores},
        ))
    else:
        # Fall back to consensus for D4
        cons_hdr = consensus.m_f.get("has_header", ConsensusField())
        cons_delim = consensus.m_f.get("delimiter", ConsensusField())
        s_hdr = 1.0 if du_output.m_f.has_header == (cons_hdr.value or True) else 0.0
        s_delim = 1.0 if du_output.m_f.delimiter == (cons_delim.value or ",") else 0.0
        d4_score = 0.5 * s_hdr + 0.5 * s_delim
        dimensions.append(DimensionScore(
            dimension="D4", score=round(d4_score, 3), method="consensus",
            confidence=_avg_confidence(consensus.m_f),
        ))

    # D5: Consensus-based
    d5 = score_d5(du_output, consensus)
    dimensions.append(d5)

    # Build TaskDUScore
    result = TaskDUScore(
        task_id=task.task_id,
        system_name=du_output.system_name,
        benchmark=task.benchmark.value,
        dimensions=dimensions,
        structured_output=score_structured_output(du_output),
        latency_s=du_output.extraction_time_s,
        token_count=sum(du_output.token_usage.values()),
    )
    result.compute_aggregates()

    return result


# ── Aggregate Scoring ──────────────────────────────────────────────

def aggregate_scores(
    task_scores: List[TaskDUScore],
) -> Dict[str, Any]:
    """Compute aggregate metrics across all tasks for a system."""
    if not task_scores:
        return {}

    system_name = task_scores[0].system_name
    n = len(task_scores)

    # Per-dimension averages
    dim_scores: Dict[str, List[float]] = {f"D{i}": [] for i in range(1, 6)}
    for ts in task_scores:
        for ds in ts.dimensions:
            if ds.dimension in dim_scores:
                dim_scores[ds.dimension].append(ds.score)

    dim_avgs = {
        d: round(sum(scores) / len(scores), 3) if scores else 0.0
        for d, scores in dim_scores.items()
    }

    # Overall aggregates
    coverage_scores = [ts.du_coverage_score for ts in task_scores]
    quality_scores = [ts.du_quality_score for ts in task_scores]

    return {
        "system_name": system_name,
        "n_tasks": n,
        "per_dimension": dim_avgs,
        "du_coverage_score": round(sum(coverage_scores) / n, 3),
        "du_quality_score": round(sum(quality_scores) / n, 3),
        "avg_latency_s": round(
            sum(ts.latency_s for ts in task_scores) / n, 1,
        ),
        "avg_tokens": round(
            sum(ts.token_count for ts in task_scores) / n,
        ),
        "structured_output_rate": round(
            sum(1 for ts in task_scores if ts.structured_output >= 0.5) / n, 3,
        ),
    }


def compute_correlation(
    du_quality_scores: List[float],
    pipeline_pass_rates: List[float],
) -> Dict[str, float]:
    """Compute Pearson r between DU quality and pipeline success."""
    import numpy as np

    if len(du_quality_scores) < 3 or len(du_quality_scores) != len(pipeline_pass_rates):
        return {"pearson_r": 0.0, "p_value": 1.0, "n": 0}

    from scipy import stats
    r, p = stats.pearsonr(du_quality_scores, pipeline_pass_rates)

    return {
        "pearson_r": round(float(r), 3),
        "p_value": round(float(p), 4),
        "n": len(du_quality_scores),
    }


def bootstrap_ci(
    scores: List[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval. Returns (mean, lower, upper)."""
    import numpy as np

    if not scores:
        return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(42)
    arr = np.array(scores)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(means, 100 * alpha))
    upper = float(np.percentile(means, 100 * (1 - alpha)))
    return (round(float(arr.mean()), 3), round(lower, 3), round(upper, 3))


def mcnemar_test(
    system_a_correct: List[bool],
    system_b_correct: List[bool],
) -> Dict[str, float]:
    """McNemar's test for pairwise system comparison."""
    if len(system_a_correct) != len(system_b_correct):
        return {"chi2": 0.0, "p_value": 1.0}

    # Build contingency table
    b_c = sum(
        1 for a, b in zip(system_a_correct, system_b_correct)
        if not a and b
    )
    c_b = sum(
        1 for a, b in zip(system_a_correct, system_b_correct)
        if a and not b
    )

    if b_c + c_b == 0:
        return {"chi2": 0.0, "p_value": 1.0}

    chi2 = (abs(b_c - c_b) - 1) ** 2 / (b_c + c_b)

    from scipy import stats
    p_value = 1 - stats.chi2.cdf(chi2, 1)

    return {"chi2": round(chi2, 3), "p_value": round(p_value, 4)}


# ── Helpers ────────────────────────────────────────────────────────

def _string_similarity(a: str, b: str) -> float:
    """Simple normalized string similarity."""
    if not a and not b:
        return 0.0  # No credit for empty-vs-empty (vacuous agreement)
    if not a or not b:
        return 0.0
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    if a_lower == b_lower:
        return 1.0
    # Containment check
    if a_lower in b_lower or b_lower in a_lower:
        return 0.7
    # Token overlap
    a_tokens = set(a_lower.split())
    b_tokens = set(b_lower.split())
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return overlap / union if union > 0 else 0.0


def _set_jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity for two lists of strings (case-insensitive)."""
    if not a and not b:
        return 0.0  # No credit for empty-vs-empty (vacuous agreement)
    if not a or not b:
        return 0.0
    set_a = {s.lower().strip() for s in a}
    set_b = {s.lower().strip() for s in b}
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _dict_overlap(a: Dict[str, str], b: Dict[str, str]) -> float:
    """Overlap score for two dicts (key match + value match)."""
    if not a and not b:
        return 0.0  # No credit for empty-vs-empty (vacuous agreement)
    if not a or not b:
        return 0.0
    a_norm = {k.lower().strip(): v.lower().strip() for k, v in a.items()}
    b_norm = {k.lower().strip(): v.lower().strip() for k, v in b.items()}
    common_keys = set(a_norm) & set(b_norm)
    all_keys = set(a_norm) | set(b_norm)
    if not all_keys:
        return 0.0
    key_overlap = len(common_keys) / len(all_keys)
    value_match = sum(
        1 for k in common_keys if a_norm[k] == b_norm[k]
    ) / max(len(common_keys), 1)
    return 0.5 * key_overlap + 0.5 * value_match


def _avg_confidence(dim_fields: Dict[str, ConsensusField]) -> float:
    """Average confidence across all fields in a dimension."""
    if not dim_fields:
        return 0.0
    confs = [cf.confidence for cf in dim_fields.values()]
    return sum(confs) / len(confs) if confs else 0.0
