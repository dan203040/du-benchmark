# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Pydantic models for DU benchmark structured outputs.

Defines the M* schema (M_Q, M_S, M_U, M_F, M_C) representing
the five DU dimensions (D1-D5), plus task and result containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── D1: Query Understanding (M_Q) ──────────────────────────────────

@dataclass
class MQ:
    """D1 — Task-conditioned query understanding."""
    target_metric: str = ""
    filters: List[str] = field(default_factory=list)
    grouping_variables: List[str] = field(default_factory=list)
    output_cardinality: str = ""          # scalar | list | table | plot
    sub_questions: List[str] = field(default_factory=list)


# ── D2: Schema & Join Understanding (M_S) ──────────────────────────

@dataclass
class JoinKey:
    """A single proposed join between two sources."""
    left_source: str = ""
    right_source: str = ""
    left_column: str = ""
    right_column: str = ""
    join_type: str = "inner"              # inner | left | right | outer


@dataclass
class MS:
    """D2 — Multi-source schema understanding."""
    sources: List[str] = field(default_factory=list)
    join_keys: List[JoinKey] = field(default_factory=list)
    schema_conflicts: List[str] = field(default_factory=list)
    column_mappings: Dict[str, str] = field(default_factory=dict)


# ── D3: Semantic / Unit Understanding (M_U) ────────────────────────

@dataclass
class MU:
    """D3 — Unit labels, scale detection, semantic annotations."""
    unit_annotations: Dict[str, str] = field(default_factory=dict)   # col -> unit
    scale_issues: List[str] = field(default_factory=list)
    cross_source_conflicts: List[str] = field(default_factory=list)


# ── D4: Format Diagnosis (M_F) ─────────────────────────────────────

@dataclass
class MF:
    """D4 — File format diagnosis."""
    has_header: bool = True
    delimiter: str = ","
    encoding: str = "utf-8"
    sentinel_values: List[str] = field(default_factory=list)
    expected_columns: int = 0
    expected_rows: int = 0
    file_format: str = "csv"              # csv | json | excel | parquet | ...
    format_notes: List[str] = field(default_factory=list)


# ── D5: Analytical Constraints (M_C) ───────────────────────────────

@dataclass
class MC:
    """D5 — Derived constraints for the analytical task."""
    constraints: List[str] = field(default_factory=list)
    derived_filters: List[str] = field(default_factory=list)
    statistical_tests: List[str] = field(default_factory=list)
    output_format_requirements: List[str] = field(default_factory=list)


# ── Complete DU Output ──────────────────────────────────────────────

@dataclass
class DUOutput:
    """Complete data understanding output across all five dimensions."""
    m_q: MQ = field(default_factory=MQ)
    m_s: MS = field(default_factory=MS)
    m_u: MU = field(default_factory=MU)
    m_f: MF = field(default_factory=MF)
    m_c: MC = field(default_factory=MC)
    raw_text: str = ""                    # Original LLM / system response
    system_name: str = ""
    extraction_time_s: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DUOutput":
        """Reconstruct from a dict (e.g. loaded from JSON)."""
        du = cls()
        if "m_q" in d:
            du.m_q = MQ(**d["m_q"])
        if "m_s" in d:
            jks = d["m_s"].pop("join_keys", [])
            du.m_s = MS(**d["m_s"])
            du.m_s.join_keys = [JoinKey(**jk) for jk in jks]
        if "m_u" in d:
            du.m_u = MU(**d["m_u"])
        if "m_f" in d:
            du.m_f = MF(**d["m_f"])
        if "m_c" in d:
            du.m_c = MC(**d["m_c"])
        du.raw_text = d.get("raw_text", "")
        du.system_name = d.get("system_name", "")
        du.extraction_time_s = d.get("extraction_time_s", 0.0)
        du.token_usage = d.get("token_usage", {})
        return du


# ── Consensus Output ────────────────────────────────────────────────

@dataclass
class ConsensusField:
    """A single consensus-resolved field with confidence."""
    value: Any = None
    confidence: float = 0.0               # 0.33, 0.67, or 1.0
    votes: List[Any] = field(default_factory=list)


@dataclass
class ConsensusDU:
    """Silver-standard DU output from multi-LLM consensus."""
    m_q: Dict[str, ConsensusField] = field(default_factory=dict)
    m_s: Dict[str, ConsensusField] = field(default_factory=dict)
    m_u: Dict[str, ConsensusField] = field(default_factory=dict)
    m_f: Dict[str, ConsensusField] = field(default_factory=dict)
    m_c: Dict[str, ConsensusField] = field(default_factory=dict)
    models_used: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConsensusDU":
        obj = cls()
        for dim in ("m_q", "m_s", "m_u", "m_f", "m_c"):
            raw = d.get(dim, {})
            setattr(obj, dim, {
                k: ConsensusField(**v) for k, v in raw.items()
            })
        obj.models_used = d.get("models_used", [])
        obj.overall_confidence = d.get("overall_confidence", 0.0)
        return obj


# ── Scoring ─────────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score for a single DU dimension on a single task."""
    dimension: str                        # D1-D5
    score: float                          # 0.0 - 1.0
    method: str                           # "deterministic" | "consensus" | "end_to_end"
    confidence: float = 1.0               # consensus confidence (1.0 for deterministic)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDUScore:
    """Complete DU evaluation for one system on one task."""
    task_id: str
    system_name: str
    benchmark: str
    dimensions: List[DimensionScore] = field(default_factory=list)
    du_coverage_score: float = 0.0        # unweighted mean D1-D5
    du_quality_score: float = 0.0         # weighted: .25*D1+.25*D2+.15*D3+.15*D4+.20*D5
    structured_output: float = 0.0        # 0.0 / 0.5 / 1.0
    latency_s: float = 0.0
    token_count: int = 0

    def compute_aggregates(self):
        """Compute coverage and quality scores from dimension scores."""
        dim_scores = {ds.dimension: ds.score for ds in self.dimensions}
        n = len(dim_scores)
        if n == 0:
            return
        self.du_coverage_score = sum(dim_scores.values()) / max(n, 1)
        weights = {"D1": 0.25, "D2": 0.25, "D3": 0.15, "D4": 0.15, "D5": 0.20}
        self.du_quality_score = sum(
            weights.get(d, 0.2) * s for d, s in dim_scores.items()
        )

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ── Benchmark-Level Task Descriptor ─────────────────────────────────

class BenchmarkSource(str, Enum):
    KRAMABENCH = "kramabench"
    AGENTBENCH = "agentbench"
    DACODE = "dacode"


@dataclass
class DUBenchmarkTask:
    """Task descriptor for the DU benchmark (wraps existing benchmark tasks)."""
    task_id: str
    benchmark: BenchmarkSource
    goal: str
    data_files: List[str] = field(default_factory=list)
    category: str = ""
    difficulty: str = ""
    primary_dimensions: List[str] = field(default_factory=list)  # which D* are relevant
    file_samples: Dict[str, str] = field(default_factory=dict)   # file -> first 20 lines
    column_lists: Dict[str, List[str]] = field(default_factory=dict)
    file_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ── Cross-Cutting Properties ────────────────────────────────────────

@dataclass
class CrossCuttingMetrics:
    """Cross-cutting properties measured for each system."""
    structured_output: float = 0.0        # 0.0 / 0.5 / 1.0
    question_conditioning: float = 0.0    # Jaccard distance when Q changes
    latency_s: float = 0.0               # wall-clock for DU step only
    token_cost: int = 0                   # input + output tokens
