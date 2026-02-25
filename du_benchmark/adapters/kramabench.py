# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
KRAMABENCH adapter for ADP-MA evaluation.

KRAMABENCH (arxiv:2506.06541) is a benchmark for data science pipelines
that tests multi-step reasoning over heterogeneous data sources.

This adapter uses the official KramaBench metrics (reimplemented to avoid
external dependencies like nltk/rouge_score).  Metric formulas are taken
directly from ``benchmark/metrics.py`` in the KramaBench repository.

Official answer_type → metric mapping (from answer_type_fixtures.json):
    numeric_exact       → success
    numeric_approximate → mean_absolute_error, mean_squared_error, rae_score
    string_exact        → success
    string_approximate  → llm_paraphrase  (requires OpenAI, optional)
    list_exact          → f1, precision, recall
    list_approximate    → f1_approximate   (requires OpenAI, optional)

Overall KramaBench score =
    sum(support × mean_metric) / total_support × 100
    across metrics: success, llm_paraphrase, rae_score, f1, f1_approximate
"""

import glob as globmod
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import pandas as pd

from .base import ExternalBenchmarkAdapter, BenchmarkTask, BenchmarkResult

logger = logging.getLogger(__name__)

# Optional ADP-MA data_tools — fall back to basic pandas readers when not available
try:
    from meta_agents_adk.tools.data_tools import (
        read_csv_robust_multi, read_excel_robust,
        read_geopackage, read_tle, read_sp3, read_cdf, read_structured_text,
    )
    _HAS_DATA_TOOLS = True
except ImportError:
    _HAS_DATA_TOOLS = False
    read_csv_robust_multi = None
    read_excel_robust = None
    read_geopackage = None
    read_tle = None
    read_sp3 = None
    read_cdf = None
    read_structured_text = None

# ---------------------------------------------------------------------------
# Answer-type → metrics mapping (mirrors answer_type_fixtures.json)
# ---------------------------------------------------------------------------

ANSWER_TYPE_METRICS: Dict[str, List[str]] = {
    "numeric_exact": ["success"],
    "numeric_approximate": ["mean_absolute_error", "mean_squared_error", "rae_score"],
    "string_exact": ["success"],
    "string_approximate": ["llm_paraphrase"],
    "list_exact": ["f1", "precision", "recall"],
    "list_approximate": ["f1_approximate"],
}

# Metrics used for the aggregate KramaBench score
AGGREGATE_METRICS = {"success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"}


# ---------------------------------------------------------------------------
# Official KramaBench metric implementations
# ---------------------------------------------------------------------------

def _str_to_float(s) -> float:
    """Convert string to float, handling percentage notation."""
    if isinstance(s, str):
        s = s.strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100
        return float(s)
    return float(s)


def metric_success(predicted, target, rel_tol: float = 1e-2) -> Optional[float]:
    """Binary exact-match.  Mirrors ``Success.__call__`` in KramaBench.

    Uses *rel_tol* (default 1e-2 = 1%) relative tolerance for numeric
    comparisons, matching the official KramaBench evaluation threshold.
    """
    try:
        # Coerce bools to strings so True/"True"/"Yes" all compare correctly
        if isinstance(predicted, (bool,)) or (hasattr(predicted, 'item') and isinstance(predicted.item(), bool)):
            predicted = _normalize_bool(predicted)
        if isinstance(target, (bool,)) or (hasattr(target, 'item') and isinstance(target.item(), bool)):
            target = _normalize_bool(target)

        if isinstance(target, float):
            pred_f = _str_to_float(predicted)
            rae = abs(pred_f - target) / abs(target) if target != 0 else abs(pred_f)
            rae_pct = abs(pred_f * 100 - target) / abs(target) if target != 0 else abs(pred_f * 100)
            return 1.0 if (rae < rel_tol or rae_pct < rel_tol) else 0.0
        elif isinstance(target, str) and isinstance(predicted, str):
            import unicodedata, re as _re
            # Normalize boolean synonyms (True/False ↔ Yes/No) before comparing
            predicted = _normalize_bool(predicted)
            target = _normalize_bool(target)
            p = unicodedata.normalize("NFKD", predicted.strip().lower())
            t = unicodedata.normalize("NFKD", target.strip().lower())
            # Try exact match first, then ASCII-folded match
            if p == t:
                return 1.0
            p_ascii = p.encode("ascii", "ignore").decode()
            t_ascii = t.encode("ascii", "ignore").decode()
            if p_ascii == t_ascii:
                return 1.0
            # Punctuation-normalized match: collapse commas/semicolons and
            # extra whitespace so minor formatting differences (e.g.
            # "Anaheim, CA" vs "Anaheim CA") don't cause false negatives.
            p_norm = _re.sub(r"[,;]+", " ", p_ascii)
            t_norm = _re.sub(r"[,;]+", " ", t_ascii)
            p_norm = _re.sub(r"\s+", " ", p_norm).strip()
            t_norm = _re.sub(r"\s+", " ", t_norm).strip()
            if p_norm == t_norm:
                return 1.0
            # Fallback: strip trailing parenthetical content
            # from the predicted answer only (not the target)
            p_stripped = _re.sub(r"\s*\([^)]*\)\s*$", "", p_norm)
            p_stripped = _re.sub(r"\s*:\s*[\d,.]+\s*$", "", p_stripped)
            p_stripped = _re.sub(r"\s+", " ", p_stripped).strip()
            if p_stripped and p_stripped == t_norm:
                return 1.0
            # Strip leading "Label: " prefix from predicted answer
            if ":" in p_norm:
                after_colon = p_norm.split(":", 1)[1].strip()
                after_colon = _re.sub(r"\s+", " ", after_colon).strip()
                if after_colon and after_colon == t_norm:
                    return 1.0
            # If target is short (<=5 words), check if predicted starts with target
            t_words = t_norm.split()
            if 1 <= len(t_words) <= 5 and p_norm.startswith(t_norm) and len(p_norm) > len(t_norm):
                return 1.0
            # K1.3: Split on common connectors and check first part
            for sep in [': ', ' - ', ' — ']:
                if sep in p_norm:
                    prefix = p_norm.split(sep, 1)[0].strip()
                    if prefix and prefix == t_norm:
                        return 1.0
            return 0.0
        elif isinstance(target, int):
            try:
                return 1.0 if int(float(str(predicted))) == target else 0.0
            except (ValueError, TypeError):
                return 0.0
        else:
            return 1.0 if predicted == target else 0.0
    except Exception as e:
        logger.warning("Success metric error: %s", e)
        return 0.0


def metric_mae(predicted, target) -> Optional[float]:
    """Mean Absolute Error (per-item)."""
    try:
        return abs(_str_to_float(predicted) - _str_to_float(target))
    except Exception:
        return None


def metric_mse(predicted, target) -> Optional[float]:
    """Mean Squared Error (per-item)."""
    try:
        diff = _str_to_float(predicted) - _str_to_float(target)
        return diff * diff
    except Exception:
        return None


def metric_rae_score(predicted, target) -> Optional[float]:
    """RAE Score = 1 / (1 + |pred-target|/|target|).  Range [0, 1]."""
    try:
        pred_f = _str_to_float(predicted)
        tgt_f = _str_to_float(target)
        rae = abs(pred_f - tgt_f) / abs(tgt_f) if tgt_f != 0 else abs(pred_f)
        return 1.0 / (1.0 + rae)
    except Exception:
        return 0.0


def _list_f1(predicted, target, approx: bool = False) -> Tuple[float, float, float]:
    """Compute F1, precision, recall for list answers.

    When *approx* is False, matching is exact (case-insensitive for
    strings, 1e-6 relative error for numerics).  When *approx* is True,
    uses 1% relative error for numerics (LLM paraphrase for strings is
    skipped since it requires an API key).
    """
    if isinstance(target, str):
        target = json.loads(target)
    if isinstance(predicted, str):
        try:
            predicted = json.loads(predicted)
        except json.JSONDecodeError:
            try:
                predicted = eval(predicted)
            except Exception:
                predicted = [predicted]
    if not isinstance(predicted, list):
        predicted = list(predicted) if hasattr(predicted, "__iter__") else [predicted]
    if not isinstance(target, list):
        target = list(target) if hasattr(target, "__iter__") else [target]

    if len(target) == 0:
        score = 1.0 if len(predicted) == 0 else 0.0
        return score, score, score

    numeric_tol = 0.01 if approx else 1e-6

    matched_pred: set = set()
    recall_cnt = 0

    for t in target:
        found = False
        for j, p in enumerate(predicted):
            if isinstance(t, str):
                if str(p).strip().lower() == t.strip().lower():
                    found = True
                    matched_pred.add(j)
                    break
            elif isinstance(t, (float, int)):
                try:
                    pp = p if isinstance(p, (float, int)) else _str_to_float(p)
                except Exception:
                    continue
                denom = max(abs(t), 1e-12)
                if abs(pp - t) / denom < numeric_tol:
                    found = True
                    matched_pred.add(j)
                    break
            else:
                if p == t:
                    found = True
                    matched_pred.add(j)
                    break
        if found:
            recall_cnt += 1

    recall = recall_cnt / len(target)
    precision = len(matched_pred) / len(predicted) if predicted else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


# ---------------------------------------------------------------------------
# LLM Paraphrase metric
# ---------------------------------------------------------------------------

# Few-shot prompt replicating the official KramaBench paraphrase evaluation
# (from benchmark/llm_tools/prompts.py)
_PARAPHRASE_FEWSHOT: List[Dict[str, str]] = [
    {"role": "user", "content": 'You will receive two sentences A and B. Do these two sentences mean the same thing? Answer with only one word "yes" or "no".'},
    {"role": "assistant", "content": "Please provide the sentences for me to evaluate."},
    {"role": "user", "content": 'A: "Amrozi accused his brother, whom he called \\"the witness\\", of deliberately distorting his evidence."; B: "Amrozi accused his brother, whom he disparagingly referred to as \'the liar witness\', of intentionally twisting his testimony."'},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": 'A: "Pennmakkal is an Indian Malayalam film from 1966, produced by J. Sasikumar and directed by KP Kottarakkara."; B: "The Indian Malayalam film \'Pennmakkal\', released in 1966, was produced by J. Sasikumar and directed by KP Kottarakkara."'},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": 'A: "Sorkin , who faces charges of conspir- acy to obstruct justice and lying to a grand jury , was to have been tried separately."; B: "Despite being accused of conspiring to obstruct justice and perjury, Sorkin was supposed to stand trial on his own."'},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": 'A: "Gilroy police and FBI agents described Gehring as cooperative , but said Saturday that he had revealed nothing about what had happened to the children ."; B: "Although Gilroy police and FBI agents reported that Gehring was cooperative , he hadn\'t disclosed any information about the children\'s whereabouts or what had happened to them as of Saturday ."'},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": 'A: "Whereas \u201ce\u201d the electric charge of the particle and A is the magnetic vector potential of the electromagnetic field."; B: "The electric charge of the particle is denoted by \u201ce\u201d, and the magnetic vector potential of the electromagnetic field is denoted by \'A\'."'},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": 'A: "The Jidanul River is a tributary of the Jiul de Vest River in Romania."; B: "The Jidanul River is a mere insignificant stream that flows into the grand Jiul de Vest River in Romania."'},
    {"role": "assistant", "content": "No"},
]


def _llm_paraphrase_heuristic(predicted: str, target: str) -> float:
    """Heuristic fallback for paraphrase detection (no LLM needed).

    Uses normalised string containment and, when available, Levenshtein
    ratio via ``rapidfuzz``.
    """
    import re as _re
    p = str(predicted).strip().lower()
    t = str(target).strip().lower()
    if p == t:
        return 1.0
    # Containment check
    if p and t and (p in t or t in p):
        shorter, longer = (p, t) if len(p) <= len(t) else (t, p)
        if len(shorter) / max(len(longer), 1) > 0.6:
            return 1.0
    # Strip "Label: value" prefix — if text after colon matches target
    if ":" in p:
        after_colon = p.split(":", 1)[1].strip()
        after_colon = _re.sub(r"\s+", " ", after_colon).strip()
        if after_colon and after_colon == t:
            return 1.0
    # Short target prefix: if target is <=5 words and predicted starts with it
    t_words = t.split()
    if 1 <= len(t_words) <= 5 and p.startswith(t) and len(p) > len(t):
        return 1.0
    # Levenshtein via rapidfuzz (already in requirements.txt)
    try:
        from rapidfuzz import fuzz
        ratio = fuzz.ratio(p, t) / 100.0
        return 1.0 if ratio >= 0.85 else 0.0
    except ImportError:
        pass
    return 0.0


def _resolve_paraphrase_llm_config():
    """Resolve which LLM provider/model to use for paraphrase evaluation.

    Checks environment variables in order:
      1. DEEPSEEK_API_KEY  → DeepSeek (deepseek-chat)
      2. OPENAI_API_KEY    → OpenAI  (gpt-4o-mini)

    Returns (api_key, base_url, model) or (None, None, None) if no key is set.
    """
    dk = os.environ.get("DEEPSEEK_API_KEY")
    if dk:
        return dk, "https://api.deepseek.com", os.environ.get("PARAPHRASE_MODEL", "deepseek-chat")

    ok = os.environ.get("OPENAI_API_KEY")
    if ok:
        return ok, None, os.environ.get("PARAPHRASE_MODEL", "gpt-4o-mini")

    return None, None, None


def _llm_paraphrase(predicted, target) -> Optional[float]:
    """LLM-based paraphrase metric replicating official KramaBench behaviour.

    Supports OpenAI and DeepSeek (or any OpenAI-compatible provider).
    Set DEEPSEEK_API_KEY or OPENAI_API_KEY in the environment.
    Optionally override the model with PARAPHRASE_MODEL env var.
    Falls back to a heuristic when no API key is available.
    """
    pred_s = str(predicted).strip()
    tgt_s = str(target).strip()
    if not pred_s or not tgt_s:
        return 0.0

    api_key, base_url, model = _resolve_paraphrase_llm_config()
    if api_key:
        try:
            import openai
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)
            messages = list(_PARAPHRASE_FEWSHOT) + [
                {"role": "user", "content": f'A: "{tgt_s}"; B: "{pred_s}"'}
            ]
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=5,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content.strip().lower()
            if answer.startswith("yes"):
                return 1.0
            elif answer.startswith("no"):
                return 0.0
            else:
                logger.warning("Unexpected paraphrase LLM response: %s", answer)
                return _llm_paraphrase_heuristic(pred_s, tgt_s)
        except Exception as exc:
            logger.warning("LLM paraphrase call failed (%s), using heuristic", exc)
            return _llm_paraphrase_heuristic(pred_s, tgt_s)

    # No API key — use heuristic
    return _llm_paraphrase_heuristic(pred_s, tgt_s)


# Metric dispatch table
METRIC_FUNCTIONS = {
    "success": lambda p, t: metric_success(p, t),
    "mean_absolute_error": lambda p, t: metric_mae(p, t),
    "mean_squared_error": lambda p, t: metric_mse(p, t),
    "rae_score": lambda p, t: metric_rae_score(p, t),
    "f1": lambda p, t: _list_f1(p, t, approx=False)[0],
    "precision": lambda p, t: _list_f1(p, t, approx=False)[1],
    "recall": lambda p, t: _list_f1(p, t, approx=False)[2],
    "f1_approximate": lambda p, t: _list_f1(p, t, approx=True)[0],
    "llm_paraphrase": _llm_paraphrase,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_bool(val) -> str:
    """Map True/False to Yes/No for string answer comparison."""
    sv = str(val).strip().lower()
    if sv in ("true", "yes"):
        return "Yes"
    if sv in ("false", "no"):
        return "No"
    return str(val).strip()


def compute_kramabench_score(
    task_evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the official KramaBench aggregate score.

    The formula (from ``evaluate.py``):
        total_score = sum(support × mean_metric)
                      for metric in {success, llm_paraphrase, rae_score,
                                     f1, f1_approximate}
        total_support = sum(total_support)
        score = total_score / total_support × 100

    Returns a dict with per-metric aggregation and the overall score.
    """
    # Collect per-metric values across all tasks
    metric_values: Dict[str, List[float]] = {}
    metric_support: Dict[str, int] = {}  # total tasks that use this metric

    for ev in task_evaluations:
        answer_type = ev.get("answer_type", "")
        metrics_for_type = ANSWER_TYPE_METRICS.get(answer_type, [])
        for m in metrics_for_type:
            if m not in metric_support:
                metric_support[m] = 0
                metric_values[m] = []
            metric_support[m] += 1
            val = ev.get("kramabench_metrics", {}).get(m)
            if val is not None:
                metric_values[m].append(float(val))

    # Aggregate
    per_metric = {}
    total_score = 0.0
    total_support = 0
    for m, values in metric_values.items():
        n_computed = len(values)
        mean_val = sum(values) / n_computed if n_computed > 0 else 0.0
        std_val = (
            math.sqrt(sum((v - mean_val) ** 2 for v in values) / n_computed)
            if n_computed > 1
            else 0.0
        )
        per_metric[m] = {
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "sum": round(sum(values), 6),
            "support": n_computed,
            "total_support": metric_support[m],
        }
        if m in AGGREGATE_METRICS:
            total_support += metric_support[m]
            total_score += n_computed * mean_val

    overall = round(total_score / total_support * 100, 2) if total_support > 0 else 0.0

    return {
        "kramabench_score": overall,
        "per_metric": per_metric,
        "total_support": total_support,
    }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class KramaBenchAdapter(ExternalBenchmarkAdapter):
    """
    Adapter for KRAMABENCH benchmark.

    KRAMABENCH tasks involve:
    - Multi-step data processing pipelines
    - Heterogeneous data sources (CSV, JSON, text files)
    - Numeric and categorical answer types
    """

    @property
    def benchmark_name(self) -> str:
        return "KRAMABENCH"

    def __init__(self, benchmark_dir: Path, output_dir: Path):
        super().__init__(benchmark_dir, output_dir)
        self.workload_dir = self.benchmark_dir / "workload"
        self.data_dir = self.benchmark_dir / "data"
        self._tasks_cache: Dict[str, Dict] = {}

    def _load_workload_files(self) -> Dict[str, List[Dict]]:
        """Load all workload JSON files."""
        workloads = {}

        for workload_file in self.workload_dir.glob("*.json"):
            if workload_file.name.startswith("quick-start"):
                continue  # Skip quickstart

            with open(workload_file) as f:
                try:
                    tasks = json.load(f)
                    workloads[workload_file.stem] = tasks
                except json.JSONDecodeError:
                    continue

        return workloads

    def list_tasks(self) -> List[str]:
        """List all available task IDs."""
        task_ids = []

        workloads = self._load_workload_files()
        for domain, tasks in workloads.items():
            for task in tasks:
                task_id = task.get("id", "")
                if task_id:
                    task_ids.append(task_id)
                    self._tasks_cache[task_id] = {
                        "domain": domain,
                        "task": task
                    }

        return sorted(task_ids)

    def load_task(self, task_id: str) -> BenchmarkTask:
        """Load a task by ID."""
        if task_id not in self._tasks_cache:
            self.list_tasks()  # Populate cache

        if task_id not in self._tasks_cache:
            raise ValueError(f"Task {task_id} not found in KRAMABENCH")

        cached = self._tasks_cache[task_id]
        task_data = cached["task"]
        domain = cached["domain"]

        # Determine difficulty from task ID
        if "-easy-" in task_id:
            difficulty = "easy"
        elif "-medium-" in task_id:
            difficulty = "medium"
        elif "-hard-" in task_id:
            difficulty = "hard"
        else:
            difficulty = "medium"

        # Build goal from query
        goal = task_data.get("query", "")

        # Resolve data files: data_sources are bare filenames or globs
        # that live somewhere under data/<domain>/input/
        data_sources = task_data.get("data_sources", [])
        resolved_files = self._resolve_data_sources(domain, data_sources)

        # Expected output
        answer = task_data.get("answer")
        answer_type = task_data.get("answer_type", "string_exact")

        return BenchmarkTask(
            task_id=task_id,
            name=task_id.replace("-", " ").title(),
            description=goal,
            goal=goal,
            difficulty=difficulty,
            category=domain,
            data_files=resolved_files,
            expected_output={
                "answer": answer,
                "answer_type": answer_type,
                "subtasks": task_data.get("subtasks", []),
            },
            source_benchmark="KRAMABENCH",
            original_config=task_data,
        )

    # Common typo corrections for directory/file names in workloads
    _TYPO_FIXES = {
        "identitiy": "identity",
    }

    def _normalize_name(self, name: str) -> str:
        """Normalise a filename/directory name for fuzzy matching."""
        result = name.lower()
        for typo, fix in self._TYPO_FIXES.items():
            result = result.replace(typo, fix)
        return result

    def _fuzzy_rglob(
        self, base_dir: Path, target: str
    ) -> List[Path]:
        """Search for *target* under *base_dir* with fuzzy fallbacks.

        1. Exact ``rglob(target)``
        2. Case-insensitive match
        3. Typo-corrected match
        4. Extension-stem match (``.txt`` ↔ ``.text``)
        5. Levenshtein closest match (if rapidfuzz available)
        """
        if not base_dir.exists():
            return []

        # 1. Exact
        found = list(base_dir.rglob(target))
        if found:
            return found

        target_norm = self._normalize_name(target)
        target_stem = Path(target).stem.lower()

        best_candidates: List[Path] = []
        for p in base_dir.rglob("*"):
            if not p.is_file() and not p.is_dir():
                continue
            rel = str(p.relative_to(base_dir))
            # 2. Case-insensitive
            if rel.lower() == target.lower():
                return [p]
            # 3. Typo-corrected
            if self._normalize_name(rel) == target_norm:
                return [p]
            # 4. Name match — same stem, different extension (.txt vs .text)
            if p.is_file() and p.stem.lower() == target_stem:
                best_candidates.append(p)

        if best_candidates:
            logger.info("Fuzzy-matched '%s' → '%s'", target, best_candidates[0])
            return [best_candidates[0]]

        # 5. Date-prefix match — for files like omni2-wu334-<dates>.csv where
        #    the workload references a different date range than what's on disk
        date_prefix = re.sub(r'-\d{8}_to_\d{8}', '', Path(target).stem.lower())
        if date_prefix != Path(target).stem.lower():
            # Target contains a date range — try matching by prefix only
            all_files = [p for p in base_dir.rglob("*") if p.is_file()]
            prefix_matches = []
            for p in all_files:
                p_prefix = re.sub(r'-\d{8}_to_\d{8}', '', p.stem.lower())
                if p_prefix == date_prefix and p.suffix.lower() == Path(target).suffix.lower():
                    prefix_matches.append(p)
            if prefix_matches:
                logger.info(
                    "Date-prefix-matched '%s' → '%s'",
                    target, prefix_matches[0].name,
                )
                return [prefix_matches[0]]

        # 6. Levenshtein closest match (files only)
        try:
            from rapidfuzz import fuzz
            all_files = all_files if 'all_files' in dir() else [
                p for p in base_dir.rglob("*") if p.is_file()
            ]
            scored = [
                (p, fuzz.ratio(target.lower(), p.name.lower()))
                for p in all_files
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored and scored[0][1] >= 75:
                logger.info(
                    "Levenshtein-matched '%s' → '%s' (score=%.0f)",
                    target, scored[0][0].name, scored[0][1],
                )
                return [scored[0][0]]
        except ImportError:
            pass

        return []

    def _resolve_data_sources(
        self, domain: str, data_sources: List[str]
    ) -> List[str]:
        """Resolve data source names to actual file paths.

        Data sources in workload JSON are bare filenames or glob patterns
        (e.g. ``"State MSA Identity Theft Data/*"``).  The actual files
        live under ``data/<domain>/input/`` in a nested directory
        structure.  This method searches recursively to find them,
        with fuzzy fallbacks for typos, case mismatches, and extension
        differences.
        """
        domain_input_dir = self.data_dir / domain / "input"
        if not domain_input_dir.exists():
            # Fallback: try the domain name with variant spellings
            for candidate in self.data_dir.iterdir():
                if candidate.is_dir() and candidate.name.lower().startswith(domain[:4].lower()):
                    domain_input_dir = candidate / "input"
                    break
            if not domain_input_dir.exists():
                logger.warning(
                    "Data directory not found for domain '%s' (tried %s)",
                    domain, domain_input_dir,
                )

        resolved: List[str] = []
        for ds in data_sources:
            # If it's a glob pattern (contains *)
            if "*" in ds:
                pattern = str(domain_input_dir / "**" / ds)
                matches = globmod.glob(pattern, recursive=True)
                if not matches:
                    # Try just the directory name portion
                    dir_part = ds.replace("/*", "").replace("\\*", "")
                    # Try exact glob first
                    pattern = str(domain_input_dir / "**" / dir_part)
                    for d in globmod.glob(pattern, recursive=True):
                        if Path(d).is_dir():
                            matches = [
                                str(f) for f in Path(d).iterdir() if f.is_file()
                            ]
                            break
                    # Fuzzy directory match
                    if not matches:
                        fuzzy_dirs = self._fuzzy_rglob(domain_input_dir, dir_part)
                        for fd in fuzzy_dirs:
                            if fd.is_dir():
                                matches = [
                                    str(f) for f in fd.iterdir() if f.is_file()
                                ]
                                break
                if not matches:
                    logger.warning(
                        "No files found for glob pattern '%s' in %s", ds, domain_input_dir,
                    )
                resolved.extend(sorted(matches))
            else:
                # Search recursively with fuzzy fallbacks
                found = self._fuzzy_rglob(domain_input_dir, ds)
                if found:
                    resolved.append(str(found[0]))
                else:
                    logger.warning(
                        "Data source '%s' not found under %s", ds, domain_input_dir,
                    )
                    # Last resort: use the original path construction
                    resolved.append(str(self.data_dir / ds))

        return resolved

    def get_task_data(self, task: BenchmarkTask) -> Dict[str, pd.DataFrame]:
        """Load input data for a task."""
        data = {}

        for data_path in task.data_files:
            path = Path(data_path)
            if not path.exists():
                logger.warning("Data file does not exist: %s", path)
                continue

            name = path.name

            if path.suffix == ".csv":
                try:
                    if read_csv_robust_multi is None:
                        data[name] = pd.read_csv(str(path)); continue
                    sections = read_csv_robust_multi(str(path))
                    if len(sections) == 1:
                        data[name] = next(iter(sections.values()))
                    else:
                        # Primary section gets the original filename
                        data[name] = sections.get("primary", next(iter(sections.values())))
                        # Extra sections get filename stem + section name
                        stem = Path(name).stem
                        for sec_name, sec_df in sections.items():
                            if sec_name != "primary":
                                data[f"{stem}__{sec_name}"] = sec_df
                except Exception:
                    logger.warning("CSV parsing failed for %s", path)
            elif path.suffix == ".json":
                try:
                    with open(path) as f:
                        json_data = json.load(f)
                    if isinstance(json_data, list):
                        data[name] = pd.DataFrame(json_data)
                    else:
                        data[name] = pd.DataFrame([json_data])
                except Exception:
                    logger.warning("JSON parsing failed for %s", path)
            elif path.suffix == ".pdf":
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(str(path))
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                    data[name] = pd.DataFrame({"text": [text]})
                except ImportError:
                    try:
                        import PyPDF2
                        with open(path, "rb") as f:
                            reader = PyPDF2.PdfReader(f)
                            text = "".join(
                                p.extract_text() or "" for p in reader.pages
                            )
                        data[name] = pd.DataFrame({"text": [text]})
                    except Exception:
                        logger.warning(
                            "PDF parsing failed for %s (install PyMuPDF or PyPDF2)",
                            path,
                        )
                except Exception:
                    logger.warning("PDF parsing failed for %s", path)
            elif path.suffix == ".html":
                try:
                    tables = pd.read_html(path)
                    if tables:
                        data[name] = tables[0]
                except Exception:
                    # Fall back to raw text
                    try:
                        with open(path) as f:
                            text = f.read()
                        data[name] = pd.DataFrame({"text": [text]})
                    except Exception:
                        logger.warning("HTML parsing failed for %s", path)
            elif path.suffix in (".xlsx", ".xls"):
                try:
                    # read_excel_robust imported at module level
                    sheets = read_excel_robust(str(path))
                    if not sheets:
                        logger.warning("Excel parsing returned no sheets for %s", path)
                        continue
                    if len(sheets) == 1:
                        data[name] = next(iter(sheets.values()))
                    else:
                        # Primary sheet keeps the filename; extras get stem__sheet
                        data[name] = sheets.get("primary", next(iter(sheets.values())))
                        stem = Path(name).stem
                        for sheet_key, sheet_df in sheets.items():
                            if sheet_key != "primary":
                                data[f"{stem}__{sheet_key}"] = sheet_df
                except Exception:
                    logger.warning("Excel parsing failed for %s", path)
            elif path.suffix == ".npz":
                try:
                    import numpy as np
                    npz_data = np.load(path, allow_pickle=True)
                    # Convert arrays to DataFrame columns where possible
                    col_dict = {}
                    max_len = 0
                    for key in npz_data.files:
                        arr = npz_data[key]
                        if arr.ndim == 0:
                            col_dict[key] = [arr.item()]
                        elif arr.ndim == 1:
                            col_dict[key] = arr
                            max_len = max(max_len, len(arr))
                        elif arr.ndim == 2:
                            for i in range(arr.shape[1]):
                                col_dict[f"{key}_{i}"] = arr[:, i]
                            max_len = max(max_len, arr.shape[0])
                        else:
                            # Store shape metadata for high-dim arrays
                            col_dict[f"{key}_shape"] = [str(arr.shape)]
                            col_dict[f"{key}_dtype"] = [str(arr.dtype)]
                            max_len = max(max_len, 1)
                    if col_dict:
                        # Pad shorter columns to max_len
                        for k, v in col_dict.items():
                            arr = np.asarray(v)
                            if len(arr) < max_len:
                                col_dict[k] = np.pad(
                                    arr.astype(float), (0, max_len - len(arr)),
                                    constant_values=np.nan,
                                )
                        data[name] = pd.DataFrame(col_dict)
                    else:
                        data[name] = pd.DataFrame({"_npz_keys": list(npz_data.files)})
                except Exception as exc:
                    logger.warning("NPZ parsing failed for %s: %s", path, exc)
            elif path.suffix == ".gpkg":
                try:
                    # Load as GeoDataFrame directly to preserve geometry for spatial ops
                    import geopandas as gpd
                    gdf = gpd.read_file(str(path))
                    if not gdf.empty:
                        data[name] = gdf
                    else:
                        logger.warning("GeoPackage returned empty GeoDataFrame for %s", path)
                except ImportError:
                    # Fallback to data_tools reader if geopandas not available
                    # read_geopackage imported at module level
                    gdf = read_geopackage(str(path), keep_geometry=True)
                    if not gdf.empty:
                        data[name] = gdf
                    else:
                        logger.warning("GeoPackage returned empty DataFrame for %s", path)
                except Exception as exc:
                    logger.warning("GeoPackage parsing failed for %s: %s", path, exc)
            elif path.suffix == ".tle":
                try:
                    # read_tle imported at module level
                    df = read_tle(str(path))
                    if df.empty:
                        with open(path, errors="replace") as f:
                            data[name] = pd.DataFrame({"text": [f.read()]})
                    else:
                        data[name] = df
                except Exception:
                    try:
                        with open(path, errors="replace") as f:
                            data[name] = pd.DataFrame({"text": [f.read()]})
                    except Exception:
                        logger.warning("TLE file reading failed for %s", path)
            elif path.suffix == ".sp3":
                try:
                    # read_sp3 imported at module level
                    df = read_sp3(str(path))
                    if df.empty:
                        with open(path, errors="replace") as f:
                            data[name] = pd.DataFrame({"text": [f.read()]})
                    else:
                        data[name] = df
                except Exception:
                    try:
                        with open(path, errors="replace") as f:
                            data[name] = pd.DataFrame({"text": [f.read()]})
                    except Exception:
                        logger.warning("SP3 file reading failed for %s", path)
            elif path.suffix == ".cdf":
                try:
                    # read_cdf imported at module level
                    df = read_cdf(str(path))
                    if df.empty:
                        data[name] = pd.DataFrame({"error": [f"CDF parse returned empty for {path.name}"]})
                    else:
                        data[name] = df
                except Exception as exc:
                    data[name] = pd.DataFrame({"error": [f"CDF parse failed: {exc}"]})
                    logger.warning("CDF file reading failed for %s: %s", path, exc)
            elif path.suffix in (".dat", ".lst"):
                try:
                    # read_structured_text imported at module level
                    df = read_structured_text(str(path))
                    # Fall back to raw text if structured parsing only produced 1 column
                    if len(df.columns) <= 1 and "text" in df.columns:
                        data[name] = df
                    elif df.empty:
                        with open(path, errors="replace") as f:
                            data[name] = pd.DataFrame({"text": [f.read()]})
                    else:
                        data[name] = df
                except Exception:
                    try:
                        with open(path, errors="replace") as f:
                            data[name] = pd.DataFrame({"text": [f.read()]})
                    except Exception:
                        logger.warning("Structured text reading failed for %s", path)
            elif path.suffix in (".txt", ".text", ".hdr", ""):
                try:
                    # read_structured_text imported at module level
                    df = read_structured_text(str(path))
                    if len(df.columns) > 1 and len(df) > 0:
                        data[name] = df
                    else:
                        with open(path, errors="replace") as f:
                            data[name] = pd.DataFrame({"text": [f.read()]})
                except Exception:
                    try:
                        with open(path, errors="replace") as f:
                            data[name] = pd.DataFrame({"text": [f.read()]})
                    except Exception:
                        logger.warning("Text file reading failed for %s", path)

        # A.1: Fill-value sanitization for astronomy domain
        # Astronomy data uses sentinel values (9999.99, 999.9, 9.99E32, etc.)
        # for missing measurements. Replace with NaN before pipeline sees them.
        if task.category and task.category.lower() == "astronomy":
            import numpy as np
            _FILL_VALUES = [9999.99, 999.9, 99999, 99999.0, 9999, 9999.0,
                            9.99e32, 1e32, -1e31, 999999, 999999.0, 99.99]
            for name, df in data.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) == 0:
                    continue
                replaced_count = 0
                for col in numeric_cols:
                    for fv in _FILL_VALUES:
                        mask = df[col] == fv
                        n = mask.sum()
                        if n > 0:
                            df.loc[mask, col] = np.nan
                            replaced_count += n
                    # Also catch very large absolute values (>1e30)
                    abs_mask = df[col].abs() > 1e30
                    n = abs_mask.sum()
                    if n > 0:
                        df.loc[abs_mask, col] = np.nan
                        replaced_count += n
                if replaced_count > 0:
                    logger.info(
                        "Astronomy fill-value sanitization: replaced %d sentinel "
                        "values with NaN in '%s'",
                        replaced_count, name,
                    )

        return data

    @staticmethod
    def _extract_answer(
        output: pd.DataFrame,
        answer_type: str,
        query: str = "",
        input_columns: Optional[List[str]] = None,
    ):
        """Extract the answer value from a pipeline output DataFrame.

        Improvements over naive heuristics:
        * Query-keyword column matching — parse the goal for column hints.
        * Prefer last-added column — rightmost non-input column is most
          likely the computed answer.
        * Numeric single-value detection for single-row results.
        * Better list answer extraction using query keywords.
        * Logging at every decision point for debuggability.
        """
        if output is None or len(output) == 0:
            logger.debug("_extract_answer: output is None or empty")
            return None

        def _is_null(val) -> bool:
            """Check if a value is null/NA/NaN so we can skip to next fallback."""
            if val is None:
                return True
            if isinstance(val, float) and pd.isna(val):
                return True
            sv = str(val).strip().lower()
            if sv in ("nan", "<na>", "nat", "none", "", "<nan>"):
                return True
            return False

        input_cols = set(input_columns) if input_columns else set()
        # Columns added by the pipeline (not present in input data)
        added_cols = [
            c for c in output.columns if c not in input_cols
        ] if input_cols else []

        cols_lower = {c: c.lower().strip() for c in output.columns}

        # --- Helper: extract keywords from query for column matching ---
        query_words = set()
        query_lower = query.lower() if query else ""
        if query:
            # Extract multi-word tokens and single words from query
            for word in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query_lower):
                if len(word) > 2 and word not in {
                    "the", "and", "for", "are", "this", "that", "with",
                    "from", "what", "which", "how", "many", "much",
                    "does", "not", "has", "have", "was", "were",
                    "each", "all", "data", "file", "using",
                }:
                    query_words.add(word)

        # --- Helper: detect if the query is asking FOR a year/date value ---
        _query_asks_for_year = bool(re.search(
            r"\b(which|what|in what)\s+(year|years)\b", query_lower,
        )) if query else False

        # --- Helper: infer the expected element type for list answers ---
        #     from the query phrasing (e.g. "which years" → int, "which states" → str)
        _list_expects_strings = False
        _list_expects_ints = False
        if answer_type in ("list_exact", "list_approximate") and query:
            if re.search(
                r"\b(which|what|list|identify|find|name)\b.*\b("
                r"states?|countries|country|names?|cities|city|"
                r"beaches?|areas?|regions?|genes?|proteins?|"
                r"agencies|agency|departments?|organizations?|"
                r"samples?|categories|category|types?|species"
                r")\b", query_lower,
            ):
                _list_expects_strings = True
            if re.search(r"\b(which|what|in which|during which)\b.*\b(years?|seasons?)\b", query_lower):
                _list_expects_ints = True

        def _query_col_match(col_name: str) -> bool:
            """Check if a column name matches keywords from the query."""
            cl = col_name.lower().replace("_", " ").replace("-", " ")
            return any(w in cl for w in query_words) if query_words else False

        # 1. Exact column name match — always wins
        for col, cl in cols_lower.items():
            if cl in ("answer", "result"):
                if answer_type in ("list_exact", "list_approximate") and len(output) > 1:
                    logger.debug("_extract_answer: exact column match '%s' (list)", col)
                    return output[col].tolist()
                val = output[col].iloc[0]
                if not _is_null(val):
                    logger.debug("_extract_answer: exact column match '%s'", col)
                    return val

        # 2. Single-column output
        if len(output.columns) == 1:
            logger.debug("_extract_answer: single-column output")
            if answer_type in ("list_exact", "list_approximate"):
                return output.iloc[:, 0].tolist()
            val = output.iloc[0, 0]
            return _normalize_bool(val) if answer_type.startswith("string") else val

        # 3. Column-name hint search (before positional fallback)
        # Note: Removed "check" and "output" as they often match metadata columns
        # (e.g., "precipitation_check_method", "output_format") rather than answer columns
        answer_hints = ["answer", "result", "minimum", "min_categories",
                        "ratio", "difference", "total", "correlation",
                        "percentage", "proportion"]
        for col, cl in cols_lower.items():
            if any(h in cl for h in answer_hints):
                val = output[col].iloc[0]
                if _is_null(val):
                    continue
                # For numeric answer types, skip string columns (they're likely metadata)
                if answer_type.startswith("numeric") and output[col].dtype == "object":
                    logger.debug("_extract_answer: skipping string col '%s' for numeric answer", col)
                    continue
                logger.debug("_extract_answer: hint column match '%s'", col)
                return _normalize_bool(val) if answer_type.startswith("string") else val

        # 3b. Query-keyword column matching on pipeline-added columns
        #     For string answer types, prefer string-typed columns.
        #     For numeric answer types, prefer numeric-typed columns.
        def _type_appropriate(col_name: str) -> bool:
            """Check if a column's dtype matches the answer type."""
            if answer_type.startswith("string"):
                return output[col_name].dtype == "object"
            if answer_type.startswith("numeric"):
                return (pd.api.types.is_numeric_dtype(output[col_name])
                        and output[col_name].dtype != "bool")
            # list types — match based on inferred element type
            if _list_expects_strings:
                return output[col_name].dtype == "object"
            if _list_expects_ints:
                return (pd.api.types.is_numeric_dtype(output[col_name])
                        and output[col_name].dtype != "bool")
            return True

        def _return_col_value(col):
            """Return the extracted value from a column, respecting answer_type."""
            if answer_type in ("list_exact", "list_approximate"):
                return output[col].tolist()
            val = output[col].iloc[0]
            if answer_type.startswith("string"):
                val = _normalize_bool(val)
                # K1.1: Strip explanation text from string answers
                # e.g. "Singapore has the highest average city population of 5,983,000.00" → "Singapore"
                if isinstance(val, str) and len(val) > 0:
                    import re as _re
                    # Remove trailing parenthetical: "U.S. Space Force (Median: $1,300)" → "U.S. Space Force"
                    val = _re.sub(r'\s*\([^)]*\)\s*$', '', val).strip()
                    # Split on common explanation connectors and take shortest meaningful prefix
                    for sep in [' — ', ': ', ' - ', ', ']:
                        if sep in val:
                            parts = val.split(sep, 1)
                            candidate = parts[0].strip()
                            # Only use prefix if it's substantially shorter (explanation removed)
                            if candidate and len(candidate) < len(val) * 0.7:
                                val = candidate
                                break
                return val
            return val

        # 3a-priority. If query asks for a year/date, check for year columns first
        #     before generic keyword matching can grab the wrong column.
        if _query_asks_for_year or _list_expects_ints:
            for col in output.columns:
                cl = col.lower().strip()
                if "year" in cl or cl == "date":
                    if pd.api.types.is_numeric_dtype(output[col]) and output[col].dtype != "bool":
                        logger.debug(
                            "_extract_answer: year-priority match col '%s'", col
                        )
                        return _return_col_value(col)
        if _list_expects_strings:
            str_cols = [c for c in output.columns if output[c].dtype == "object"]
            if str_cols:
                # Prefer string columns matching query keywords
                if query_words:
                    for col in str_cols:
                        if _query_col_match(col):
                            logger.debug(
                                "_extract_answer: string-priority query-match col '%s'", col
                            )
                            return _return_col_value(col)
                # Skip ID-like columns, pick first meaningful string column
                skip_str = {"id", "index", "unnamed: 0", "code"}
                for col in str_cols:
                    if col.lower().strip() not in skip_str:
                        logger.debug(
                            "_extract_answer: string-priority first col '%s'", col
                        )
                        return _return_col_value(col)

        # 3b. Query-keyword column matching — type-appropriate first, then any.
        #     Search added columns first (3b), then all columns (3c).
        #     Within each scope: prefer type-appropriate match, fall back to any.
        #     When multiple columns match, prefer the one with the most keyword hits
        #     (breaks ties by rightmost position).
        def _query_col_score(col_name: str) -> int:
            """Count how many query keywords appear in the column name."""
            cl = col_name.lower().replace("_", " ").replace("-", " ")
            return sum(1 for w in query_words if w in cl)

        for scope_label, scope_cols in [
            ("added", list(added_cols) if added_cols else []),
            ("all", list(output.columns)),
        ]:
            if not scope_cols or not query_words:
                continue
            # Pass 1: type-appropriate — pick the column with the most keyword hits
            best_col, best_score = None, 0
            for col in scope_cols:
                if _query_col_match(col) and _type_appropriate(col):
                    score = _query_col_score(col)
                    if score > best_score:
                        best_col, best_score = col, score
            if best_col is not None:
                logger.debug(
                    "_extract_answer: query-keyword match on %s col '%s' "
                    "(type-appropriate, score=%d)",
                    scope_label, best_col, best_score,
                )
                return _return_col_value(best_col)
            # Pass 2: any column — but only if NOT a string/numeric type
            #          that would pick the wrong dtype.
            if not answer_type.startswith("string") and not answer_type.startswith("numeric"):
                best_col, best_score = None, 0
                for col in scope_cols:
                    if _query_col_match(col):
                        score = _query_col_score(col)
                        if score > best_score:
                            best_col, best_score = col, score
                if best_col is not None:
                    logger.debug(
                        "_extract_answer: query-keyword match on %s col '%s' (score=%d)",
                        scope_label, best_col, best_score,
                    )
                    return _return_col_value(best_col)

        # 4. Single-row output
        if len(output) == 1:
            # 4a. Prefer the last pipeline-added column (rightmost non-input)
            #     For string types, prefer type-appropriate added column.
            if added_cols:
                if answer_type.startswith("string"):
                    str_added = [c for c in added_cols
                                 if output[c].dtype == "object"
                                 and not _is_null(output[c].iloc[0])]
                    if str_added:
                        col = str_added[-1]
                        logger.debug(
                            "_extract_answer: single-row, last added string col '%s'", col
                        )
                        return _normalize_bool(output[col].iloc[0])
                    # No string added col — fall through to 4b to check ALL string cols
                elif answer_type.startswith("numeric"):
                    num_added = [c for c in added_cols
                                 if pd.api.types.is_numeric_dtype(output[c])
                                 and output[c].dtype != "bool"
                                 and not _is_null(output[c].iloc[0])]
                    if num_added:
                        col = num_added[-1]
                        logger.debug(
                            "_extract_answer: single-row, last added numeric col '%s'", col
                        )
                        return output[col].iloc[0]
                    # Fall through to 4b instead of returning possibly-null value
                else:
                    col = added_cols[-1]
                    val = output[col].iloc[0]
                    if not _is_null(val):
                        logger.debug(
                            "_extract_answer: single-row, last added col '%s'", col
                        )
                        return _normalize_bool(val) if answer_type.startswith("string") else val

            # 4b. String types: prefer string columns (from any column, not just added)
            if answer_type.startswith("string"):
                str_cols = [c for c in output.columns if output[c].dtype == "object"]
                if str_cols:
                    # Prefer query-matching string columns
                    if query_words:
                        for col in reversed(str_cols):
                            if _query_col_match(col) and not _is_null(output[col].iloc[0]):
                                logger.debug(
                                    "_extract_answer: single-row string query-match col '%s'", col
                                )
                                return _normalize_bool(output[col].iloc[0])
                    # Skip typical non-answer columns
                    # Prioritize column literally named "answer"
                    for col in str_cols:
                        if col.lower().strip() == "answer" and not _is_null(output[col].iloc[0]):
                            logger.debug(
                                "_extract_answer: single-row string 'answer' col '%s'", col
                            )
                            return _normalize_bool(output[col].iloc[0])
                    skip_words = [
                        "id", "iso2", "iso3", "capital", "admin",
                        "agent_type", "substep_id", "status", "phase_id",
                        "phase_name", "agent_name", "objective",
                    ]
                    for col in reversed(str_cols):
                        cl = col.lower().strip()
                        if any(s in cl for s in skip_words):
                            continue
                        if _is_null(output[col].iloc[0]):
                            continue
                        logger.debug(
                            "_extract_answer: single-row string col '%s'", col
                        )
                        return _normalize_bool(output[col].iloc[0])
                    # Any non-null string column as fallback
                    for col in str_cols:
                        if not _is_null(output[col].iloc[0]):
                            return _normalize_bool(output[col].iloc[0])

            if answer_type in ("numeric_exact", "numeric_approximate"):
                # If query asks for a year, prefer year/date columns
                if _query_asks_for_year:
                    for col in output.columns:
                        cl = col.lower().strip()
                        if "year" in cl or "date" in cl:
                            try:
                                val = output[col].iloc[0]
                                if _is_null(val):
                                    continue
                                float(val)
                                logger.debug(
                                    "_extract_answer: single-row, query asks for year, col '%s'", col
                                )
                                return val
                            except (ValueError, TypeError):
                                continue

                skip_words = ["year", "id", "name", "state", "country",
                              "region", "category", "type", "date",
                              "agent_type", "substep_id", "status", "phase_id",
                              "phase_name", "agent_name", "objective"]
                # Don't skip year/date if the query explicitly asks for them
                if _query_asks_for_year:
                    skip_words = [w for w in skip_words if w not in ("year", "date")]
                for col in reversed(list(output.columns)):
                    if output[col].dtype == "bool":
                        continue
                    cl = col.lower().strip() if isinstance(col, str) else str(col)
                    if any(s in cl for s in skip_words):
                        continue
                    try:
                        val = output[col].iloc[0]
                        if _is_null(val):
                            continue
                        float(val)
                        logger.debug(
                            "_extract_answer: single-row numeric col '%s'", col
                        )
                        return val
                    except (ValueError, TypeError):
                        continue
                for col in reversed(list(output.columns)):
                    if output[col].dtype == "bool":
                        continue
                    try:
                        val = output[col].iloc[0]
                        if _is_null(val):
                            continue
                        float(val)
                        return val
                    except (ValueError, TypeError):
                        continue

            # 4-final. Type-enforced fallback for single-row output.
            #   For string answers, only return a string column.
            #   For numeric answers, only return a numeric column.
            if answer_type.startswith("string"):
                str_cols = [c for c in output.columns
                            if output[c].dtype == "object"
                            and not _is_null(output[c].iloc[0])]
                if str_cols:
                    logger.debug("_extract_answer: single-row string fallback col '%s'", str_cols[-1])
                    return _normalize_bool(output[str_cols[-1]].iloc[0])
            elif answer_type.startswith("numeric"):
                for col in reversed(list(output.columns)):
                    if output[col].dtype == "bool":
                        continue
                    val = output[col].iloc[0]
                    if _is_null(val):
                        continue
                    try:
                        float(val)
                        logger.debug("_extract_answer: single-row numeric fallback col '%s'", col)
                        return val
                    except (ValueError, TypeError):
                        continue

            # Ultimate fallback — return first non-null value
            for col in output.columns:
                val = output[col].iloc[0]
                if not _is_null(val):
                    logger.debug("_extract_answer: single-row ultimate fallback col '%s'", col)
                    return _normalize_bool(val) if answer_type.startswith("string") else val
            logger.debug("_extract_answer: single-row all values null")
            return None

        # 5. Multi-row output — strategy depends on answer_type
        if answer_type in ("string_exact", "string_approximate"):
            str_cols = [c for c in output.columns if output[c].dtype == "object"]
            num_cols = [c for c in output.columns
                        if pd.api.types.is_numeric_dtype(output[c])
                        and output[c].dtype != "bool"]

            # 5a. If only 1 row after dropping NaN in string cols, treat as single-row
            if str_cols:
                # Prefer query-keyword matched string columns
                if query_words:
                    for col in reversed(str_cols):
                        if _query_col_match(col):
                            val = output[col].dropna().iloc[0] if len(output[col].dropna()) > 0 else None
                            if val is not None and not _is_null(val):
                                logger.debug(
                                    "_extract_answer: multi-row string, query-match col '%s'", col
                                )
                                return _normalize_bool(val)

                # Prefer added string columns
                str_added = [c for c in str_cols if c in set(added_cols)]
                if str_added:
                    val = output[str_added[-1]].dropna().iloc[0] if len(output[str_added[-1]].dropna()) > 0 else None
                    if val is not None and not _is_null(val):
                        logger.debug(
                            "_extract_answer: multi-row string, added col '%s'", str_added[-1]
                        )
                        return _normalize_bool(val)

                # idxmax heuristic: row with max numeric → string value
                if num_cols:
                    max_idx = output[num_cols[-1]].idxmax()
                    val = str(output.loc[max_idx, str_cols[0]])
                    if not _is_null(val):
                        logger.debug(
                            "_extract_answer: multi-row string, max-of '%s' → row %s → '%s'",
                            num_cols[-1], max_idx, val,
                        )
                        return _normalize_bool(val)

            # Check for yes/no/true/false values
            for col in reversed(list(output.columns)):
                vals = output[col].dropna().unique()
                for v in vals:
                    sv = str(v).strip().lower()
                    if sv in ("yes", "no", "true", "false"):
                        logger.debug("_extract_answer: multi-row yes/no in '%s'", col)
                        return _normalize_bool(v)

            # Fallback: first non-null string value, or last column
            if str_cols:
                for col in str_cols:
                    val = output[col].dropna().iloc[0] if len(output[col].dropna()) > 0 else None
                    if val is not None and not _is_null(val):
                        logger.debug("_extract_answer: multi-row string fallback col '%s'", col)
                        return _normalize_bool(val)
            logger.debug("_extract_answer: multi-row string fallback last col")
            val = output.iloc[0, -1]
            return _normalize_bool(val) if not _is_null(val) else None

        if answer_type in ("numeric_exact", "numeric_approximate"):
            # Prefer last added non-bool numeric column if available
            if added_cols:
                num_added = [c for c in added_cols
                             if pd.api.types.is_numeric_dtype(output[c])
                             and output[c].dtype != "bool"
                             and not _is_null(output[c].iloc[0])]
                if num_added:
                    logger.debug(
                        "_extract_answer: multi-row numeric, last added numeric col '%s'",
                        num_added[-1],
                    )
                    return output[num_added[-1]].iloc[0]
            # Fallback: last non-bool numeric column with non-null value
            for col in reversed(list(output.columns)):
                if pd.api.types.is_numeric_dtype(output[col]) and output[col].dtype != "bool":
                    val = output[col].iloc[0]
                    if not _is_null(val):
                        logger.debug("_extract_answer: multi-row numeric fallback col '%s'", col)
                        return val
            logger.debug("_extract_answer: multi-row numeric fallback last col")
            val = output.iloc[0, -1]
            return val if not _is_null(val) else None

        if answer_type in ("list_exact", "list_approximate"):
            # K1.2: Skip pipeline metadata columns that should never be answer columns
            _meta_cols = {"agent_type", "substep_id", "status", "phase_id",
                          "phase_name", "agent_name", "objective", "unnamed: 0"}
            _data_cols = [c for c in output.columns
                          if c.lower().strip() not in _meta_cols]
            if _data_cols and len(_data_cols) < len(output.columns):
                logger.debug(
                    "_extract_answer: list, filtered %d metadata cols, %d data cols remain",
                    len(output.columns) - len(_data_cols), len(_data_cols),
                )

            # --- Type-aware column selection for list answers ---
            # If we know the expected element type from query analysis,
            # prefer columns of that type.
            if _list_expects_ints:
                # Query asks "which years" → prefer Year-like integer columns
                for col in output.columns:
                    cl = col.lower().strip()
                    if "year" in cl and pd.api.types.is_numeric_dtype(output[col]):
                        logger.debug(
                            "_extract_answer: list, query expects years, col '%s'", col
                        )
                        return output[col].tolist()

            if _list_expects_strings:
                # Query asks "which states/countries" → prefer string columns
                str_cols = [c for c in _data_cols
                            if output[c].dtype == "object"]
                if str_cols:
                    # Prefer columns matching query keywords
                    if query_words:
                        for col in str_cols:
                            if _query_col_match(col):
                                logger.debug(
                                    "_extract_answer: list, query expects strings, matched col '%s'", col
                                )
                                return output[col].tolist()
                    # Fall back to first string column
                    logger.debug(
                        "_extract_answer: list, query expects strings, first str col '%s'",
                        str_cols[0],
                    )
                    return output[str_cols[0]].tolist()

            # Prefer columns matching query keywords, but skip boolean columns
            if query_words:
                # Pass 1: non-boolean query-matched data columns
                for col in _data_cols:
                    if output[col].dtype == "bool":
                        continue
                    if _query_col_match(col):
                        logger.debug(
                            "_extract_answer: list, query-matched non-bool col '%s'", col
                        )
                        return output[col].tolist()
                # Pass 2: any query-matched data column (including bool)
                for col in _data_cols:
                    if _query_col_match(col):
                        logger.debug(
                            "_extract_answer: list, query-matched col '%s'", col
                        )
                        return output[col].tolist()
            # Prefer added columns (skip boolean)
            if added_cols:
                non_bool_added = [c for c in added_cols if output[c].dtype != "bool"]
                if non_bool_added:
                    logger.debug(
                        "_extract_answer: list, first added non-bool col '%s'", non_bool_added[0]
                    )
                    return output[non_bool_added[0]].tolist()
                logger.debug(
                    "_extract_answer: list, first added col '%s'", added_cols[0]
                )
                return output[added_cols[0]].tolist()
            # Fall back to first non-ID, non-boolean data column
            skip = {"id", "index", "unnamed: 0"}
            for col in _data_cols:
                if col.lower().strip() in skip:
                    continue
                if output[col].dtype == "bool":
                    continue
                logger.debug("_extract_answer: list, first non-ID non-bool col '%s'", col)
                return output[col].tolist()
            # Ultimate fallback (data columns first, then all)
            for col in _data_cols:
                if col.lower().strip() not in skip:
                    logger.debug("_extract_answer: list, first non-ID col '%s'", col)
                    return output[col].tolist()
            return output.iloc[:, 0].tolist()

        # 6. Fallback
        logger.debug("_extract_answer: ultimate fallback")
        return output.iloc[0, -1]

    def evaluate_output(
        self,
        task: BenchmarkTask,
        output: pd.DataFrame,
        input_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate pipeline output using the official KramaBench metrics.

        Returns a dict with:
        - ``passed``: bool — True if the primary metric indicates success
        - ``score``: float — primary metric value (0–1 range)
        - ``kramabench_metrics``: dict — all applicable metric values
        - ``expected_answer``, ``actual_answer``, ``answer_type``
        """
        expected = task.expected_output
        answer = expected.get("answer")
        answer_type = expected.get("answer_type", "string_exact")

        actual_answer = self._extract_answer(
            output, answer_type, query=task.goal,
            input_columns=input_columns,
        )

        # Flatten nested single-element lists (e.g. [[2010.0], [2011.0]] → [2010.0, 2011.0])
        if answer_type in ("list_exact", "list_approximate") and isinstance(actual_answer, list):
            flat = []
            for item in actual_answer:
                if isinstance(item, list) and len(item) == 1:
                    flat.append(item[0])
                else:
                    flat.append(item)
            actual_answer = flat

        # Determine which metrics apply
        metric_names = ANSWER_TYPE_METRICS.get(answer_type, ["success"])

        # Fast-path: for approximate types, try metric_success first.
        # If the answer is an exact match, skip expensive LLM paraphrase.
        if answer_type in ("string_approximate", "list_approximate") and actual_answer is not None:
            try:
                if answer_type == "string_approximate":
                    exact_score = metric_success(actual_answer, answer)
                    if exact_score == 1.0:
                        return {
                            "passed": True,
                            "score": 1.0,
                            "kramabench_metrics": {"llm_paraphrase": 1.0},
                            "expected_answer": answer,
                            "actual_answer": actual_answer,
                            "answer_type": answer_type,
                        }
                elif answer_type == "list_approximate":
                    f1, prec, rec = _list_f1(actual_answer, answer, approx=True)
                    if f1 == 1.0:
                        return {
                            "passed": True,
                            "score": 1.0,
                            "kramabench_metrics": {"f1_approximate": 1.0},
                            "expected_answer": answer,
                            "actual_answer": actual_answer,
                            "answer_type": answer_type,
                        }
            except Exception:
                pass  # Fall through to normal evaluation

        # Compute each metric
        kramabench_metrics: Dict[str, Optional[float]] = {}
        for metric_name in metric_names:
            fn = METRIC_FUNCTIONS.get(metric_name)
            if fn is None:
                kramabench_metrics[metric_name] = None
                continue
            if actual_answer is None:
                kramabench_metrics[metric_name] = 0.0
                continue
            try:
                kramabench_metrics[metric_name] = fn(actual_answer, answer)
            except Exception as e:
                logger.warning("Metric %s failed for task %s: %s",
                               metric_name, task.task_id, e)
                kramabench_metrics[metric_name] = 0.0

        # Determine pass/fail and primary score
        if "success" in kramabench_metrics:
            passed = kramabench_metrics["success"] == 1.0
            score = kramabench_metrics["success"] or 0.0
        elif "rae_score" in kramabench_metrics:
            score = kramabench_metrics["rae_score"] or 0.0
            passed = score >= 0.9  # High RAE score indicates close match
        elif "f1" in kramabench_metrics:
            score = kramabench_metrics["f1"] or 0.0
            passed = score == 1.0
        elif "f1_approximate" in kramabench_metrics:
            score = kramabench_metrics["f1_approximate"] or 0.0
            passed = score >= 0.9
        elif "llm_paraphrase" in kramabench_metrics:
            val = kramabench_metrics["llm_paraphrase"]
            passed = val == 1.0 if val is not None else False
            score = val if val is not None else 0.0
        else:
            passed = False
            score = 0.0

        return {
            "passed": passed,
            "score": float(score),
            "kramabench_metrics": kramabench_metrics,
            "expected_answer": answer,
            "actual_answer": actual_answer,
            "answer_type": answer_type,
        }

    def get_task_data_type(self, task: BenchmarkTask) -> str:
        """Detect the natural data type for a task based on files and content.

        Returns one of:
        - "geodataframe" — for spatial/geographic operations (GeoPackage, shapefiles)
        - "numpy_array" — for numpy array computations
        - "dataframe" — default for standard pandas operations

        Detection is based on:
        1. File extensions (.gpkg, .shp → geodataframe; .npz → numpy_array)
        2. Task query/description keywords
        3. Domain-specific patterns
        """
        # Check file extensions
        file_extensions = [Path(f).suffix.lower() for f in task.data_files]

        # GeoDataFrame indicators - spatial file formats
        geo_extensions = {".gpkg", ".shp", ".geojson", ".kml", ".gml"}
        if any(ext in geo_extensions for ext in file_extensions):
            return "geodataframe"

        # Numpy indicators - array-based file formats
        numpy_extensions = {".npz", ".npy"}
        if any(ext in numpy_extensions for ext in file_extensions):
            return "numpy_array"

        # Check task content for keyword indicators
        query = task.goal.lower() if task.goal else ""
        description = task.description.lower() if task.description else ""
        combined = query + " " + description

        # Geographic/spatial keywords
        geo_keywords = [
            "geodataframe", "geopandas", "gpd.", "spatial",
            "geometry", "polygon", "multipolygon", "point", "linestring",
            "shapefile", "geopackage", "coordinate", "crs", "epsg",
            "latitude", "longitude", "lat", "lon", "centroid",
            "buffer", "intersection", "union", "within", "contains",
            "distance", "area", "boundary", "geospatial"
        ]
        for kw in geo_keywords:
            if kw in combined:
                return "geodataframe"

        # Numpy keywords (similar to DSEval adapter)
        numpy_keywords = [
            "np.array", "numpy", "np.dot", "np.linalg", "np.sum",
            "np.mean", "np.std", "np.var", "np.reshape", "np.transpose",
            "ndarray", "matrix", "eigenvalue", "eigenvector", "svd",
            "fft", "convolution", "np.where", "np.argmax", "np.argmin",
            "vectorize", "broadcasting", "element-wise"
        ]
        for kw in numpy_keywords:
            if kw in combined:
                return "numpy_array"

        # Domain-based detection
        domain = task.category.lower() if task.category else ""

        # Geographic domains
        geo_domains = ["geography", "geo", "spatial", "gis", "mapping", "cartography"]
        if any(d in domain for d in geo_domains):
            return "geodataframe"

        # Default to DataFrame for standard pandas operations
        return "dataframe"
