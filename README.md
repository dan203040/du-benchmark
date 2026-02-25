> **NOTE:** This repository is made available only to the reviewers of VLDB for this manuscript due to the required rule. Please do not actively disseminate this as it is not currently intended to be done so.

## Reproducibility Statement

The submitted repository supports reproducibility at two levels:

**(1) Validation of reported results (no external dependencies or API keys required):** All experimental outputs are included as pre-computed JSON files in the `results/` directory — the 3-model consensus silver standard (257 tasks), per-system per-dimension scores with bootstrap 95% confidence intervals, McNemar's pairwise statistical tests, and the 5-condition causal ablation results. All paper tables and figures can be regenerated deterministically from these files using the provided analysis scripts (`analyze_results.py`, `rescore_decontaminated.py`, `per_benchmark_breakdown.py`, `consensus_stability.py`).

**(2) Full re-execution of the benchmark pipeline (requires LLM API keys):** The consensus generation, DU extraction, and scoring pipeline can be re-run end-to-end. Benchmark task data is obtained via the included `download_data.sh` script from public repositories (KramaBench, AgentBench DBBench, DACode). Fresh consensus generation requires API keys for three LLM providers (DeepSeek, Anthropic, Google). The causal ablation experiment (Section 5) additionally requires the ADP-MA agentic pipeline; since this component is under separate review, pre-computed ablation results are included for independent validation.

The M\* scoring taxonomy, multi-LLM consensus protocol, and all 257 benchmark task definitions are fully contained in the repository with no proprietary dependencies for result validation.

---

# DU Benchmark: Benchmarking Automated Data Understanding

This repository contains the benchmark suite, scoring infrastructure, and pre-computed results for evaluating automated **Data Understanding (DU)** systems across the M\* taxonomy (D1--D5).

> **Paper:** *Towards Principled Benchmarking of Automated Data Understanding* (submitted to VLDB EA&B 2026)

## Overview

The DU Benchmark evaluates how well data-analysis systems "understand" their input data before generating code. It defines five dimensions:

| Dimension | Name | Evaluation Method |
|-----------|------|-------------------|
| **D1** | Query Understanding | Multi-LLM consensus |
| **D2** | Schema & Join Understanding | Deterministic verification |
| **D3** | Semantic / Unit Understanding | Multi-LLM consensus |
| **D4** | Format Diagnosis | Deterministic verification |
| **D5** | Analytical Constraints | Multi-LLM consensus |

The benchmark aggregates 257 tasks from three existing benchmarks:
- **KramaBench** (105 tasks) -- multi-step data science pipelines
- **AgentBench DBBench** (100 tasks) -- natural-language-to-database queries
- **DACode** (52 tasks) -- data analysis code generation

## Repository Structure

```
du-benchmark/
  du_benchmark/          # Python package
    schema.py            # M* taxonomy dataclasses (D1-D5)
    consensus.py         # Multi-LLM consensus generation
    metrics.py           # D1-D5 scoring, Quality Score, bootstrap CIs
    deterministic_verify.py  # D2/D4 ground-truth verification
    run_benchmark.py     # Main orchestrator
    analyze_results.py   # Heatmaps, radar charts, statistical tests
    extractors/          # DU extraction systems (LLM, profiler, hybrid, ...)
    adapters/            # Benchmark task loaders (KramaBench, AgentBench, DACode)
    llm/                 # Minimal multi-provider LLM client
  results/               # Pre-computed results for validation
  paper/                 # VLDB paper source and figures
  data/                  # Benchmark data (download script provided)
  scripts/               # Convenience shell scripts
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download benchmark data

```bash
bash data/download_data.sh
```

### 3. Validate pre-computed results

```python
import json
from du_benchmark.schema import DUOutput
from du_benchmark.metrics import score_du_quality

# Load pre-computed results
with open("results/du_benchmark_final.json") as f:
    results = json.load(f)
print(f"Loaded {len(results)} system results")

# Verify schema imports
du = DUOutput()
print("Schema OK")
```

### 4. Generate fresh consensus (requires API keys)

```bash
export DEEPSEEK_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."

python -m du_benchmark.consensus \
    --tasks all \
    --models deepseek,claude-sonnet,gemini-flash \
    --output results/consensus/
```

### 5. Run full benchmark

```bash
bash scripts/run_benchmark.sh
```

## Scoring

The DU Quality Score is a weighted combination of per-dimension scores:

```
Q = 0.25*D1 + 0.25*D2 + 0.15*D3 + 0.15*D4 + 0.20*D5
```

- **D2 and D4** are scored deterministically (join keys actually work, file formats load correctly)
- **D1, D3, D5** are scored against a multi-LLM consensus silver standard (3 independent models)
- **Bootstrap 95% CIs** are provided for all aggregate scores

## Pre-Computed Results

The `results/` directory contains all experimental outputs needed to reproduce the paper's tables and figures:

- `consensus_decontaminated/` -- 3-model silver standard (DeepSeek + Claude + Gemini)
- `analysis_decontaminated/` -- Score tables, heatmaps, radar charts
- `ablation/` -- Causal DU ablation results (5 conditions)
- `du_benchmark_final.json` -- Consolidated per-system, per-task scores

## Causal Ablation

The ablation experiment (Section 5 of the paper) tests the causal impact of DU quality on downstream pipeline success by injecting different quality levels into the ADP-MA pipeline:

| Condition | Description |
|-----------|-------------|
| consensus | 3-LLM silver M\* (near-oracle upper bound) |
| llm-only | Single-model M\* |
| profiler | ydata-profiling (D2/D4 only) |
| no-du | Empty metadata (lower bound) |
| scrambled | Wrong task's M\* (noise floor) |

> **Note:** Running the ablation experiment requires the full ADP-MA pipeline (`pip install adp-ma`). Pre-computed results are provided in `results/ablation/`.

## Systems Evaluated

| System | Type | DU Quality |
|--------|------|------------|
| ADP-MA | Agentic pipeline | 0.523 |
| DR-DU (ours) | Dimension-routed hybrid | 0.501 |
| Profile+LLM | Profiler + LLM hybrid | 0.445 |
| LLM-only | Single LLM prompt | 0.394 |
| DIN-SQL | Text-to-SQL | 0.371 |
| ydata-profiling | Statistical profiler | 0.299 |

## Citation

```bibtex
@article{dubenchmark2026,
  title={Towards Principled Benchmarking of Automated Data Understanding},
  author={...},
  journal={Proceedings of the VLDB Endowment},
  year={2026}
}
```

## License

This benchmark code is released under the MIT License. The underlying benchmark datasets (KramaBench, AgentBench, DACode) retain their original licenses.
