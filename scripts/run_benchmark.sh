#!/usr/bin/env bash
# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
# Run full DU benchmark evaluation.
#
# Requires:
#   - Benchmark data downloaded (bash data/download_data.sh)
#   - At least one LLM API key set
#
# Usage:
#   bash scripts/run_benchmark.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running DU benchmark..."
python -m du_benchmark.run_benchmark \
    --benchmarks all \
    --output results/ \
    "$@"

echo "Done. Results in results/"
