#!/usr/bin/env bash
# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
# Run causal DU ablation experiment.
#
# NOTE: This requires the full ADP-MA pipeline (pip install adp-ma).
# Pre-computed results are available in results/ablation/.
#
# Usage:
#   DEEPSEEK_API_KEY="sk-..." bash scripts/run_ablation.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running DU ablation experiment..."
echo "NOTE: Requires ADP-MA pipeline to be installed."
python -m du_benchmark.run_ablation \
    --conditions all \
    --provider deepseek \
    --model deepseek-chat \
    "$@"

echo "Done. Results in results/ablation/"
