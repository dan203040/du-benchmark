#!/usr/bin/env bash
# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
# Generate multi-LLM consensus silver standard.
#
# Requires API keys for at least 3 providers:
#   export DEEPSEEK_API_KEY="sk-..."
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export GOOGLE_API_KEY="AI..."
#
# Usage:
#   bash scripts/run_consensus.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Generating multi-LLM consensus..."
python -m du_benchmark.consensus \
    --tasks all \
    --models deepseek,claude-sonnet,gemini-flash \
    --output results/consensus/ \
    --max-concurrent 5

echo "Done. Results in results/consensus/"
