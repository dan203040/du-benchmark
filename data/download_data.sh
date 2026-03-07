#!/usr/bin/env bash
# Download benchmark datasets for DU Benchmark.
#
# This script clones/downloads the three upstream benchmarks:
#   - KramaBench (105 tasks)
#   - AgentBench DBBench (100 tasks)
#   - DACode (52 tasks)
#   - DSEval (299 tasks)
#
# Usage:
#   cd du-benchmark
#   bash data/download_data.sh

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Downloading benchmark data to ${DATA_DIR} ==="

# ── KramaBench ─────────────────────────────────────────────────────
echo ""
echo "[1/3] KramaBench..."
if [ -d "${DATA_DIR}/kramabench" ]; then
    echo "  Already exists, skipping."
else
    git clone --depth 1 https://github.com/mitdbg/Kramabench.git "${DATA_DIR}/kramabench"
    echo "  Done."
fi

# ── AgentBench ────────────────────────────────────────────────────
echo ""
echo "[2/3] AgentBench DBBench..."
if [ -d "${DATA_DIR}/agentbench" ]; then
    echo "  Already exists, skipping."
else
    git clone --depth 1 https://github.com/THUDM/AgentBench.git "${DATA_DIR}/agentbench"
    echo "  Done."
fi

# ── DACode ─────────────────────────────────────────────────────────
echo ""
echo "[3/3] DACode..."
if [ -d "${DATA_DIR}/da_code" ]; then
    echo "  Already exists, skipping."
else
    git clone --depth 1 https://github.com/yiyihum/da-code.git "${DATA_DIR}/da_code"
    echo "  Done."
fi

# ── DSEval ─────────────────────────────────────────────────────────
echo ""
echo "[4/4] DSEval..."
if [ -d "${DATA_DIR}/dseval" ]; then
    echo "  Already exists, skipping."
else
    git clone --depth 1 https://github.com/MetaCopilot/dseval.git "${DATA_DIR}/dseval"
    echo "  Done."
fi

echo ""
echo "=== All benchmark data downloaded ==="
echo "Total tasks: 556 (KramaBench 105 + AgentBench 100 + DACode 52 + DSEval 299)"
