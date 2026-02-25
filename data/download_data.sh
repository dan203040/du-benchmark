#!/usr/bin/env bash
# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
# Download benchmark datasets for DU Benchmark.
#
# This script clones/downloads the three upstream benchmarks:
#   - KramaBench (105 tasks)
#   - AgentBench DBBench (100 tasks)
#   - DACode (52 tasks)
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
    git clone --depth 1 https://github.com/UIUC-Chatbot/KramaBench.git "${DATA_DIR}/kramabench"
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
    git clone --depth 1 https://github.com/yiyihum/DACode.git "${DATA_DIR}/da_code"
    echo "  Done."
fi

echo ""
echo "=== All benchmark data downloaded ==="
echo "Total tasks: 257 (KramaBench 105 + AgentBench 100 + DACode 52)"
