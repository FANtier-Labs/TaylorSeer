#!/usr/bin/env bash
set -euo pipefail

WARMUP=${WARMUP:-3}
MODEL=${MODEL:-black-forest-labs/FLUX.1-dev}
OUTPUT=${OUTPUT:-outputs/drawbench/flux}
PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# Usage: bash scripts/flux/run_base.sh [extra generation options]
# Example: bash scripts/flux/run_base.sh --limit 5 --steps 20
# Multi-GPU: CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/flux/run_base.sh
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 1
fi

"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node "$NPROC_PER_NODE" --module eval.drawbench.generate \
    --model "$MODEL" --warmup "$WARMUP" "$@" \
    --method baseline --output "$OUTPUT/baseline"
