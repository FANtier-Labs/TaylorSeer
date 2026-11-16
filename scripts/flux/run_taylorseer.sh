#!/usr/bin/env bash
set -euo pipefail

# Pair entries by index. Keep both arrays the same length.
ORDERS=(2 2 2 2 1)
INTERVALS=(3 4 5 6 6)
WARMUP=${WARMUP:-3}
MODEL=${MODEL:-black-forest-labs/FLUX.1-dev}
OUTPUT=${OUTPUT:-outputs/drawbench/flux}
PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# Generate baseline first with run_base.sh using the same model and generation options.
# Usage: bash scripts/flux/run_taylorseer.sh [extra generation options]
# Example: bash scripts/flux/run_taylorseer.sh --limit 5 --steps 20
# Multi-GPU: CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/flux/run_taylorseer.sh
# Overrides: MODEL=/path/to/FLUX WARMUP=3 OUTPUT=outputs/test bash scripts/flux/run_taylorseer.sh
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ "${#ORDERS[@]}" -ne "${#INTERVALS[@]}" ]; then
    echo "ORDERS and INTERVALS must have the same length" >&2
    exit 1
fi

if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 1
fi

# Each worker loads a full model and processes its own shard of samples.
LAUNCH=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node "$NPROC_PER_NODE" --module)

COMMON=(--model "$MODEL" --warmup "$WARMUP" "$@")
BASELINE="$OUTPUT/baseline"

for i in "${!ORDERS[@]}"; do
    order=${ORDERS[$i]}
    interval=${INTERVALS[$i]}
    RUN="$OUTPUT/order_${order}_interval_${interval}_warmup_${WARMUP}"
    echo "Running order=$order interval=$interval warmup=$WARMUP"
    "${LAUNCH[@]}" eval.drawbench.generate "${COMMON[@]}" \
        --method taylorseer --order "$order" --interval "$interval" \
        --output "$RUN/images"
    "${LAUNCH[@]}" eval.drawbench.score \
        --baseline "$BASELINE" --candidate "$RUN/images" --output "$RUN/scores"
    "$PYTHON" -m eval.drawbench.export \
        --scores "$RUN/scores" --output "$RUN/report"
done
