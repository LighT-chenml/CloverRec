#!/usr/bin/env bash

set -euo pipefail

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-quick}"

MODEL_IP="${MODEL_IP:-10.0.0.5}"
EMB_POOL_IP="${EMB_POOL_IP:-10.0.0.11}"
EMB_POOL_HOST="${EMB_POOL_HOST:-192.168.123.7}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/cml/CloverRec}"
COORDINATOR_PYTHON="${COORDINATOR_PYTHON:-/home/cml/anaconda3/envs/CloverRec/bin/python}"
MODEL_PYTHON="${MODEL_PYTHON:-/home/cml/anaconda3/envs/CloverRec/bin/python}"
EMB_POOL_PYTHON="${EMB_POOL_PYTHON:-/home/cml/miniconda3/envs/CloverRec/bin/python}"

usage() {
    cat <<'USAGE'
Usage: scripts/run_artifact.sh [quick|full]

  quick  Check both servers, build all four systems, and run one RM1 point per
         system. This is the recommended artifact sanity check.
  full   Check both servers, build all four systems, and run the recommended
         RM1-RM4 knee matrix.
USAGE
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

case "$MODE" in
    quick)
        WORKLOADS=(RM1)
        BATCH_PROFILE="smoke"
        ;;
    full)
        WORKLOADS=(RM1 RM2 RM3 RM4)
        BATCH_PROFILE="knee"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac

for python_bin in "$COORDINATOR_PYTHON" "$MODEL_PYTHON"; do
    if [[ ! -x "$python_bin" ]]; then
        echo "Python executable not found: $python_bin" >&2
        exit 1
    fi
done

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOGS_DIR="$REPO_ROOT/results/artifact_${MODE}_${RUN_STAMP}"
SYSTEMS=(cloverrec remote_emb naive_pim_emb local_emb)

echo "CloverRec artifact evaluation"
echo "Mode: $MODE"
echo "Results: $LOGS_DIR"
echo

echo "Checking PM2 model/coordinator environment..."
PYTHON="$COORDINATOR_PYTHON" "$REPO_ROOT/scripts/check_env.sh" --role model

echo
echo "Checking PM5 embedding-pool environment..."
ssh -o BatchMode=yes -o ConnectTimeout=5 "$EMB_POOL_HOST" \
    "cd '$REMOTE_REPO_ROOT' && PYTHON='$EMB_POOL_PYTHON' scripts/check_env.sh --role emb_pool"

echo
echo "Building and running the $MODE evaluation..."
"$COORDINATOR_PYTHON" "$REPO_ROOT/scripts/run_e2e.py" \
    --systems "${SYSTEMS[@]}" \
    --workloads "${WORKLOADS[@]}" \
    --batch-profile "$BATCH_PROFILE" \
    --num-batches 100 \
    --table-size 100000 \
    --model-ip "$MODEL_IP" \
    --emb-pool-ip "$EMB_POOL_IP" \
    --emb-pool-host "$EMB_POOL_HOST" \
    --remote-repo-root "$REMOTE_REPO_ROOT" \
    --coordinator-python "$COORDINATOR_PYTHON" \
    --model-python "$MODEL_PYTHON" \
    --emb-pool-python "$EMB_POOL_PYTHON" \
    --logs-dir "$LOGS_DIR"

echo
echo "Evaluation complete. Summary: $LOGS_DIR/summary.csv"
