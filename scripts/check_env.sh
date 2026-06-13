#!/usr/bin/env bash

set -u

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

ROLE="all"
PYTHON_BIN="${PYTHON:-python}"

usage() {
    cat <<'USAGE'
Usage: scripts/check_env.sh [--role all|model|coordinator|emb_pool|pim]

Checks the local machine for the software expected by CloverRec.
Use --role model on a GPU server, --role coordinator on the host/coordinator
server, and --role emb_pool or --role pim on the UPMEM PIM server.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role)
            ROLE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$ROLE" in
    all|model|coordinator|emb_pool|pim) ;;
    *)
        echo "Invalid role: $ROLE" >&2
        usage >&2
        exit 2
        ;;
esac

FAILURES=0
WARNINGS=0

ok() {
    echo "[OK]   $*"
}

warn() {
    echo "[WARN] $*"
    WARNINGS=$((WARNINGS + 1))
}

fail() {
    echo "[FAIL] $*"
    FAILURES=$((FAILURES + 1))
}

check_cmd() {
    local cmd="$1"
    local label="$2"
    local required="${3:-yes}"

    if command -v "$cmd" >/dev/null 2>&1; then
        ok "$label: $(command -v "$cmd")"
    elif [[ "$required" == "yes" ]]; then
        fail "$label: command '$cmd' not found"
    else
        warn "$label: command '$cmd' not found"
    fi
}

check_import() {
    local module="$1"
    local label="$2"
    local required="${3:-yes}"

    if "$PYTHON_BIN" - "$module" >/dev/null 2>&1 <<'PY'
import importlib
import sys

importlib.import_module(sys.argv[1])
PY
    then
        ok "$label: import $module"
    elif [[ "$required" == "yes" ]]; then
        fail "$label: cannot import $module"
    else
        warn "$label: cannot import $module"
    fi
}

echo "CloverRec environment check"
echo "Role: $ROLE"
echo "Python: $PYTHON_BIN"
echo

check_cmd "$PYTHON_BIN" "Python"
check_cmd g++ "g++"
check_import numpy "NumPy"
check_import torch "PyTorch"
check_import pybind11 "pybind11"
check_import pandas "pandas"
check_import rpyc "RPyC"

if [[ "$ROLE" == "all" || "$ROLE" == "model" ]]; then
    echo
    echo "GPU server checks"
    check_cmd nvidia-smi "NVIDIA driver"
    check_cmd nvcc "CUDA compiler" "no"
fi

if [[ "$ROLE" == "all" || "$ROLE" == "model" || "$ROLE" == "coordinator" || "$ROLE" == "emb_pool" || "$ROLE" == "pim" ]]; then
    echo
    echo "RDMA checks"
    check_import pyverbs "pyverbs/RDMA Python bindings"
    check_cmd ibv_devinfo "ibverbs device info" "no"
    check_cmd rdma "rdma tool" "no"
    if command -v rdma >/dev/null 2>&1; then
        echo
        echo "RDMA links:"
        rdma link || true
    fi
fi

if [[ "$ROLE" == "all" || "$ROLE" == "emb_pool" || "$ROLE" == "pim" ]]; then
    echo
    echo "UPMEM PIM checks"
    check_cmd dpu-upmem-dpurte-clang "UPMEM DPU compiler"
    check_cmd dpu-ls "UPMEM DPU listing tool" "no"
    if [[ -d /usr/include/dpu ]]; then
        ok "UPMEM headers: /usr/include/dpu"
    else
        fail "UPMEM headers: /usr/include/dpu not found"
    fi
fi

echo
echo "Network interfaces visible on this machine:"
if command -v ip >/dev/null 2>&1; then
    ip -brief addr || true
    echo
    echo "Use the InfiniBand/RDMA interface IP for --model-ip and --emb-pool-ip."
else
    warn "ip command not available"
fi

echo
echo "Summary: $FAILURES failure(s), $WARNINGS warning(s)"
if [[ "$FAILURES" -gt 0 ]]; then
    exit 1
fi
