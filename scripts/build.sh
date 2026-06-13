#!/usr/bin/env bash

set -euo pipefail

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

SYSTEM=""
COMPONENT="auto"
PYTHON_BIN="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage: scripts/build.sh --system cloverrec|local_emb|remote_emb|naive_pim_emb [--component auto|client|pim|all]

Components:
  auto    Build what is available on this machine. PIM code is skipped if UPMEM tools are absent.
  client  Build the client_cache extension when the selected system has one.
  pim     Build the UPMEM DPU binary and pim_module extension.
  all     Build both client and PIM components, failing if a requested component cannot build.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system)
            SYSTEM="$2"
            shift 2
            ;;
        --component)
            COMPONENT="$2"
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

case "$SYSTEM" in
    cloverrec|local_emb|remote_emb|naive_pim_emb) ;;
    "")
        echo "--system is required" >&2
        usage >&2
        exit 2
        ;;
    *)
        echo "Invalid system: $SYSTEM" >&2
        usage >&2
        exit 2
        ;;
esac

case "$COMPONENT" in
    auto|client|pim|all) ;;
    *)
        echo "Invalid component: $COMPONENT" >&2
        usage >&2
        exit 2
        ;;
esac

SYSTEM_DIR="$REPO_ROOT/$SYSTEM"

build_client() {
    if [[ -f "$SYSTEM_DIR/client_cache_setup.py" ]]; then
        echo "Building client_cache for $SYSTEM"
        (cd "$SYSTEM_DIR" && "$PYTHON_BIN" client_cache_setup.py build_ext --inplace)
    else
        echo "No client_cache extension for $SYSTEM"
    fi
}

can_build_pim() {
    [[ -f "$SYSTEM_DIR/setup.py" && -f "$SYSTEM_DIR/pim_dpu.c" ]] || return 1
    command -v dpu-upmem-dpurte-clang >/dev/null 2>&1 || return 1
    [[ -d /usr/include/dpu ]] || return 1
}

build_pim() {
    if [[ ! -f "$SYSTEM_DIR/setup.py" || ! -f "$SYSTEM_DIR/pim_dpu.c" ]]; then
        echo "No PIM module for $SYSTEM"
        return 0
    fi

    if ! command -v dpu-upmem-dpurte-clang >/dev/null 2>&1; then
        echo "UPMEM compiler not found; cannot build PIM module for $SYSTEM" >&2
        return 1
    fi

    echo "Building UPMEM DPU binary for $SYSTEM"
    (
        cd "$SYSTEM_DIR"
        dpu-upmem-dpurte-clang -DNR_TASKLETS=16 -DSTACK_SIZE_DEFAULT=256 -O2 -o pim_dpu pim_dpu.c
        "$PYTHON_BIN" setup.py build_ext --inplace
    )
}

case "$COMPONENT" in
    client)
        build_client
        ;;
    pim)
        build_pim
        ;;
    all)
        build_client
        build_pim
        ;;
    auto)
        build_client
        if can_build_pim; then
            build_pim
        elif [[ -f "$SYSTEM_DIR/setup.py" || -f "$SYSTEM_DIR/pim_dpu.c" ]]; then
            echo "Skipping PIM build for $SYSTEM because UPMEM tools/headers are not available"
        fi
        ;;
esac
