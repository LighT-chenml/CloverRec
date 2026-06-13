#!/usr/bin/env bash

set -euo pipefail

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$REPO_ROOT/scripts/run_smoke.py" "$@"
