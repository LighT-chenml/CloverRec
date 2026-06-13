#!/usr/bin/env bash

set -euo pipefail

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

PYTHON_BIN="${PYTHON:-python}"
WORK_DIR="${TMPDIR:-/tmp}/cloverrec_pyverbs_apt"

usage() {
    cat <<'USAGE'
Usage: scripts/install_pyverbs_from_apt.sh

Downloads the Ubuntu python3-pyverbs package, extracts it without root, and
copies the pyverbs Python bindings into the active Python environment.

Run it after activating the CloverRec Conda environment, or pass PYTHON:

  conda activate CloverRec
  scripts/install_pyverbs_from_apt.sh

  PYTHON=/path/to/env/bin/python scripts/install_pyverbs_from_apt.sh

This is intended for Ubuntu 22.04 / Python 3.10 systems where pyverbs is needed
but sudo is unavailable. If you have sudo, installing python3-pyverbs system-wide
is simpler.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get is required for this helper." >&2
    exit 1
fi

if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "dpkg-deb is required for this helper." >&2
    exit 1
fi

SITE_PACKAGES="$("$PYTHON_BIN" - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"

PY_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "$PY_VERSION" != "3.10" ]]; then
    echo "python3-pyverbs from Ubuntu 22.04 targets Python 3.10; found Python $PY_VERSION." >&2
    exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

(
    cd "$WORK_DIR"
    apt-get download python3-pyverbs
    dpkg-deb -x python3-pyverbs_*.deb extract
)

SRC="$WORK_DIR/extract/usr/lib/python3/dist-packages/pyverbs"
if [[ ! -d "$SRC" ]]; then
    echo "Downloaded package did not contain pyverbs bindings." >&2
    exit 1
fi

rm -rf "$SITE_PACKAGES/pyverbs"
cp -a "$SRC" "$SITE_PACKAGES/"

"$PYTHON_BIN" - <<'PY'
from pyverbs.device import Context

ctx = Context(name="mlx5_0")
port = ctx.query_port(1)
print(f"pyverbs installed; opened {ctx.name}, lid={port.lid}")
PY
