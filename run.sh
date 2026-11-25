#!/usr/bin/env bash
# One-command runner: creates venv (first time), installs deps, processes photos.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "Creating virtual environment .venv ..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip -q
pip install -q -r requirements.txt

python remove_watermark.py --input photos --output photos_cleaned "$@"
