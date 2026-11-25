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
# simple-lama-inpainting pins numpy<2 / pillow<10, which break resolution on
# Python 3.13+; install it without deps (runtime code is compatible).
pip install -q --no-deps simple-lama-inpainting

python remove_watermark.py --input photos --output photos_cleaned "$@"
