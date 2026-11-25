@echo off
REM One-command runner for Windows: creates venv (first time), installs deps, processes photos.
cd /d %~dp0

if not exist .venv (
    echo Creating virtual environment .venv ...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip -q
pip install -q -r requirements.txt

python remove_watermark.py --input photos --output photos_cleaned %*
