@echo off
REM One-command runner for Windows: creates venv (first time), installs deps, processes photos.
cd /d %~dp0

REM --- Pick a venv creator -------------------------------------------------
REM The system Python may be too new for the dependency wheels (e.g. 3.15 has
REM no torchvision/ultralytics stack yet). If uv is available we use it to
REM fetch a standalone Python 3.13; otherwise fall back to "python -m venv".
where uv >nul 2>nul
if %errorlevel%==0 (set "HAVE_UV=1") else (set "HAVE_UV=0")

if not exist .venv (
    echo Creating virtual environment .venv ...
    if "%HAVE_UV%"=="1" (
        uv venv --python 3.13 --seed .venv
    ) else (
        python -m venv .venv
    )
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip -q
pip install -q -r requirements.txt
REM simple-lama-inpainting pins numpy<2 / pillow<10, which break resolution on
REM Python 3.13+; install it without deps (runtime code is compatible).
pip install -q --no-deps simple-lama-inpainting

python remove_watermark.py --input photos --output photos_cleaned %*
