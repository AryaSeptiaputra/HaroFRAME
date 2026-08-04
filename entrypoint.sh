#!/bin/bash
# HaroFRAME — vast.ai "On-start Script"
#
# For a vast.ai instance created from the official "PyTorch" template (not
# the root Dockerfile's custom-image path -- see VAST_GUIDE.md for that
# alternative). Paste this script into the instance/template's "On-start
# Script" field. Runs every time the instance (re)starts -- idempotent, safe
# on a fresh instance, a restarted one, or one stopped/started repeatedly.
# Prepares the environment so that once you connect (Instance Portal / exec),
# you can go straight to:
#   python3 scripts/generate_video.py test_images/alex.jpg "a person smiling"
#   python3 scripts/interactive_generate.py
# See VAST_GUIDE.md for the full walkthrough this automates.
#
# Model weights (base SDXL checkpoint, LoRAs, SAM) are prefetched in step 7 from
# whatever the config says will be used — set HAROFRAME_PREFETCH=false to skip.
#
# Secrets are deliberately NOT set here: export
# HAROFRAME_IDENTITY__HF_TOKEN and (optionally)
# HAROFRAME_GENERATION__LORA__CIVITAI_API_KEY via vast.ai's own "Environment
# Variables" field on the instance/template — never hardcode a real token
# value in this script, it's version-controlled.

set -uo pipefail

REPO_URL="https://github.com/AryaSeptiaputra/HaroFRAME.git"
REPO_DIR="$HOME/HaroFRAME"
BRANCH="main"

echo "=== [entrypoint] $(date) starting ==="

# --- 1. Activate the template's pre-installed PyTorch venv, if present ---
# The official vast.ai PyTorch template ships PyTorch pre-installed at
# /venv/main, pre-built against that instance's CUDA/driver -- activating it
# (instead of falling back to plain `python3`, which may resolve to a
# different, unaccelerated interpreter in a non-interactive on-start-script
# shell) is what makes step 3 below reuse that build rather than pulling a
# second, possibly mismatched torch wheel from PyPI.
if [ -f /venv/main/bin/activate ]; then
    echo "[entrypoint] activating vast.ai PyTorch template venv (/venv/main)..."
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

# --- 2. Clone or update the repo ---
if [ -d "$REPO_DIR/.git" ]; then
    echo "[entrypoint] repo exists at $REPO_DIR, pulling latest $BRANCH..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "[entrypoint] cloning $REPO_URL into $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout "$BRANCH"
fi

# --- 3. System deps (mirrors the apt-get layer in the root Dockerfile) ---
# opencv-python needs libgl1/libglib2.0-0 even headless; build-essential
# covers any package that needs to compile from sdist. imageio-ffmpeg bundles
# its own ffmpeg binary, so no system ffmpeg install is needed here.
MISSING_APT_PKGS=""
for pkg in libgl1 libglib2.0-0 build-essential; do
    dpkg -s "$pkg" &> /dev/null || MISSING_APT_PKGS="$MISSING_APT_PKGS $pkg"
done
if [ -n "$MISSING_APT_PKGS" ]; then
    echo "[entrypoint] installing system deps:$MISSING_APT_PKGS"
    apt-get update && apt-get install -y --no-install-recommends $MISSING_APT_PKGS
fi

# --- 4. Python deps ---
# pyproject.toml pins torch>=2.4 as a hard dependency, but pip only reinstalls
# a package if the currently-active interpreter's installed version doesn't
# already satisfy that constraint -- since step 1 activated the template's
# venv first, its pre-installed (CUDA-matched) torch build already satisfies
# >=2.4 and is left untouched; only the project's own deps and onnxruntime-gpu
# (from the gpu extra) actually get installed/changed here. The "garment" extra
# (segment-anything) IS installed, matching the Dockerfile, because the stage-1
# inpaint pass is enabled by default (GenerationConfig.inpaint.enabled) and
# nothing runs without it; its SAM checkpoint is fetched in step 6. The
# "restoration" extra stays opt-in -- see VAST_GUIDE.md's Troubleshooting
# section for its manual install command.
echo "[entrypoint] installing python deps (pip install -e .[gpu,garment])..."
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[gpu,garment]"

# --- 5. Cache/output directories ---
# .cache/models is where SDXL base/InstantID/IP-Adapter/LoRA weights get
# lazily downloaded to on first use (see IdentityConfig.cache_dir) -- created
# up front so a persistent-disk mount at this path (if configured) is used
# from the very first run instead of the first run creating it mid-download.
mkdir -p .cache/models outputs test_images

# --- 6. Sanity checks ---
echo "[entrypoint] torch/cuda check:"
python3 -c "import torch; print(' torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

if [ -n "${HAROFRAME_IDENTITY__HF_TOKEN:-}" ]; then
    echo "[entrypoint] HAROFRAME_IDENTITY__HF_TOKEN is set."
else
    echo "[entrypoint] WARNING: HAROFRAME_IDENTITY__HF_TOKEN is NOT set — gated/"
    echo "[entrypoint]          rate-limited HF models may fail to download."
fi

IPADAPTER_ON="${HAROFRAME_IDENTITY__IPADAPTER__ENABLED:-false}"
INSTANTID_ON="${HAROFRAME_IDENTITY__INSTANTID__ENABLED:-false}"
if [ "$IPADAPTER_ON" = "true" ] && [ "$INSTANTID_ON" = "true" ]; then
    echo "[entrypoint] WARNING: both HAROFRAME_IDENTITY__IPADAPTER__ENABLED and"
    echo "[entrypoint]          HAROFRAME_IDENTITY__INSTANTID__ENABLED are true —"
    echo "[entrypoint]          this raises ConflictingAdapterConfigError at runtime, pick one."
elif [ "$IPADAPTER_ON" != "true" ] && [ "$INSTANTID_ON" != "true" ]; then
    echo "[entrypoint] WARNING: neither HAROFRAME_IDENTITY__IPADAPTER__ENABLED nor"
    echo "[entrypoint]          HAROFRAME_IDENTITY__INSTANTID__ENABLED is true — set exactly"
    echo "[entrypoint]          one before running the pipeline."
fi

# --- 7. Prefetch model weights ---
# Pulls the base SDXL checkpoint, every enabled LoRA, and (while stage-1
# inpainting is enabled) the SAM checkpoint, so the first generate() doesn't
# stall for tens of GB mid-run. Driven entirely by config — prefetch_models.py
# chooses nothing itself, it warms whatever HAROFRAME_IDENTITY__BASE_SDXL_MODEL /
# HAROFRAME_GENERATION__LORA__ENTRIES / HAROFRAME_GENERATION__INPAINT__* already
# say will be used, and no-ops on anything already cached (so restarts are
# cheap). Deliberately non-fatal: a download failure leaves a usable instance
# that just re-downloads lazily at generation time.
#
# Set HAROFRAME_PREFETCH=false to skip — e.g. on a metered connection, or when
# .cache/models is a persistent volume you have already populated.
if [ "${HAROFRAME_PREFETCH:-true}" = "true" ]; then
    echo "[entrypoint] prefetching model weights (HAROFRAME_PREFETCH=false to skip)..."
    if ! python3 scripts/prefetch_models.py; then
        echo "[entrypoint] WARNING: prefetch did not fully succeed — see the summary above."
        echo "[entrypoint]          The instance is still usable; missing weights download"
        echo "[entrypoint]          lazily on first use instead."
    fi
else
    echo "[entrypoint] HAROFRAME_PREFETCH=false — skipping model prefetch."
fi

echo "=== [entrypoint] $(date) done — ready for: python3 scripts/generate_video.py <photo> \"<prompt>\" --inpaint-prompt \"<outfit>\" --out outputs/clip.mp4 ==="
