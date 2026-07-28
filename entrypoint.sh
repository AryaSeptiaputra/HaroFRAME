#!/bin/bash
# HaroFRAME — vast.ai "On-start Script"
#
# Alternative to the root Dockerfile: instead of building/pushing a custom
# image, point a vast.ai instance at a stock PyTorch template and paste this
# script into "On-start Script". Runs every time the instance (re)starts —
# idempotent, safe on a fresh instance, a restarted one, or one stopped/
# started repeatedly. Prepares the environment so that once you connect
# (Instance Portal / exec), you can go straight to:
#   python3 scripts/generate_video.py test_images/alex.jpg "a person smiling"
#   python3 scripts/interactive_generate.py
# See VAST_GUIDE.md for the full walkthrough this automates (and for the
# Dockerfile-based deploy path, which is the other supported option).
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

# --- 1. Clone or update the repo ---
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

# --- 2. System deps (mirrors the apt-get layer in the root Dockerfile) ---
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

# --- 3. Python deps ---
# torch is a hard dependency in pyproject.toml (torch>=2.4, installed from
# PyPI with its own bundled CUDA runtime) rather than assumed to come from
# the template, so this pulls/upgrades it like any other project dependency
# -- unlike templates that pre-pin torch, there's no need to keep it out of
# this install. The gpu extra pulls onnxruntime-gpu; the optional
# restoration/garment extras are intentionally NOT installed here (matches
# the Dockerfile) since they're large and only some setups need them -- see
# VAST_GUIDE.md's Troubleshooting section for the manual install commands.
echo "[entrypoint] installing python deps (pip install -e .[gpu])..."
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[gpu]"

# --- 4. Cache/output directories ---
# .cache/models is where SDXL base/InstantID/IP-Adapter/LoRA weights get
# lazily downloaded to on first use (see IdentityConfig.cache_dir) -- created
# up front so a persistent-disk mount at this path (if configured) is used
# from the very first run instead of the first run creating it mid-download.
mkdir -p .cache/models outputs test_images

# --- 5. Sanity checks ---
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

echo "=== [entrypoint] $(date) done — ready for: python3 scripts/generate_video.py <photo> \"<prompt>\" --out outputs/clip.mp4 ==="
