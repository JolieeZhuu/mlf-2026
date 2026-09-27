#!/usr/bin/env bash
#
# One-click setup for the MASt3R 3D reconstruction pipeline on macOS (Apple Silicon / Intel).
#
# Steps:
#   1. Clones naver/mast3r recursively (includes dust3r submodule)
#   2. Creates or updates 'mast3r' conda environment with PyTorch (MPS support)
#   3. Installs MASt3R, DUSt3R, and pipeline dependencies
#   4. Downloads the MASt3R model checkpoint (~700 MB)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== MASt3R Pipeline Setup (macOS) ==="
echo "Working directory: $SCRIPT_DIR"

# ─── Step 1: Clone MASt3R ───────────────────────────────────────────
MAST3R_DIR="$SCRIPT_DIR/mast3r"
if [ -d "$MAST3R_DIR" ] && [ -n "$(ls -A "$MAST3R_DIR" 2>/dev/null)" ]; then
    echo "[1/4] mast3r/ already exists, skipping clone."
else
    echo "[1/4] Cloning naver/mast3r (with --recursive for DUSt3R submodule)..."
    rm -rf "$MAST3R_DIR"
    git clone --recursive https://github.com/naver/mast3r "$MAST3R_DIR"
fi

# ─── Step 2: Initialize Conda & Create environment ──────────────────
echo "[2/4] Setting up conda environment 'mast3r'..."

# Find conda
if [ -z "${CONDA_EXE:-}" ]; then
    if [ -f "/opt/miniconda3/bin/conda" ]; then
        CONDA_EXE="/opt/miniconda3/bin/conda"
    elif [ -f "$HOME/miniconda3/bin/conda" ]; then
        CONDA_EXE="$HOME/miniconda3/bin/conda"
    elif [ -f "/opt/homebrew/bin/conda" ]; then
        CONDA_EXE="/opt/homebrew/bin/conda"
    else
        CONDA_EXE="$(which conda || true)"
    fi
fi

if [ -z "$CONDA_EXE" ] || [ ! -x "$CONDA_EXE" ]; then
    echo "ERROR: conda executable not found. Please ensure conda is installed and available in PATH."
    exit 1
fi

CONDA_BASE="$("$CONDA_EXE" info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -qE "^\bmast3r\b"; then
    echo "  Environment 'mast3r' already exists. Activating..."
    conda activate mast3r
else
    echo "  Creating 'mast3r' conda environment with Python 3.11..."
    conda create -n mast3r python=3.11 cmake -y
    conda activate mast3r
fi

# Open3D on macOS ARM64 links dynamically to Homebrew's libusb
if [ ! -f "/opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib" ]; then
    echo "  Installing libusb via Homebrew (required by Open3D on macOS)..."
    if command -v brew >/dev/null 2>&1; then
        brew install libusb
    else
        echo "  WARNING: Homebrew not found. Open3D may require libusb from /opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib."
    fi
fi

echo "  Installing PyTorch, torchvision, and image libraries..."
conda install pytorch torchvision jpeg libjpeg-turbo -c pytorch -c conda-forge -y

echo "  Installing MASt3R and DUSt3R requirements..."
pip install -r "$MAST3R_DIR/requirements.txt"
pip install -r "$MAST3R_DIR/dust3r/requirements.txt"

echo "  Installing pipeline dependencies (Open3D, trimesh, Pillow, opencv, etc.)..."
pip install open3d trimesh Pillow numpy opencv-python matplotlib tqdm scipy

# ─── Step 3: RoPE CUDA kernel info ─────────────────────────────────
echo "[3/4] Checking RoPE implementation..."
echo "  Note: On Apple Silicon, RoPE CUDA kernels (curope) are omitted."
echo "  DUSt3R and CroCo will automatically fall back to pure PyTorch."

# ─── Step 4: Download model checkpoint ─────────────────────────────
CHECKPOINT_DIR="$MAST3R_DIR/checkpoints"
CHECKPOINT_FILE="$CHECKPOINT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "[4/4] Checkpoint already exists at: $CHECKPOINT_FILE"
else
    echo "[4/4] Downloading MASt3R checkpoint (~700 MB)..."
    mkdir -p "$CHECKPOINT_DIR"
    curl -L -o "$CHECKPOINT_FILE" "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    echo "  Checkpoint downloaded successfully."
fi

# ─── Verify Environment ─────────────────────────────────────────────
echo ""
echo "=== Verifying Environment ==="
python -c "
import torch
print(f'PyTorch version:       {torch.__version__}')
print(f'Apple Silicon MPS:     {torch.backends.mps.is_available()}')
import open3d as o3d
print(f'Open3D version:        {o3d.__version__}')
"

echo ""
echo "=== Setup Complete! ==="
echo "To activate the environment:"
echo "    conda activate mast3r"
echo ""
echo "Tip: To allow PyTorch to fall back to CPU if an MPS operator is unsupported:"
echo "    export PYTORCH_ENABLE_MPS_FALLBACK=1"
