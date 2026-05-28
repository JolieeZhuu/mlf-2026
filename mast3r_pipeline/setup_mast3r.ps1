<#
.SYNOPSIS
    One-click setup for the MASt3R 3D reconstruction pipeline.

.DESCRIPTION
    This script:
      1. Clones the naver/mast3r repository (with DUSt3R submodule)
      2. Creates a conda environment with PyTorch + CUDA
      3. Installs Python dependencies (MASt3R + Open3D + extras)
      4. Downloads the pre-trained MASt3R model checkpoint

    Run this script ONCE before using reconstruct_mast3r.py.

.NOTES
    Prerequisites:
      - Git
      - Conda (Anaconda or Miniconda)
      - NVIDIA GPU with CUDA support
      - ~3 GB disk space for model checkpoint
#>

param(
    [string]$CudaVersion = "12.1",
    [switch]$SkipClone,
    [switch]$SkipEnv,
    [switch]$SkipCheckpoint
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== MASt3R Pipeline Setup ===" -ForegroundColor Cyan
Write-Host "Working directory: $ScriptDir" -ForegroundColor Gray

# ─── Step 1: Clone MASt3R ───────────────────────────────────────────
if (-not $SkipClone) {
    $mast3rDir = Join-Path $ScriptDir "mast3r"
    if (Test-Path $mast3rDir) {
        Write-Host "[1/4] mast3r/ already exists, skipping clone." -ForegroundColor Yellow
    } else {
        Write-Host "[1/4] Cloning naver/mast3r (with --recursive for DUSt3R submodule)..." -ForegroundColor Green
        git clone --recursive https://github.com/naver/mast3r "$mast3rDir"
        if ($LASTEXITCODE -ne 0) { throw "git clone failed." }
    }
} else {
    Write-Host "[1/4] Skipping clone (--SkipClone)." -ForegroundColor Yellow
}

# ─── Step 2: Create conda environment ──────────────────────────────
if (-not $SkipEnv) {
    Write-Host "[2/4] Creating conda environment 'mast3r'..." -ForegroundColor Green

    # Check if environment already exists
    $envList = conda env list 2>&1
    if ($envList -match "\bmast3r\b") {
        Write-Host "  Environment 'mast3r' already exists. Updating..." -ForegroundColor Yellow
        conda activate mast3r
    } else {
        conda create -n mast3r python=3.11 cmake=3.14.0 -y
        conda activate mast3r
    }

    Write-Host "  Installing PyTorch with CUDA $CudaVersion..." -ForegroundColor Green
    conda install pytorch torchvision "pytorch-cuda=$CudaVersion" -c pytorch -c nvidia -y

    Write-Host "  Installing MASt3R requirements..." -ForegroundColor Green
    $mast3rDir = Join-Path $ScriptDir "mast3r"
    pip install -r (Join-Path $mast3rDir "requirements.txt")
    pip install -r (Join-Path (Join-Path $mast3rDir "dust3r") "requirements.txt")

    Write-Host "  Installing pipeline dependencies (Open3D, trimesh, etc.)..." -ForegroundColor Green
    pip install open3d trimesh Pillow numpy opencv-python matplotlib tqdm scipy
} else {
    Write-Host "[2/4] Skipping environment setup (--SkipEnv)." -ForegroundColor Yellow
}

# ─── Step 3: (Optional) Compile CUDA kernels for RoPE ─────────────
Write-Host "[3/4] Attempting to compile CUDA kernels for RoPE (optional, for speed)..." -ForegroundColor Green
$curopeDir = Join-Path $ScriptDir "mast3r" "dust3r" "croco" "models" "curope"
if (Test-Path $curopeDir) {
    Push-Location $curopeDir
    try {
        python setup.py build_ext --inplace 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  CUDA kernel compilation failed (non-critical). Falling back to pure PyTorch." -ForegroundColor Yellow
        } else {
            Write-Host "  CUDA kernels compiled successfully." -ForegroundColor Green
        }
    } catch {
        Write-Host "  CUDA kernel compilation failed (non-critical). Falling back to pure PyTorch." -ForegroundColor Yellow
    }
    Pop-Location
} else {
    Write-Host "  curope directory not found, skipping." -ForegroundColor Yellow
}

# ─── Step 4: Download model checkpoint ─────────────────────────────
if (-not $SkipCheckpoint) {
    $checkpointDir = Join-Path $ScriptDir "mast3r" "checkpoints"
    $checkpointFile = Join-Path $checkpointDir "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

    if (Test-Path $checkpointFile) {
        Write-Host "[4/4] Checkpoint already exists, skipping download." -ForegroundColor Yellow
    } else {
        Write-Host "[4/4] Downloading MASt3R checkpoint (~700 MB)..." -ForegroundColor Green
        New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null

        $url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        try {
            Invoke-WebRequest -Uri $url -OutFile $checkpointFile -UseBasicParsing
            Write-Host "  Checkpoint downloaded to: $checkpointFile" -ForegroundColor Green
        } catch {
            Write-Host "  Download via Invoke-WebRequest failed. Trying curl..." -ForegroundColor Yellow
            curl.exe -L -o $checkpointFile $url
            if ($LASTEXITCODE -ne 0) { throw "Checkpoint download failed." }
        }
    }
} else {
    Write-Host "[4/4] Skipping checkpoint download (--SkipCheckpoint)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "To activate the environment:  conda activate mast3r" -ForegroundColor White
Write-Host "To run the pipeline:          python reconstruct_mast3r.py --image_dir ..\Example\images" -ForegroundColor White
