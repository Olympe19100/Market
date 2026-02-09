#!/bin/bash
# =============================================================================
# RunPod Setup Script for PPO Market Maker Training
# =============================================================================
# Usage: bash scripts/runpod_setup.sh
#
# Recommended RunPod configs:
#   - A100 40GB: ~4x speedup vs RTX 3060
#   - H100 80GB: ~8x speedup vs RTX 3060
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "PPO Market Maker - RunPod Setup"
echo "=============================================="

# 1. Clone repo
echo "[1/5] Cloning repository..."
if [ ! -d "Market" ]; then
    git clone https://github.com/Olympe19100/Market.git
fi
cd Market

# 2. Install dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy rich tensorboard
pip install mamba-ssm causal-conv1d  # Mamba dependencies

# 3. Download data (you need to upload your data or download it)
echo "[3/5] Setting up data directory..."
mkdir -p data/raw_methusdt
echo "NOTE: Upload your data to data/raw_methusdt/"
echo "      Or use: scp -r local_data/ runpod:~/Market/data/raw_methusdt/"

# 4. Verify GPU
echo "[4/5] Verifying GPU..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Memory: {props.total_memory / 1024**3:.1f} GB')
"

# 5. Test training config
echo "[5/5] Testing configuration..."
python -c "
from core.auto_config import create_training_config
# This will fail if data is not uploaded yet
try:
    config = create_training_config('data/raw_methusdt')
    print(f'Config loaded successfully!')
    print(f'  n_envs:     {config[\"training\"][\"n_envs\"]}')
    print(f'  batch_size: {config[\"training\"][\"batch_size\"]}')
except Exception as e:
    print(f'Config test skipped (upload data first): {e}')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Upload your data:  scp -r data/raw_methusdt/ runpod:~/Market/data/"
echo "  2. Start training:    python train.py"
echo "  3. Monitor with:      tensorboard --logdir runs/"
echo ""
echo "GPU-optimized settings will be auto-detected."
echo "Expected speedup on A100: ~4x, on H100: ~8x"
