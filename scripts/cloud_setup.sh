#!/bin/bash
# Setup script for Cloud GPU Instance (Ubuntu)

echo "🚀 Starting Cloud Environment Setup..."

# Update and install system dependencies
sudo apt-get update
sudo apt-get install -y git wget curl build-essential

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements_cloud.txt

# Install specific CUDA-optimized Mamba kernels
echo "⚙️ Installing Mamba-SSM kernels (this may take a few minutes)..."
pip install causal-conv1d>=1.4.0
pip install mamba-ssm

# Verify Installation
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# W&B Login (will prompt for API key)
echo "🔑 Please login to Weights & Biases:"
wandb login

echo "✅ Setup Complete. You can now launch training with:"
echo "accelerate launch scripts/pretrain_mamba.py -d ./processed_data --use-wandb"
