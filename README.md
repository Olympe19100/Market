# PPO Market Maker

Deep Reinforcement Learning market maker using PPO with Mamba-based LOB encoder.

## Features

- **100% Data-Driven**: All hyperparameters derived from market data
- **Avellaneda-Stoikov**: Theoretically grounded reward function
- **GPU Auto-Scaling**: Automatically optimizes for your GPU
- **torch.compile**: 20-40% speedup on Linux

## Quick Start

```bash
# Local training (auto-detects GPU)
python train.py

# With custom settings
python train.py --n-envs 32 --batch-size 1024
```

## GPU Scaling

| GPU | VRAM | n_envs | batch_size | Speedup |
|-----|------|--------|------------|---------|
| RTX 3060 | 12GB | 16 | 512 | 1x |
| RTX 4090 | 24GB | 32 | 1024 | 2x |
| A100 | 40GB | 64 | 2048 | 4x |
| H100 | 80GB | 128 | 4096 | 8x |

## RunPod Deployment

```bash
# On RunPod instance
git clone https://github.com/Olympe19100/Market.git
cd Market
bash scripts/runpod_setup.sh

# Upload your data
scp -r data/raw_methusdt/ runpod:~/Market/data/

# Start training
python train.py
```

## Architecture

```
LOB Data → Mamba Encoder → Actor-Critic Heads → PPO Update
              ↓
    Market Features (iVPIN, TFI, Volatility)
```

## Key Components

- `train.py`: Main training loop with async vectorized environments
- `core/auto_config.py`: Data-driven hyperparameter derivation
- `models/rl/PPO_agent.py`: PPO with adaptive controllers
- `environment/sim_env.py`: Market simulation with A-S reward

## License

MIT
