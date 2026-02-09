#!/usr/bin/env python
"""
Mamba-LOB SSL Pre-training Script - ULTRA OPTIMIZED

Optimizations:
- Multi-GPU with Accelerate (DDP)
- Mixed Precision (bf16/fp16) - ~2x speedup
- torch.compile (PyTorch 2.0+) - ~30% speedup
- Gradient Checkpointing - -50% VRAM
- 8-bit Adam (bitsandbytes) - -30% VRAM
- Cosine Warmup Scheduler - better convergence
- EMA Model - better final weights
- Channel-last memory format - +10% CNN
- Prefetch DataLoader - overlap data/compute
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
import gc
import copy
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from models.mamba_lob.model import LOBModel
from core.config import ModelConfig
from torch.utils.data import Dataset, DataLoader

# Try imports for optimizations
try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    print("⚠️ accelerate not found. pip install accelerate")

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
LOB_FEATURE_DIM = 40
MAX_WINDOW_SIZE = 200


# ============================================================================
# EMA Model (Exponential Moving Average)
# ============================================================================

class EMA:
    """Maintains EMA of model weights for better final model."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()
    
    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_avg
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ============================================================================
# Cosine Warmup Scheduler
# ============================================================================

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / max(1, self.warmup_steps) 
                    for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return [self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
                    for base_lr in self.base_lrs]


# ============================================================================
# Dataset
# ============================================================================

class LOBSequenceDataset(Dataset):
    def __init__(self, data_dir: str, max_window: int = MAX_WINDOW_SIZE, stride: int = MAX_WINDOW_SIZE,
                 split: str = 'train', val_ratio: float = 0.1):
        self.max_window = max_window
        self.data_dir = Path(data_dir)
        
        self.features = np.load(self.data_dir / "features.npy", mmap_mode='r')
        
        total_records = len(self.features)
        max_start_idx = total_records - max_window - 1
        all_indices = np.arange(0, max_start_idx, stride)
        
        split_idx = int(len(all_indices) * (1 - val_ratio))
        self.valid_indices = all_indices[:split_idx] if split == 'train' else all_indices[split_idx:]
        
        logger.info(f"{split.upper()}: {len(self.valid_indices):,} sequences")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        input_seq = np.array(self.features[start_idx:start_idx + self.max_window])
        target = np.array(self.features[start_idx + self.max_window])
        return torch.from_numpy(input_seq), torch.from_numpy(target)


# ============================================================================
# Training
# ============================================================================

def train_ssl_optimized(config: dict):
    """Ultra-optimized training loop."""
    
    # Accelerator
    if HAS_ACCELERATE:
        accelerator = Accelerator(
            mixed_precision='no', # AMP Disabled by user request
            gradient_accumulation_steps=config['grad_accum_steps'],
        )
        device = accelerator.device
        is_main = accelerator.is_main_process
    else:
        accelerator = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
    
    if is_main:
        logger.info(f"🖥️  Device: {device}")
        logger.info(f"🎮 GPUs: {torch.cuda.device_count()}")
        logger.info(f"📊 Mixed Precision: Disabled")
        
        # Initialize W&B
        if config['use_wandb'] and HAS_WANDB:
            wandb.init(
                project=config['project_name'],
                config=config,
                name=f"mamba-lob-ssl-{datetime.now().strftime('%m%d-%H%M')}"
            )
            logger.info("🚀 Weights & Biases initialized")
    
    # Data
    train_dataset = LOBSequenceDataset(config['data_dir'], config['max_window'], config['stride'], 'train')
    val_dataset = LOBSequenceDataset(config['data_dir'], config['max_window'], config['stride'], 'val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True, 
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    # Model
    model_config = ModelConfig(
        n_features=config['n_features'],
        window_size=config['max_window'],
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers']
    )
    model = LOBModel(model_config)
    
    # Gradient Checkpointing for memory efficiency
    if config['gradient_checkpointing']:
        if hasattr(model, 'mamba'):
            model.mamba.gradient_checkpointing = True
        if is_main:
            logger.info("✅ Gradient checkpointing enabled")
    
    # Channel-last memory format for CNN (10% speedup)
    model = model.to(memory_format=torch.channels_last)
    
    # torch.compile
    if config['use_compile'] and hasattr(torch, 'compile'):
        if is_main:
            logger.info("🔧 Compiling model...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # Optimizer: 8-bit Adam if available
    if HAS_BNB and config['use_8bit_adam']:
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        if is_main:
            logger.info("✅ Using 8-bit Adam optimizer")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Scheduler with warmup
    total_steps = len(train_loader) * config['epochs'] // config['grad_accum_steps']
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    
    # EMA
    if config['use_ema']:
        ema = EMA(model, decay=0.999)
        if is_main:
            logger.info("✅ EMA enabled")
    else:
        ema = None
    
    # Prepare
    if HAS_ACCELERATE:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    else:
        model = model.to(device)
    
    if is_main:
        params = sum(p.numel() for p in model.parameters())
        batch_eff = config['batch_size'] * config['grad_accum_steps'] * torch.cuda.device_count()
        logger.info(f"📐 Model: {params:,} params")
        logger.info(f"📦 Effective batch: {batch_eff}")
        logger.info(f"🚀 Starting {config['epochs']} epochs...")
    
    # Lagrangian
    lagrangian_lambda = config['lagrangian_lambda']
    target_window = config['target_window']
    
    def compute_loss(pred, target, model):
        n_features = target.shape[-1]
        mean, log_var = pred[:, :n_features], torch.clamp(pred[:, n_features:], -10, 10)
        nll = 0.5 * (log_var + (target - mean) ** 2 / (torch.exp(log_var) + 1e-8))
        recon = nll.mean()
        
        # Safe access to window_cutoff even when model is wrapped (torch.compile, accelerate, etc.)
        if hasattr(model, 'module'):
            cutoff = model.module.window_cutoff
        elif hasattr(model, '_orig_mod'):  # torch.compile wrapper
            cutoff = model._orig_mod.window_cutoff
        else:
            cutoff = model.window_cutoff
            
        window_loss = lagrangian_lambda * (cutoff - target_window) ** 2
        return recon + window_loss, recon
    
    # Training
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')
    # AMP Disabled
    scaler = None 
    
    for epoch in range(config['epochs']):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if not HAS_ACCELERATE:
                inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward without AMP
            if HAS_ACCELERATE:
                with accelerator.accumulate(model):
                    pred = model.forward_ssl(inputs)
                    loss, recon = compute_loss(pred, targets, model)
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config['grad_clip'])
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    if ema:
                        ema.update()
            else:
                # AMP Disabled manual loop
                pred = model.forward_ssl(inputs)
                loss, recon = compute_loss(pred, targets, model)
                
                loss.backward()
                
                if (batch_idx + 1) % config['grad_accum_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if ema:
                        ema.update()
            
            losses.append(recon.item())
            
            cutoff = model.module.window_cutoff if hasattr(model, 'module') else model.window_cutoff
            pbar.set_postfix({'loss': f"{np.mean(losses[-100:]):.4f}", 'cut': f"{cutoff.item():.0f}"})
        
        # Validation (use EMA weights if available)
        if ema:
            ema.apply_shadow()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                if not HAS_ACCELERATE:
                    inputs, targets = inputs.to(device), targets.to(device)
                pred = model.forward_ssl(inputs)
                _, recon = compute_loss(pred, targets, model)
                val_losses.append(recon.item())
        
        if ema:
            ema.restore()
        
        avg_train, avg_val = np.mean(losses), np.mean(val_losses)
        cutoff = model.module.window_cutoff if hasattr(model, 'module') else model.window_cutoff
        
        if is_main:
            logger.info(f"Epoch {epoch+1}: Train={avg_train:.4f} Val={avg_val:.4f} Cutoff={cutoff.item():.0f}")
            
            # Logging to W&B (moved here after val_losses is computed)
            if config['use_wandb'] and HAS_WANDB:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "window_cutoff": cutoff.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        
        if avg_val < best_val and is_main:
            best_val = avg_val
            unwrapped = accelerator.unwrap_model(model) if HAS_ACCELERATE else model
            # Save EMA weights if available
            if ema:
                ema.apply_shadow()
            torch.save({
                'model_state_dict': unwrapped.state_dict(),
                'val_loss': avg_val,
                'cutoff': cutoff.item(),
                'epoch': epoch,
                'config': config
            }, checkpoint_dir / 'mamba_ssl_best.pt')
            if ema:
                ema.restore()
            logger.info(f"💾 Saved best (val={avg_val:.4f})")
    
    if is_main:
        logger.info(f"✅ Done! Best: {best_val:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mamba-LOB SSL Pre-training (Ultra Optimized)")
    parser.add_argument("--data-dir", "-d", default="data/processed")
    parser.add_argument("--max-window", "-w", type=int, default=MAX_WINDOW_SIZE)
    parser.add_argument("--target-window", "-t", type=int, default=50)
    parser.add_argument("--stride", "-s", type=int, default=200)
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--grad-accum", "-g", type=int, default=32)
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lagrangian-lambda", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-8bit", action="store_true")
    parser.add_argument("--no-gradient-ckpt", action="store_true")
    parser.add_argument("--checkpoint-dir", "-c", default="./checkpoints/ssl")
    parser.add_argument("--use-wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--project-name", default="market-mamba-ssl", help="W&B Project Name")
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'max_window': args.max_window,
        'target_window': args.target_window,
        'stride': args.stride,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum,
        'epochs': args.epochs,
        'lr': args.lr,
        'lagrangian_lambda': args.lagrangian_lambda,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'num_workers': args.num_workers,
        'use_compile': not args.no_compile,
        'use_ema': not args.no_ema,
        'use_8bit_adam': not args.no_8bit,
        'gradient_checkpointing': not args.no_gradient_ckpt,
        'checkpoint_dir': args.checkpoint_dir,
        'use_wandb': args.use_wandb,
        'project_name': args.project_name,
        'n_features': LOB_FEATURE_DIM,
        'embedding_dim': 192,
        'num_layers': 3
    }
    
    train_ssl_optimized(config)


if __name__ == "__main__":
    main()
