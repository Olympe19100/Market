# models/rl/PPO_agent.py
# Refactored to use LOBModel as backbone instead of duplicating architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque
from dataclasses import dataclass, field
from core.config import RLConfig, ModelConfig
from .memory import PPOMemory, RunningMeanStd
from models.mamba_lob.model import LOBModel
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# MATHEMATICAL BOUNDS FOR NUMERICAL STABILITY
# =============================================================================

def compute_log_prob_bounds(logstd: torch.Tensor, k_sigma: float = 6.0) -> Tuple[float, float]:
    """
    Compute theoretically-grounded log_prob bounds for Gaussian policy.

    For d-dimensional Gaussian with std σ:
        log_prob = Σᵢ [-½((aᵢ-μᵢ)/σᵢ)² - log(σᵢ) - ½log(2π)]

    Maximum (action = mean):
        log_prob_max = d × (-log(σ) - ½log(2π))

    Minimum (action at k×σ from mean, probability < 2e-9 for k=6):
        log_prob_min = d × (-½k² - log(σ) - ½log(2π))

    Args:
        logstd: Log standard deviation tensor [batch, action_dim] or [action_dim]
        k_sigma: Number of std deviations for minimum bound (6 = 99.9999998% coverage)

    Returns:
        (log_prob_min, log_prob_max) tuple
    """
    # Get action dimension and mean logstd
    if logstd.dim() > 1:
        d = logstd.shape[-1]
        mean_logstd = logstd.mean().item()
    else:
        d = logstd.shape[0]
        mean_logstd = logstd.mean().item()

    # Constants
    half_log_2pi = 0.5 * math.log(2 * math.pi)  # ≈ 0.919

    # Per-dimension contributions
    # At mean: -log(σ) - ½log(2π)
    # At k*σ: -½k² - log(σ) - ½log(2π)
    per_dim_max = -mean_logstd - half_log_2pi
    per_dim_min = -0.5 * k_sigma**2 - mean_logstd - half_log_2pi

    log_prob_max = d * per_dim_max
    log_prob_min = d * per_dim_min

    return log_prob_min, log_prob_max


def compute_ratio_bounds(clip_ratio: float = 0.2, safety_factor: float = 10.0) -> Tuple[float, float]:
    """
    Compute ratio bounds for PPO update.

    The PPO clip objective already bounds the effective ratio to [1-ε, 1+ε].
    Extra clamping is a safety net for extreme outliers.

    Mathematical reasoning:
    - Normal operation: ratio ∈ [1-ε, 1+ε] = [0.8, 1.2] for ε=0.2
    - Safety bound: ratio ∈ [1/(safety_factor/ε), safety_factor/ε]
    - With safety_factor=10, ε=0.2: ratio ∈ [0.02, 50]

    Args:
        clip_ratio: PPO clip ratio ε (default 0.2)
        safety_factor: Multiplier for safety bounds (default 10x beyond clip)

    Returns:
        (ratio_min, ratio_max) tuple
    """
    # Safety bounds are safety_factor times wider than clip bounds
    ratio_max = 1.0 + safety_factor * clip_ratio  # 1 + 10*0.2 = 3.0... too tight
    # Actually, use multiplicative bounds
    ratio_max = (1 + clip_ratio) ** safety_factor  # 1.2^10 ≈ 6.2
    ratio_min = (1 - clip_ratio) ** safety_factor  # 0.8^10 ≈ 0.107

    # Ensure symmetric in log-space: min = 1/max
    ratio_max = max(ratio_max, 1.0 / ratio_min)
    ratio_min = 1.0 / ratio_max

    return ratio_min, ratio_max


# =============================================================================
# SOTA OPTIMIZERS
# =============================================================================

class Lion(Optimizer):
    """
    Lion optimizer (Google 2023) - "Evolved Sign Momentum"

    Paper: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
    https://arxiv.org/abs/2302.06675

    === THEORETICAL FOUNDATION ===

    Lion uses the sign of momentum for updates, discovered via program search:

        m_t = β1 * m_{t-1} + (1 - β1) * g_t          # Update momentum
        update = sign(β2 * m_{t-1} + (1 - β2) * g_t)  # Update direction (sign only)
        θ_t = θ_{t-1} - lr * (update + λ * θ_{t-1})   # Weight update with decay

    Key properties:
    1. Uses sign(·) instead of adaptive scaling → uniform step sizes
    2. Decoupled weight decay (like AdamW)
    3. Lower memory than Adam (no v_t second moment)
    4. Better for transformers/attention architectures

    Recommended hyperparameters (from paper):
    - lr: 1/10 to 1/3 of Adam lr (Lion uses larger effective updates)
    - β1: 0.9 (momentum decay)
    - β2: 0.99 (update interpolation)
    - weight_decay: 10x Adam's weight decay (Lion needs more regularization)
    """

    def __init__(self, params, lr: float = 1e-4, betas: tuple = (0.9, 0.99),
                 weight_decay: float = 0.0):
        """
        Args:
            params: Model parameters
            lr: Learning rate (use 3-10x smaller than Adam)
            betas: (β1, β2) for momentum and update interpolation
            weight_decay: Decoupled weight decay (use 3-10x larger than Adam)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]

                # Initialize momentum
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay (decoupled, applied before update)
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update direction: sign of interpolated momentum
                # update = sign(β2 * m + (1 - β2) * g)
                update = exp_avg.mul(beta2).add(grad, alpha=1 - beta2).sign_()

                # Apply update
                p.add_(update, alpha=-group['lr'])

                # Update momentum state for next iteration
                # m = β1 * m + (1 - β1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        return loss


class Sophia(Optimizer):
    """
    Sophia optimizer (Stanford 2023) - Second-order clipped stochastic optimization

    Paper: "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
    https://arxiv.org/abs/2305.14342

    === THEORETICAL FOUNDATION ===

    Sophia uses diagonal Hessian estimates for adaptive learning rates:

        m_t = β1 * m_{t-1} + (1 - β1) * g_t              # EMA of gradients
        h_t = β2 * h_{t-1} + (1 - β2) * diag(H_t)        # EMA of Hessian diagonal
        update = clip(m_t / max(h_t, ε), ρ)              # Clipped Newton step
        θ_t = θ_{t-1} - lr * update - λ * θ_{t-1}        # Update with decay

    Hessian diagonal estimation (Hutchinson's method):
        diag(H) ≈ g * z where z ~ Rademacher(±1) and g = ∂L/∂θ * z

    Key properties:
    1. Adapts per-parameter learning rates using curvature
    2. Clipping prevents instability from small Hessian values
    3. 2x faster convergence than Adam on LLMs
    4. Requires periodic Hessian estimation (every k steps)
    """

    def __init__(self, params, lr: float = 1e-4, betas: tuple = (0.965, 0.99),
                 rho: float = 0.04, weight_decay: float = 0.0,
                 hessian_update_interval: int = 10):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: (β1, β2) for gradient and Hessian EMA
            rho: Clipping threshold for update magnitude
            weight_decay: Weight decay coefficient
            hessian_update_interval: Steps between Hessian updates
        """
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.hessian_update_interval = hessian_update_interval
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None, hessian_vector=None):
        """
        Perform optimization step.

        Args:
            closure: Optional closure for loss evaluation
            hessian_vector: Pre-computed Hessian diagonal (optional)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1
        update_hessian = (self.step_count % self.hessian_update_interval == 0)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.ones_like(p)  # Initialize to 1 (like Adam's v)

                exp_avg = state['exp_avg']
                hessian = state['hessian']
                beta1, beta2 = group['betas']

                # Update gradient EMA
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update Hessian estimate (Hutchinson's method)
                if update_hessian:
                    # Approximate Hessian diagonal: h ≈ g² (Gauss-Newton approx)
                    # This is a simplification; full Hutchinson needs a backward pass
                    hessian_estimate = grad.pow(2)
                    hessian.mul_(beta2).add_(hessian_estimate, alpha=1 - beta2)

                # Clipped update: clip(m / max(h, ε), ρ)
                update = exp_avg / hessian.clamp(min=1e-8)
                update.clamp_(-group['rho'], group['rho'])

                # Apply update with weight decay
                p.add_(update, alpha=-group['lr'])
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

        return loss

    def update_hessian(self, loss_fn, params):
        """
        Compute Hessian diagonal using Hutchinson's estimator.
        Call periodically (e.g., every 10 steps) for more accurate estimates.

        This requires a separate backward pass with Rademacher vectors.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'hessian' not in state:
                    continue

                # Hutchinson's trace estimator: E[z^T H z] = tr(H)
                # For diagonal: E[z * (H @ z)] = diag(H) when z is Rademacher
                z = torch.randint_like(p, 0, 2) * 2 - 1  # Rademacher: {-1, +1}

                # This is a simplified version; full implementation needs
                # torch.autograd.grad(loss_fn(params), p, create_graph=True)
                # then another backward with z

                # For now, use Gauss-Newton approximation: diag(H) ≈ g²
                if p.grad is not None:
                    hessian_estimate = p.grad.pow(2)
                    beta2 = group['betas'][1]
                    state['hessian'].mul_(beta2).add_(hessian_estimate, alpha=1 - beta2)


class Muon(Optimizer):
    """
    Muon optimizer - Momentum Orthogonalized by Newton-schulz

    Designed for neural networks with structured weight matrices.
    Particularly effective for:
    - State Space Models (Mamba, S4, etc.)
    - Transformers with weight tying
    - Any model with near-orthogonal weight matrices

    === THEORETICAL FOUNDATION ===

    Muon applies Newton-Schulz iteration to orthogonalize momentum:

        m_t = β * m_{t-1} + g_t                    # Momentum accumulation
        m_orth = newton_schulz(m_t, steps=5)       # Orthogonalize momentum
        θ_t = θ_{t-1} - lr * m_orth                # Update

    Newton-Schulz iteration (converges to polar decomposition):
        X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k^T @ X_k

    Why it works for Mamba:
    1. SSM matrices (A, B, C) benefit from orthogonal structure
    2. Preserves spectral properties of state transitions
    3. Better gradient flow through recurrent dynamics

    Reference: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        """
        Args:
            params: Model parameters
            lr: Learning rate (Muon uses higher LR than Adam, ~0.02)
            momentum: Momentum coefficient
            nesterov: Use Nesterov momentum
            ns_steps: Newton-Schulz iteration steps (5 is usually enough)
        """
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Newton-Schulz iteration for computing orthogonal component.

        Converges to U where M = U @ S @ V^T (polar decomposition).
        For square matrices, this gives the nearest orthogonal matrix.
        """
        # Only apply to 2D matrices
        if M.dim() != 2:
            return M

        # Scale for numerical stability
        a, b = M.shape
        if a > b:
            M = M.T
            transpose = True
        else:
            transpose = False

        # Normalize
        scale = (M.norm() / (a * b) ** 0.5).clamp(min=1e-8)
        M = M / scale

        # Newton-Schulz iteration: X_{k+1} = 1.5*X - 0.5*X@X^T@X
        for _ in range(steps):
            M = 1.5 * M - 0.5 * M @ M.T @ M

        if transpose:
            M = M.T

        return M * scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                # Momentum update
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf

                # Apply Newton-Schulz orthogonalization to 2D params only
                if p.dim() == 2 and min(p.shape) > 1:
                    update = self.newton_schulz(update, steps=ns_steps)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


class AdamW8bit:
    """
    Placeholder for 8-bit AdamW from bitsandbytes.
    Use this for memory-efficient training on large models.

    Requires: pip install bitsandbytes
    """
    pass  # Import from bitsandbytes if available


def get_optimizer(name: str, params, lr: float, weight_decay: float = 0.0, **kwargs):
    """
    Factory function for SOTA optimizers.

    Args:
        name: Optimizer name ('muon', 'lion', 'sophia', 'adamw', 'adam')
        params: Model parameters
        lr: Learning rate (will be adjusted per optimizer)
        weight_decay: Weight decay
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer instance

    Recommendations by architecture:
        - Mamba/SSM: 'muon' or 'adamw' (preserves structured matrices)
        - Transformer: 'lion' or 'adamw'
        - General: 'adamw' (safe default)
    """
    name = name.lower()

    if name == 'muon':
        # Muon: Best for Mamba/SSM architectures
        # Uses Newton-Schulz orthogonalization on weight matrices
        # For RL: use conservative 2x scaling (not 10x like supervised learning)
        # RL is more sensitive to large updates due to on-policy nature
        muon_lr = lr * 2  # Conservative for RL stability
        return Muon(params, lr=muon_lr, momentum=kwargs.get('momentum', 0.95),
                   nesterov=kwargs.get('nesterov', True),
                   ns_steps=kwargs.get('ns_steps', 5))

    elif name == 'lion':
        # Lion: Good for transformers, uses sign of momentum
        # Needs smaller LR and larger weight decay than Adam
        lion_lr = lr / 3
        lion_wd = weight_decay * 10 if weight_decay > 0 else 1e-2
        return Lion(params, lr=lion_lr, weight_decay=lion_wd,
                   betas=kwargs.get('betas', (0.9, 0.99)))

    elif name == 'sophia':
        # Sophia: Second-order optimizer, good for LLMs
        return Sophia(params, lr=lr, weight_decay=weight_decay,
                     betas=kwargs.get('betas', (0.965, 0.99)),
                     rho=kwargs.get('rho', 0.04))

    elif name == 'adamw':
        # AdamW: Standard baseline, what Mamba authors used
        # For Mamba: betas=(0.9, 0.95), weight_decay=0.1
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                          betas=kwargs.get('betas', (0.9, 0.95)),
                          eps=kwargs.get('eps', 1e-8))

    elif name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay,
                         betas=kwargs.get('betas', (0.9, 0.999)),
                         eps=kwargs.get('eps', 1e-8))

    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from: muon, lion, sophia, adamw, adam")


class GradientNoiseScale:
    """
    Computes the Gradient Noise Scale (GNS) to derive optimal learning rate.

    === THEORETICAL FOUNDATION ===

    From McCandlish et al. (2018) "An Empirical Model of Large-Batch Training":

    The gradient noise scale B_noise is defined as:
        B_noise = tr(Σ) / ||G||²

    Where:
        - Σ = covariance matrix of per-sample gradients
        - G = true gradient (expected value)
        - tr(Σ) = trace of covariance = sum of variances

    In practice, we estimate:
        - G ≈ mean of mini-batch gradients
        - tr(Σ) ≈ variance of gradient norms across mini-batches

    The optimal learning rate is then:
        lr_opt = ε_max * B_noise / (B + B_noise)

    Where:
        - ε_max = maximum stable learning rate (from loss curvature)
        - B = batch size

    For B << B_noise: lr_opt ≈ ε_max (small batch, high noise)
    For B >> B_noise: lr_opt ≈ ε_max * B_noise / B (large batch, low noise)
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.grad_norms: deque = deque(maxlen=window_size)
        self.grad_norm_sq: deque = deque(maxlen=window_size)
        self.loss_values: deque = deque(maxlen=window_size)
        self.is_calibrated = False
        self.b_noise = 1.0  # Critical batch size
        self.eps_max = 1e-3  # Maximum stable LR

    def update(self, model_params, loss_value: float):
        """
        Update statistics with current gradients.
        Call after loss.backward() but before optimizer.step().
        """
        # Compute gradient norm
        total_norm = 0.0
        for p in model_params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        self.grad_norms.append(total_norm)
        self.grad_norm_sq.append(total_norm ** 2)
        self.loss_values.append(loss_value)

    def compute_optimal_lr(self, batch_size: int) -> dict:
        """
        Compute optimal learning rate from gradient statistics.

        Returns dict with:
            - lr_optimal: recommended learning rate
            - b_noise: critical batch size
            - eps_max: maximum stable LR
            - signal_to_noise: gradient signal-to-noise ratio
        """
        if len(self.grad_norms) < 10:
            return {'lr_optimal': 3e-4, 'calibrated': False}

        grad_norms = np.array(list(self.grad_norms))
        grad_norm_sq = np.array(list(self.grad_norm_sq))
        losses = np.array(list(self.loss_values))

        # Estimate ||G||² (squared norm of expected gradient)
        # Using (E[g])² as estimator
        mean_grad_norm = np.mean(grad_norms)
        g_norm_sq = mean_grad_norm ** 2

        # Estimate tr(Σ) (trace of gradient covariance)
        # tr(Σ) = E[||g||²] - ||E[g]||²
        mean_grad_norm_sq = np.mean(grad_norm_sq)
        trace_sigma = max(mean_grad_norm_sq - g_norm_sq, 1e-10)

        # Gradient Noise Scale: B_noise = tr(Σ) / ||G||²
        self.b_noise = trace_sigma / max(g_norm_sq, 1e-10)

        # Estimate ε_max from loss curvature
        # Simple heuristic: ε_max ≈ 2 / L where L is Lipschitz constant
        # Approximate L from gradient norm / loss change
        if len(losses) > 1:
            loss_changes = np.abs(np.diff(losses))
            mean_loss_change = np.mean(loss_changes) + 1e-10
            lipschitz_estimate = mean_grad_norm / mean_loss_change
            self.eps_max = min(2.0 / max(lipschitz_estimate, 1.0), 1e-2)
        else:
            self.eps_max = 1e-3

        # Optimal learning rate (McCandlish et al. 2018, Eq. 1)
        # lr_opt = ε_max * B_noise / (B + B_noise)
        lr_optimal = self.eps_max * self.b_noise / (batch_size + self.b_noise)

        # Clamp to reasonable range
        lr_optimal = np.clip(lr_optimal, 1e-6, 1e-2)

        self.is_calibrated = len(self.grad_norms) >= self.window_size

        # Signal-to-noise ratio
        snr = g_norm_sq / trace_sigma if trace_sigma > 0 else 0

        return {
            'lr_optimal': float(lr_optimal),
            'b_noise': float(self.b_noise),
            'eps_max': float(self.eps_max),
            'signal_to_noise': float(snr),
            'grad_norm_mean': float(mean_grad_norm),
            'calibrated': self.is_calibrated
        }

    def state_dict(self) -> dict:
        return {
            'grad_norms': list(self.grad_norms),
            'grad_norm_sq': list(self.grad_norm_sq),
            'loss_values': list(self.loss_values),
            'b_noise': self.b_noise,
            'eps_max': self.eps_max,
            'is_calibrated': self.is_calibrated
        }

    def load_state_dict(self, state: dict):
        self.grad_norms = deque(state.get('grad_norms', []), maxlen=self.window_size)
        self.grad_norm_sq = deque(state.get('grad_norm_sq', []), maxlen=self.window_size)
        self.loss_values = deque(state.get('loss_values', []), maxlen=self.window_size)
        self.b_noise = state.get('b_noise', 1.0)
        self.eps_max = state.get('eps_max', 1e-3)
        self.is_calibrated = state.get('is_calibrated', False)


class AdaptiveCoverage:
    """
    Adaptive entropy target based on training progress.

    === THEORETICAL FOUNDATION ===

    The exploration-exploitation tradeoff is captured by action space coverage:
    - Early training: High coverage (80%) = more exploration
    - Late training: High precision (95%) = more exploitation

    The schedule follows the principle of simulated annealing (Kirkpatrick 1983):
        coverage(t) = coverage_min + (coverage_max - coverage_min) * decay(t)

    Where decay follows the reward improvement rate:
        decay(t) = exp(-λ * cumulative_improvement)

    This is DATA-DRIVEN: coverage adapts based on actual learning progress,
    not arbitrary episode counts.

    === ENTROPY TARGET DERIVATION ===

    For Gaussian policy with actions in [-1, 1]:
        H_target = dim(A) × [0.5·log(2πe) - log(z_p)]

    Where z_p = Φ⁻¹((1+coverage)/2) is the Gaussian quantile.
    """

    def __init__(self, action_dim: int = 9, window_size: int = 50):
        self.action_dim = action_dim
        self.window_size = window_size

        # Coverage bounds (from exploration to exploitation)
        self.coverage_max = 0.80  # Early: 80% coverage (z=1.28, σ=0.78)
        self.coverage_min = 0.95  # Late: 95% coverage (z=1.96, σ=0.51)

        # Track reward improvements
        self.reward_history: deque = deque(maxlen=window_size)
        self.best_reward = float('-inf')
        self.cumulative_improvement = 0.0

        # Decay rate (learned from data)
        self.decay_rate = 0.1  # Will be calibrated

        # Current state
        self.current_coverage = self.coverage_max
        self.current_target_entropy = self._compute_target_entropy(self.coverage_max)

    def _compute_target_entropy(self, coverage: float) -> float:
        """Compute target entropy from coverage using Gaussian theory."""
        # z_p = Φ⁻¹((1+coverage)/2) - using rational approximation (Abramowitz & Stegun)
        # For common coverages, use precomputed values for precision
        z_table = {
            0.80: 1.2816,  # 80% coverage
            0.85: 1.4395,  # 85% coverage
            0.90: 1.6449,  # 90% coverage
            0.95: 1.9600,  # 95% coverage
            0.99: 2.5758,  # 99% coverage
        }

        # Find closest precomputed value or interpolate
        if coverage in z_table:
            z_p = z_table[coverage]
        else:
            # Linear interpolation between nearest points
            coverages = sorted(z_table.keys())
            for i, c in enumerate(coverages[:-1]):
                if c <= coverage <= coverages[i + 1]:
                    t = (coverage - c) / (coverages[i + 1] - c)
                    z_p = z_table[c] * (1 - t) + z_table[coverages[i + 1]] * t
                    break
            else:
                z_p = 1.6449  # Default to 90%

        # H = dim × [0.5·log(2πe) - log(z_p)]
        h_per_dim = 0.5 * np.log(2 * np.pi * np.e) - np.log(z_p)
        return self.action_dim * h_per_dim

    def update(self, mean_reward: float) -> dict:
        """
        Update coverage based on reward improvement.

        Returns dict with current adaptive parameters.
        """
        self.reward_history.append(mean_reward)

        # Track improvement (only count after first reward to avoid -inf issue)
        if self.best_reward > float('-inf') and mean_reward > self.best_reward:
            improvement = mean_reward - self.best_reward
            # Clamp improvement to avoid numerical issues
            self.cumulative_improvement += np.clip(improvement, 0, 10.0)
        # Update best reward (including first time)
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward

        # Compute decay based on cumulative improvement
        # As model improves, reduce exploration
        if len(self.reward_history) >= 10:
            # Normalize improvement by reward scale (robust to negative rewards)
            reward_scale = max(
                np.std(list(self.reward_history)),
                abs(np.mean(list(self.reward_history))) + 1e-4,
                1e-4
            )
            normalized_improvement = self.cumulative_improvement / reward_scale
            # Clamp to prevent numerical overflow
            normalized_improvement = np.clip(normalized_improvement, 0, 100)

            # Exponential decay: coverage decreases as improvement accumulates
            decay = np.exp(-self.decay_rate * normalized_improvement)

            # Update coverage
            self.current_coverage = self.coverage_min + (self.coverage_max - self.coverage_min) * decay
            self.current_target_entropy = self._compute_target_entropy(self.current_coverage)

        return {
            'coverage': self.current_coverage,
            'target_entropy': self.current_target_entropy,
            'cumulative_improvement': self.cumulative_improvement,
            'best_reward': self.best_reward
        }

    def state_dict(self) -> dict:
        return {
            'reward_history': list(self.reward_history),
            'best_reward': self.best_reward,
            'cumulative_improvement': self.cumulative_improvement,
            'current_coverage': self.current_coverage,
            'current_target_entropy': self.current_target_entropy
        }

    def load_state_dict(self, state: dict):
        self.reward_history = deque(state.get('reward_history', []), maxlen=self.window_size)
        self.best_reward = state.get('best_reward', float('-inf'))
        self.cumulative_improvement = state.get('cumulative_improvement', 0.0)
        self.current_coverage = state.get('current_coverage', self.coverage_max)
        self.current_target_entropy = state.get('current_target_entropy',
                                                  self._compute_target_entropy(self.coverage_max))


class AdaptiveClipRatio:
    """
    Adaptive PPO clip ratio based on policy ratio statistics.

    === THEORETICAL FOUNDATION ===

    From Engstrom et al. (2020) "Implementation Matters in Deep RL":

    The clip ratio ε controls the trust region size:
        L_clip = min(r*A, clip(r, 1-ε, 1+ε)*A)

    Optimal ε depends on the actual distribution of ratios r = π_new/π_old:
        - If ratios are tightly clustered around 1: can use smaller ε (more aggressive)
        - If ratios have high variance: need larger ε (more conservative)

    === DATA-DRIVEN DERIVATION ===

    We set ε such that most ratios (e.g., 95%) fall within [1-ε, 1+ε]:
        ε = z_0.95 × std(r) = 1.96 × std(r)

    This ensures the clipping rarely activates for typical updates,
    while still protecting against catastrophically large updates.

    Bounds:
        - ε_min = 0.1 (minimum trust region from TRPO theory)
        - ε_max = 0.3 (PPO paper recommendation)
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.ratio_stds: deque = deque(maxlen=window_size)

        # Bounds from PPO theory
        self.clip_min = 0.1   # Minimum clip (conservative)
        self.clip_max = 0.3   # Maximum clip (aggressive)

        # Coverage quantile
        self.z_coverage = 1.96  # 95% coverage

        # Current adaptive value
        self.current_clip = 0.2  # PPO default

    def update(self, ratios: torch.Tensor) -> float:
        """
        Update clip ratio based on observed policy ratios.

        Args:
            ratios: Tensor of π_new/π_old ratios from current batch

        Returns:
            Adaptive clip ratio for next update
        """
        with torch.no_grad():
            ratio_std = ratios.std().item()
            self.ratio_stds.append(ratio_std)

            if len(self.ratio_stds) >= 10:
                # Compute adaptive clip from ratio statistics
                mean_std = np.mean(list(self.ratio_stds))

                # ε = z × std(r) to cover 95% of ratios
                adaptive_clip = self.z_coverage * mean_std

                # Clamp to reasonable bounds
                self.current_clip = np.clip(adaptive_clip, self.clip_min, self.clip_max)

        return self.current_clip

    def state_dict(self) -> dict:
        return {
            'ratio_stds': list(self.ratio_stds),
            'current_clip': self.current_clip
        }

    def load_state_dict(self, state: dict):
        self.ratio_stds = deque(state.get('ratio_stds', []), maxlen=self.window_size)
        self.current_clip = state.get('current_clip', 0.2)


@dataclass
class AdaptiveKLController:
    """
    Adaptive KL divergence controller with theoretically grounded initialization.

    === THEORETICAL FOUNDATIONS ===

    1. KL Target Derivation (Schulman 2015, TRPO):
       For Gaussian policy π(a|s) = N(μ, σ²I), the KL divergence is:
       D_KL(π_old || π_new) = 0.5 * Σ_i [(σ_old_i/σ_new_i)² + (μ_new_i - μ_old_i)²/σ_new_i² - 1 + 2*log(σ_new_i/σ_old_i)]

       For small changes (Δμ << σ, Δσ << σ):
       D_KL ≈ 0.5 * Σ_i [(Δμ_i/σ_i)² + 2*(Δσ_i/σ_i)²]
            = 0.5 * d * [mean((Δμ/σ)²) + 2*mean((Δσ/σ)²)]

       If we want each dimension to change by at most δ in units of σ:
       target_kl = 0.5 * d * δ²

       For δ = 0.1 (10% of std per update), d = 9:
       target_kl = 0.5 * 9 * 0.01 = 0.045

    2. Beta Initialization (Lagrangian Duality):
       We solve: min_θ L(θ) s.t. D_KL ≤ ε
       Lagrangian: L(θ) + β * (D_KL - ε)

       Optimal β* satisfies complementary slackness.
       Initial β should be O(1/ε) to have meaningful constraint.

    3. PID Gains (Ziegler-Nichols-inspired):
       For a first-order system with time constant τ:
       Kp = 1/τ, Ki = Kp/(2τ), Kd = Kp*τ/8

       We use τ = calibration_window for adaptation timescale.
    """
    action_dim: int = 9
    window_size: int = 100

    # Calibration
    calibration_episodes: int = 20

    # PID gains derived from Ziegler-Nichols for τ = 20
    # Kp = 1/τ = 0.05, but we use log-scale so multiply by ln(2) ≈ 0.7
    kp: float = field(default=0.0)  # Will be computed
    ki: float = field(default=0.0)
    kd: float = field(default=0.0)

    # Bounds (derived from theory)
    beta_min: float = field(default=0.0)  # Will be computed
    beta_max: float = field(default=0.0)
    target_percentile: float = 50.0

    # State
    kl_history: deque = field(default_factory=lambda: deque(maxlen=100))
    beta: float = field(default=0.0)
    target_kl: float = field(default=0.0)
    is_calibrated: bool = False
    integral_error: float = 0.0
    last_error: float = 0.0
    calibration_kls: list = field(default_factory=list)

    def __post_init__(self):
        self.kl_history = deque(maxlen=self.window_size)
        self.calibration_kls = []

        # === THEORETICAL INITIALIZATION ===

        # Target KL from trust region theory (Schulman 2015)
        # δ = 0.2 means each action dimension changes by 20% of its std per update
        # Now that action mask is fixed, we can allow larger policy changes
        delta_per_dim = 0.2
        self.target_kl = 0.5 * self.action_dim * (delta_per_dim ** 2)
        # For 9D: target_kl = 0.5 * 9 * 0.04 = 0.18

        # Beta initialization: β = 1/target_kl (Lagrangian scaling)
        self.beta = 1.0 / self.target_kl
        # For target_kl = 0.045: beta ≈ 22

        # Beta bounds from constraint strength
        # beta_min: constraint is very loose (10x target)
        # beta_max: constraint is very tight (0.1x target)
        self.beta_min = 1.0 / (10 * self.target_kl)  # ≈ 2.2
        self.beta_max = 1.0 / (0.1 * self.target_kl)  # ≈ 222

        # PID gains from Ziegler-Nichols, adapted for log-scale control
        tau = float(self.calibration_episodes)
        self.kp = 0.7 / tau  # 0.7 = ln(2), for doubling/halving beta
        self.ki = self.kp / (2 * tau)
        self.kd = self.kp * tau / 8

        logger.debug(f"AdaptiveKL initialized: target_kl={self.target_kl:.4f}, "
                    f"beta={self.beta:.1f}, bounds=[{self.beta_min:.1f}, {self.beta_max:.1f}]")

    def update(self, observed_kl: float, episode: int) -> dict:
        """
        Update controller with observed KL and return adjusted parameters.

        Args:
            observed_kl: KL divergence from last update
            episode: Current episode number

        Returns:
            dict with 'beta', 'target_kl', 'early_stop' (bool)
        """
        self.kl_history.append(observed_kl)

        # Calibration phase: gather statistics
        if episode < self.calibration_episodes:
            self.calibration_kls.append(observed_kl)
            # During calibration, use conservative high beta
            self.beta = max(self.beta, 50.0)
            return {
                'beta': self.beta,
                'target_kl': self.target_kl,
                'early_stop_threshold': self.target_kl * 20,  # Very permissive during calibration
                'calibrating': True
            }

        # Calibrate target KL from observed distribution
        if not self.is_calibrated and len(self.calibration_kls) >= self.calibration_episodes:
            self._calibrate()

        # PID control for beta adjustment
        error = observed_kl - self.target_kl

        # Update integral (with anti-windup)
        self.integral_error = np.clip(
            self.integral_error + error,
            -10.0, 10.0
        )

        # Derivative
        derivative = error - self.last_error
        self.last_error = error

        # PID output (log-scale for beta)
        log_beta_adjustment = (
            self.kp * error +
            self.ki * self.integral_error +
            self.kd * derivative
        )

        # Apply adjustment (multiplicative in linear scale)
        self.beta = np.clip(
            self.beta * np.exp(log_beta_adjustment),
            self.beta_min,
            self.beta_max
        )

        # Adaptive early stopping threshold based on recent KL variance
        if len(self.kl_history) >= 10:
            kl_std = np.std(list(self.kl_history)[-20:])
            # Early stop at target + 3 sigma (99.7% of normal distribution)
            early_stop_threshold = self.target_kl + 3 * max(kl_std, 0.01)
        else:
            early_stop_threshold = self.target_kl * 10

        return {
            'beta': self.beta,
            'target_kl': self.target_kl,
            'early_stop_threshold': early_stop_threshold,
            'calibrating': False
        }

    def _calibrate(self):
        """Calibrate target KL from observed distribution."""
        kls = np.array(self.calibration_kls)

        # Remove outliers (top 10%) for robust estimation
        p90 = np.percentile(kls, 90)
        filtered_kls = kls[kls <= p90]

        if len(filtered_kls) < 5:
            filtered_kls = kls

        # Set target at specified percentile of filtered distribution
        self.target_kl = np.percentile(filtered_kls, self.target_percentile)

        # Ensure target is reasonable
        min_target = 0.005 * self.action_dim  # Minimum: 0.005 per dim
        max_target = 0.1 * self.action_dim    # Maximum: 0.1 per dim
        self.target_kl = np.clip(self.target_kl, min_target, max_target)

        # Set initial beta based on how far current KL is from target
        median_kl = np.median(filtered_kls)
        if median_kl > self.target_kl:
            # KL too high, need higher beta
            self.beta = 20.0 * (median_kl / self.target_kl)
        else:
            # KL acceptable, use moderate beta
            self.beta = 10.0

        self.beta = np.clip(self.beta, self.beta_min, self.beta_max)
        self.is_calibrated = True

        logger.info(f"[AdaptiveKL] Calibrated: target_kl={self.target_kl:.4f}, "
                   f"initial_beta={self.beta:.1f}, "
                   f"observed_median={median_kl:.4f}, observed_p90={p90:.4f}")

    def get_stats(self) -> dict:
        """Get current statistics for logging."""
        if len(self.kl_history) == 0:
            return {}

        kls = np.array(list(self.kl_history))
        return {
            'kl_mean': float(np.mean(kls)),
            'kl_std': float(np.std(kls)),
            'kl_p50': float(np.percentile(kls, 50)),
            'kl_p90': float(np.percentile(kls, 90)),
            'beta': float(self.beta),
            'target_kl': float(self.target_kl),
            'is_calibrated': self.is_calibrated
        }

    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        return {
            'beta': self.beta,
            'target_kl': self.target_kl,
            'is_calibrated': self.is_calibrated,
            'integral_error': self.integral_error,
            'last_error': self.last_error,
            'kl_history': list(self.kl_history),
            'calibration_kls': self.calibration_kls
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.beta = state.get('beta', self.beta)
        self.target_kl = state.get('target_kl', self.target_kl)
        self.is_calibrated = state.get('is_calibrated', False)
        self.integral_error = state.get('integral_error', 0.0)
        self.last_error = state.get('last_error', 0.0)
        if 'kl_history' in state:
            self.kl_history = deque(state['kl_history'], maxlen=self.window_size)
        if 'calibration_kls' in state:
            self.calibration_kls = state['calibration_kls']


class PopArtLayer(nn.Module):
    """
    PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets).
    Van Hasselt et al. 2016 — adaptive normalization for value function outputs.
    Prevents critic instability when reward scale changes across episodes.
    """

    def __init__(self, input_dim: int, output_dim: int = 1, beta: float = 3e-4):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(input_dim, output_dim)
        # Running statistics (not parameters — not in optimizer)
        self.register_buffer('mu', torch.zeros(output_dim))
        self.register_buffer('sigma', torch.ones(output_dim))
        self.register_buffer('nu', torch.zeros(output_dim))       # second moment
        self.register_buffer('count', torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns UN-normalized value prediction (in original reward scale)."""
        normalized_output = self.linear(x)
        return normalized_output * self.sigma + self.mu

    def normalized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized value prediction (for loss computation against normalized targets)."""
        return self.linear(x)

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets to the PopArt scale for critic loss."""
        return (targets - self.mu) / self.sigma

    def update_stats(self, targets: torch.Tensor):
        """Update running mean/variance and preserve outputs (the 'Art' in PopArt)."""
        with torch.no_grad():
            batch_mean = targets.mean(dim=0)
            batch_second = (targets ** 2).mean(dim=0)
            self.count += 1

            old_mu = self.mu.clone()
            old_sigma = self.sigma.clone()

            # Exponential moving average
            self.mu = (1 - self.beta) * self.mu + self.beta * batch_mean
            self.nu = (1 - self.beta) * self.nu + self.beta * batch_second
            new_sigma = torch.sqrt(torch.clamp(self.nu - self.mu ** 2, min=1e-6))

            # Preserve outputs: adjust weights and biases so f(x) stays the same
            self.linear.weight.data.mul_(old_sigma / new_sigma)
            self.linear.bias.data.mul_(old_sigma / new_sigma)
            self.linear.bias.data.add_((old_mu - self.mu) / new_sigma)

            self.sigma = new_sigma


class ActorCriticHead(nn.Module):
    """
    Thin Actor/Critic heads that sit on top of LOBModel's fused features.
    LOBModel outputs 576-dim features (512 LOB + 64 aux).
    """

    def __init__(self, input_dim: int = 576, hidden_dim: int = 256, action_dim: int = 9, position_dim: int = 11):
        super().__init__()

        # Position state encoder (inventory, time, cash_norm, inv_norm,
        #   room_to_buy, room_to_sell, last_bid_fill, last_ask_fill, spread_bps,
        #   bid_order_age, ask_order_age)
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, 48),
            nn.LayerNorm(48),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_dim = input_dim + 48
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Actor head (9D continuous actions: bid_spread, ask_spread, bid_qty, ask_qty,
        #   flatten, ladder_step, ladder_decay, hold_bid, hold_ask)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        # State-dependent logstd with theoretically derived bounds
        #
        # THEORETICAL DERIVATION:
        # Actions are tanh-bounded to [-1, 1]. For Gaussian sampling:
        #   - σ_min: Should allow precise targeting. For 99% of samples within ±0.1:
        #     0.1 = 2.58σ → σ = 0.039 → log(σ) = -3.24
        #     But too small causes gradient issues, so use σ = 0.1 → log(σ) = -2.3
        #
        #   - σ_max: Should cover action range. For 95% of samples within ±1:
        #     1.0 = 1.96σ → σ = 0.51 → log(σ) = -0.67
        #
        # === LOGSTD BOUNDS FROM ACTION SPACE COVERAGE ===
        #
        # THEORETICAL DERIVATION:
        # For actions in [-1, 1] with coverage p:
        #   - σ_max: Want ~70% coverage at max exploration → z_0.70=1.04 → σ=0.96
        #   - σ_min: Want ~99% coverage at convergence → z_0.99=2.58 → σ=0.39
        #
        # But we also want the policy to be able to be more precise when needed:
        #   - σ_min = 0.1 allows 99.99% within ±0.4 (very precise)
        #   - σ_max = 1.0 allows uniform-like exploration
        #
        # log(σ) bounds:
        self.logstd_min = -2.3  # σ = 0.10 → 99% within ±0.26
        self.logstd_max = 0.0   # σ = 1.00 → 68% within ±1.0 (full exploration)
        self.actor_logstd_net = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Critic head with PopArt normalization
        self.critic_backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.critic_popart = PopArtLayer(hidden_dim // 2, 1)
        
    def forward(self,
                backbone_features: torch.Tensor,
                inventory: torch.Tensor,
                time_remaining: torch.Tensor,
                cash_normalized: torch.Tensor = None,
                inventory_normalized: torch.Tensor = None,
                room_to_buy: torch.Tensor = None,
                room_to_sell: torch.Tensor = None,
                last_bid_fill: torch.Tensor = None,
                last_ask_fill: torch.Tensor = None,
                spread_bps: torch.Tensor = None,
                bid_order_age: torch.Tensor = None,
                ask_order_age: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            backbone_features: (B, 576) from LOBModel
            inventory: (B,)
            time_remaining: (B,)
            cash_normalized: (B,) cash / initial_cash
            inventory_normalized: (B,) inventory / max_position
            room_to_buy: (B,) normalized residual buy capacity
            room_to_sell: (B,) normalized residual sell capacity
            last_bid_fill: (B,) 1.0 if bid was filled last step
            last_ask_fill: (B,) 1.0 if ask was filled last step
            spread_bps: (B,) current spread in basis points (scaled /100 for network)
            bid_order_age: (B,) bid order age / 100 (FIX 3)
            ask_order_age: (B,) ask order age / 100 (FIX 3)
        """
        if cash_normalized is None:
            cash_normalized = torch.zeros_like(inventory)
        if inventory_normalized is None:
            inventory_normalized = torch.zeros_like(inventory)
        if room_to_buy is None:
            room_to_buy = torch.ones_like(inventory)
        if room_to_sell is None:
            room_to_sell = torch.ones_like(inventory)
        if last_bid_fill is None:
            last_bid_fill = torch.zeros_like(inventory)
        if last_ask_fill is None:
            last_ask_fill = torch.zeros_like(inventory)
        if spread_bps is None:
            spread_bps = torch.zeros_like(inventory)
        if bid_order_age is None:
            bid_order_age = torch.zeros_like(inventory)
        if ask_order_age is None:
            ask_order_age = torch.zeros_like(inventory)

        # Stack position info (11D)
        pos_info = torch.cat([
            inventory.view(-1, 1),
            time_remaining.view(-1, 1),
            cash_normalized.view(-1, 1),
            inventory_normalized.view(-1, 1),
            room_to_buy.view(-1, 1),
            room_to_sell.view(-1, 1),
            last_bid_fill.view(-1, 1),
            last_ask_fill.view(-1, 1),
            (spread_bps / 100.0).view(-1, 1),  # Scale bps to ~[0, 0.2] range
            bid_order_age.view(-1, 1),
            ask_order_age.view(-1, 1),
        ], dim=-1)
        
        pos_encoded = self.position_encoder(pos_info)
        
        # Fuse with backbone features
        x = torch.cat([backbone_features, pos_encoded], dim=-1)
        x = self.fusion(x)
        
        # Actor/Critic outputs
        action_mean = self.actor_mean(x)
        # State-dependent logstd: map tanh output to [logstd_min, logstd_max]
        tanh_out = self.actor_logstd_net(x)
        action_logstd = self.logstd_min + 0.5 * (self.logstd_max - self.logstd_min) * (tanh_out + 1.0)
        critic_features = self.critic_backbone(x)
        value = self.critic_popart(critic_features)

        return action_mean, action_logstd, value

    def forward_with_features(self,
                backbone_features: torch.Tensor,
                inventory: torch.Tensor,
                time_remaining: torch.Tensor,
                cash_normalized: torch.Tensor = None,
                inventory_normalized: torch.Tensor = None,
                room_to_buy: torch.Tensor = None,
                room_to_sell: torch.Tensor = None,
                last_bid_fill: torch.Tensor = None,
                last_ask_fill: torch.Tensor = None,
                spread_bps: torch.Tensor = None,
                bid_order_age: torch.Tensor = None,
                ask_order_age: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as forward() but also returns critic_features for PopArt training.
        This avoids recomputing the forward pass in update().

        Returns:
            action_mean, action_logstd, value, critic_features
        """
        if cash_normalized is None:
            cash_normalized = torch.zeros_like(inventory)
        if inventory_normalized is None:
            inventory_normalized = torch.zeros_like(inventory)
        if room_to_buy is None:
            room_to_buy = torch.ones_like(inventory)
        if room_to_sell is None:
            room_to_sell = torch.ones_like(inventory)
        if last_bid_fill is None:
            last_bid_fill = torch.zeros_like(inventory)
        if last_ask_fill is None:
            last_ask_fill = torch.zeros_like(inventory)
        if spread_bps is None:
            spread_bps = torch.zeros_like(inventory)
        if bid_order_age is None:
            bid_order_age = torch.zeros_like(inventory)
        if ask_order_age is None:
            ask_order_age = torch.zeros_like(inventory)

        pos_info = torch.cat([
            inventory.view(-1, 1),
            time_remaining.view(-1, 1),
            cash_normalized.view(-1, 1),
            inventory_normalized.view(-1, 1),
            room_to_buy.view(-1, 1),
            room_to_sell.view(-1, 1),
            last_bid_fill.view(-1, 1),
            last_ask_fill.view(-1, 1),
            (spread_bps / 100.0).view(-1, 1),
            bid_order_age.view(-1, 1),
            ask_order_age.view(-1, 1),
        ], dim=-1)

        pos_encoded = self.position_encoder(pos_info)
        x = torch.cat([backbone_features, pos_encoded], dim=-1)
        x = self.fusion(x)

        action_mean = self.actor_mean(x)
        tanh_out = self.actor_logstd_net(x)
        action_logstd = self.logstd_min + 0.5 * (self.logstd_max - self.logstd_min) * (tanh_out + 1.0)
        critic_features = self.critic_backbone(x)
        value = self.critic_popart(critic_features)

        return action_mean, action_logstd, value, critic_features



class PPOAgent:
    """
    PPO Agent using LOBModel as backbone for feature extraction.
    Supports loading pretrained Mamba-LOB weights.
    """

    def __init__(self,
                 rl_config: RLConfig,
                 model_config: Optional[ModelConfig] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 pretrained_path: Optional[str] = None,
                 freeze_backbone: bool = False,
                 n_episodes: int = 1000):
        
        self.rl_config = rl_config
        self.device = device
        self.model_config = model_config or ModelConfig()
        
        # Use LOBModel as backbone (no duplication!)
        self.backbone = LOBModel(self.model_config).to(device)
        
        # Load pretrained weights if provided
        if pretrained_path:
            logger.info(f"Loading pretrained backbone from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info("Pretrained weights loaded successfully")
        
        # Optionally freeze backbone for fine-tuning heads only
        if freeze_backbone:
            logger.info("Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Actor/Critic heads on top of backbone
        # LOBModel outputs 576-dim (512 LOB + 64 aux when aux enabled)
        backbone_output_dim = 512 + (64 if self.model_config.n_aux_features > 0 else 0)
        self.heads = ActorCriticHead(
            input_dim=backbone_output_dim,
            hidden_dim=256,
            action_dim=9,
            position_dim=11
        ).to(device)

        # === TORCH.COMPILE OPTIMIZATION (PyTorch 2.0+) ===
        # Provides 20-40% speedup via graph compilation and kernel fusion
        # NOTE: Requires Triton which is NOT available on Windows
        import platform
        is_windows = platform.system() == 'Windows'

        self._use_compile = getattr(rl_config, 'use_torch_compile', True)
        if is_windows and self._use_compile:
            logger.info("torch.compile disabled on Windows (Triton not supported)")
            self._use_compile = False

        if self._use_compile and hasattr(torch, 'compile'):
            try:
                # 'reduce-overhead' mode is best for inference-heavy RL workloads
                compile_mode = 'reduce-overhead'
                self.backbone = torch.compile(self.backbone, mode=compile_mode)
                self.heads = torch.compile(self.heads, mode=compile_mode)
                logger.info(f"torch.compile enabled (mode={compile_mode}) - expect 20-40% speedup")
            except Exception as e:
                logger.warning(f"torch.compile failed, falling back to eager mode: {e}")
                self._use_compile = False
        elif not is_windows:
            logger.info("torch.compile disabled or not available")

        # Combine parameters for optimizer
        params = list(self.heads.parameters())
        if not freeze_backbone:
            params += list(self.backbone.parameters())
        
        # === SOTA OPTIMIZER SELECTION ===
        # AdamW: Standard baseline, stable for RL (what Mamba authors used)
        # Muon: Best for Mamba/SSM but can cause NaN in RL due to aggressive updates
        # Lion (Google 2023): Good for transformers
        # Sophia (Stanford 2023): Second-order, 2x faster on LLMs
        optimizer_name = getattr(rl_config, 'optimizer', 'adamw')
        self.optimizer = get_optimizer(
            name=optimizer_name,
            params=params,
            lr=rl_config.learning_rate,
            weight_decay=getattr(rl_config, 'weight_decay', 0.1),
        )
        self.optimizer_name = optimizer_name
        logger.info(f"Using {optimizer_name.upper()} optimizer (lr={rl_config.learning_rate})")
        self.scheduler = self._build_scheduler(n_episodes)
        self.entropy_coef = 0.05  # Increased for better exploration (was 0.01)

        # Mixed precision support - DISABLED to prevent KL divergence from precision mismatch
        # (data collected in fp32, but update in fp16/bf16 causes different outputs)
        self._amp_device = 'cuda' if torch.device(device).type == 'cuda' else 'cpu'
        self._amp_enabled = False  # Disabled - was causing KL=2.0+ on first batch
        self._amp_dtype = torch.float32
        self.scaler = torch.amp.GradScaler(self._amp_device, enabled=False)
        self.memory = PPOMemory(rl_config.batch_size, rl_config.buffer_size, device)
        
        # History buffers for temporal processing
        self.history_size = getattr(model_config, 'window_size', 50)
        self.lob_history = deque(maxlen=self.history_size)
        self.market_history = deque(maxlen=self.history_size)

        # Per-env histories for vectorized data collection (initialized lazily)
        self._vec_lob_histories: Dict[int, deque] = {}
        self._vec_market_histories: Dict[int, deque] = {}
        
        # Lagrangian multipliers for constrained optimization (M3ORL-style)
        self.lagrangian_risk = nn.Parameter(torch.tensor(rl_config.lagrangian_initial_lambda))
        self.lagrangian_ent = nn.Parameter(torch.tensor(rl_config.lagrangian_initial_lambda))
        self.lagrangian_as = nn.Parameter(torch.tensor(rl_config.lagrangian_initial_lambda))
        self.risk_target = rl_config.lagrangian_target_inventory
        self.ent_target = rl_config.lagrangian_target_entropy
        self.toxic_fill_target = getattr(rl_config, 'lagrangian_target_toxic_fill', 0.3)

        # === SAC-STYLE AUTOMATIC ENTROPY TUNING (Haarnoja 2018) ===
        # Theorem 1: Optimal temperature α satisfies: E[-α·log(π) - α·H*] = 0
        #
        # === THEORETICAL DERIVATION OF TARGET ENTROPY ===
        #
        # For Gaussian policy with actions in [-1, 1] (tanh bounded):
        #   - We want coverage p of action space (e.g., 95%)
        #   - Gaussian: p% of samples within ±z_p·σ where z_p = Φ⁻¹((1+p)/2)
        #   - For actions to stay in [-1,1]: 1 = z_p·σ → σ = 1/z_p
        #
        # Entropy of Gaussian: H = 0.5·log(2πe·σ²) = 0.5·log(2πe) + log(σ)
        #                        = 1.4189 + log(1/z_p) = 1.4189 - log(z_p)
        #
        # For p=0.95: z_0.95 = 1.96 → H_per_dim = 1.4189 - 0.673 = 0.746
        # For p=0.90: z_0.90 = 1.645 → H_per_dim = 1.4189 - 0.498 = 0.921
        # For p=0.80: z_0.80 = 1.28 → H_per_dim = 1.4189 - 0.247 = 1.172
        #
        # Total: H_target = dim(A) × H_per_dim
        #
        self.auto_entropy_tuning = getattr(rl_config, 'auto_entropy_tuning', True)
        action_dim = 9
        coverage = getattr(rl_config, 'entropy_coverage', 0.90)  # 90% action space coverage
        z_p = 1.645 if coverage == 0.90 else (1.96 if coverage == 0.95 else 1.28)  # Gaussian quantile
        h_per_dim = 0.5 * np.log(2 * np.pi * np.e) - np.log(z_p)  # = 1.4189 - log(z_p)
        self.target_entropy = action_dim * h_per_dim
        logger.info(f"Target entropy (coverage={coverage:.0%}): {self.target_entropy:.3f} "
                    f"(H/dim={h_per_dim:.3f}, σ_target={1/z_p:.3f})")
        if self.auto_entropy_tuning:
            # log(alpha) is learnable for numerical stability
            self.log_alpha = nn.Parameter(torch.tensor(getattr(rl_config, 'log_alpha_init', 0.0)))
            # Use same optimizer type for alpha (but simpler - just one param)
            alpha_lr = getattr(rl_config, 'alpha_lr', 3e-4)
            if optimizer_name == 'lion':
                self.alpha_optimizer = Lion([self.log_alpha], lr=alpha_lr / 3, weight_decay=0)
            else:
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self._alpha_scheduler_enabled = True
        else:
            self.log_alpha = torch.tensor(0.0).to(device)
            self._alpha_scheduler_enabled = False

        # === ADAPTIVE KL CONTROLLER (Data-driven calibration) ===
        # Auto-calibrates KL thresholds from observed distribution
        self.adaptive_kl_controller = AdaptiveKLController(
            action_dim=9,
            window_size=100,
            calibration_episodes=getattr(rl_config, 'kl_calibration_episodes', 20),
            target_percentile=getattr(rl_config, 'kl_target_percentile', 50.0),
            beta_min=getattr(rl_config, 'kl_beta_min', 0.5),
            beta_max=getattr(rl_config, 'kl_beta_max', 100.0),
        )

        # Legacy attributes for backward compatibility
        self.adaptive_kl_penalty = True
        self.kl_beta = self.adaptive_kl_controller.beta
        self.kl_target = self.adaptive_kl_controller.target_kl
        self.kl_early_stop_mult = 10.0  # Fallback, overridden by adaptive controller

        # === AVELLANEDA-STOIKOV RISK AVERSION (2008) ===
        # gamma in HJB: V = x + q*s - gamma*sigma^2*q^2*(T-t)
        # Learned via Lagrangian dual for variance constraint
        self.risk_aversion_learnable = getattr(rl_config, 'risk_aversion_learnable', True)
        self.risk_aversion = nn.Parameter(torch.tensor(getattr(rl_config, 'risk_aversion_init', 0.1)))
        self.risk_aversion_lr = getattr(rl_config, 'risk_aversion_lr', 1e-3)
        self.target_pnl_variance = getattr(rl_config, 'target_pnl_variance', 0.01)
        self._pnl_history = deque(maxlen=1000)  # For variance estimation

        # Rolling tracker for toxic fill ratio (updated per episode)
        self._toxic_fills = 0
        self._total_fills = 0
        
        # Running normalization for observations (13 market features) and rewards
        n_aux = getattr(self.model_config, 'n_aux_features', 15)
        self.obs_rms = RunningMeanStd(shape=(n_aux,))
        self.reward_rms = RunningMeanStd(shape=())

        self.train_steps = 0
        self.episodes = 0

        # === GRADIENT NOISE SCALE FOR ADAPTIVE LR (McCandlish 2018) ===
        # Derives optimal learning rate from gradient statistics
        self.grad_noise_scale = GradientNoiseScale(window_size=50)
        self.adaptive_lr = getattr(rl_config, 'adaptive_lr', True)
        self.base_lr = rl_config.learning_rate  # Store initial LR

        # === ADAPTIVE COVERAGE FOR TARGET ENTROPY (Data-driven) ===
        # Adapts exploration→exploitation based on reward improvement
        self.adaptive_coverage = AdaptiveCoverage(action_dim=9, window_size=50)
        self.target_entropy = self.adaptive_coverage.current_target_entropy

        # === ADAPTIVE CLIP RATIO (Engstrom 2020) ===
        # Adapts clip ratio based on policy ratio variance
        self.adaptive_clip = AdaptiveClipRatio(window_size=100)
        self.current_clip_ratio = rl_config.clip_ratio  # Will be updated adaptively

        logger.info(f"PPOAgent initialized with unified LOBModel backbone (freeze={freeze_backbone})")
        logger.info(f"  Adaptive systems: LR={self.adaptive_lr}, Coverage=True, ClipRatio=True")

    def _build_scheduler(self, n_episodes: int) -> LambdaLR:
        """Data-driven warmup + cosine decay LR scheduler.

        Warmup is adaptive based on gradient noise calibration:
        - Start at 1% LR until GradientNoiseScale is calibrated (~10-20 episodes)
        - Then ramp to full LR over remaining warmup period
        - Cosine decay for the rest of training
        """
        # Base warmup = sqrt(n_episodes) - scales sub-linearly with training length
        # For 1000 episodes: sqrt(1000) ≈ 32 warmup episodes
        # For 100 episodes: sqrt(100) = 10 warmup episodes
        base_warmup = int(math.sqrt(n_episodes))
        warmup_episodes = max(10, min(50, base_warmup))  # Clamp to [10, 50]

        def lr_lambda(ep):
            # Phase 1: Initial warmup (1% to 100%)
            if ep < warmup_episodes:
                # Linear warmup from 1% to 100%
                return max(ep / max(warmup_episodes, 1), 0.01)
            # Phase 2: Cosine decay from 100% to ~0%
            progress = (ep - warmup_episodes) / max(n_episodes - warmup_episodes, 1)
            # Cosine annealing with minimum LR floor of 1%
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def reset_episode(self):
        """Clear temporal history between episodes to avoid cross-episode data leakage."""
        self.lob_history.clear()
        self.market_history.clear()
        self._toxic_fills = 0
        self._total_fills = 0

    def init_vec_histories(self, n_envs: int):
        """Initialize per-env history deques for vectorized data collection."""
        self._vec_lob_histories = {i: deque(maxlen=self.history_size) for i in range(n_envs)}
        self._vec_market_histories = {i: deque(maxlen=self.history_size) for i in range(n_envs)}

    def reset_env_history(self, env_id: int):
        """Clear history for a specific vectorized env."""
        if env_id in self._vec_lob_histories:
            self._vec_lob_histories[env_id].clear()
            self._vec_market_histories[env_id].clear()

    def select_action_for_env(self, state: Dict, env_id: int) -> Tuple[np.ndarray, float, float]:
        """Select action using env-specific history (for vectorized data collection).

        Identical to select_action() but uses per-env history deques to avoid
        cross-env data leakage when collecting data from multiple environments.
        """
        self.backbone.eval()
        self.heads.eval()

        with torch.no_grad():
            # Use per-env history (not the shared one)
            lob_hist = self._vec_lob_histories[env_id]
            mkt_hist = self._vec_market_histories[env_id]

            lob_hist.append(state['lob_features'])
            n_aux = getattr(self.model_config, 'n_aux_features', 15)
            mkt_feat = state.get('market_features', np.zeros(n_aux))

            # SKIP obs_rms normalization — it causes KL divergence because stats
            # change during collection, making early samples normalized differently
            # than late samples. Market features from env are already reasonable.
            # Just clip to prevent extreme values.
            mkt_feat = np.clip(mkt_feat, -100.0, 100.0)

            mkt_hist.append(mkt_feat)

            # Build sequence from per-env history
            def get_seq(hist, current_feat):
                seq = list(hist)
                if len(seq) < self.history_size:
                    padding = [np.zeros_like(current_feat)] * (self.history_size - len(seq))
                    seq = padding + seq
                return np.array(seq)

            lob_seq = get_seq(lob_hist, state['lob_features'])

            # Prepare tensors
            lob_t = torch.FloatTensor(lob_seq).unsqueeze(0).to(self.device)
            mkt_t = torch.FloatTensor(mkt_feat).unsqueeze(0).to(self.device)
            inv_t = torch.FloatTensor([state['inventory']]).to(self.device)
            time_t = torch.FloatTensor([state['time_remaining']]).to(self.device)

            cash_norm_t = torch.FloatTensor([state.get('cash_normalized', 1.0)]).to(self.device)
            inv_norm_t = torch.FloatTensor([state.get('inventory_normalized', 0.0)]).to(self.device)
            rtb_t = torch.FloatTensor([state.get('room_to_buy', 1.0)]).to(self.device)
            rts_t = torch.FloatTensor([state.get('room_to_sell', 1.0)]).to(self.device)
            lbf_t = torch.FloatTensor([state.get('last_bid_fill', 0.0)]).to(self.device)
            laf_t = torch.FloatTensor([state.get('last_ask_fill', 0.0)]).to(self.device)
            spd_t = torch.FloatTensor([state.get('spread_bps', 0.0)]).to(self.device)
            boa_t = torch.FloatTensor([state.get('bid_order_age', 0.0)]).to(self.device)
            aoa_t = torch.FloatTensor([state.get('ask_order_age', 0.0)]).to(self.device)

            backbone_features = self._get_backbone_features(lob_t, mkt_t)

            mu, logstd, value = self.heads(
                backbone_features, inv_t, time_t,
                cash_norm_t, inv_norm_t, rtb_t, rts_t,
                lbf_t, laf_t, spd_t,
                boa_t, aoa_t
            )

            mu, logstd = self._apply_action_mask(mu, logstd, state)

            dist = torch.distributions.Normal(mu, torch.exp(logstd))
            action = dist.sample()
            # Clamp log_prob with theoretically-derived bounds (6-sigma coverage)
            lp_min, lp_max = compute_log_prob_bounds(logstd, k_sigma=6.0)
            log_prob = torch.clamp(dist.log_prob(action).sum(dim=-1), min=lp_min, max=lp_max)

            # Store sequence for memory
            state['lob_seq'] = lob_seq

            return action.cpu().numpy()[0], log_prob.item(), value.item()

    def select_action_batch(self, states: List[Dict], dones: List[bool]) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """Batched action selection for vectorized environments.

        CRITICAL OPTIMIZATION: Instead of N separate forward passes (one per env),
        this method batches all states and runs a single forward pass.
        Speedup: ~Nx on GPU where N = number of environments.

        Args:
            states: List of N state dicts from each environment
            dones: List of N booleans indicating if each env is done

        Returns:
            actions: List of N action arrays
            log_probs: List of N log probability floats
            values: List of N value floats
        """
        n_envs = len(states)
        self.backbone.eval()
        self.heads.eval()

        # Pre-allocate outputs for done envs
        actions = [np.zeros(9) for _ in range(n_envs)]
        log_probs = [0.0] * n_envs
        values = [0.0] * n_envs

        # Get indices of active (non-done) environments
        active_indices = [i for i, done in enumerate(dones) if not done]
        if not active_indices:
            return actions, log_probs, values

        with torch.no_grad():
            n_aux = getattr(self.model_config, 'n_aux_features', 15)

            # Build sequences helper
            def get_seq(hist, current_feat):
                seq = list(hist)
                if len(seq) < self.history_size:
                    padding = [np.zeros_like(current_feat)] * (self.history_size - len(seq))
                    seq = padding + seq
                return np.array(seq)

            # Collect batch data
            lob_batch = []
            mkt_batch = []
            inv_batch = []
            time_batch = []
            cash_batch = []
            inv_norm_batch = []
            rtb_batch = []
            rts_batch = []
            lbf_batch = []
            laf_batch = []
            spd_batch = []
            boa_batch = []
            aoa_batch = []

            for i in active_indices:
                state = states[i]

                # Update per-env history
                lob_hist = self._vec_lob_histories[i]
                mkt_hist = self._vec_market_histories[i]

                lob_hist.append(state['lob_features'])
                mkt_feat = state.get('market_features', np.zeros(n_aux))
                mkt_feat = np.clip(mkt_feat, -100.0, 100.0)
                mkt_hist.append(mkt_feat)

                lob_seq = get_seq(lob_hist, state['lob_features'])
                state['lob_seq'] = lob_seq  # Store for memory

                lob_batch.append(lob_seq)
                mkt_batch.append(mkt_feat)
                inv_batch.append(state['inventory'])
                time_batch.append(state['time_remaining'])
                cash_batch.append(state.get('cash_normalized', 1.0))
                inv_norm_batch.append(state.get('inventory_normalized', 0.0))
                rtb_batch.append(state.get('room_to_buy', 1.0))
                rts_batch.append(state.get('room_to_sell', 1.0))
                lbf_batch.append(state.get('last_bid_fill', 0.0))
                laf_batch.append(state.get('last_ask_fill', 0.0))
                spd_batch.append(state.get('spread_bps', 0.0))
                boa_batch.append(state.get('bid_order_age', 0.0))
                aoa_batch.append(state.get('ask_order_age', 0.0))

            # Convert to tensors (batch dimension first)
            lob_t = torch.FloatTensor(np.array(lob_batch)).to(self.device)  # (B, T, F)
            mkt_t = torch.FloatTensor(np.array(mkt_batch)).to(self.device)  # (B, n_aux)
            inv_t = torch.FloatTensor(inv_batch).to(self.device)
            time_t = torch.FloatTensor(time_batch).to(self.device)
            cash_t = torch.FloatTensor(cash_batch).to(self.device)
            inv_norm_t = torch.FloatTensor(inv_norm_batch).to(self.device)
            rtb_t = torch.FloatTensor(rtb_batch).to(self.device)
            rts_t = torch.FloatTensor(rts_batch).to(self.device)
            lbf_t = torch.FloatTensor(lbf_batch).to(self.device)
            laf_t = torch.FloatTensor(laf_batch).to(self.device)
            spd_t = torch.FloatTensor(spd_batch).to(self.device)
            boa_t = torch.FloatTensor(boa_batch).to(self.device)
            aoa_t = torch.FloatTensor(aoa_batch).to(self.device)

            # Single batched forward pass
            backbone_features = self._get_backbone_features(lob_t, mkt_t)

            mu, logstd, value = self.heads(
                backbone_features, inv_t, time_t,
                cash_t, inv_norm_t, rtb_t, rts_t,
                lbf_t, laf_t, spd_t,
                boa_t, aoa_t
            )

            # Apply action masks (batched)
            mu, logstd = self._apply_action_mask_batch(mu, logstd, rtb_t, rts_t)

            # Sample actions
            dist = torch.distributions.Normal(mu, torch.exp(logstd))
            action_t = dist.sample()
            # Clamp log_prob with theoretically-derived bounds (6-sigma coverage)
            lp_min, lp_max = compute_log_prob_bounds(logstd, k_sigma=6.0)
            log_prob_t = torch.clamp(dist.log_prob(action_t).sum(dim=-1), min=lp_min, max=lp_max)

            # Convert to numpy and distribute to output lists
            action_np = action_t.cpu().numpy()
            log_prob_np = log_prob_t.cpu().numpy()
            value_np = value.cpu().numpy().squeeze(-1)

            for j, i in enumerate(active_indices):
                actions[i] = action_np[j]
                log_probs[i] = float(log_prob_np[j])
                values[i] = float(value_np[j])

        return actions, log_probs, values

    def track_adverse_selection(self, info: Dict):
        """Track toxic fill ratio from step info for Lagrangian constraint.
        Call after each env.step() with the info dict."""
        fill_qty = info.get('fill_qty', 0.0)
        ivpin = info.get('ivpin_fast', 0.0)
        if fill_qty > 0:
            self._total_fills += 1
            if ivpin > 0.7:
                self._toxic_fills += 1

    def track_pnl(self, pnl: float):
        """Track PnL for variance estimation (Avellaneda-Stoikov risk learning).

        Used for the Lagrangian dual: max_γ min_π E[R] - γ(Var[PnL] - ε)
        γ is increased when Var[PnL] > ε (too much risk)
        γ is decreased when Var[PnL] < ε (can take more risk)
        """
        self._pnl_history.append(pnl)

    def get_risk_aversion(self) -> float:
        """Get current risk aversion γ for syncing to environment.

        This value is learned via Lagrangian dual optimization,
        not set arbitrarily.
        """
        return self.risk_aversion.item()

    @property
    def lambda_as(self) -> float:
        """Current adverse selection Lagrangian multiplier (for env reward sync)."""
        return self.lagrangian_as.item()

    @staticmethod
    def _apply_action_mask(mu: torch.Tensor, logstd: torch.Tensor, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft action masking for continuous actions (single sample inference).
        Reduces exploration noise and biases mean for constrained action dimensions.
        Action dims: [bid_spread, ask_spread, bid_qty, ask_qty, flatten,
                      ladder_step, ladder_decay, hold_bid, hold_ask]
        """
        room_to_buy = state.get('room_to_buy', 1.0)
        room_to_sell = state.get('room_to_sell', 1.0)

        # If no room to buy: shrink bid_qty (dim 2) std, push mean to -1
        if room_to_buy < 0.01:
            mu = mu.clone()
            logstd = logstd.clone()
            mu[..., 2] = -1.0   # min qty
            logstd[..., 2] = -4.0  # near-zero std

        # If no room to sell: shrink ask_qty (dim 3) std, push mean to -1
        if room_to_sell < 0.01:
            mu = mu.clone()
            logstd = logstd.clone()
            mu[..., 3] = -1.0
            logstd[..., 3] = -4.0

        return mu, logstd

    @staticmethod
    def _apply_action_mask_batch(mu: torch.Tensor, logstd: torch.Tensor,
                                  room_to_buy: torch.Tensor, room_to_sell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply action masking to batch tensors during update.
        CRITICAL: Must match logic in _apply_action_mask exactly to avoid phantom KL divergence.

        Args:
            mu: (B, 9) action means
            logstd: (B, 9) action log standard deviations
            room_to_buy: (B,) normalized buy capacity
            room_to_sell: (B,) normalized sell capacity

        Returns:
            Masked (mu, logstd) tensors
        """
        # Clone to avoid in-place modification issues with autograd
        mu = mu.clone()
        logstd = logstd.clone()

        # Mask buy: if room_to_buy < 0.01, force bid_qty (dim 2) to min
        mask_buy = (room_to_buy < 0.01)
        if mask_buy.any():
            mu[mask_buy, 2] = -1.0
            logstd[mask_buy, 2] = -4.0

        # Mask sell: if room_to_sell < 0.01, force ask_qty (dim 3) to min
        mask_sell = (room_to_sell < 0.01)
        if mask_sell.any():
            mu[mask_sell, 3] = -1.0
            logstd[mask_sell, 3] = -4.0

        return mu, logstd

    def _get_backbone_features(self, lob_seq: torch.Tensor, market_features: torch.Tensor = None, training: bool = False) -> torch.Tensor:
        """
        Extract features from LOBModel backbone.

        Args:
            lob_seq: (B, L, 40) LOB sequence
            market_features: (B, 13) microstructure features (optional)
            training: if True, keep gradients for backprop; if False, use no_grad (inference)

        Returns:
            (B, 576) fused features
        """
        if training:
            # Training path: extract features WITH gradients (for backprop).
            # Use eval() mode to disable dropout — ensures outputs match data collection.
            # Only LayerNorm (no BatchNorm), so eval mode is safe. Gradients still flow.
            was_training = self.backbone.training
            self.backbone.eval()
            x = self.backbone.input_norm(lob_seq)
            x = self.backbone.embedding(x)
            x = self.backbone.mamba(x)
            x = x.permute(0, 2, 1)
            for conv_layer in self.backbone.conv_layers:
                x = conv_layer(x)
            x = x.permute(0, 2, 1)
            lob_features = x.mean(1)  # (B, 512)
        else:
            was_training = False
            # Inference path: extract_features() uses eval()+no_grad() internally
            lob_features = self.backbone.extract_features(lob_seq)  # (B, 512)

        # Add auxiliary features if available (backbone still in eval mode for dropout consistency)
        if market_features is not None and self.backbone.aux_input_dim > 0:
            aux_embedding = self.backbone.aux_encoder(market_features)  # (B, 64)
            fused = torch.cat([lob_features, aux_embedding], dim=-1)  # (B, 576)
        else:
            if self.backbone.aux_input_dim > 0:
                zeros = torch.zeros(lob_features.size(0), 64, device=lob_features.device)
                fused = torch.cat([lob_features, zeros], dim=-1)
            else:
                fused = lob_features

        # Restore train mode after feature extraction
        if was_training:
            self.backbone.train()

        return fused

    def select_action(self, state: Dict) -> Tuple[np.ndarray, float, float]:
        """Select action based on current state and history."""
        self.backbone.eval()
        self.heads.eval()
        
        with torch.no_grad():
            # Update history
            self.lob_history.append(state['lob_features'])
            n_aux = getattr(self.model_config, 'n_aux_features', 15)
            mkt_feat = state.get('market_features', np.zeros(n_aux))

            # SKIP obs_rms normalization — it causes KL divergence because stats
            # change during collection, making early samples normalized differently
            # than late samples. Market features from env are already reasonable.
            # Just clip to prevent extreme values.
            mkt_feat = np.clip(mkt_feat, -100.0, 100.0)

            self.market_history.append(mkt_feat)

            # Build sequence
            def get_seq(hist, current_feat):
                seq = list(hist)
                if len(seq) < self.history_size:
                    padding = [np.zeros_like(current_feat)] * (self.history_size - len(seq))
                    seq = padding + seq
                return np.array(seq)

            lob_seq = get_seq(self.lob_history, state['lob_features'])

            # Prepare tensors
            lob_t = torch.FloatTensor(lob_seq).unsqueeze(0).to(self.device)
            mkt_t = torch.FloatTensor(mkt_feat).unsqueeze(0).to(self.device)
            inv_t = torch.FloatTensor([state['inventory']]).to(self.device)
            time_t = torch.FloatTensor([state['time_remaining']]).to(self.device)

            # Portfolio state info
            cash_norm_t = torch.FloatTensor([state.get('cash_normalized', 1.0)]).to(self.device)
            inv_norm_t = torch.FloatTensor([state.get('inventory_normalized', 0.0)]).to(self.device)
            rtb_t = torch.FloatTensor([state.get('room_to_buy', 1.0)]).to(self.device)
            rts_t = torch.FloatTensor([state.get('room_to_sell', 1.0)]).to(self.device)
            lbf_t = torch.FloatTensor([state.get('last_bid_fill', 0.0)]).to(self.device)
            laf_t = torch.FloatTensor([state.get('last_ask_fill', 0.0)]).to(self.device)
            spd_t = torch.FloatTensor([state.get('spread_bps', 0.0)]).to(self.device)
            boa_t = torch.FloatTensor([state.get('bid_order_age', 0.0)]).to(self.device)
            aoa_t = torch.FloatTensor([state.get('ask_order_age', 0.0)]).to(self.device)

            # Get backbone features
            backbone_features = self._get_backbone_features(lob_t, mkt_t)

            # Forward through heads
            mu, logstd, value = self.heads(
                backbone_features, inv_t, time_t,
                cash_norm_t, inv_norm_t, rtb_t, rts_t,
                lbf_t, laf_t, spd_t,
                boa_t, aoa_t
            )

            # Soft action masking for constrained dims
            mu, logstd = self._apply_action_mask(mu, logstd, state)

            dist = torch.distributions.Normal(mu, torch.exp(logstd))
            action = dist.sample()
            # Clamp log_prob with theoretically-derived bounds (6-sigma coverage)
            lp_min, lp_max = compute_log_prob_bounds(logstd, k_sigma=6.0)
            log_prob = torch.clamp(dist.log_prob(action).sum(dim=-1), min=lp_min, max=lp_max)

            # Store sequence for memory
            state['lob_seq'] = lob_seq

            return action.cpu().numpy()[0], log_prob.item(), value.item()

    def select_action_deterministic(self, state: Dict) -> Tuple[np.ndarray, float, float]:
        """Select action deterministically for evaluation (uses mean, no sampling noise)."""
        self.backbone.eval()
        self.heads.eval()

        with torch.no_grad():
            # Update history
            self.lob_history.append(state['lob_features'])
            n_aux = getattr(self.model_config, 'n_aux_features', 15)
            mkt_feat = state.get('market_features', np.zeros(n_aux))

            # SKIP obs_rms normalization during eval to match training behavior
            # (training also skips normalization to avoid KL divergence issues)
            mkt_feat = np.clip(mkt_feat, -100.0, 100.0)

            self.market_history.append(mkt_feat)

            # Build sequence
            def get_seq(hist, current_feat):
                seq = list(hist)
                if len(seq) < self.history_size:
                    padding = [np.zeros_like(current_feat)] * (self.history_size - len(seq))
                    seq = padding + seq
                return np.array(seq)

            lob_seq = get_seq(self.lob_history, state['lob_features'])

            # Prepare tensors
            lob_t = torch.FloatTensor(lob_seq).unsqueeze(0).to(self.device)
            mkt_t = torch.FloatTensor(mkt_feat).unsqueeze(0).to(self.device)
            inv_t = torch.FloatTensor([state['inventory']]).to(self.device)
            time_t = torch.FloatTensor([state['time_remaining']]).to(self.device)

            cash_norm_t = torch.FloatTensor([state.get('cash_normalized', 1.0)]).to(self.device)
            inv_norm_t = torch.FloatTensor([state.get('inventory_normalized', 0.0)]).to(self.device)
            rtb_t = torch.FloatTensor([state.get('room_to_buy', 1.0)]).to(self.device)
            rts_t = torch.FloatTensor([state.get('room_to_sell', 1.0)]).to(self.device)
            lbf_t = torch.FloatTensor([state.get('last_bid_fill', 0.0)]).to(self.device)
            laf_t = torch.FloatTensor([state.get('last_ask_fill', 0.0)]).to(self.device)
            spd_t = torch.FloatTensor([state.get('spread_bps', 0.0)]).to(self.device)
            boa_t = torch.FloatTensor([state.get('bid_order_age', 0.0)]).to(self.device)
            aoa_t = torch.FloatTensor([state.get('ask_order_age', 0.0)]).to(self.device)

            backbone_features = self._get_backbone_features(lob_t, mkt_t)

            mu, _logstd, value = self.heads(
                backbone_features, inv_t, time_t,
                cash_norm_t, inv_norm_t, rtb_t, rts_t,
                lbf_t, laf_t, spd_t,
                boa_t, aoa_t
            )

            # Soft action masking for constrained dims
            mu, _logstd = self._apply_action_mask(mu, _logstd, state)

            return mu.cpu().numpy()[0], 0.0, value.item()

    def store_transition(self, state: Dict, action: np.ndarray, reward: float,
                        next_state: Dict, log_prob: float, value: float, done: bool):
        """Store transition in memory with reward normalization."""
        # Normalize reward using RunningMeanStd (auto-scales across market conditions)
        self.reward_rms.update(np.array([reward]))
        reward = float((reward - self.reward_rms.mean) / np.sqrt(self.reward_rms.var + 1e-8))

        n_aux = getattr(self.model_config, 'n_aux_features', 15)
        stored_state = {
            'lob_features': state.get('lob_seq', state['lob_features']),
            'market_features': state.get('market_features', np.zeros(n_aux)),
            'inventory': state['inventory'],
            'time_remaining': state['time_remaining'],
            'cash_normalized': state.get('cash_normalized', 1.0),
            'inventory_normalized': state.get('inventory_normalized', 0.0),
            'room_to_buy': state.get('room_to_buy', 1.0),
            'room_to_sell': state.get('room_to_sell', 1.0),
            'last_bid_fill': state.get('last_bid_fill', 0.0),
            'last_ask_fill': state.get('last_ask_fill', 0.0),
            'spread_bps': state.get('spread_bps', 0.0),
            'bid_order_age': state.get('bid_order_age', 0.0),
            'ask_order_age': state.get('ask_order_age', 0.0),
        }
        self.memory.store(
            state=stored_state, action=action, reward=reward,
            next_state=next_state, log_prob=log_prob, value=value, done=done
        )

    def store_trajectory(self, trajectory: List[Dict]):
        """Store an entire trajectory at once (OPTIMIZED: 10x faster than individual store() calls).

        Args:
            trajectory: List of dicts with keys 'state', 'action', 'reward',
                       'next_state', 'log_prob', 'value', 'done'
        """
        if not trajectory:
            return

        n_aux = getattr(self.model_config, 'n_aux_features', 15)

        # Pre-process trajectory: normalize rewards and prepare states
        processed = []
        for t in trajectory:
            state = t['state']
            reward = t['reward']

            # Normalize reward
            self.reward_rms.update(np.array([reward]))
            norm_reward = float((reward - self.reward_rms.mean) / np.sqrt(self.reward_rms.var + 1e-8))

            stored_state = {
                'lob_features': state.get('lob_seq', state['lob_features']),
                'market_features': state.get('market_features', np.zeros(n_aux)),
                'inventory': state['inventory'],
                'time_remaining': state['time_remaining'],
                'cash_normalized': state.get('cash_normalized', 1.0),
                'inventory_normalized': state.get('inventory_normalized', 0.0),
                'room_to_buy': state.get('room_to_buy', 1.0),
                'room_to_sell': state.get('room_to_sell', 1.0),
                'last_bid_fill': state.get('last_bid_fill', 0.0),
                'last_ask_fill': state.get('last_ask_fill', 0.0),
                'spread_bps': state.get('spread_bps', 0.0),
                'bid_order_age': state.get('bid_order_age', 0.0),
                'ask_order_age': state.get('ask_order_age', 0.0),
            }

            processed.append({
                'state': stored_state,
                'action': t['action'],
                'reward': norm_reward,
                'next_state': t['next_state'],
                'log_prob': t['log_prob'],
                'value': t['value'],
                'done': t['done']
            })

        # Bulk store to memory
        self.memory.store_trajectory(processed)

    def update(self) -> Dict[str, float]:
        """Update model using collected experience."""
        if len(self.memory) < self.rl_config.batch_size:
            return {}

        self.backbone.train()
        self.heads.train()

        # === DATA-DRIVEN BATCH SIZE ===
        # Optimal batch size from Gradient Noise Scale (McCandlish 2018):
        # B_opt = B_noise (gradient noise scale)
        # Larger batches = lower variance but diminishing returns past B_noise
        buffer_len = len(self.memory)
        base_batch_size = self.rl_config.batch_size

        if self.grad_noise_scale.is_calibrated:
            # Use optimal batch size from GNS, clamped to reasonable range
            gns_info = self.grad_noise_scale.compute_optimal_lr(base_batch_size)
            optimal_batch = int(gns_info.get('b_noise', base_batch_size))
            # Clamp between base/2 and base*2, and ensure divisibility
            adaptive_batch = max(base_batch_size // 2, min(base_batch_size * 2, optimal_batch))
            adaptive_batch = (adaptive_batch // 32) * 32  # Round to multiple of 32 for GPU efficiency
            adaptive_batch = max(32, min(buffer_len // 4, adaptive_batch))  # Ensure at least 4 batches
        else:
            adaptive_batch = base_batch_size

        # Temporarily update memory's batch size for this update
        original_batch_size = self.memory.batch_size
        self.memory.batch_size = adaptive_batch

        epoch_metrics = {'actor_loss': [], 'critic_loss': [], 'entropy': [], 'approx_kl': []}
        
        try:
            rewards, values, dones = self.memory.get_rewards_values_dones()
            next_value = values[-1] if len(values) > 0 else 0.0

            # === ADAPTIVE COVERAGE UPDATE (Data-driven entropy target) ===
            mean_reward = float(np.mean(rewards))
            coverage_info = self.adaptive_coverage.update(mean_reward)
            self.target_entropy = coverage_info['target_entropy']
            
            advantages = self.memory.compute_advantages(
                rewards, values, dones,
                next_value=next_value,
                gamma=self.rl_config.gamma,
                gae_lambda=self.rl_config.gae_lambda
            )
            
            advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            advantages = advantages.squeeze(-1)  # Fix: ensure shape is (N,) not (N, 1)
            values_t = torch.tensor(values, dtype=torch.float32, device=self.device).squeeze(-1)

            # CRITICAL: Normalize advantages BEFORE computing returns
            # This ensures returns = V(s) + A_normalized(s) relationship holds
            adv_std = advantages.std()
            if adv_std < 1e-6:
                # If all advantages identical (flat rewards), use identity scaling
                adv_std = torch.tensor(1.0, device=self.device)
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            # Compute returns from normalized advantages
            returns = advantages + values_t
            
            kl_early_stop = False
            # Get adaptive KL parameters from controller
            kl_params = self.adaptive_kl_controller.update(0.0, self.episodes)  # Will be updated with real KL later
            kl_early_stop_threshold = kl_params['early_stop_threshold']
            optimizer_stepped = False

            # === DATA-DRIVEN NUM_EPOCHS ===
            # Each sample should be seen ~3 times on average (Schulman 2017)
            # num_epochs = ceil(target_sample_reuse * buffer_size / (n_batches * batch_size))
            # With KL early stopping, we can start high and stop early if needed
            buffer_size = len(self.memory.rewards_buffer)
            n_batches = max(1, buffer_size // self.rl_config.batch_size)
            target_sample_reuse = 3.0  # Each sample seen ~3 times
            adaptive_num_epochs = max(3, min(15, int(np.ceil(target_sample_reuse * n_batches))))
            # Use adaptive epochs if more data collected, else use config default
            num_epochs = adaptive_num_epochs if buffer_size > self.rl_config.batch_size * 4 else self.rl_config.num_epochs

            for epoch in range(num_epochs):
                if kl_early_stop:
                    break

                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_entropies = []

                batches = self.memory.get_batches()
                accum_steps = getattr(self.rl_config, 'gradient_accumulation_steps', 1)
                self.optimizer.zero_grad()
                for batch_idx, batch in enumerate(batches):
                    batch_indices = batch['indices']
                    batch_adv = advantages[batch_indices].squeeze(-1)   # (B,) — must be 1D for element-wise with ratio
                    batch_ret = returns[batch_indices].squeeze(-1)       # (B,)
                    
                    lob_f = batch['lob_features']
                    market_f = batch['market_features']  # Already normalized at collection time
                    inv_f = batch['inventory']
                    time_f = batch['time_remaining']
                    cash_norm_f = batch['cash_normalized']
                    inv_norm_f = batch['inventory_normalized']
                    rtb_f = batch['room_to_buy']
                    rts_f = batch['room_to_sell']
                    lbf_f = batch['last_bid_fill']
                    laf_f = batch['last_ask_fill']
                    spd_f = batch['spread_bps']
                    boa_f = batch['bid_order_age']
                    aoa_f = batch['ask_order_age']

                    # Mixed precision: autocast only for backbone + heads forward
                    with torch.amp.autocast(self._amp_device, dtype=self._amp_dtype, enabled=self._amp_enabled):
                        backbone_features = self._get_backbone_features(lob_f, market_f, training=True)

                        # OPTIMIZED: Use forward_with_features to get critic_features in single pass
                        # (avoids redundant forward for PopArt loss)
                        action_mean, action_logstd, values_pred, critic_features = self.heads.forward_with_features(
                            backbone_features, inv_f, time_f,
                            cash_norm_f, inv_norm_f, rtb_f, rts_f,
                            lbf_f, laf_f, spd_f,
                            boa_f, aoa_f
                        )

                        # [FIX] Apply action masking to match data collection behavior
                        # If we don't mask here, the "new" policy predicts unmasked actions for states where 
                        # we had zero room to trade, causing massive KL divergence against the "old" masked policy.
                        action_mean, action_logstd = self._apply_action_mask_batch(
                            action_mean, action_logstd, 
                            rtb_f, rts_f
                        )

                    # Distribution math in fp32 (avoids NaN from fp16 overflow)
                    action_mean = action_mean.float()
                    action_logstd = action_logstd.float()
                    values_pred = values_pred.float().squeeze(-1)

                    # NaN check after forward pass (masking already applied above)
                    if torch.isnan(backbone_features).any():
                        logger.warning(f"NaN in backbone features at epoch {epoch}, batch {batch_idx}. Skipping.")
                        continue
                    if torch.isnan(action_mean).any() or torch.isnan(action_logstd).any():
                        logger.warning(f"NaN in heads output at epoch {epoch}, batch {batch_idx}. Skipping.")
                        continue
                    # NOTE: Action mask already applied above (lines 2063-2066), no need to apply again

                    # NaN protection: skip batch if model outputs NaN
                    if torch.isnan(action_mean).any() or torch.isnan(action_logstd).any():
                        logger.warning(f"NaN detected in policy output at epoch {epoch}, batch {batch_idx}. Skipping batch.")
                        continue

                    dist = torch.distributions.Normal(action_mean, torch.exp(action_logstd))
                    # Clamp log_probs with theoretically-derived bounds (6-sigma coverage)
                    lp_min, lp_max = compute_log_prob_bounds(action_logstd, k_sigma=6.0)
                    new_log_probs = torch.clamp(dist.log_prob(batch['actions']).sum(-1), min=lp_min, max=lp_max)
                    # Total entropy = sum over action dimensions, then mean over batch
                    # For 9D Gaussian with σ=0.6: H ≈ 9 × 0.9 = 8.1
                    entropy = dist.entropy().sum(-1).mean()

                    # Clamp ratio with theoretically-derived bounds
                    # Based on PPO clip ratio ε with safety factor for outliers
                    r_min, r_max = compute_ratio_bounds(self.rl_config.clip_ratio, safety_factor=10.0)
                    ratio = torch.clamp(torch.exp(new_log_probs - batch['log_probs']), min=r_min, max=r_max)

                    # Update adaptive clip ratio based on ratio statistics (Engstrom 2020)
                    self.current_clip_ratio = self.adaptive_clip.update(ratio)

                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.current_clip_ratio,
                                       1.0 + self.current_clip_ratio) * batch_adv

                    # Dual-Clip PPO (Ye et al. 2020): lower bound for negative advantages
                    dual_clip_c = getattr(self.rl_config, 'dual_clip_c', 3.0)
                    clipped_obj = torch.min(surr1, surr2)
                    neg_adv_mask = (batch_adv < 0).float()
                    clipped_obj = torch.max(clipped_obj, dual_clip_c * batch_adv * neg_adv_mask + clipped_obj * (1 - neg_adv_mask))
                    actor_loss = -clipped_obj.mean()

                    # PopArt critic loss: normalize targets to PopArt scale
                    self.heads.critic_popart.update_stats(batch_ret.unsqueeze(-1))
                    normalized_ret = self.heads.critic_popart.normalize_targets(batch_ret.unsqueeze(-1)).squeeze(-1)

                    # OPTIMIZED: Reuse critic_features from forward_with_features (no redundant forward pass)
                    normalized_vpred = self.heads.critic_popart.normalized_forward(critic_features.float()).squeeze(-1)

                    # Value clipping in normalized space
                    old_values_norm = self.heads.critic_popart.normalize_targets(
                        batch['values'].squeeze(-1).unsqueeze(-1)
                    ).squeeze(-1)
                    clip_ratio = self.rl_config.clip_ratio
                    values_clipped = old_values_norm + torch.clamp(
                        normalized_vpred - old_values_norm, -clip_ratio, clip_ratio
                    )
                    critic_loss_unclipped = (normalized_vpred - normalized_ret) ** 2
                    critic_loss_clipped = (values_clipped - normalized_ret) ** 2
                    critic_loss = 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

                    # Lagrangian constraints — only penalize violations (clamp min=0)
                    avg_inv_norm = torch.abs(inv_norm_f).mean()
                    risk_violation = torch.clamp(avg_inv_norm - self.risk_target, min=0)

                    # Adverse selection constraint (M3ORL): penalize toxic fill ratio > target
                    toxic_fill_ratio = (self._toxic_fills / max(1, self._total_fills))
                    as_violation = torch.clamp(
                        torch.tensor(toxic_fill_ratio - self.toxic_fill_target, device=self.device),
                        min=0
                    )

                    risk_loss = self.lagrangian_risk.detach() * risk_violation
                    as_loss = self.lagrangian_as.detach() * as_violation

                    # NOTE: ent_loss disabled when using SAC auto-entropy tuning
                    # They have different targets and conflict with each other
                    if self.auto_entropy_tuning:
                        ent_loss = torch.tensor(0.0, device=self.device)
                    else:
                        ent_violation = torch.clamp(self.ent_target - entropy, min=0)
                        ent_loss = self.lagrangian_ent.detach() * ent_violation

                    # === SAC AUTOMATIC ENTROPY (Haarnoja 2018, Eq. 11) ===
                    #
                    # Objective: J(α) = E[α·H(π) - α·H_target] = α·(H - H_target)
                    # Gradient:  ∇_α J = H - H_target
                    #
                    # Gradient descent on log_alpha (for numerical stability):
                    #   log_α ← log_α - lr·∇J = log_α - lr·(H - H_target)
                    #
                    # If H < H_target: gradient < 0, log_α increases, α increases → more exploration ✓
                    # If H > H_target: gradient > 0, log_α decreases, α decreases → less exploration ✓
                    #
                    # CRITICAL FIX: Use actual Gaussian entropy, NOT -log_prob!
                    current_entropy = dist.entropy().sum(-1).mean()  # Sum over 9 action dims

                    if self.auto_entropy_tuning:
                        alpha = self.log_alpha.exp()
                        # Loss for gradient descent: minimize α·(H - H_target)
                        # When H < target: loss < 0, gradient makes log_α increase
                        # DETACH current_entropy: we only want gradient through alpha, not through dist
                        alpha_loss = alpha * (current_entropy.detach() - self.target_entropy)
                    else:
                        alpha = torch.tensor(self.entropy_coef).to(self.device)
                        alpha_loss = torch.tensor(0.0).to(self.device)

                    # Entropy term: add entropy bonus to encourage exploration
                    # Keep current_entropy attached here so gradient flows to policy
                    entropy_term = -alpha.detach() * current_entropy

                    # === ADAPTIVE KL PENALTY (Data-driven calibration) ===
                    # L = L_clip + beta * D_KL where beta is auto-calibrated
                    # PPO uses reverse KL: D_KL(π_new || π_old) to penalize large changes
                    with torch.no_grad():
                        # Reverse KL approximation: E_new[log(π_new/π_old)]
                        # = E_new[log_prob_new - log_prob_old]
                        # We want to MINIMIZE this, so add it to loss
                        kl_div = (new_log_probs - batch['log_probs']).mean()
                    # Use beta from adaptive controller (kl_div should be positive when policy changes)
                    kl_penalty = self.adaptive_kl_controller.beta * torch.abs(kl_div)

                    # Standard PPO loss weighting (Schulman 2017):
                    # L = L_actor + c1 * L_critic + c2 * L_entropy
                    # c1 = 0.5, c2 handled by alpha in SAC
                    critic_weight = 0.5
                    loss = (actor_loss + critic_weight * critic_loss + entropy_term
                            + risk_loss + ent_loss + as_loss + kl_penalty) / accum_steps

                    self.scaler.scale(loss).backward()

                    # Update alpha (SAC temperature)
                    if self.auto_entropy_tuning:
                        self.alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer.step()

                    # Step optimizer every accum_steps or at last batch
                    if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(batches):
                        self.scaler.unscale_(self.optimizer)

                        # === GRADIENT NOISE SCALE UPDATE ===
                        # Track gradient statistics for adaptive LR (before clipping)
                        all_params = list(self.backbone.parameters()) + list(self.heads.parameters())
                        self.grad_noise_scale.update(all_params, loss.item())

                        nn.utils.clip_grad_norm_(all_params, self.rl_config.max_grad_norm)

                        # === ADAPTIVE LEARNING RATE (McCandlish 2018) ===
                        if self.adaptive_lr and self.grad_noise_scale.is_calibrated:
                            lr_info = self.grad_noise_scale.compute_optimal_lr(self.rl_config.batch_size)
                            optimal_lr = lr_info['lr_optimal']
                            # Blend with scheduler: use min of scheduled and optimal
                            scheduled_lr = self.base_lr * self.scheduler.get_last_lr()[0] / self.base_lr
                            new_lr = min(optimal_lr, scheduled_lr * 2)  # Allow 2x boost max
                            new_lr = max(new_lr, self.base_lr * 0.01)  # Floor at 1% of base
                            for pg in self.optimizer.param_groups:
                                pg['lr'] = new_lr

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        optimizer_stepped = True

                    # KL diagnostic (computed after gradient step)
                    with torch.no_grad():
                        # Reverse KL: E[log(π_new/π_old)] - positive when policy changes
                        approx_kl = abs((new_log_probs - batch['log_probs']).mean().item())

                    # Record metrics BEFORE early stop check (so they're not lost on break)
                    epoch_actor_losses.append(actor_loss.item())
                    epoch_critic_losses.append(critic_loss.item())
                    epoch_entropies.append(entropy.item())

                    # === ADAPTIVE KL CONTROL (Data-driven) ===
                    # Update controller with observed KL at end of each batch
                    kl_params = self.adaptive_kl_controller.update(approx_kl, self.episodes)
                    kl_early_stop_threshold = kl_params['early_stop_threshold']

                    # Sync legacy attributes for logging/checkpointing
                    self.kl_beta = self.adaptive_kl_controller.beta
                    self.kl_target = self.adaptive_kl_controller.target_kl

                    # Early stopping if KL exceeds adaptive threshold
                    if abs(approx_kl) > kl_early_stop_threshold:
                        kl_early_stop = True
                        if kl_params.get('calibrating', False):
                            logger.info(f"KL early stop (calibrating): KL={approx_kl:.4f} > {kl_early_stop_threshold:.4f}")
                        else:
                            logger.warning(f"KL early stop: KL={approx_kl:.4f} > {kl_early_stop_threshold:.4f} "
                                          f"(target={self.kl_target:.4f}, beta={self.kl_beta:.1f})")
                        break

                    # Update Lagrangian multipliers — capped to [0.0001, 1.0]
                    with torch.no_grad():
                        risk_grad = (avg_inv_norm - self.risk_target).item()
                        new_risk = self.lagrangian_risk.item() + self.rl_config.lagrangian_lambda_lr * risk_grad
                        self.lagrangian_risk.copy_(torch.tensor(max(0.0001, min(1.0, new_risk))))

                        ent_grad = (self.ent_target - entropy).item()
                        new_ent = self.lagrangian_ent.item() + self.rl_config.lagrangian_lambda_lr * ent_grad
                        self.lagrangian_ent.copy_(torch.tensor(max(0.0001, min(1.0, new_ent))))

                        as_grad = as_violation.item()
                        new_as = self.lagrangian_as.item() + self.rl_config.lagrangian_lambda_lr * as_grad
                        self.lagrangian_as.copy_(torch.tensor(max(0.0001, min(1.0, new_as))))

                    # === AVELLANEDA-STOIKOV RISK AVERSION LEARNING ===
                    # Lagrangian dual: max_γ min_π E[R] - γ(Var[PnL] - ε)
                    # Update γ based on constraint violation
                    if self.risk_aversion_learnable and len(self._pnl_history) >= 100:
                        with torch.no_grad():
                            pnl_tensor = torch.tensor(list(self._pnl_history), dtype=torch.float32)
                            pnl_variance = pnl_tensor.var().item()
                            variance_violation = pnl_variance - self.target_pnl_variance
                            # Gradient ascent on γ (dual variable)
                            new_gamma = self.risk_aversion.item() + self.risk_aversion_lr * variance_violation
                            # Clamp to reasonable range [0.01, 10.0]
                            new_gamma = max(0.01, min(10.0, new_gamma))
                            self.risk_aversion.copy_(torch.tensor(new_gamma))

                if epoch_actor_losses:  # may be empty if KL stopped on first batch
                    epoch_metrics['actor_loss'].append(np.mean(epoch_actor_losses))
                    epoch_metrics['critic_loss'].append(np.mean(epoch_critic_losses))
                    epoch_metrics['entropy'].append(np.mean(epoch_entropies))
                    epoch_metrics['approx_kl'].append(approx_kl)
                self.train_steps += 1
            
            # Get adaptive KL stats for logging
            kl_stats = self.adaptive_kl_controller.get_stats()

            # Get gradient noise scale stats for adaptive LR
            gns_stats = self.grad_noise_scale.compute_optimal_lr(self.rl_config.batch_size)

            metrics = {
                'actor_loss': float(np.mean(epoch_metrics['actor_loss'])) if epoch_metrics['actor_loss'] else 0.0,
                'critic_loss': float(np.mean(epoch_metrics['critic_loss'])) if epoch_metrics['critic_loss'] else 0.0,
                'entropy': float(np.mean(epoch_metrics['entropy'])) if epoch_metrics['entropy'] else 0.0,
                'approx_kl': float(np.mean(epoch_metrics['approx_kl'])) if epoch_metrics['approx_kl'] else 0.0,
                # Data-driven training metrics
                'num_epochs_used': num_epochs,  # Adaptive epochs based on buffer size
                'buffer_size': buffer_size,
                # Adaptive KL metrics
                'kl_target': kl_stats.get('target_kl', self.kl_target),
                'kl_beta': kl_stats.get('beta', self.kl_beta),
                'kl_calibrated': 1.0 if kl_stats.get('is_calibrated', False) else 0.0,
                # Gradient Noise Scale metrics (McCandlish 2018)
                'lr_optimal': gns_stats.get('lr_optimal', self.base_lr),
                'b_noise': gns_stats.get('b_noise', 1.0),
                'grad_snr': gns_stats.get('signal_to_noise', 0.0),
                'lr_calibrated': 1.0 if gns_stats.get('calibrated', False) else 0.0,
                # Adaptive Coverage metrics (exploration→exploitation)
                'coverage': self.adaptive_coverage.current_coverage,
                'target_entropy': self.target_entropy,
                # Adaptive Clip Ratio metrics (Engstrom 2020)
                'clip_ratio': self.current_clip_ratio,
                # Data-driven batch size
                'batch_size_used': adaptive_batch,
            }

            # Increment episode counter for adaptive KL calibration
            self.episodes += 1

            # Step LR scheduler only if optimizer actually stepped (avoids PyTorch warning)
            if optimizer_stepped:
                self.scheduler.step()
                # Also decay alpha LR to prevent entropy collapse late in training
                if self._alpha_scheduler_enabled:
                    # Scale alpha LR proportionally to main LR
                    alpha_lr_scale = self.scheduler.get_last_lr()[0] / self.base_lr
                    for pg in self.alpha_optimizer.param_groups:
                        pg['lr'] = 3e-4 * max(alpha_lr_scale, 0.1)  # Floor at 10%

            # Restore original batch size
            self.memory.batch_size = original_batch_size
            self.memory.clear()
            return metrics

        except Exception as e:
            logger.error(f"Error in update: {e}")
            raise

    def state_dict(self):
        """Capture agent state for checkpointing."""
        state = {
            'backbone_state': self.backbone.state_dict(),
            'heads_state': self.heads.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
            'episodes': self.episodes,
            'lagrangian_risk': self.lagrangian_risk.item(),
            'lagrangian_ent': self.lagrangian_ent.item(),
            'entropy_coef': self.entropy_coef,
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
            'obs_rms_count': self.obs_rms.count,
            'reward_rms_mean': self.reward_rms.mean,
            'reward_rms_var': self.reward_rms.var,
            'reward_rms_count': self.reward_rms.count,
            'scaler_state': self.scaler.state_dict(),
            'lagrangian_as': self.lagrangian_as.item(),
            # New theoretically-grounded parameters
            'risk_aversion': self.risk_aversion.item(),
            'kl_beta': self.kl_beta,
            # Adaptive KL controller state
            'adaptive_kl_controller': self.adaptive_kl_controller.state_dict(),
            # Gradient Noise Scale state
            'grad_noise_scale': self.grad_noise_scale.state_dict(),
            # Adaptive Coverage state (data-driven entropy target)
            'adaptive_coverage': self.adaptive_coverage.state_dict(),
            # Adaptive Clip Ratio state (Engstrom 2020)
            'adaptive_clip': self.adaptive_clip.state_dict(),
            'current_clip_ratio': self.current_clip_ratio,
        }
        if self.auto_entropy_tuning:
            state['log_alpha'] = self.log_alpha.item()
            state['alpha_optimizer_state'] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load agent state from checkpoint."""
        try:
            self.backbone.load_state_dict(state_dict['backbone_state'])
            logger.info("Backbone state loaded")
        except Exception as e:
            logger.error(f"Error loading backbone: {e}")
            raise
        
        try:
            self.heads.load_state_dict(state_dict['heads_state'])
            logger.info("Heads state loaded")
        except Exception as e:
            logger.error(f"Error loading heads: {e}")
            raise
        
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        if 'scheduler_state' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.train_steps = state_dict.get('train_steps', 0)
        self.episodes = state_dict.get('episodes', 0)
        self.entropy_coef = state_dict.get('entropy_coef', 0.01)

        if 'lagrangian_risk' in state_dict:
            self.lagrangian_risk.data.copy_(torch.tensor(state_dict['lagrangian_risk']))
        if 'lagrangian_ent' in state_dict:
            self.lagrangian_ent.data.copy_(torch.tensor(state_dict['lagrangian_ent']))
        if 'lagrangian_as' in state_dict:
            self.lagrangian_as.data.copy_(torch.tensor(state_dict['lagrangian_as']))

        # Restore RunningMeanStd states
        if 'obs_rms_mean' in state_dict:
            self.obs_rms.mean = state_dict['obs_rms_mean']
            self.obs_rms.var = state_dict['obs_rms_var']
            self.obs_rms.count = state_dict['obs_rms_count']
        if 'reward_rms_mean' in state_dict:
            self.reward_rms.mean = state_dict['reward_rms_mean']
            self.reward_rms.var = state_dict['reward_rms_var']
            self.reward_rms.count = state_dict['reward_rms_count']
        if 'scaler_state' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler_state'])

        # Restore theoretically-grounded parameters
        if 'risk_aversion' in state_dict:
            self.risk_aversion.data.copy_(torch.tensor(state_dict['risk_aversion']))
        if 'kl_beta' in state_dict:
            self.kl_beta = state_dict['kl_beta']
        if 'log_alpha' in state_dict and self.auto_entropy_tuning:
            self.log_alpha.data.copy_(torch.tensor(state_dict['log_alpha']))
        if 'alpha_optimizer_state' in state_dict and self.auto_entropy_tuning:
            self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer_state'])

        # Restore adaptive KL controller state
        if 'adaptive_kl_controller' in state_dict:
            self.adaptive_kl_controller.load_state_dict(state_dict['adaptive_kl_controller'])
            self.kl_beta = self.adaptive_kl_controller.beta
            self.kl_target = self.adaptive_kl_controller.target_kl
            logger.info(f"AdaptiveKL restored: target={self.kl_target:.4f}, beta={self.kl_beta:.1f}, "
                       f"calibrated={self.adaptive_kl_controller.is_calibrated}")

        # Restore gradient noise scale state
        if 'grad_noise_scale' in state_dict:
            self.grad_noise_scale.load_state_dict(state_dict['grad_noise_scale'])
            if self.grad_noise_scale.is_calibrated:
                gns = self.grad_noise_scale.compute_optimal_lr(self.rl_config.batch_size)
                logger.info(f"GradientNoiseScale restored: lr_opt={gns['lr_optimal']:.2e}, "
                           f"B_noise={gns['b_noise']:.1f}")

        # Restore adaptive coverage state (data-driven entropy target)
        if 'adaptive_coverage' in state_dict:
            self.adaptive_coverage.load_state_dict(state_dict['adaptive_coverage'])
            self.target_entropy = self.adaptive_coverage.current_target_entropy
            logger.info(f"AdaptiveCoverage restored: coverage={self.adaptive_coverage.current_coverage:.2%}, "
                       f"H_target={self.target_entropy:.2f}")

        # Restore adaptive clip ratio state
        if 'adaptive_clip' in state_dict:
            self.adaptive_clip.load_state_dict(state_dict['adaptive_clip'])
            self.current_clip_ratio = state_dict.get('current_clip_ratio', 0.2)
            logger.info(f"AdaptiveClipRatio restored: clip={self.current_clip_ratio:.3f}")
