# market_maker/core/config.py
from dataclasses import dataclass
from typing import Tuple
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class MarketConfig:
    def __init__(self, symbol="XRPUSDC", min_qty=10.0, tick_size=0.00001, min_notional=10.0,
                 max_position=1000, update_interval=0.1, window_size=50, max_order_size=100.0):
        self.symbol: str = symbol
        self.min_qty: float = min_qty
        self.tick_size: float = tick_size
        self.min_notional: float = min_notional
        self.max_position: int = max_position
        # SOTA: Timebase Definitions
        self.update_interval: float = update_interval  # e.g., 0.1s (100ms)
        self.steps_per_second = 1.0 / self.update_interval
        self.steps_per_minute = self.steps_per_second * 60
        self.steps_per_hour = self.steps_per_minute * 60
        self.steps_per_day = self.steps_per_hour * 24
        self.steps_per_year_crypto = self.steps_per_day * 365
        self.steps_per_year_equity = self.steps_per_hour * 6.5 * 252 # standard equity hours

        self.window_size = window_size
        self.max_order_size = max_order_size
        self.fee_rate = 0.001 # 0.1% fee
        self.fractal_windows = [8, 34, 55, 144, 233, 610] # Multi-scale Fibonacci windows
        # iVPIN uses exponential decay — no bucket calibration needed
        self.stale_data_timeout = 5.0 # seconds
        self.record_db_path = "data/db/market_data.db"
        self.record_depth = 20
        self.record_batch_size = 1000
        
        # SOTA: Processor settings
        self.lob_scaling_factor = 100.0
        self.trade_history_maxlen = 2000
        logger.debug(f"MarketConfig initialized with symbol={symbol}, min_qty={min_qty}, tick_size={tick_size}, "
                     f"min_notional={min_notional}, max_position={max_position}, update_interval={update_interval}, "
                     f"steps_per_second={self.steps_per_second}")

@dataclass
class RLConfig:
    def __init__(self,
                 zeta=0.01,
                 gamma=0.99,
                 max_bias=0.05,
                 max_spread=0.1,
                 min_spread=0.001,
                 max_bid_adjustment=0.5, 
                 max_ask_adjustment=0.5, 
                 batch_size=64, 
                 learning_rate=3e-4,
                  max_steps: int = None, # If None, derived from episode_duration_minutes
                  episode_duration_minutes: float = 15.0, # Target 15 min episodes
                  execution_intensity=10.0, 
                  num_epochs=4,
                  clip_ratio=0.2,
                  target_kl=0.01, 
                  gae_lambda=0.95, 
                  max_grad_norm=0.5,
                  buffer_size=10000,
                  optimizer='adamw',  # 'adamw' (stable), 'muon' (Mamba), 'lion' (transformer)
                  weight_decay=0.1,  # Mamba paper uses 0.1
                  max_equity_exposure=0.50,  # 50% Equity Exposure Limit (only dynamic limit)
                  initial_cash=100.0):  # Starting capital
                  
        self.zeta = zeta
        self.gamma = gamma
        self.max_bias = max_bias
        self.max_spread = max_spread
        self.min_spread = min_spread
        self.max_bid_adjustment = max_bid_adjustment
        self.max_ask_adjustment = max_ask_adjustment
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # SOTA: Meaningful Horizon
        self.episode_duration_minutes = episode_duration_minutes
        if max_steps is None:
            # Default to 100ms interval if not specified via market_config context
            self.max_steps = int(self.episode_duration_minutes * 60 / 0.1)
        else:
            self.max_steps = max_steps
            
        self.execution_intensity = execution_intensity
        self.num_epochs = num_epochs
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.max_equity_exposure = max_equity_exposure
        self.initial_cash = initial_cash

        # === SOTA OPTIMIZER (Muon for Mamba) ===
        # Muon: Newton-Schulz orthogonalization, best for SSM (Mamba, S4)
        # Lion: Sign momentum, best for transformers
        # AdamW: What Mamba paper used (betas=0.9,0.95, wd=0.1)
        self.optimizer = optimizer  # 'muon', 'lion', 'sophia', 'adamw', 'adam'
        self.weight_decay = weight_decay  # Mamba paper uses 0.1
        self.use_torch_compile = True  # torch.compile for 20-40% speedup (PyTorch 2.0+)
        self.dsr_alpha = 0.05 # DSR moving average decay
        self.log_frequency = 100 # Steps between logging metrics
        self.retry_interval = 1.0 # Seconds between retries on error
        self.normalize_observations = True # SOTA Normalization
        self.inventory_penalty_scaling = "relative" # "quadratic" or "relative"
        self.dsr_scaling = 100.0 # Scaling factor for reward signal
        self.lagrangian_target_window = 32.0 # Target seq length
        self.lagrangian_lambda_lr = 1e-3 # Learning rate for multipliers
        self.lagrangian_initial_lambda = 0.001
        self.gradient_accumulation_steps = 4  # Effective batch = batch_size * accum_steps
        self.dual_clip_c = 3.0  # Dual-Clip PPO lower bound (Ye et al. 2020)
        
        # SOTA: Risk & Exploration Lagrangian Targets
        self.lagrangian_target_inventory = 0.4 # Target 40% of MaxPos (soft limit, allows trading)
        self.lagrangian_target_toxic_fill = 0.3 # Target: max 30% of fills during toxic flow (iVPIN > 0.7)

        # === THEORETICALLY GROUNDED PARAMETERS ===

        # === SAC AUTOMATIC ENTROPY (Haarnoja et al. 2018) ===
        #
        # THEORETICAL DERIVATION:
        # Theorem 1: The optimal temperature α satisfies the constraint:
        #   E_π[-log π(a|s)] = H* (target entropy)
        #
        # For continuous actions with Gaussian policy:
        #   H(π) = 0.5 * d * (1 + log(2π)) + Σ_i log(σ_i)
        #        = 0.5 * d * log(2πe) + Σ_i log(σ_i)
        #
        # For uniform exploration (σ = 1 for all dims covering [-1, 1]):
        #   H_max ≈ 0.5 * d * log(2πe) ≈ 1.42 * d
        #
        # Haarnoja recommends H* = -dim(A) as heuristic, but this is arbitrary.
        # Better: H* = fraction of max entropy, e.g., 50% of H_max
        #
        # For d = 9:
        #   H_max ≈ 12.8
        #   H* = 0.5 * H_max ≈ 6.4
        #
        # But SAC convention uses negative (minimizing -H), so:
        _action_dim = 9
        _h_max = 0.5 * _action_dim * math.log(2 * math.pi * math.e)
        self.target_entropy = -0.5 * _h_max  # 50% of max entropy
        self.auto_entropy_tuning = True
        self.log_alpha_init = 0.0  # α = 1 initially
        self.alpha_lr = 3e-4  # Same as actor (co-learning)

        # === ADAPTIVE KL CONTROLLER (Theoretically grounded) ===
        #
        # THEORETICAL DERIVATION:
        # For Gaussian π(a|s) = N(μ, σ²I), trust region constraint D_KL ≤ ε
        # where ε = target_kl.
        #
        # From TRPO (Schulman 2015), for small policy changes:
        #   D_KL ≈ 0.5 * d * [(Δμ/σ)² + 2*(Δσ/σ)²]
        #
        # If we want max δ = 10% change per dimension per update:
        #   target_kl = 0.5 * action_dim * δ²
        #
        # For action_dim = 9, δ = 0.1:
        #   target_kl = 0.5 * 9 * 0.01 = 0.045
        #
        self.adaptive_kl_penalty = True
        self.kl_calibration_episodes = 20  # Episodes for distribution estimation
        self.kl_target_percentile = 50.0  # Median of observed KL distribution

        # Beta bounds from Lagrangian duality: β* = 1/ε at constraint boundary
        # Allow β to vary by 10x in each direction
        # Beta bounds from Lagrangian duality: β* = 1/ε at constraint boundary
        # Allow β to vary by 10x in each direction
        _action_dim = 9
        _delta_per_dim = 0.1  # 10% change per dim per update
        _theoretical_target = 0.5 * _action_dim * (_delta_per_dim ** 2)
        
        self.kl_target = target_kl if target_kl is not None else _theoretical_target
        self.kl_beta_min = 1.0 / (10 * self.kl_target)  # Loose constraint
        self.kl_beta_max = 1.0 / (0.1 * self.kl_target)  # Tight constraint
        self.kl_beta_init = 1.0 / self.kl_target  # Optimal at boundary

        # Avellaneda-Stoikov Risk Aversion (2008)
        # gamma in the HJB solution: V = x + q*s - gamma*sigma^2*q^2*(T-t)
        # Learned via Lagrangian dual for variance constraint
        self.risk_aversion_learnable = True
        self.risk_aversion_init = 0.1  # Initial gamma
        self.risk_aversion_lr = 1e-3
        self.target_pnl_variance = 0.01  # Constraint: Var[PnL] <= epsilon

        # Guéant-Lehalle-Fernandez-Tapia (2013) fill probability
        # P(fill) = A * exp(-k * delta / sigma) where k depends on order flow
        # k is estimated from realized data, not arbitrary
        self.fill_model_use_realized = True

        # Legacy entropy target (for backward compat, but auto_entropy_tuning overrides)
        _d_ref = 5
        _sigma_floor = math.atanh(0.5) * (_d_ref / _action_dim) ** 0.25
        self.lagrangian_target_entropy = 0.5 * math.log(2 * math.pi * math.e) + math.log(_sigma_floor)
        logger.debug(f"RLConfig initialized with zeta={zeta}, gamma={gamma}, max_bias={max_bias}, "
                     f"max_spread={max_spread}, batch_size={batch_size}, learning_rate={learning_rate}, "
                     f"max_steps={max_steps}, execution_intensity={execution_intensity}, num_epochs={num_epochs}, "
                     f"max_equity_exposure={max_equity_exposure}, initial_cash={initial_cash}")

@dataclass
class ModelConfig:
    def __init__(self, input_shape=(50, 40), window_size=50, embedding_dim=192, num_heads=4, num_layers=3, dropout=0.1, n_features=40, n_aux_features=15):
        self.input_shape: Tuple[int, int] = input_shape  # (sequence_length, features_per_level)
        self.window_size: int = window_size
        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.num_layers: int = num_layers
        self.dropout: float = dropout
        self.n_features: int = n_features  # Nombre total de features par niveau du LOB
        self.n_aux_features: int = n_aux_features # Features auxiliaires (15 features from MarketFeatureProcessor)
        
        # SOTA: Mamba Specifics
        self.mamba_d_state = 16
        self.mamba_d_conv = 4
        self.mamba_expand = 2
        self.mamba_n_layers = num_layers
        
        # SOTA: Training
        self.learning_rate = 3e-4
        self.grad_clip_norm = 1.0
        self.ssl_forecast_dim = n_features # Match input dim for reconstruction
        logger.debug(f"ModelConfig initialized with input_shape={input_shape}, window_size={window_size}, "
                     f"embedding_dim={embedding_dim}, num_heads={num_heads}, num_layers={num_layers}, "
                     f"dropout={dropout}, n_features={n_features}, n_aux_features={n_aux_features}")

@dataclass
class SimulationConfig:
    """Configuration spécifique pour la simulation."""
    def __init__(self, min_latency=0.086, max_latency=0.700, maker_fee=0.001, taker_fee=0.001,
                 partial_fill_prob=0.3, min_fill_ratio=0.1, max_fill_ratio=0.9, order_cancel_prob=0.1,
                 price_tolerance=0.002, min_tick_size=0.00001, min_order_size=100.0, max_order_size=10000.0,
                 fill_lambda_decay=0.5,  # Fill probability decay rate (per basis point)
                 fee_curriculum=True):  # Enable curriculum learning on fees
        # Latence réseau
        self.min_latency: float = min_latency  # 86ms minimum
        self.max_latency: float = max_latency  # 700ms maximum
        
        # Frais de trading
        self.maker_fee: float = maker_fee  # 0.1% maker fee
        self.taker_fee: float = taker_fee  # 0.1% taker fee
        
        # Probabilités d'exécution et annulation
        self.partial_fill_prob: float = partial_fill_prob  # Probabilité de remplissage partiel
        self.min_fill_ratio: float = min_fill_ratio    # Remplissage minimum en cas de partial fill
        self.max_fill_ratio: float = max_fill_ratio    # Remplissage maximum en cas de partial fill
        self.order_cancel_prob: float = order_cancel_prob  # Probabilité d'annulation d'ordre
        
        # Contraintes de prix
        self.price_tolerance: float = price_tolerance  # 0.2% déviation max du mid price
        self.min_tick_size: float = min_tick_size  # Tick size minimum
        
        # Contraintes de volume
        self.min_order_size: float = min_order_size   # Taille minimum d'ordre
        self.max_order_size: float = max_order_size  # Taille maximum d'ordre
        
        # Fill probability parameters
        self.fill_lambda_decay: float = fill_lambda_decay  # Decay rate per basis point for fill prob

        # === FEE CURRICULUM LEARNING ===
        # Theoretical basis: Start with easy conditions (low fees) to learn basic strategy,
        # then gradually increase difficulty. This is standard curriculum learning.
        # XRP/USDC spread is ~2.5 bps, so we start with fees that allow profitability.
        self.fee_curriculum = fee_curriculum
        self.base_maker_fee = maker_fee
        self.base_taker_fee = taker_fee

        # Fee schedule: (episode_threshold, maker_fee, taker_fee)
        # Designed so spread (2.5 bps) > round-trip fees initially
        self.fee_schedule = [
            (0, 0.00005, 0.00005),      # 0-99:   0.5 bps each (1 bps RT) - easy
            (100, 0.0001, 0.0001),      # 100-199: 1 bps each (2 bps RT) - still profitable
            (200, 0.00015, 0.00015),    # 200-299: 1.5 bps each (3 bps RT) - challenging
            (300, 0.0002, 0.0002),      # 300-399: 2 bps each (4 bps RT) - hard
            (400, 0.0003, 0.0003),      # 400-499: 3 bps each (6 bps RT) - very hard
            (500, 0.0005, 0.0005),      # 500-699: 5 bps each (10 bps RT) - realistic VIP
            (700, 0.00075, 0.00075),    # 700-899: 7.5 bps each (15 bps RT)
            (900, 0.001, 0.001),        # 900+:   10 bps each (20 bps RT) - standard
        ]

        logger.debug(f"SimulationConfig initialized with min_latency={min_latency}, max_latency={max_latency}, "
                     f"maker_fee={maker_fee}, taker_fee={taker_fee}, partial_fill_prob={partial_fill_prob}, "
                     f"min_fill_ratio={min_fill_ratio}, max_fill_ratio={max_fill_ratio}, order_cancel_prob={order_cancel_prob}, "
                     f"price_tolerance={price_tolerance}, min_tick_size={min_tick_size}, "
                     f"min_order_size={min_order_size}, max_order_size={max_order_size}, fill_lambda_decay={fill_lambda_decay}, "
                     f"fee_curriculum={fee_curriculum}")

    def get_fees_for_episode(self, episode: int) -> tuple:
        """Get (maker_fee, taker_fee) for the given episode based on curriculum.

        Curriculum learning: Start with low fees to let agent learn profitable strategies,
        then gradually increase to realistic levels.

        Returns:
            (maker_fee, taker_fee) tuple
        """
        if not self.fee_curriculum:
            return (self.base_maker_fee, self.base_taker_fee)

        # Find the appropriate fee level for this episode
        current_fees = self.fee_schedule[0][1:3]  # Default to first level
        for threshold, maker, taker in self.fee_schedule:
            if episode >= threshold:
                current_fees = (maker, taker)
            else:
                break

        return current_fees

    def update_fees(self, episode: int):
        """Update current fees based on episode (for curriculum learning)."""
        maker, taker = self.get_fees_for_episode(episode)
        self.maker_fee = maker
        self.taker_fee = taker
        return maker, taker

@dataclass
class MonitoringConfig:
    def __init__(self, history_maxlen=10000, metrics_maxlen=100, update_interval=0.1):
        self.history_maxlen = history_maxlen
        self.metrics_maxlen = metrics_maxlen
        # Correct SOTA annualization for Crypto (24/7/365)
        self.steps_per_year = int((365 * 24 * 3600) / update_interval)
        self.min_position_value = 1e-8

@dataclass
class DashboardConfig:
    def __init__(self, port=8081, update_freq=0.5, tape_limit=50, large_trade_threshold=10000):
        self.port = port
        self.update_freq = update_freq
        self.tape_limit = tape_limit
        self.large_trade_threshold = large_trade_threshold
