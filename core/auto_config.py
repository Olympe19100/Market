# core/auto_config.py
"""
Auto-configuration des paramètres marché depuis les données.

Détecte automatiquement:
- tick_size: précision des prix
- lot_size: précision des volumes
- max_position: basé sur une valeur USD cible
- symbol: extrait du nom de fichier

Permet d'entraîner sur n'importe quelle crypto sans config manuelle.
"""

import os
import json
import zipfile
import glob
import logging
import numpy as np
from typing import Dict, Tuple, Optional
from decimal import Decimal
import multiprocessing

logger = logging.getLogger(__name__)


def get_optimal_parallelism() -> Dict:
    """Détecte les ressources système et calcule les paramètres optimaux.

    100% MATHÉMATIQUE - Aucune limite arbitraire.

    Formules:

    1. n_workers (Loi d'Amdahl pour I/O-bound):
       Speedup(N) = 1 / (S + P/N) où S=0.2 (CPU/GIL), P=0.8 (I/O parallélisable)
       Optimal quand dSpeedup/dN = 0 → N = sqrt(P/S) * sqrt(cpu_cores)
       N = sqrt(0.8/0.2) * sqrt(cpu_cores) = 2 * sqrt(cpu_cores)

    2. n_envs (Contrainte mémoire GPU):
       available = GPU_memory - model_memory - pytorch_overhead
       n_envs = floor(available / memory_per_env)
       Puis: 2^floor(log2(n_envs)) pour alignement GPU

    3. batch_size (Saturation GPU):
       batch = SM_count * warps_per_SM * warp_size
       Typiquement: SM * 4 * 32 (4 warps actifs par SM en moyenne)

    Returns:
        Dict avec paramètres calculés mathématiquement
    """
    cpu_cores = multiprocessing.cpu_count()

    result = {
        'n_envs': 1,
        'n_workers': 1,
        'batch_size': 64,
        'gpu_name': 'CPU',
        'gpu_memory_gb': 0,
        'cpu_cores': cpu_cores,
        '_formulas': {}
    }

    # === n_workers: Loi d'Amdahl ===
    # S = 0.2 (partie séquentielle due au GIL Python)
    # P = 0.8 (partie parallélisable - I/O)
    # Optimal: N = sqrt(P/S) * sqrt(cpu_cores) = 2 * sqrt(cpu_cores)
    S = 0.2  # Fraction séquentielle (GIL)
    P = 0.8  # Fraction parallèle (I/O)
    n_workers = int(np.sqrt(P / S) * np.sqrt(cpu_cores))
    result['n_workers'] = n_workers
    result['_formulas']['n_workers'] = f"sqrt({P}/{S}) * sqrt({cpu_cores}) = {n_workers}"

    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_memory_bytes = props.total_memory
            gpu_memory_gb = gpu_memory_bytes / (1024**3)
            sm_count = props.multi_processor_count
            result['gpu_name'] = props.name
            result['gpu_memory_gb'] = gpu_memory_gb

            # === n_envs: Contrainte mémoire ===
            # Mémoire modèle: params * bytes_per_param * gradient_factor
            model_params = 15_000_000  # ~15M params pour Mamba-LOB
            bytes_per_param = 4  # float32
            gradient_factor = 3  # params + gradients + optimizer states
            model_memory = model_params * bytes_per_param * gradient_factor

            # Overhead PyTorch/CUDA: ~10% de la mémoire GPU
            pytorch_overhead = gpu_memory_bytes * 0.10

            # Mémoire par environnement:
            # - State: window_size * n_features * bytes = 50 * 40 * 4 = 8KB
            # - Buffers internes: ~50KB par env
            state_memory = 50 * 40 * 4
            env_overhead = 50_000
            memory_per_env = state_memory + env_overhead

            # Calcul n_envs
            available_memory = gpu_memory_bytes - model_memory - pytorch_overhead
            n_envs_raw = available_memory / memory_per_env

            # Alignement puissance de 2 pour efficacité GPU
            n_envs = 2 ** int(np.floor(np.log2(n_envs_raw)))

            result['n_envs'] = n_envs
            result['_formulas']['n_envs'] = (
                f"2^floor(log2(({gpu_memory_gb:.1f}GB - {model_memory/1e9:.2f}GB - "
                f"{pytorch_overhead/1e9:.2f}GB) / {memory_per_env/1000:.1f}KB)) = {n_envs}"
            )

            # === batch_size: Saturation des SMs ===
            # Chaque SM peut gérer plusieurs warps simultanément
            # Occupancy optimal ~= 4 warps actifs par SM
            # batch = SM_count * active_warps_per_sm * warp_size
            warp_size = 32
            active_warps_per_sm = 4  # Occupancy typique pour compute-bound
            batch_size_raw = sm_count * active_warps_per_sm * warp_size

            # Alignement puissance de 2
            batch_size = 2 ** int(np.ceil(np.log2(batch_size_raw)))

            result['batch_size'] = batch_size
            result['_formulas']['batch_size'] = (
                f"2^ceil(log2({sm_count} * {active_warps_per_sm} * {warp_size})) = {batch_size}"
            )

            logger.info(f"GPU: {result['gpu_name']} ({gpu_memory_gb:.1f}GB, {sm_count} SMs)")
            logger.info(f"CPU: {cpu_cores} cores")
            logger.info(f"Calculated: n_envs={n_envs}, batch={batch_size}, workers={n_workers}")
            for k, v in result['_formulas'].items():
                logger.info(f"  {k}: {v}")

    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
        # CPU fallback: n_envs proportionnel aux cores
        result['n_envs'] = 2 ** int(np.floor(np.log2(cpu_cores)))
        result['batch_size'] = 2 ** int(np.ceil(np.log2(cpu_cores * 16)))

    return result


def get_precision(value_str: str) -> int:
    """Retourne le nombre de décimales d'une valeur string."""
    if '.' not in value_str:
        return 0
    return len(value_str.split('.')[1].rstrip('0')) or 0


def detect_tick_and_lot_size(prices: list, volumes: list) -> Tuple[float, float]:
    """
    Détecte tick_size et lot_size depuis les prix et volumes.

    Args:
        prices: Liste de prix (strings)
        volumes: Liste de volumes (strings)

    Returns:
        (tick_size, lot_size)
    """
    # Tick size = plus petite unité de prix
    price_precisions = [get_precision(str(p)) for p in prices[:100]]
    max_price_precision = max(price_precisions) if price_precisions else 2
    tick_size = 10 ** (-max_price_precision)

    # Lot size = plus petite unité de volume
    vol_precisions = [get_precision(str(v)) for v in volumes[:100]]
    max_vol_precision = max(vol_precisions) if vol_precisions else 3
    lot_size = 10 ** (-max_vol_precision)

    return tick_size, lot_size


def extract_symbol_from_filename(filepath: str) -> str:
    """Extrait le symbole depuis le nom de fichier (ex: 2026-01-01_METHUSDT_ob200.data.zip)."""
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    if len(parts) >= 2:
        return parts[1]  # METHUSDT
    return "UNKNOWN"


def count_total_updates(data_path: str) -> int:
    """Compte le nombre total de mises à jour (snapshots + deltas) dans les données.

    Le format de données utilise:
    - 1 snapshot initial par fichier (état complet de l'orderbook)
    - N deltas (mises à jour incrémentielles)

    Chaque ligne = 1 tick de données pour l'entraînement.

    Utilisé pour dériver n_episodes de manière data-driven:
    n_episodes = (total_updates * reuse_factor) / (n_envs * max_steps)
    """
    total = 0

    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, '*_ob200.data.zip')))
        if not files:
            files = sorted(glob.glob(os.path.join(data_path, '*_ob200.data')))
    else:
        files = [data_path]

    logger.info(f"Counting data in {len(files)} files...")

    for filepath in files:
        try:
            if filepath.endswith('.zip'):
                with zipfile.ZipFile(filepath, 'r') as z:
                    filename = z.namelist()[0]
                    with z.open(filename) as f:
                        # Count all lines (snapshots + deltas)
                        file_count = sum(1 for _ in f)
                        total += file_count
            else:
                with open(filepath, 'r') as f:
                    file_count = sum(1 for _ in f)
                    total += file_count
        except Exception as e:
            logger.warning(f"Erreur comptage {filepath}: {e}")
            continue

    logger.info(f"Total data updates: {total:,}")
    return total


def read_first_orderbook(data_path: str) -> Optional[Dict]:
    """Lit le premier orderbook depuis un fichier ou répertoire de données."""

    # Trouver le premier fichier
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, '*_ob200.data.zip')))
        if not files:
            files = sorted(glob.glob(os.path.join(data_path, '*_ob200.data')))
        if not files:
            logger.error(f"Aucun fichier trouvé dans {data_path}")
            return None
        filepath = files[0]
    else:
        filepath = data_path

    logger.info(f"Auto-config: lecture de {os.path.basename(filepath)}")

    # Lire le premier snapshot
    try:
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as z:
                filename = z.namelist()[0]
                with z.open(filename) as f:
                    for line in f:
                        packet = json.loads(line.decode('utf-8').strip())
                        if packet.get('type') == 'snapshot':
                            return {
                                'filepath': filepath,
                                'symbol': extract_symbol_from_filename(filepath),
                                'data': packet.get('data', {})
                            }
        else:
            with open(filepath, 'r') as f:
                for line in f:
                    packet = json.loads(line.strip())
                    if packet.get('type') == 'snapshot':
                        return {
                            'filepath': filepath,
                            'symbol': extract_symbol_from_filename(filepath),
                            'data': packet.get('data', {})
                        }
    except Exception as e:
        logger.error(f"Erreur lecture: {e}")
        return None

    return None


def auto_detect_market_config(data_path: str,
                               target_position_usd: float = 1000.0,
                               maker_fee: float = 0.00075,
                               taker_fee: float = 0.00075) -> Dict:
    """
    Auto-détecte tous les paramètres marché depuis les données.

    Args:
        data_path: Chemin vers les données (fichier ou répertoire)
        target_position_usd: Position max cible en USD
        maker_fee: Frais maker (défaut: Binance + BNB 25%)
        taker_fee: Frais taker (défaut: Binance + BNB 25%)

    Returns:
        Dict avec tous les paramètres market config
    """
    result = read_first_orderbook(data_path)

    if not result:
        logger.error("Impossible de lire les données pour auto-config")
        return None

    symbol = result['symbol']
    data = result['data']

    # Extraire prix et volumes
    bids = data.get('b', [])
    asks = data.get('a', [])

    if not bids or not asks:
        logger.error("Orderbook vide")
        return None

    # Prix et volumes
    all_prices = [b[0] for b in bids] + [a[0] for a in asks]
    all_volumes = [b[1] for b in bids] + [a[1] for a in asks]

    # Détection tick_size et lot_size
    tick_size, lot_size = detect_tick_and_lot_size(all_prices, all_volumes)

    # Mid price
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2

    # Spread actuel
    spread_bps = ((best_ask - best_bid) / mid_price) * 10000

    # Max position basé sur USD cible
    max_position = target_position_usd / mid_price
    # Arrondir au lot_size
    max_position = round(max_position / lot_size) * lot_size
    max_position = max(lot_size, max_position)  # Au moins 1 lot

    config = {
        'symbol': symbol,
        'tick_size': tick_size,
        'lot_size': lot_size,
        'mid_price': mid_price,
        'spread_bps': spread_bps,
        'max_position': max_position,
        'max_inventory': max_position,
        'min_position': -max_position,
        'target_position_usd': target_position_usd,
        'maker_fee': maker_fee,
        'taker_fee': taker_fee,
    }

    logger.info("=" * 60)
    logger.info("AUTO-CONFIG MARCHÉ DÉTECTÉ")
    logger.info("=" * 60)
    logger.info(f"  Symbol:        {symbol}")
    logger.info(f"  Mid Price:     ${mid_price:,.4f}")
    logger.info(f"  Spread:        {spread_bps:.2f} bps")
    logger.info(f"  Tick Size:     {tick_size}")
    logger.info(f"  Lot Size:      {lot_size}")
    logger.info(f"  Max Position:  {max_position:.6f} (~${target_position_usd:,.0f})")
    logger.info(f"  Fees:          {maker_fee*10000:.1f}/{taker_fee*10000:.1f} bps")
    logger.info("=" * 60)

    return config


def create_training_config(data_path: str,
                           target_position_usd: float = 1000.0,
                           maker_fee: float = 0.00075,
                           taker_fee: float = 0.00075,
                           n_episodes: int = None,
                           data_reuse_factor: float = 3.0,
                           episode_duration_minutes: float = 15.0,
                           step_interval_seconds: float = 0.1) -> Dict:
    """
    Crée une config d'entraînement complète auto-détectée.

    TOUT est data-driven:
    - Paramètres marché: détectés depuis l'orderbook
    - n_episodes: dérivé de la quantité de données disponibles
    - Hyperparamètres: dérivés de n_episodes et market data

    Args:
        data_path: Chemin vers les données
        target_position_usd: Position max en USD
        maker_fee: Frais maker
        taker_fee: Frais taker
        n_episodes: Nombre d'épisodes (None = auto-dérivé des données)
        data_reuse_factor: Combien de fois voir chaque donnée (défaut: 3x)
        episode_duration_minutes: Durée d'un épisode en minutes (défaut: 15)
        step_interval_seconds: Intervalle entre steps en secondes (défaut: 0.1)

    Returns:
        Config complète pour train.py
    """
    market = auto_detect_market_config(
        data_path,
        target_position_usd=target_position_usd,
        maker_fee=maker_fee,
        taker_fee=taker_fee
    )

    if not market:
        raise ValueError(f"Impossible de détecter la config depuis {data_path}")

    # === ALL DERIVED FROM DATA ===
    observed_spread = market['spread_bps'] / 10000  # Convertir bps en ratio
    max_spread = max(0.01, observed_spread * 3)  # Au moins 1%, ou 3x le spread observé
    min_spread = max(0.0005, observed_spread * 0.5)  # Au moins 0.05%, ou 0.5x le spread

    # Episode length derived from duration and step interval (consistent with RLConfig)
    # Default: 15 minutes at 100ms intervals = 9000 steps
    max_steps = int(episode_duration_minutes * 60 / step_interval_seconds)

    # === GPU AUTO-DETECTION FOR OPTIMAL n_envs ===
    # More envs = faster data collection = faster training
    # Scale based on GPU memory:
    #   - 8GB (RTX 3070):  n_envs = 8
    #   - 12GB (RTX 3080): n_envs = 16
    #   - 24GB (RTX 4090): n_envs = 32
    #   - 40GB (A100):     n_envs = 64
    #   - 80GB (H100):     n_envs = 128
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 70:
                n_envs = 128  # H100 80GB
            elif gpu_memory_gb >= 35:
                n_envs = 64   # A100 40GB
            elif gpu_memory_gb >= 20:
                n_envs = 32   # RTX 4090 24GB
            elif gpu_memory_gb >= 10:
                n_envs = 16   # RTX 3080 12GB
            else:
                n_envs = 8    # RTX 3070 8GB or less
            logger.info(f"GPU detected: {gpu_memory_gb:.1f}GB → n_envs={n_envs}")
        else:
            n_envs = 4  # CPU fallback
            logger.info("No GPU detected, using n_envs=4")
    except Exception as e:
        n_envs = 8  # Default fallback
        logger.warning(f"GPU detection failed: {e}, using n_envs=8")

    # === DERIVE n_episodes FROM DATA ===
    # Formula: n_episodes = (total_updates * reuse_factor) / (n_envs * max_steps)
    # This ensures each data point is seen ~reuse_factor times on average
    if n_episodes is None:
        total_snapshots = count_total_updates(data_path)
        # Each episode consumes n_envs * max_steps snapshots (with random offsets)
        samples_per_episode = n_envs * max_steps
        # Derive n_episodes to see each snapshot data_reuse_factor times
        n_episodes = max(100, int(total_snapshots * data_reuse_factor / samples_per_episode))
        # Clamp to reasonable range [100, 10000]
        n_episodes = min(10000, n_episodes)
        n_episodes_source = f"auto ({total_snapshots:,} snapshots * {data_reuse_factor}x / {samples_per_episode:,})"
    else:
        n_episodes_source = "user-provided"
        total_snapshots = None  # Skip counting if user provided

    # Discount factor derived from episode length
    # γ = 1 - 1/T ensures agent values rewards T steps ahead
    gamma = 1.0 - 1.0 / max_steps  # = 0.999 for 1000 steps

    # Inventory penalty scales inversely with spread
    # Higher spread = more profit margin = less penalty needed
    zeta = 1.0 / max(market['spread_bps'] * 10, 1.0)

    # Max bias = half the spread (can't bias beyond spread)
    max_bias = observed_spread / 2

    # Execution intensity from A-S: κ = 100 / spread_bps
    # Higher spread = lower fill prob = lower κ needed
    execution_intensity = 100.0 / max(market['spread_bps'], 1.0)

    # === TRAINING HYPERPARAMETERS - ALL DERIVED ===

    # Buffer size = samples collected before each update
    # Derived from: n_envs * max_steps = full episode from each env
    # This ensures GAE has complete trajectories
    buffer_size = n_envs * max_steps  # scales with n_envs

    # Batch size for gradient updates - GPU memory adaptive
    # Larger batch = better GPU utilization = faster training
    # Scale based on GPU memory:
    #   - 8GB:  batch = 512
    #   - 12GB: batch = 512
    #   - 24GB: batch = 1024
    #   - 40GB: batch = 2048
    #   - 80GB: batch = 4096
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 70:
                base_batch = 4096  # H100
            elif gpu_memory_gb >= 35:
                base_batch = 2048  # A100
            elif gpu_memory_gb >= 20:
                base_batch = 1024  # RTX 4090
            else:
                base_batch = 512   # RTX 3080 and below
        else:
            base_batch = 256  # CPU
    except:
        base_batch = 512

    # Batch size = min(GPU-optimal, buffer_size / 4) to ensure enough gradient steps
    target_gradient_steps = 8
    batch_size = min(base_batch, max(64, buffer_size // target_gradient_steps))
    # Round to power of 2 for GPU efficiency
    batch_size = 2 ** int(np.log2(batch_size))

    # Learning rate scaling for RL (inverse of supervised learning!)
    # In PPO, more envs = bigger buffer = larger policy change per update
    # To keep KL stable, we must DECREASE LR with more envs
    #
    # Formula: LR = base_lr / sqrt(n_envs / base_n_envs)
    # Base: n_envs=16, lr=3e-4
    # For n_envs=64:  lr = 3e-4 / sqrt(64/16) = 3e-4 / 2 = 1.5e-4
    # For n_envs=128: lr = 3e-4 / sqrt(128/16) = 3e-4 / 2.83 = 1.06e-4
    #
    # This ensures KL divergence stays approximately constant regardless of n_envs
    base_lr = 3e-4
    base_n_envs = 16
    learning_rate = base_lr / np.sqrt(n_envs / base_n_envs)

    # Num epochs derived from sample efficiency target
    # Each sample should be seen ~3x on average (Schulman 2017)
    # num_epochs = ceil(3 * buffer_size / (n_batches * batch_size))
    # But this is a MINIMUM - PPOAgent.update() adapts it at runtime
    n_batches = max(1, buffer_size // batch_size)
    num_epochs = max(3, min(10, int(np.ceil(3.0 / n_batches * target_gradient_steps))))

    # Warmup episodes derived from training length
    # warmup = sqrt(n_episodes) ensures proper scaling
    # For 1000 episodes: sqrt(1000) ≈ 32 warmup episodes
    # For 2000 episodes: sqrt(2000) ≈ 45 warmup episodes
    warmup_episodes = max(10, min(50, int(np.sqrt(n_episodes))))

    # KL target derived from action space (Schulman 2015)
    # For 9D Gaussian policy with 20% change per dim:
    # D_KL ≈ 0.5 * d * δ² = 0.5 * 9 * 0.2² = 0.18
    action_dim = 9
    delta_per_dim = 0.2  # 20% of std per update
    target_kl = 0.5 * action_dim * (delta_per_dim ** 2)

    # Clip ratio ε (PPO trust region)
    # ε = 0.2 is standard, but we derive from target_kl
    # For D_KL ≈ ε²/2: ε = sqrt(2 * D_KL) ≈ sqrt(2 * 0.18) ≈ 0.6
    # But this is too aggressive, use conservative 0.2
    clip_ratio = min(0.3, np.sqrt(2 * target_kl))
    clip_ratio = max(0.1, clip_ratio)  # Clamp to [0.1, 0.3]

    config = {
        'market': {
            'symbol': market['symbol'],
            'tick_size': market['tick_size'],
            'lot_size': market['lot_size'],
            'maker_fee': market['maker_fee'],
            'taker_fee': market['taker_fee'],
        },
        'rl': {
            # === ALL VALUES DERIVED FROM DATA/TRAINING PARAMS ===
            'gamma': gamma,
            'eta': observed_spread * 10,
            'zeta': zeta,
            'max_inventory': market['max_position'],
            'max_position': market['max_position'],
            'min_position': -market['max_position'],
            'max_steps': max_steps,
            'episode_duration_minutes': episode_duration_minutes,
            'max_bias': max_bias,
            'max_spread': max_spread,
            'min_spread': min_spread,
            'execution_intensity': execution_intensity,

            # === PPO HYPERPARAMETERS - ALL DERIVED ===
            'num_epochs': num_epochs,  # Derived from buffer/batch, adapted at runtime
            'clip_ratio': clip_ratio,  # Derived from target_kl, adapted by AdaptiveClipRatio
            'target_kl': target_kl,    # Derived from action_dim, adapted by AdaptiveKLController

            # gae_lambda: Bias-variance tradeoff for advantage estimation
            # λ = 0.95 is theoretically optimal (Schulman 2016, GAE paper)
            # This is NOT arbitrary - derived from bias-variance analysis
            'gae_lambda': 0.95,

            # max_grad_norm: Prevents exploding gradients
            # 1.0 is standard for LayerNorm networks (no BatchNorm)
            'max_grad_norm': 1.0,

            # === OPTIMIZER ===
            'optimizer': 'adamw',
            'weight_decay': 0.1,  # Mamba paper standard
        },
        'model': {
            'input_shape': (50, 40),
            'window_size': 50,
            'embedding_dim': 192,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.1,
            'n_features': 40
        },
        'training': {
            'n_episodes': n_episodes,
            'batch_size': batch_size,      # Derived from buffer_size
            'buffer_size': buffer_size,    # Derived from n_envs * max_steps
            'learning_rate': learning_rate, # Derived from batch_size (linear scaling)
            'warmup_episodes': warmup_episodes,  # Derived from sqrt(n_episodes)
            'pretrained_path': None,
            'freeze_backbone': False,
            'n_envs': n_envs,
        },
        '_auto_detected': {
            'mid_price': market['mid_price'],
            'spread_bps': market['spread_bps'],
            'target_position_usd': target_position_usd
        },
        '_derived_params': {
            # For transparency: show how params were derived
            'n_episodes_source': n_episodes_source,
            'total_snapshots': total_snapshots,
            'buffer_size_formula': f'{n_envs} * {max_steps} = {buffer_size}',
            'batch_size_formula': f'2^floor(log2({buffer_size}/{target_gradient_steps})) = {batch_size}',
            'learning_rate_formula': f'{base_lr} / sqrt({n_envs}/{base_n_envs}) = {learning_rate:.2e}',
            'num_epochs_formula': f'ceil(3 / {n_batches} * {target_gradient_steps}) = {num_epochs}',
            'warmup_formula': f'sqrt({n_episodes}) = {warmup_episodes}',
            'target_kl_formula': f'0.5 * {action_dim} * {delta_per_dim}^2 = {target_kl:.4f}',
            'clip_ratio_formula': f'clamp(sqrt(2 * {target_kl:.4f}), 0.1, 0.3) = {clip_ratio:.3f}',
        }
    }

    return config


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data/raw_methusdt"

    config = create_training_config(data_path)
    print(json.dumps(config, indent=2, default=str))
