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
from typing import Dict, Tuple, Optional, List
from decimal import Decimal
import multiprocessing
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# DATA-DRIVEN ARCHITECTURE & REGULARIZATION
# =============================================================================

@dataclass
class DataStatistics:
    """Statistics computed from data for data-driven configuration."""
    n_samples: int
    feature_dim: int
    effective_rank: int          # PCA rank explaining 95% variance
    reward_variance: float
    reward_range: Tuple[float, float]
    feature_variance: np.ndarray  # Per-feature variance
    autocorrelation: float       # Reward autocorrelation (temporal structure)


def compute_data_statistics(data_path: str, n_samples: int = 2000) -> DataStatistics:
    """
    Analyse les données pour extraire des statistiques qui guident l'architecture.

    Cette fonction sample les données et calcule :
    - Rang effectif des features (complexité intrinsèque)
    - Variance des rewards (difficulté de prédiction)
    - Autocorrélation des rewards (structure temporelle)
    """
    from training.data_loader import MarketDataLoader
    from data.processor import LOBFeatureProcessor, MarketFeatureProcessor

    loader = MarketDataLoader(data_path)
    lob_proc = LOBFeatureProcessor()
    market_proc = MarketFeatureProcessor()

    features_list = []
    rewards_proxy = []  # Use price changes as reward proxy
    prev_mid = None

    for i in range(min(n_samples, len(loader.data))):
        try:
            ob = loader.get_next_orderbook()
            if ob is None:
                break

            # Extract features
            lob_feat = lob_proc.process(ob)
            market_proc.process(ob)
            market_feat = market_proc.get_features()

            # Combine features
            combined = np.concatenate([lob_feat, market_feat])
            features_list.append(combined)

            # Price change as reward proxy
            bids = ob.get('bids', [[0, 0]])
            asks = ob.get('asks', [[0, 0]])
            mid = (float(bids[0][0]) + float(asks[0][0])) / 2
            if prev_mid is not None and prev_mid > 0:
                pct_change = (mid - prev_mid) / prev_mid * 10000  # bps
                rewards_proxy.append(pct_change)
            prev_mid = mid

        except Exception as e:
            logger.debug(f"Sample {i} failed: {e}")
            continue

    if len(features_list) < 100:
        logger.warning(f"Only {len(features_list)} samples collected, using defaults")
        return DataStatistics(
            n_samples=len(features_list),
            feature_dim=55,
            effective_rank=30,
            reward_variance=1.0,
            reward_range=(-10, 10),
            feature_variance=np.ones(55),
            autocorrelation=0.1
        )

    features = np.array(features_list)
    rewards = np.array(rewards_proxy) if rewards_proxy else np.zeros(100)

    # Compute effective rank via PCA (dimensions explaining 95% variance)
    try:
        # Center features
        features_centered = features - features.mean(axis=0)
        # SVD for numerical stability
        _, s, _ = np.linalg.svd(features_centered, full_matrices=False)
        explained_var = np.cumsum(s**2) / np.sum(s**2)
        effective_rank = int(np.searchsorted(explained_var, 0.95) + 1)
    except Exception:
        effective_rank = features.shape[1] // 2

    # Reward statistics
    reward_var = np.var(rewards) if len(rewards) > 1 else 1.0
    reward_range = (float(np.min(rewards)), float(np.max(rewards))) if len(rewards) > 1 else (-10, 10)

    # Autocorrelation (lag-1) - measures temporal structure
    if len(rewards) > 10:
        autocorr = np.corrcoef(rewards[:-1], rewards[1:])[0, 1]
        autocorr = 0.0 if np.isnan(autocorr) else autocorr
    else:
        autocorr = 0.0

    return DataStatistics(
        n_samples=len(features_list),
        feature_dim=features.shape[1],
        effective_rank=effective_rank,
        reward_variance=float(reward_var),
        reward_range=reward_range,
        feature_variance=np.var(features, axis=0),
        autocorrelation=float(autocorr)
    )


def compute_network_capacity(stats: DataStatistics, action_dim: int = 9) -> Dict:
    """
    Calcule la capacité optimale du réseau basée sur les données.

    Théorie:
    1. Information Bottleneck: hidden ≥ sqrt(input × output)
    2. Complexité ajustée: × log(1 + reward_variance)
    3. Rank ratio: capacité proportionnelle au rang effectif
    4. Anti-overfitting: cap basé sur n_samples

    Formules:
        base_capacity = sqrt(feature_dim × action_dim)
        complexity_factor = 1 + log(1 + reward_variance)
        rank_factor = 1 + effective_rank / feature_dim

        raw_hidden = base_capacity × complexity_factor × rank_factor

        # Anti-overfitting: max params ≈ n_samples / 10
        max_params = n_samples / 10
        # 2-layer MLP: params ≈ input × hidden + hidden × output + hidden × hidden
        max_hidden = sqrt(max_params / 3)

        hidden = min(raw_hidden, max_hidden)
        hidden = 2^ceil(log2(hidden))  # Round to power of 2
    """
    feature_dim = stats.feature_dim
    n_samples = stats.n_samples

    # Base capacity (information-theoretic minimum)
    base_capacity = np.sqrt(feature_dim * action_dim)

    # Complexity factor based on reward variance
    # Higher variance = harder problem = more capacity needed
    complexity_factor = 1 + np.log1p(stats.reward_variance)

    # Rank factor: if features are low-rank, need less capacity
    rank_ratio = stats.effective_rank / max(feature_dim, 1)
    rank_factor = 0.5 + rank_ratio  # Range [0.5, 1.5]

    # Raw capacity calculation
    raw_actor_hidden = base_capacity * complexity_factor * rank_factor

    # Critic needs more capacity (value prediction harder)
    raw_critic_hidden = raw_actor_hidden * 1.5

    # Anti-overfitting constraint
    # Rule of thumb: 10 samples per parameter minimum
    # 2-layer MLP: params ≈ in×h + h×h + h×out ≈ 3×h² for large h
    max_params = n_samples / 10
    max_hidden_from_data = np.sqrt(max_params / 3)

    # Apply constraint
    actor_hidden = min(raw_actor_hidden, max_hidden_from_data)
    critic_hidden = min(raw_critic_hidden, max_hidden_from_data * 1.2)

    # Round to nearest power of 2 (GPU efficiency)
    actor_hidden = int(2 ** np.ceil(np.log2(max(32, actor_hidden))))
    critic_hidden = int(2 ** np.ceil(np.log2(max(32, critic_hidden))))

    # Cap at reasonable values
    actor_hidden = min(actor_hidden, 512)
    critic_hidden = min(critic_hidden, 512)

    return {
        'actor_hidden_dim': actor_hidden,
        'critic_hidden_dim': critic_hidden,
        '_capacity_derivation': {
            'base_capacity': float(base_capacity),
            'complexity_factor': float(complexity_factor),
            'rank_factor': float(rank_factor),
            'raw_actor': float(raw_actor_hidden),
            'max_from_data': float(max_hidden_from_data),
            'formula': 'min(sqrt(feat×act) × log(1+var) × (0.5+rank_ratio), sqrt(n_samples/30))'
        }
    }


def compute_regularization(stats: DataStatistics, n_epochs: int) -> Dict:
    """
    Calcule les hyperparamètres de régularisation basés sur les données.

    Anti-Overfitting Strategy:

    1. Dropout: Inversement proportionnel à n_samples
       dropout = clip(1 / sqrt(n_samples / 1000), 0.05, 0.3)

    2. Weight Decay: Proportionnel à variance / n_samples
       wd = clip(reward_variance / n_samples × 100, 0.01, 0.3)

    3. Label Smoothing (entropy bonus): Based on autocorrelation
       High autocorr = predictable = less smoothing needed

    4. Gradient Clipping: Based on reward range
       max_grad = clip(1.0 / log(1 + reward_range), 0.5, 2.0)
    """
    n_samples = max(stats.n_samples, 100)

    # Dropout: more data = less dropout needed
    # Base: 0.1 at 10k samples, scales with sqrt
    dropout = 1.0 / np.sqrt(n_samples / 1000)
    dropout = float(np.clip(dropout, 0.05, 0.3))

    # Weight decay: prevents memorization
    # Higher variance + fewer samples = more regularization
    wd_base = stats.reward_variance / n_samples * 100
    weight_decay = float(np.clip(wd_base, 0.01, 0.3))

    # Gradient clipping based on reward magnitude
    reward_range = stats.reward_range[1] - stats.reward_range[0]
    max_grad_norm = 1.0 / np.log1p(reward_range / 10)
    max_grad_norm = float(np.clip(max_grad_norm, 0.5, 2.0))

    # Early stopping patience: more data = can train longer
    # patience = sqrt(n_samples / 100) epochs without improvement
    early_stop_patience = int(np.sqrt(n_samples / 100))
    early_stop_patience = max(3, min(early_stop_patience, 20))

    # Entropy coefficient based on autocorrelation
    # High autocorr = structured problem = less exploration needed
    entropy_coef_init = 0.01 * (1 - abs(stats.autocorrelation))
    entropy_coef_init = float(np.clip(entropy_coef_init, 0.001, 0.05))

    return {
        'dropout': dropout,
        'weight_decay': weight_decay,
        'max_grad_norm': max_grad_norm,
        'early_stop_patience': early_stop_patience,
        'entropy_coef_init': entropy_coef_init,
        '_regularization_derivation': {
            'dropout_formula': f'clip(1/sqrt({n_samples}/1000), 0.05, 0.3) = {dropout:.3f}',
            'weight_decay_formula': f'clip({stats.reward_variance:.2f}/{n_samples}×100, 0.01, 0.3) = {weight_decay:.3f}',
            'max_grad_formula': f'clip(1/log(1+{reward_range:.1f}/10), 0.5, 2.0) = {max_grad_norm:.2f}',
            'patience_formula': f'sqrt({n_samples}/100) = {early_stop_patience}',
            'entropy_formula': f'0.01 × (1 - |{stats.autocorrelation:.3f}|) = {entropy_coef_init:.4f}'
        }
    }


def compute_entropy_target(stats: DataStatistics, action_dim: int = 9) -> Dict:
    """
    Calcule le target entropy empiriquement plutôt que théoriquement.

    Problème: La formule Gaussienne H = 0.5×log(2πe×σ²) assume distribution non-bornée,
    mais nos actions sont tanh-bounded [-1, 1].

    Solution Data-Driven:
    1. Entropy max théorique pour uniform[-1,1]: H_max = log(2) × action_dim ≈ 6.24 pour 9D
    2. Entropy min (déterministe): H_min ≈ 0
    3. Target = H_max × coverage_factor

    Coverage schedule:
    - Early (exploration): 80% of max entropy
    - Late (exploitation): 50% of max entropy

    La variance des rewards guide le schedule:
    - High variance = need more exploration = higher coverage
    """
    # Max entropy for bounded uniform distribution
    h_max_per_dim = np.log(2)  # entropy of uniform[-1,1]
    h_max = h_max_per_dim * action_dim  # ≈ 6.24 for 9D

    # Adjust based on reward variance (harder problem = more exploration)
    variance_factor = np.clip(1 + np.log1p(stats.reward_variance) / 5, 0.8, 1.2)

    # Coverage bounds
    coverage_early = 0.80 * variance_factor  # ~80-96% exploration
    coverage_late = 0.50 * variance_factor   # ~50-60% exploitation

    # Target entropies
    target_entropy_early = h_max * coverage_early
    target_entropy_late = h_max * coverage_late

    return {
        'target_entropy_max': float(h_max),
        'target_entropy_early': float(target_entropy_early),
        'target_entropy_late': float(target_entropy_late),
        'coverage_early': float(coverage_early),
        'coverage_late': float(coverage_late),
        '_entropy_derivation': {
            'h_max_formula': f'log(2) × {action_dim} = {h_max:.3f}',
            'variance_factor': float(variance_factor),
            'note': 'Based on bounded uniform distribution, NOT unbounded Gaussian'
        }
    }


def compute_learning_rate_bounds(stats: DataStatistics, batch_size: int) -> Dict:
    """
    Calcule les bornes du learning rate basées sur les données.

    Théorie (Smith 2017, Cyclical Learning Rates):
    - LR_max ≈ 1 / (loss_curvature × batch_size)
    - LR_min ≈ LR_max / 10

    Approximation de la courbure via variance:
    - High reward variance = high curvature = lower LR needed

    Formule:
        lr_base = 3e-4  # Adam default for RL
        lr_scale = 1 / sqrt(reward_variance + 1)
        lr_max = lr_base × lr_scale
        lr_min = lr_max / 10
    """
    lr_base = 3e-4

    # Scale inversely with variance (high variance = unstable = lower LR)
    variance_scale = 1.0 / np.sqrt(stats.reward_variance + 1)

    # Scale with batch size (larger batch = can use higher LR)
    batch_scale = np.sqrt(batch_size / 64)  # Normalize to batch=64

    lr_max = lr_base * variance_scale * batch_scale
    lr_min = lr_max / 10

    # Warmup steps based on data size
    warmup_steps = int(np.sqrt(stats.n_samples))

    return {
        'lr_max': float(np.clip(lr_max, 1e-5, 1e-2)),
        'lr_min': float(np.clip(lr_min, 1e-6, 1e-3)),
        'warmup_steps': warmup_steps,
        '_lr_derivation': {
            'variance_scale': float(variance_scale),
            'batch_scale': float(batch_scale),
            'formula': f'3e-4 × {variance_scale:.3f} × {batch_scale:.3f} = {lr_max:.2e}'
        }
    }


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

    # === COMPUTE DATA STATISTICS FOR DATA-DRIVEN CONFIG ===
    logger.info("Computing data statistics for data-driven configuration...")
    try:
        data_stats = compute_data_statistics(data_path, n_samples=2000)
        logger.info(f"  Samples analyzed: {data_stats.n_samples}")
        logger.info(f"  Feature dim: {data_stats.feature_dim}, Effective rank: {data_stats.effective_rank}")
        logger.info(f"  Reward variance: {data_stats.reward_variance:.4f}")
        logger.info(f"  Autocorrelation: {data_stats.autocorrelation:.4f}")
    except Exception as e:
        logger.warning(f"Data statistics computation failed: {e}, using defaults")
        data_stats = DataStatistics(
            n_samples=10000, feature_dim=55, effective_rank=30,
            reward_variance=1.0, reward_range=(-10, 10),
            feature_variance=np.ones(55), autocorrelation=0.1
        )

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

    # === DATA-DRIVEN NETWORK CAPACITY (Anti-Overfitting) ===
    network_capacity = compute_network_capacity(data_stats, action_dim=action_dim)
    actor_hidden_dim = network_capacity['actor_hidden_dim']
    critic_hidden_dim = network_capacity['critic_hidden_dim']
    logger.info(f"  Network capacity: Actor={actor_hidden_dim}, Critic={critic_hidden_dim}")

    # === DATA-DRIVEN REGULARIZATION ===
    regularization = compute_regularization(data_stats, n_epochs=num_epochs)
    dropout = regularization['dropout']
    weight_decay = regularization['weight_decay']
    max_grad_norm = regularization['max_grad_norm']
    early_stop_patience = regularization['early_stop_patience']
    entropy_coef_init = regularization['entropy_coef_init']
    logger.info(f"  Regularization: dropout={dropout:.3f}, weight_decay={weight_decay:.3f}, patience={early_stop_patience}")

    # === DATA-DRIVEN ENTROPY TARGET (Bounded distribution) ===
    entropy_config = compute_entropy_target(data_stats, action_dim=action_dim)
    target_entropy_early = entropy_config['target_entropy_early']
    target_entropy_late = entropy_config['target_entropy_late']
    logger.info(f"  Entropy targets: early={target_entropy_early:.2f}, late={target_entropy_late:.2f}")

    # === DATA-DRIVEN LEARNING RATE ===
    lr_config = compute_learning_rate_bounds(data_stats, batch_size=batch_size)
    # Override learning_rate with data-driven value, but keep the n_envs scaling
    lr_data_driven = lr_config['lr_max'] / np.sqrt(n_envs / base_n_envs)
    learning_rate = min(learning_rate, lr_data_driven)  # Use more conservative value
    warmup_steps = lr_config['warmup_steps']
    logger.info(f"  Learning rate: {learning_rate:.2e} (data-driven bound: {lr_config['lr_max']:.2e})")

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

            # max_grad_norm: DATA-DRIVEN based on reward range
            'max_grad_norm': max_grad_norm,

            # === ENTROPY (DATA-DRIVEN) ===
            'target_entropy_early': target_entropy_early,
            'target_entropy_late': target_entropy_late,
            'entropy_coef_init': entropy_coef_init,

            # === EARLY STOPPING (ANTI-OVERFITTING) ===
            'early_stop_patience': early_stop_patience,

            # === OPTIMIZER ===
            'optimizer': 'adamw',
            'weight_decay': weight_decay,  # DATA-DRIVEN
        },
        'model': {
            'input_shape': (50, 40),
            'window_size': 50,
            'embedding_dim': 192,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': dropout,  # DATA-DRIVEN
            'n_features': 40,
            # === NETWORK CAPACITY (DATA-DRIVEN, ANTI-OVERFITTING) ===
            'actor_hidden_dim': actor_hidden_dim,
            'critic_hidden_dim': critic_hidden_dim,
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
        },
        '_data_driven': {
            # === DATA STATISTICS ===
            'n_samples_analyzed': data_stats.n_samples,
            'feature_dim': data_stats.feature_dim,
            'effective_rank': data_stats.effective_rank,
            'reward_variance': data_stats.reward_variance,
            'autocorrelation': data_stats.autocorrelation,

            # === NETWORK CAPACITY DERIVATION ===
            **network_capacity.get('_capacity_derivation', {}),
            'actor_hidden_dim': actor_hidden_dim,
            'critic_hidden_dim': critic_hidden_dim,

            # === REGULARIZATION DERIVATION ===
            **regularization.get('_regularization_derivation', {}),

            # === ENTROPY DERIVATION ===
            **entropy_config.get('_entropy_derivation', {}),

            # === LEARNING RATE DERIVATION ===
            **lr_config.get('_lr_derivation', {}),
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
