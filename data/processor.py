# processor.py

import numpy as np
import pickle
import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
from core.config import MarketConfig, ModelConfig

# Configuration du logging pour le suivi des processus de traitement
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LOBFeatureProcessor:
    """Stateless processor for LOB (Limit Order Book) feature extraction."""
    def __init__(self, market_config: Optional[MarketConfig] = None):
        self.market_config = market_config
    
    def get_attn_lob_features(self, orderbook: Dict) -> List[float]:
        """Extract 40-dim Attn-LOB features using SOTA Invariant Normalization.
        
        Features: bid_prices(10) + bid_volumes(10) + ask_prices(10) + ask_volumes(10)
        Note: spread and imbalance removed as per user request.
        """
        try:
            bids = np.array(orderbook['bids'][:10], dtype=float)
            asks = np.array(orderbook['asks'][:10], dtype=float)
            if len(bids) < 10 or len(asks) < 10:
                return [0.0] * 40
            
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid = (best_bid + best_ask) / 2
            if mid == 0:
                return [0.0] * 40
            
            scale_price = self.market_config.lob_scaling_factor if self.market_config else 100.0
            bp = ((bids[:, 0] - mid) / mid) * scale_price
            ap = ((asks[:, 0] - mid) / mid) * scale_price
            bv = np.log1p(bids[:, 1])
            av = np.log1p(asks[:, 1])
            
            # Spread and imbalance removed
            return np.concatenate([bp, bv, ap, av]).tolist()
        except Exception as e:
            logger.error(f"Error extracting LOB features: {e}")
            return [0.0] * 40

    def process(self, orderbook: Dict) -> np.ndarray:
        """Standard process method for LOB features."""
        return np.array(self.get_attn_lob_features(orderbook))


class MarketFeatureProcessor:
    def __init__(self, market_config: Optional[MarketConfig] = None, **kwargs):
        self.market_config = market_config
        # Note: We use SOTA Invariant Normalization directly in get_attn_lob_features()
        # No external scaler needed
        self.scaler = None  # Deprecated - kept for compatibility

        self.price_history = []
        self.orderbook_history = []
        # Max lengths to prevent memory leaks over long training runs
        # Hurst needs max 610 prices (largest window), OFI needs max 233 orderbooks
        self._max_price_history = 1000
        self._max_orderbook_history = 500

        trade_len = market_config.trade_history_maxlen if market_config else 2000

        self.trade_history = deque(maxlen=trade_len)

        # SOTA: Use the stateless LOB processor internally
        self.lob_processor = LOBFeatureProcessor(market_config)

        # Windows from config (Fibonacci sequence)
        self.windows = [8, 34, 55, 144, 233, 610]

        # Feature Index Mapping (for sim_env coherence)
        # iVPIN replaces VPIN — same indices, same dimension
        self.FEAT_NOISE = 0
        self.FEAT_IVPIN_FAST = 1
        self.FEAT_IVPIN_SLOW = 2
        # Backward compat aliases
        self.FEAT_VPIN_FAST = 1
        self.FEAT_VPIN_SLOW = 2
        self.FEAT_OFI_MID = 3
        self.FEAT_OFI_SLOW = 4
        self.FEAT_TFI_FAST = 5
        self.FEAT_TFI_SLOW = 6
        self.FEAT_SLOPE = 7
        self.FEAT_ENTROPY = 8
        self.FEAT_ILLIQUIDITY = 9
        self.FEAT_HURST_FAST = 10
        self.FEAT_HURST_MID = 11
        self.FEAT_HURST_SLOW = 12
        self.FEAT_SPREAD_BPS = 13     # Current spread in basis points
        self.FEAT_WARMUP = 14         # 0 during warm-up, 1 when trade-based features are live
        
        # Indicators
        self._mid_price = None
        self._volatility = None
        self._rsi = {10: None, 60: None, 300: None}
        self._osi = {10: None, 60: None, 300: None}
        
        # RSI helpers
        self.avg_gain = {10: None, 60: None, 300: None}
        self.avg_loss = {10: None, 60: None, 300: None}
        
        logger.debug("MarketFeatureProcessor initialized (iVPIN mode)")

    def get_attn_lob_features(self, orderbook: Dict) -> List[float]:
        """Proxy to central LOBFeatureProcessor."""
        return self.lob_processor.get_attn_lob_features(orderbook)

    def reset(self):
        """Reset all temporal state between episodes to prevent cross-episode leakage."""
        self.price_history.clear()
        self.orderbook_history.clear()
        self.trade_history.clear()
        self._mid_price = None
        self._volatility = None
        self.avg_gain = {k: None for k in self.avg_gain}
        self.avg_loss = {k: None for k in self.avg_loss}
        self._rsi = {k: None for k in self._rsi}
        self._osi = {k: None for k in self._osi}
        logger.debug("MarketFeatureProcessor reset for new episode")

    def update_orderbook(self, orderbook: Dict, timestamp: Optional[datetime] = None):
        """Met à jour le carnet d'ordres, calcule le mid price. Utilise un timestamp externe si fourni."""
        # Mise à jour du carnet d'ordres et ajout de mid_price à l'historique des prix
        self._mid_price = self._calculate_mid_price(orderbook)
        self.price_history.append(self._mid_price)
        
        # Store orderbook with its timestamp
        ts = timestamp if timestamp else datetime.now()
        self.orderbook_history.append({'data': orderbook, 'ts': ts})

        # Trim to prevent unbounded memory growth
        if len(self.price_history) > self._max_price_history:
            self.price_history = self.price_history[-self._max_price_history:]
        if len(self.orderbook_history) > self._max_orderbook_history:
            self.orderbook_history = self.orderbook_history[-self._max_orderbook_history:]

        logger.debug(f"Mid-price calculated and added to history: {self._mid_price}")

        # Calcul de la volatilité ("Noise")
        if len(self.price_history) > 20:
             self._volatility = self.get_microstructure_noise()

    @property
    def mid_price(self) -> float:
        if self._mid_price is None:
            raise ValueError("Le mid_price n'est pas encore calculé.")
        return self._mid_price

    @property
    def volatility(self) -> float:
        if self._volatility is None:
            logger.warning("Volatilité non calculée, valeur par défaut de 0.0001 utilisée.")
            return 0.0001
        return self._volatility

    def get_rsi(self, window_size: int) -> float:
        if self._rsi[window_size] is None:
            logger.warning(f"RSI pour la fenêtre {window_size} non calculé, valeur par défaut de 50.0 utilisée.")
            return 50.0
        return self._rsi[window_size]

    def get_osi(self, window_size: int) -> float:
        if self._osi[window_size] is None:
            logger.warning(f"OSI pour la fenêtre {window_size} non calculé, valeur par défaut de 0.0 utilisée.")
            return 0.0
        return self._osi[window_size]

    def _calculate_mid_price(self, orderbook: Dict) -> float:
        """Calcule le mid price à partir du premier niveau de prix bid et ask."""
        return (float(orderbook['bids'][0][0]) + float(orderbook['asks'][0][0])) / 2

    def _calculate_rv(self, prices: List[float]) -> float:
        """Calcule la volatilité réalisée (RV) sur une liste de prix."""
        returns = np.diff(np.log(prices))
        if len(returns) == 0:
            logger.warning("No returns available for RV calculation.")
            return 0.0001  # Valeur par défaut
        rv = np.sqrt(np.sum(returns**2) * (252 * 24 * 60 * 60 / len(returns)))
        logger.debug(f"Realized volatility (RV) calculated: {rv}")
        return rv

    def _calculate_rsi(self, window_size: int) -> float:
        """Calcule le RSI en utilisant une moyenne exponentielle pour une fenêtre donnée."""
        changes = np.diff(self.price_history[-(window_size + 1):])
        gains = np.maximum(changes, 0)
        losses = -np.minimum(changes, 0)

        if self.avg_gain[window_size] is None:
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
        else:
            avg_gain = (self.avg_gain[window_size] * (window_size - 1) + gains[-1]) / window_size
            avg_loss = (self.avg_loss[window_size] * (window_size - 1) + losses[-1]) / window_size

        self.avg_gain[window_size] = avg_gain
        self.avg_loss[window_size] = avg_loss

        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi

    def _calculate_osi(self, orderbooks: List[Dict]) -> float:
        """Calcule l'Order Strength Index (OSI) basé sur le volume des ordres bid et ask."""
        total_bid_volume = sum(sum(float(level[1]) for level in ob['bids'][:10]) for ob in orderbooks)
        total_ask_volume = sum(sum(float(level[1]) for level in ob['asks'][:10]) for ob in orderbooks)

        total_volume = total_bid_volume + total_ask_volume
        osi = (total_bid_volume - total_ask_volume) / total_volume if total_volume != 0 else 0.0
        return osi
    
    def process(self, orderbook: Dict, timestamp: Optional[datetime] = None) -> np.ndarray:
        """
        Retourne le vecteur de Microstructure Features (SOTA Fractal Scaling).
        Returns 15-dim vector.
        """
        self.update_orderbook(orderbook, timestamp=timestamp)

        # Multi-Scale iVPIN (intensity-based, replaces bucket VPIN)
        ivpin_fast = self.get_ivpin(self.windows[0], timestamp)  # 8s half-life
        ivpin_slow = self.get_ivpin(self.windows[2], timestamp)  # 55s half-life

        # Multi-Scale Hurst (Mid/Slow/Ext from windows)
        h_fast = self.get_hurst_exponent(self.windows[2])
        h_mid = self.get_hurst_exponent(self.windows[4])
        h_slow = self.get_hurst_exponent(self.windows[5])

        # Trade-based features
        trade_entropy = self.get_trade_entropy(current_timestamp=timestamp)
        amihud = self.get_amihud_illiquidity(current_timestamp=timestamp)
        tfi_fast = self.get_tfi(self.windows[1], current_timestamp=timestamp)
        tfi_slow = self.get_tfi(self.windows[3], current_timestamp=timestamp)

        # Spread in basis points — critical for MM pricing decisions
        best_bid = float(orderbook['bids'][0][0])
        best_ask = float(orderbook['asks'][0][0])
        mid = (best_bid + best_ask) / 2
        spread_bps = ((best_ask - best_bid) / mid) * 10000.0 if mid > 0 else 0.0

        # Warm-up flag: 1.0 when trade-based features (entropy, illiquidity, TFI, iVPIN)
        # have real data, 0.0 when they're returning zero due to insufficient history
        has_trades = len(self.trade_history) >= 10
        warmup = 1.0 if has_trades else 0.0

        # Raw features
        noise = self.get_microstructure_noise()
        ofi_mid = self.get_ofi(self.windows[2])
        ofi_slow = self.get_ofi(self.windows[4])
        slope = self.get_orderbook_slope()

        # === SCALE-INVARIANT NORMALIZATION ===
        # Each feature normalized to roughly [-5, 5] or [0, 1] range
        # This enables transfer learning across assets with different prices/volumes

        features = [
            np.clip(noise / 10.0, -5, 5),          # 0: FEAT_NOISE (bps/10 -> ~[-5,5])
            ivpin_fast,                            # 1: FEAT_IVPIN_FAST (already [0,1])
            ivpin_slow,                            # 2: FEAT_IVPIN_SLOW (already [0,1])
            np.clip(ofi_mid / 2.0, -5, 5),         # 3: FEAT_OFI_MID (normalized, /2 -> ~[-5,5])
            np.clip(ofi_slow / 2.0, -5, 5),        # 4: FEAT_OFI_SLOW (normalized, /2 -> ~[-5,5])
            tfi_fast,                              # 5: FEAT_TFI_FAST (already [-1,1])
            tfi_slow,                              # 6: FEAT_TFI_SLOW (already [-1,1])
            np.clip(slope / 10.0, 0, 5),           # 7: FEAT_SLOPE (normalized, /10 -> ~[0,5])
            np.clip(trade_entropy / 2.0, 0, 5),    # 8: FEAT_ENTROPY (/2 -> ~[0,5])
            np.clip(amihud / 20.0, 0, 5),          # 9: FEAT_ILLIQUIDITY (bps/vol, /20 -> ~[0,5])
            h_fast,                                # 10: FEAT_HURST_FAST (already [0,1])
            h_mid,                                 # 11: FEAT_HURST_MID (already [0,1])
            h_slow,                                # 12: FEAT_HURST_SLOW (already [0,1])
            np.clip(spread_bps / 50.0, 0, 5),      # 13: FEAT_SPREAD_BPS (/50 -> ~[0,5] for most assets)
            warmup,                                # 14: FEAT_WARMUP (0 or 1)
        ]
        return np.array(features)
        
    def update_trades(self, trade, timestamp: Optional[datetime] = None):
        """Ingest a new trade for microstructure calculations. Utilise un timestamp externe si fourni."""
        try:
            try:
                price = float(trade.price)
                qty = float(trade.quantity)
                is_buyer_maker = trade.is_buyer_maker 
            except AttributeError:
                price = float(trade['p'])
                qty = float(trade['q'])
                is_buyer_maker = trade['m']

            direction = -1 if is_buyer_maker else 1 
            ts = timestamp if timestamp else datetime.now()
            
            # Calculate Effective Spread (Cost)
            eff_spread = 2 * abs(price - self._mid_price) if self._mid_price else 0.0

            self.trade_history.append({
                'timestamp': ts,
                'signed_volume': direction * qty,
                'volume': qty,
                'direction': direction,
                'price': price,
                'eff_spread': eff_spread
            })

        except Exception as e:
            logger.error(f"Error in update_trades: {e}")

    def get_ivpin(self, half_life_seconds: float, current_timestamp: Optional[datetime] = None) -> float:
        """
        Intensity-based VPIN (iVPIN).

        Uses exponential decay on trade arrival intensity instead of volume buckets.
        Each trade counts as 1 event regardless of size, weighted by recency.

        iVPIN = |λ_buy - λ_sell| / (λ_buy + λ_sell)

        Returns value in [0, 1]: 0 = balanced flow, 1 = fully directional (toxic).
        Returns 0.5 (neutral prior) when insufficient data.
        """
        if len(self.trade_history) < 5:
            return 0.5

        now = current_timestamp if current_timestamp else datetime.now()
        alpha = np.log(2) / half_life_seconds
        cutoff_seconds = 5.0 * half_life_seconds

        lambda_buy = 0.0
        lambda_sell = 0.0

        for trade in reversed(self.trade_history):
            dt = (now - trade['timestamp']).total_seconds()
            if dt > cutoff_seconds:
                break
            if dt < 0:
                continue
            weight = np.exp(-alpha * dt)
            if trade['direction'] == 1:
                lambda_buy += weight
            else:
                lambda_sell += weight

        total = lambda_buy + lambda_sell
        if total < 1e-9:
            return 0.5

        return abs(lambda_buy - lambda_sell) / total

    def get_hurst_exponent(self, window: int = 100) -> float:
        """
        Simplified R/S analysis for Hurst Exponent.
        H > 0.5: Trending (Persistent)
        H < 0.5: Mean-Reverting (Anti-persistent)
        H = 0.5: Random Walk
        """
        if len(self.price_history) < window:
            return 0.5
            
        prices = np.array(self.price_history[-window:])
        returns = np.diff(np.log(prices))
        
        if len(returns) < 10:
            return 0.5
            
        # Simplified Hurst: Log(R/S) / Log(T)
        # We use a single scale window instead of full regression for CPU efficiency
        # This is a proxy for the actual exponent.
        R = np.max(np.cumsum(returns - np.mean(returns))) - np.min(np.cumsum(returns - np.mean(returns)))
        S = np.std(returns)
        
        if S == 0 or R == 0:
            return 0.5
            
        hurst = np.log(R / S) / np.log(len(returns))
        return np.clip(hurst, 0.0, 1.0)

    def get_tfi(self, window_seconds: int, current_timestamp: Optional[datetime] = None) -> float:
        """Trade Flow Imbalance: (BuyVol - SellVol) / TotalVol over time window."""
        now = current_timestamp if current_timestamp else datetime.now()
        relevant_trades = [t for t in self.trade_history if (now - t['timestamp']).total_seconds() <= window_seconds]
        
        if not relevant_trades:
            return 0.0
            
        buy_vol = sum(t['volume'] for t in relevant_trades if t['direction'] == 1)
        sell_vol = sum(t['volume'] for t in relevant_trades if t['direction'] == -1)
        total_vol = buy_vol + sell_vol
        
        return (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

    def get_ofi(self, window_steps: int) -> float:
        """Order Flow Imbalance over the last N steps.

        NORMALIZED: Returns OFI as a ratio of average BBO volume, giving
        scale-invariant values typically in [-10, 10] range across all assets.
        """
        if len(self.orderbook_history) < window_steps + 1:
            return 0.0

        ofi_sum = 0.0
        total_bbo_volume = 0.0

        # Calculate OFI for the window
        for i in range(-window_steps, 0):
            ob_t = self.orderbook_history[i]['data']
            ob_prev = self.orderbook_history[i-1]['data']

            # Best Bid
            bid_t = float(ob_t['bids'][0][0])
            bid_q_t = float(ob_t['bids'][0][1])
            bid_prev = float(ob_prev['bids'][0][0])
            bid_q_prev = float(ob_prev['bids'][0][1])

            e_bid = 0.0
            if bid_t > bid_prev:
                e_bid = bid_q_t
            elif bid_t < bid_prev:
                e_bid = -bid_q_prev
            else:
                e_bid = bid_q_t - bid_q_prev

            # Best Ask
            ask_t = float(ob_t['asks'][0][0])
            ask_q_t = float(ob_t['asks'][0][1])
            ask_prev = float(ob_prev['asks'][0][0])
            ask_q_prev = float(ob_prev['asks'][0][1])

            e_ask = 0.0
            if ask_t > ask_prev:
                e_ask = ask_q_prev
            elif ask_t < ask_prev:
                e_ask = -ask_q_t
            else:
                e_ask = ask_q_t - ask_q_prev

            # OFI = e_bid - e_ask
            ofi_sum += (e_bid - e_ask)

            # Track average BBO volume for normalization
            total_bbo_volume += (bid_q_t + ask_q_t)

        # Normalize by average BBO volume to get scale-invariant metric
        avg_bbo_volume = total_bbo_volume / window_steps if window_steps > 0 else 1.0
        if avg_bbo_volume < 1e-9:
            return 0.0

        return ofi_sum / avg_bbo_volume

    def get_amihud_illiquidity(self, window_seconds: int = 60, current_timestamp: Optional[datetime] = None) -> float:
        """
        Amihud Illiquidity Proxy: Average of |Return| / Volume.

        NORMALIZED: Returns in basis points per unit of normalized volume.
        We normalize volume by the median trade size in the window, giving
        scale-invariant values typically in [0, 100] range across all assets.
        """
        now = current_timestamp if current_timestamp else datetime.now()
        relevant_trades = [t for t in self.trade_history if (now - t['timestamp']).total_seconds() <= window_seconds]

        if len(relevant_trades) < 2:
            return 0.0

        # Get median volume for normalization (robust to outliers)
        volumes = [t['volume'] for t in relevant_trades if t['volume'] > 0]
        if not volumes:
            return 0.0
        median_vol = np.median(volumes)
        if median_vol < 1e-12:
            median_vol = 1.0

        returns_vol_ratio = []
        for i in range(1, len(relevant_trades)):
            # Return in basis points
            ret_bps = abs(np.log(relevant_trades[i]['price'] / relevant_trades[i-1]['price'])) * 10000
            # Volume normalized by median
            vol_norm = relevant_trades[i]['volume'] / median_vol
            if vol_norm > 0:
                returns_vol_ratio.append(ret_bps / vol_norm)

        return np.mean(returns_vol_ratio) if returns_vol_ratio else 0.0

    def get_orderbook_slope(self) -> float:
        """Calculates the slope of liquidity.

        NORMALIZED: Returns slope as volume per basis point of depth,
        normalized by BBO volume. This gives scale-invariant values
        typically in [0, 100] range across all assets.
        """
        if not self.orderbook_history:
            return 0.0

        ob = self.orderbook_history[-1]['data']

        # Mid price for bps calculation
        best_bid = float(ob['bids'][0][0])
        best_ask = float(ob['asks'][0][0])
        mid = (best_bid + best_ask) / 2
        if mid <= 0:
            return 0.0

        # BBO volume for normalization
        bbo_vol = float(ob['bids'][0][1]) + float(ob['asks'][0][1])
        if bbo_vol < 1e-9:
            bbo_vol = 1.0

        # Bid Slope: volume per bps of depth
        deep_bid = float(ob['bids'][-1][0])
        vol_bid = sum(float(x[1]) for x in ob['bids'])
        depth_bid_bps = abs(best_bid - deep_bid) / mid * 10000  # depth in bps
        bid_slope = (vol_bid / bbo_vol) / max(depth_bid_bps, 0.1)  # normalized slope

        # Ask Slope: volume per bps of depth
        deep_ask = float(ob['asks'][-1][0])
        vol_ask = sum(float(x[1]) for x in ob['asks'])
        depth_ask_bps = abs(deep_ask - best_ask) / mid * 10000  # depth in bps
        ask_slope = (vol_ask / bbo_vol) / max(depth_ask_bps, 0.1)  # normalized slope

        # Return average slope (now in normalized units)
        return (bid_slope + ask_slope) / 2

    def get_trade_entropy(self, window_seconds: int = 60, current_timestamp: Optional[datetime] = None) -> float:
        """Shannon Entropy of trade sizes."""
        now = current_timestamp if current_timestamp else datetime.now()
        relevant_trades = [t for t in self.trade_history if (now - t['timestamp']).total_seconds() <= window_seconds]
        
        if not relevant_trades:
            return 0.0
            
        sizes = [t['volume'] for t in relevant_trades]
        total_vol = sum(sizes)
        if total_vol == 0: return 0.0
        
        # Normalize to probabilities
        probs = [s / total_vol for s in sizes]
        import math
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        return entropy

    def get_microstructure_noise(self, window_size: int = 20) -> float:
        """
        Estimates microstructure noise using the volatility of price changes.
        Proxy: Standard deviation of mid-price changes in basis points.
        """
        if len(self.price_history) < window_size:
            return 0.0
        
        prices = np.array(self.price_history[-window_size:])
        # Use simple returns for noise if prices are close (BPS calculation)
        returns = np.diff(prices) / prices[:-1]
        
        return np.std(returns) * 10000

    def get_effective_spread(self, window_seconds: int = 60, current_timestamp: Optional[datetime] = None) -> float:
        """Average Effective Spread over the last N seconds."""
        now = current_timestamp if current_timestamp else datetime.now()
        relevant_trades = [t for t in self.trade_history if (now - t['timestamp']).total_seconds() <= window_seconds]
        
        if not relevant_trades:
            return 0.0
            
        spreads = []
        for t in relevant_trades:
            if isinstance(t, dict):
                spreads.append(t.get('eff_spread', 0.0))
            else:
                # If it's an object, try attribute access
                spreads.append(getattr(t, 'eff_spread', 0.0))
        return np.mean(spreads) if spreads else 0.0

    def get_order_imbalance(self, depth: int = 5) -> float:
        """OBI = (BidVol - AskVol) / (BidVol + AskVol)"""
        if not self.orderbook_history:
            return 0.0
            
        ob = self.orderbook_history[-1]['data']
        
        bid_vol = sum(float(x[1]) for x in ob['bids'][:depth])
        ask_vol = sum(float(x[1]) for x in ob['asks'][:depth])
        
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0



