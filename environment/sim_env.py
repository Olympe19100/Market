import asyncio
import json
import traceback
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time
from environment.base_environment import BaseMarketMakerEnv, MarketMakerState
from core.config import SimulationConfig, RLConfig, MarketConfig
from data.processor import LOBFeatureProcessor, MarketFeatureProcessor
import logging
from training.data_loader import MarketDataLoader
import pandas as pd
import vectorbt as vbt

# Configuration du logger pour le suivi de l'environnement de simulation
logger = logging.getLogger(__name__)
        
class SimulationMarketMakerEnv(BaseMarketMakerEnv):
    """
    Environnement de simulation avec contraintes réalistes:
    - Latence réseau
    - Frais de trading
    - Remplissages partiels
    - Annulations d'ordres
    - Rejet d'ordres
    """
    # Shared processors cache (class-level) - avoids creating 128 copies
    _shared_lob_processor = None
    _shared_market_processor = None

    def __init__(self, rl_config: RLConfig, market_config: MarketConfig, sim_config: SimulationConfig, data_path: str, split: str = None):
        # Use shared processors (created once, reused by all envs)
        if SimulationMarketMakerEnv._shared_lob_processor is None:
            SimulationMarketMakerEnv._shared_lob_processor = LOBFeatureProcessor(market_config=market_config)
            SimulationMarketMakerEnv._shared_market_processor = MarketFeatureProcessor(market_config=market_config)

        lob_processor = SimulationMarketMakerEnv._shared_lob_processor
        market_processor = SimulationMarketMakerEnv._shared_market_processor

        super().__init__(
            rl_config=rl_config,
            market_config=market_config,
            lob_processor=lob_processor,
            market_processor=market_processor
        )
        self.data_loader = MarketDataLoader(data_path, split=split)
        self.sim_config = sim_config
        self._current_ts = None


        # Buffers pour le calcul des métriques
        self.price_history = []
        self.volume_history = []
        self.execution_times = []

        # Portfolio value tracking for PnL reward
        initial_cash = getattr(rl_config, 'initial_cash', 100.0)
        self.last_portfolio_value = initial_cash

        # Fill tracking for observation (last step fill indicators)
        self.last_bid_fill = 0.0
        self.last_ask_fill = 0.0

        # Adverse selection (M3ORL) — λ_as synced from PPO Lagrangian
        self._lambda_as = 0.0
        self._last_ivpin_fast = 0.0
        self._last_fill_qty = 0.0
        self._last_quotes_levels = []  # For ER_t: track all quote levels

        # Avellaneda-Stoikov risk aversion γ (learned by PPOAgent)
        self._gamma_risk = 0.1  # Default, will be synced from agent

        # Non-quotation penalty scale (Guéant-Lehalle 2013)
        # This represents the opportunity cost of not providing liquidity
        # Can be learned via Lagrangian if needed
        self._nq_penalty_scale = 1.0

        # Round-trip bonus scale (Reward Shaping - Ng et al. 1999)
        # Amplifies the spread capture signal to balance continuous penalties
        self._rt_bonus_scale = 1.0

        # Realized PnL tracking — avg entry price for spread capture
        self._avg_entry_price = 0.0
        self._last_realized_pnl = 0.0  # cumulative, to compute per-step delta

        # Mark-to-market: previous mid price for dense unrealized PnL signal
        self._prev_mid_price = 0.0

        logger.debug(f"SimulationMarketMakerEnv initialized with initial_cash={initial_cash}")

    def set_risk_aversion(self, gamma: float):
        """Sync risk aversion γ from PPOAgent (learned via Lagrangian).

        This parameter comes from Avellaneda-Stoikov (2008) optimal control:
        γ determines the trade-off between expected return and inventory risk.
        It's learned, not set arbitrarily.
        """
        self._gamma_risk = max(0.001, gamma)  # Floor to prevent zero penalty

    def set_nq_penalty_scale(self, scale: float):
        """Set non-quotation penalty scale (Guéant-Lehalle 2013).

        This parameter controls the opportunity cost of not providing liquidity.
        Higher values encourage more active quoting.

        Theoretical basis: A market maker that doesn't quote misses expected
        spread capture opportunities. The penalty is proportional to:
        - Market spread (potential profit per trade)
        - Fill probability (likelihood of capturing the spread)
        """
        self._nq_penalty_scale = max(0.0, scale)

    def set_rt_bonus_scale(self, scale: float):
        """Set round-trip bonus scale (Reward Shaping - Ng et al. 1999).

        This parameter amplifies the spread capture signal when completing
        profitable round-trips (closing positions for profit).

        Theoretical basis: Potential-based reward shaping F(s,s') = γΦ(s') - Φ(s)
        preserves optimal policy while accelerating learning. The penalties
        (IP, ER, NQ) are continuous but spread capture is discrete - this
        bonus balances the learning signal.

        Args:
            scale: Multiplier for spread capture bonus. Default 1.0 means
                   the bonus equals the realized profit (effectively doubling
                   the signal from profitable round-trips).
        """
        self._rt_bonus_scale = max(0.0, scale)

    def set_fees(self, maker_fee: float, taker_fee: float):
        """Sync fee curriculum to this environment.

        Called by train.py to sync updated fees from curriculum learning
        to all vectorized environments.

        Args:
            maker_fee: New maker fee (e.g., 0.0001 = 1bps)
            taker_fee: New taker fee
        """
        self.sim_config.maker_fee = maker_fee
        self.sim_config.taker_fee = taker_fee

    def _load_historical_data(self, path: str) -> List[Dict]:
        """DEPRECATED: Use data_loader directly."""
        return []

    async def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Exécute une étape de simulation avec latence et contraintes."""
        try:
            # 1. Simuler la latence réseau
            latency = await self._simulate_network_latency()
            logger.debug(f"Simulated network latency: {latency}")

            # 2. Obtenir le prochain orderbook de marché
            orderbook = self.data_loader.get_next_orderbook()
            if orderbook is None:
                logger.warning("No more data available - ending simulation.")
                return self._get_observation(), 0, True, {'reason': 'end_of_data'}
                
            # 3. Mettre à jour l'état du marché
            await self.update_market_state(orderbook)
            logger.debug("Market state updated with new orderbook data.")

            # 4. Calculer les quotes
            quotes = self.calculate_quotes(action, {'orderbook': orderbook})
            self._last_quotes_levels = quotes.get('levels', [])
            logger.debug(f"Quotes calculated: {quotes}")

            # 5. Simuler l'exécution des ordres
            executions = await self._simulate_execution(quotes)
            logger.debug(f"Executions simulated: {executions}")

            # 5b. Update trade history for market features (TFI, Entropy, Illiquidity, iVPIN)
            for ex in executions:
                # Format matches MarketFeatureProcessor.update_trades expectation (Binance-style keys)
                # is_buyer_maker (m): True when the buyer placed the limit order (maker).
                # Our bid fill = we are buyer+maker → m=True
                # Our ask fill = buyer was taker → m=False
                trade_data = {
                    'p': ex['price'],
                    'q': ex['quantity'],
                    'm': ex['side'] == 'bid'
                }
                self.market_processor.update_trades(trade_data, timestamp=self._current_ts)

            # 6. Calculer la récompense
            # (Note: L'état est déjà mis à jour dans _simulate_execution)
            reward = self._calculate_reward(executions)
            logger.debug(f"Reward calculated: {reward}")

            # 7. Update fill indicators for next observation
            self.last_bid_fill = 1.0 if any(ex['side'] == 'bid' for ex in executions) else 0.0
            self.last_ask_fill = 1.0 if any(ex['side'] == 'ask' for ex in executions) else 0.0

            # 8. Mise à jour des métriques
            self._update_metrics(latency, executions)

            # 9. Vérifier fin d'épisode
            self.episode_step += 1
            done = self.episode_step >= self.max_steps
            logger.debug(f"Step {self.episode_step} completed. Done: {done}")

            # For metrics tracking, expose Level 1 as flat keys for backward compat
            l1 = quotes['levels'][0] if quotes.get('levels') else {}
            info = {
                'latency': latency,
                'quotes': {
                    'bid_price': l1.get('bid_price', 0),
                    'ask_price': l1.get('ask_price', 0),
                    'bid_qty': l1.get('bid_qty', 0),
                    'ask_qty': l1.get('ask_qty', 0),
                    'levels': quotes.get('levels', []),
                },
                'executions': executions,
                'metrics': self.get_info(),
                'ivpin_fast': self._last_ivpin_fast,
                'fill_qty': self._last_fill_qty,
            }

            return self._get_observation(), reward, done, info
            
        except Exception as e:
            logger.error(f"Error in simulation step: {str(e)}")
            return self._get_observation(), -1.0, True, {'reason': 'error'}

    async def _simulate_network_latency(self) -> float:
        """Simule une latence réseau réaliste (sans sleep — simulation offline)."""
        latency = np.random.uniform(self.sim_config.min_latency, self.sim_config.max_latency)
        # NOTE: No asyncio.sleep() — we record the latency for metrics but don't wait.
        # Real-time sleeping was burning ~6 min/episode of pure wall-clock waste.
        return latency
    


    async def reset(self) -> Dict:
        """Reset l'environnement pour un nouvel épisode."""
        try:
            # Reset data loader with random offset for episode diversity
            self.data_loader.reset(random_offset=True, episode_length=self.max_steps + 1)

            orderbook = self._get_next_orderbook()
            if orderbook is None:
                raise ValueError("No data available")

            self.episode_step = 0
            self.execution_times.clear()
            self.price_history.clear()
            self.volume_history.clear()

            # Reset market processor state to avoid cross-episode feature leakage
            self.market_processor.reset()

            # Reset portfolio value tracking
            initial_cash = getattr(self.rl_config, 'initial_cash', 100.0)
            self.last_portfolio_value = initial_cash

            # Reset fill indicators
            self.last_bid_fill = 0.0
            self.last_ask_fill = 0.0

            # Reset realized PnL tracking
            self._avg_entry_price = 0.0
            self._last_realized_pnl = 0.0
            self._prev_mid_price = 0.0  # will be set after _initialize_state

            # Reset order persistence (FIX 3)
            self.active_quotes = {
                'bid_price': 0.0, 'bid_qty': 0.0,
                'ask_price': 0.0, 'ask_qty': 0.0,
                'bid_age': 0, 'ask_age': 0
            }
            self._last_quotes_levels = []

            # Sync timestamp before initializing state (needed for market features)
            self._current_ts = self._parse_timestamp(orderbook)

            # Initialize state — this calls market_processor.process() internally
            self.current_state = self._initialize_state(orderbook, timestamp=self._current_ts)

            # Set fields that _initialize_state doesn't set
            bids = np.array([[float(p), float(q)] for p, q in orderbook['bids']])
            asks = np.array([[float(p), float(q)] for p, q in orderbook['asks']])
            self.current_state.best_bid = bids[0][0]
            self.current_state.best_ask = asks[0][0]
            self.current_state.current_spread = asks[0][0] - bids[0][0]

            # Créer le portfolio vectorbt initial
            self.portfolio = vbt.Portfolio.from_holding(
                close=pd.Series([self.current_state.mid_price]),
                init_cash=initial_cash,
                fees=self.sim_config.maker_fee,
                freq='1T'
            )
            self.current_state.portfolio = self.portfolio

            # NOTE: Do NOT call update_market_state here — _initialize_state already
            # called market_processor.process() which updates internal state.
            # Calling it again would double-count the first orderbook.

            self.price_history.append(self.current_state.mid_price)
            self._prev_mid_price = self.current_state.mid_price

            return self._get_observation()

        except Exception as e:
            logger.error(f"Error in reset: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _parse_timestamp(self, orderbook: Dict) -> datetime:
        """Parse local_timestamp robustly (handles ms, us, and ns)."""
        raw_ts = orderbook.get('local_timestamp', 0)
        # Heuristic: Unix seconds are ~1.7e9 (2024). Milliseconds ~1.7e12. Microseconds ~1.7e15. Nanoseconds ~1.7e18.
        if raw_ts > 1e15:  # nanoseconds
            ts_sec = raw_ts / 1e9
        elif raw_ts > 1e12:  # microseconds
            ts_sec = raw_ts / 1e6
        elif raw_ts > 1e9:  # milliseconds
            ts_sec = raw_ts / 1e3
        else:  # already seconds
            ts_sec = raw_ts
        return datetime.fromtimestamp(ts_sec)

    async def update_market_state(self, orderbook: Dict):
        """Met à jour l'état du marché avec synchro temporelle."""
        try:
            # Sync time with data
            self._current_ts = self._parse_timestamp(orderbook)
            
            # Pass TS to processors for deterministic math
            lob_features = self.lob_processor.process(orderbook)
            market_features = self.market_processor.process(orderbook, timestamp=self._current_ts)

            self.current_state.lob_features = lob_features
            self.current_state.market_features = market_features

            bids = np.array([[float(price), float(qty)] for price, qty in orderbook['bids']])
            asks = np.array([[float(price), float(qty)] for price, qty in orderbook['asks']])

            self.current_state.mid_price = (bids[0][0] + asks[0][0]) / 2
            self.current_state.current_spread = asks[0][0] - bids[0][0]
            self.current_state.best_bid = bids[0][0]
            self.current_state.best_ask = asks[0][0]

            self.price_history.append(self.current_state.mid_price)
            logger.debug(f"Market state updated: mid_price={self.current_state.mid_price}, spread={self.current_state.current_spread}")

        except Exception as e:
            logger.error(f"Error updating market state: {e}")
            raise

    def _update_metrics(self, latency: float, executions: List[Dict]):
        """Met à jour les métriques de simulation."""
        self.execution_times.append(latency)
        execution_volume = sum(ex['quantity'] * ex['price'] for ex in executions)
        self.volume_history.append(execution_volume)
        logger.debug(f"Metrics updated - Latency: {latency}, Execution volume: {execution_volume}")

    def get_info(self) -> Dict:
        """Retourne les informations de monitoring."""
        base_metrics = {
            'inventory': self.current_state.inventory,
            'cash': self.current_state.cash,
            'unrealized_pnl': self.current_state.unrealized_pnl,
            'total_pnl': self.current_state.cash + self.current_state.inventory * (self.current_state.mid_price or 0) - getattr(self.rl_config, 'initial_cash', 100.0),
        }

        sim_metrics = {
            'latency_stats': {
                'mean': np.mean(self.execution_times) if self.execution_times else 0,
                'std': np.std(self.execution_times) if self.execution_times else 0,
                'min': np.min(self.execution_times) if self.execution_times else 0,
                'max': np.max(self.execution_times) if self.execution_times else 0
            },
            'execution_stats': {
                'total_volume': sum(self.volume_history),
                'avg_price': np.mean(self.price_history) if self.price_history else 0,
                'price_volatility': np.std(self.price_history) if len(self.price_history) > 1 else 0
            },
            'simulation_progress': {
                'current_step': self.episode_step,
                'total_steps': self.max_steps,
                'data_progress': f"{self.data_loader.current_idx}/{len(self.data_loader.data)}"
            }
        }

        logger.debug(f"Simulation metrics: {sim_metrics}")
        return {**base_metrics, **sim_metrics}
    

    async def _simulate_execution(self, quotes: Dict) -> List[Dict]:
        """Simule l'exécution SOTA avec multi-level quotes, Queue Priority, Partial Fills, Size Impact."""
        current_time = self._get_current_timestamp()
        executions = []

        mid_price = self.current_state.mid_price

        # Volatility for fill model
        mf = self.current_state.market_features
        volatility = float(mf[self.market_processor.FEAT_NOISE]) if mf is not None else 0.0001

        # --- Helpers ---
        def _limit_fill_prob(distance_from_mid: float, order_qty: float, order_age: int = 0) -> float:
            """Fill probability with queue priority bonus from order age."""
            distance_bps = (distance_from_mid / mid_price) * 10000.0 if mid_price > 0 else 100.0

            lambda_decay_bps = self.sim_config.fill_lambda_decay
            base_prob = np.exp(-lambda_decay_bps * distance_bps)

            typical_trade_size = self.market_config.min_qty * 5
            size_ratio = order_qty / typical_trade_size
            size_penalty = 1.0 / (1.0 + size_ratio * 0.5)

            vol_normalized = np.clip(volatility / 0.01, 0, 1)
            vol_factor = 1.0 + vol_normalized * 0.5

            # FIX 3: Queue priority bonus — older orders get filled more easily
            age_bonus = min(order_age * 0.02, 0.15)

            prob = base_prob * size_penalty * vol_factor + age_bonus
            return float(np.clip(prob, 0.0, 0.95))

        def _maybe_partial(qty: float) -> float:
            if np.random.random() < self.sim_config.partial_fill_prob:
                return qty * np.random.uniform(self.sim_config.min_fill_ratio,
                                               self.sim_config.max_fill_ratio)
            return qty

        # --- FIX 2: Loop over all quote levels ---
        levels = quotes.get('levels', [])
        for lvl_idx, level in enumerate(levels):
            bid_price = level.get('bid_price', 0)
            ask_price = level.get('ask_price', 0)
            bid_qty = level.get('bid_qty', 0)
            ask_qty = level.get('ask_qty', 0)
            bid_age = level.get('bid_age', 0)
            ask_age = level.get('ask_age', 0)

            # --- Simulation BID ---
            if bid_qty > 0 and bid_price > 0:
                distance = abs(mid_price - bid_price)
                fill_prob = _limit_fill_prob(distance, bid_qty, bid_age)

                if np.random.random() < fill_prob:
                    filled_qty = _maybe_partial(bid_qty)
                    executions.append({
                        'side': 'bid', 'price': bid_price, 'quantity': filled_qty,
                        'timestamp': current_time, 'type': 'limit', 'level': lvl_idx,
                        'fee': bid_price * filled_qty * self.sim_config.maker_fee
                    })
                    logger.debug(f"L{lvl_idx} Bid FILL (Prob={fill_prob:.4f}, Qty={filled_qty:.1f}/{bid_qty}) at {bid_price}")

            # --- Simulation ASK ---
            if ask_qty > 0 and ask_price > 0:
                distance = abs(ask_price - mid_price)
                fill_prob = _limit_fill_prob(distance, ask_qty, ask_age)

                if np.random.random() < fill_prob:
                    filled_qty = _maybe_partial(ask_qty)
                    executions.append({
                        'side': 'ask', 'price': ask_price, 'quantity': filled_qty,
                        'timestamp': current_time, 'type': 'limit', 'level': lvl_idx,
                        'fee': ask_price * filled_qty * self.sim_config.maker_fee
                    })
                    logger.debug(f"L{lvl_idx} Ask FILL (Prob={fill_prob:.4f}, Qty={filled_qty:.1f}/{ask_qty}) at {ask_price}")

        # --- FLATTEN market order (taker — guaranteed fill with slippage) ---
        flatten_qty = quotes.get('flatten_qty', 0)
        flatten_side = quotes.get('flatten_side')
        if flatten_qty > 0 and flatten_side is not None:
            if flatten_side == 'sell':
                fill_price = float(self.current_state.best_bid) if self.current_state.best_bid else mid_price * 0.9999
                executions.append({
                    'side': 'ask', 'price': fill_price, 'quantity': float(flatten_qty),
                    'timestamp': current_time, 'type': 'market', 'level': -1,
                    'fee': fill_price * float(flatten_qty) * self.sim_config.taker_fee
                })
                logger.debug(f"FLATTEN SELL market order: qty={flatten_qty} at {fill_price}")
            else:
                fill_price = float(self.current_state.best_ask) if self.current_state.best_ask else mid_price * 1.0001
                executions.append({
                    'side': 'bid', 'price': fill_price, 'quantity': float(flatten_qty),
                    'timestamp': current_time, 'type': 'market', 'level': -1,
                    'fee': fill_price * float(flatten_qty) * self.sim_config.taker_fee
                })
                logger.debug(f"FLATTEN BUY market order: qty={flatten_qty} at {fill_price}")

        # Mise à jour après exécutions
        for execution in executions:
            if execution['side'] == 'bid':
                self.current_state.inventory += execution['quantity']
                self.current_state.cash -= (execution['price'] * execution['quantity'] + execution['fee'])
            else:
                self.current_state.inventory -= execution['quantity']
                self.current_state.cash += (execution['price'] * execution['quantity'] - execution['fee'])

            initial_cash = getattr(self.rl_config, 'initial_cash', 100.0)
            self.current_state.unrealized_pnl = (
                self.current_state.inventory * self.current_state.mid_price +
                self.current_state.cash - initial_cash
            )

        return executions



    def _calculate_reward(self, executions: List[Dict]) -> float:
        """Avellaneda-Stoikov (2008) based reward function.

        Theoretically grounded reward derived from HJB solution:
        R_t = ΔWealth_t - γ·σ²·q²·Δt

        Components:
        1. PnL_t: Spread capture (realized) + Mark-to-Market (ΔWealth)
        2. IP_t:  Quadratic inventory penalty from optimal control theory
                  -γ·σ²·q²·Δt where σ is realized volatility (observable)
        3. ER_t:  Execution risk from Guéant-Lehalle-Fernandez-Tapia (2013)

        REMOVED: c_t (fill compensation) - no theoretical foundation,
        was causing reward/PnL divergence.
        """

        try:
            mid_price = self.current_state.mid_price

            # Track portfolio value for logging
            current_value = self.current_state.cash + self.current_state.inventory * mid_price
            pnl_change = current_value - self.last_portfolio_value
            self.last_portfolio_value = current_value

            # ── 1. PnL_t: Spread capture + MTM ──────────────────────────
            # 1a. Realized PnL from fills (FIFO avg-entry tracking)
            realized_pnl = 0.0
            inventory = self.current_state.inventory
            # Reconstruct pre-execution inventory
            pre_inv = inventory
            for ex in executions:
                if ex['side'] == 'bid':
                    pre_inv -= ex['quantity']
                else:
                    pre_inv += ex['quantity']

            running_inv = pre_inv
            avg_price = self._avg_entry_price

            # Track round-trip completions for RT bonus
            total_closed_qty = 0.0
            spread_captured = 0.0  # PnL from closing positions (spread capture)

            for ex in executions:
                price = ex['price']
                qty = ex['quantity']
                fee = ex.get('fee', 0)

                if ex['side'] == 'bid':  # BUY
                    if running_inv < 0:
                        close_qty = min(qty, abs(running_inv))
                        pnl_from_close = (avg_price - price) * close_qty - fee
                        realized_pnl += pnl_from_close
                        # Track round-trip metrics
                        total_closed_qty += close_qty
                        spread_captured += pnl_from_close
                        open_qty = qty - close_qty
                        if open_qty > 0 and (abs(running_inv) + open_qty) > 1e-9:
                            avg_price = price
                        elif close_qty < abs(running_inv):
                            pass
                        else:
                            avg_price = price if open_qty > 0 else 0.0
                    else:
                        total_qty = running_inv + qty
                        if total_qty > 1e-9:
                            avg_price = (avg_price * running_inv + price * qty) / total_qty
                        else:
                            avg_price = price
                        realized_pnl -= fee
                    running_inv += qty

                else:  # SELL
                    if running_inv > 0:
                        close_qty = min(qty, running_inv)
                        pnl_from_close = (price - avg_price) * close_qty - fee
                        realized_pnl += pnl_from_close
                        # Track round-trip metrics
                        total_closed_qty += close_qty
                        spread_captured += pnl_from_close
                        open_qty = qty - close_qty
                        if open_qty > 0:
                            avg_price = price
                        elif close_qty < running_inv:
                            pass
                        else:
                            avg_price = price if open_qty > 0 else 0.0
                    else:
                        total_qty = abs(running_inv) + qty
                        if total_qty > 1e-9:
                            avg_price = (avg_price * abs(running_inv) + price * qty) / total_qty
                        else:
                            avg_price = price
                        realized_pnl -= fee
                    running_inv -= qty

            self._avg_entry_price = avg_price

            # 1b. Mark-to-market on post-execution inventory: (m_{t+1} - m_t) × Q_{t+1}
            price_delta = mid_price - self._prev_mid_price if self._prev_mid_price > 0 else 0.0
            mtm_unrealized = price_delta * inventory  # post-fill inventory (RELAER)
            self._prev_mid_price = mid_price

            pnl_t = realized_pnl + mtm_unrealized

            # ── 2. IP_t: Avellaneda-Stoikov Quadratic Inventory Penalty ──
            # From HJB solution: V(t,x,q,s) = x + q·s - γ·σ²·q²·(T-t)
            #
            # SCALING FIX: All reward components normalized to "spread units"
            # Reference scale = half_spread × min_qty (typical per-trade profit)
            # This ensures all terms are comparable in magnitude.

            max_pos = float(getattr(self.market_config, 'max_position', 1000.0))
            min_qty = float(getattr(self.market_config, 'min_qty', 1.0))
            mf = self.current_state.market_features

            # Get realized volatility from market features
            # FEATURE INDEX 0 = microstructure noise / 10 (see processor.py)
            # So we multiply by 10 to get actual volatility in BPS
            # Range after clipping: mf[0] ∈ [-5, 5] → sigma_bps ∈ [0, 50]
            raw_noise = float(mf[0]) if mf is not None and len(mf) > 0 else 1.0
            sigma_bps = max(1.0, abs(raw_noise) * 10.0)  # Undo /10 scaling, ensure positive
            sigma = sigma_bps / 10000.0  # Convert BPS to decimal

            # Reference scale: typical spread capture per trade
            # All penalties scaled relative to this so they're meaningful
            market_spread = self.current_state.current_spread or (mid_price * 0.005)
            spread_bps = (market_spread / mid_price * 10000) if mid_price > 0 else 50.0  # Spread in BPS
            reward_scale = (market_spread / 2) * min_qty  # Half-spread × min_qty
            reward_scale = max(reward_scale, 1e-6)  # Floor to prevent division issues

            # Risk aversion γ is synced from PPOAgent (learned via Lagrangian)
            gamma_risk = getattr(self, '_gamma_risk', 0.1)

            # Time discretization
            dt = 1.0 / self.max_steps

            # Normalized inventory for numerical stability
            q_normalized = inventory / max_pos
            q_abs = abs(q_normalized)

            # ═══════════════════════════════════════════════════════════════
            # DATA-DRIVEN AVELLANEDA-STOIKOV INVENTORY PENALTY
            # ═══════════════════════════════════════════════════════════════
            #
            # From A-S optimal control: V(t,x,q,s) = x + q·s - γ·σ²·q²·(T-t)
            #
            # ALL PARAMETERS DERIVED FROM MARKET DATA:
            #
            # 1. THRESHOLD = spread / (spread + volatility)
            #    - High spread, low vol → can hold more inventory (higher threshold)
            #    - Low spread, high vol → risky to hold (lower threshold)
            #
            # 2. SHARPNESS = volatility / spread
            #    - High vol → penalty kicks in faster (sharper)
            #    - Low vol → more gradual penalty (softer)
            #
            # 3. TIME_URGENCY_SLACK = spread / volatility (clamped)
            #    - High spread/vol ratio → more time to flatten
            #    - Low spread/vol ratio → must flatten quickly
            #
            # ═══════════════════════════════════════════════════════════════

            # Time remaining in episode [0, 1]
            time_remaining = getattr(self.current_state, 'time_remaining', 0.5)

            # Data-driven time urgency slack
            # slack = spread_bps / sigma_bps, clamped to [0.3, 0.9]
            # High spread/vol → more slack (can hold inventory longer)
            spread_vol_ratio = spread_bps / max(sigma_bps, 1.0)
            time_slack = np.clip(spread_vol_ratio / 10.0, 0.3, 0.9)

            # Time urgency: starts at (1-slack), increases to 1.0 at episode end
            time_urgency = 1.0 - time_slack * max(0.0, min(1.0, time_remaining))

            # Data-driven threshold: spread / (spread + volatility)
            # Range: [0.1, 0.6] - higher spread means can hold more inventory
            threshold = np.clip(spread_bps / (spread_bps + sigma_bps + 1e-6), 0.1, 0.6)

            # Data-driven sharpness: volatility / spread
            # Higher vol relative to spread → sharper penalty transition
            # Range: [2, 10]
            sharpness = np.clip(sigma_bps / max(spread_bps, 1.0), 2.0, 10.0)

            # Softplus-based smooth threshold: ~0 below threshold, ~linear above
            # f(q) = (1/k) * log(1 + exp(k * (|q| - threshold)))
            x = sharpness * (q_abs - threshold)
            if x > 20:  # Prevent overflow
                soft_excess = q_abs - threshold
            elif x < -20:  # Prevent underflow
                soft_excess = 0.0
            else:
                soft_excess = (1.0 / sharpness) * np.log(1.0 + np.exp(x))

            # Final A-S penalty: γ × soft_excess × time_urgency × scale
            # ALL components are now data-driven:
            # - gamma_risk: learned via Lagrangian
            # - soft_excess: threshold & sharpness from spread/vol
            # - time_urgency: slack from spread/vol ratio
            # - reward_scale: from market spread
            ip_t = -gamma_risk * soft_excess * time_urgency * reward_scale

            # ── 3. C_t: REMOVED ────────────────────────────────────────
            # Fill compensation had no theoretical foundation and caused
            # reward to increase while PnL decreased (misalignment).
            # The realized spread capture in PnL_t already rewards fills.

            # ── 4. ER_t: Execution risk (stale order penalty) ──────────
            # Stale orders accumulate adverse selection risk as price moves.
            # All coefficients derived from market data:
            #
            # t_w = spread_bps / σ_bps = characteristic time for price to move 1 spread
            #       (data-driven: faster markets have shorter windows)
            #
            # λ_er = σ_bps / 100 = volatility-scaled risk coefficient
            #        (data-driven: higher vol = more execution risk)
            #
            # At 10bps vol and 50bps spread: t_w=5 steps, λ_er=0.1
            # At 50bps vol and 50bps spread: t_w=1 step, λ_er=0.5
            t_w = max(1.0, spread_bps / max(sigma_bps, 1.0))  # Data-driven reference window
            lambda_er = min(1.0, sigma_bps / 100.0)  # Data-driven risk coefficient
            er_t = 0.0

            # L1: tracked via active_quotes with real ages
            for side in ['bid', 'ask']:
                age = self.active_quotes.get(f'{side}_age', 0)
                vol = self.active_quotes.get(f'{side}_qty', 0)
                if vol > 0 and age > 0:
                    vol_normalized = vol / max_pos
                    er_t -= lambda_er * vol_normalized * (1.0 + age / t_w) * reward_scale

            # L2/L3: fresh orders (age=0) still carry base risk
            if hasattr(self, '_last_quotes_levels'):
                for lvl in self._last_quotes_levels[1:]:  # skip L1
                    for side_key in ['bid_qty', 'ask_qty']:
                        vol = lvl.get(side_key, 0)
                        if vol > 0:
                            vol_normalized = vol / max_pos
                            er_t -= lambda_er * vol_normalized * reward_scale

            # ── 5. NQ_t: Non-Quotation Penalty (Guéant-Lehalle 2013) ────
            # Opportunity cost of not providing liquidity, scaled to reward_scale.
            #
            # NQ_t = -λ_nq × P(fill) × reward_scale × 𝟙[no quote on side]
            # At typical fill probability, penalty ≈ λ_nq × reward_scale per side

            # Fill probability decreases with spread (from fill model)
            # spread_bps already calculated above
            p_fill_estimate = np.exp(-0.5 * spread_bps / 10.0)  # Decay with spread

            # Check if agent is quoting on each side
            has_bid_quote = False
            has_ask_quote = False
            if hasattr(self, '_last_quotes_levels') and self._last_quotes_levels:
                for lvl in self._last_quotes_levels:
                    if lvl.get('bid_qty', 0) > 0 and lvl.get('bid_price', 0) > 0:
                        has_bid_quote = True
                    if lvl.get('ask_qty', 0) > 0 and lvl.get('ask_price', 0) > 0:
                        has_ask_quote = True

            # Opportunity cost of not quoting (per side) in spread units
            # nq_penalty_scale = 1.0 is theoretically grounded:
            # Missing a quote = missing exactly 1 spread capture opportunity
            # (scaled by fill probability p_fill_estimate)
            nq_penalty_scale = getattr(self, '_nq_penalty_scale', 1.0)  # Theoretically = 1.0

            nq_t = 0.0
            if not has_bid_quote:
                nq_t -= nq_penalty_scale * reward_scale * p_fill_estimate
            if not has_ask_quote:
                nq_t -= nq_penalty_scale * reward_scale * p_fill_estimate

            # Both sides missing = extra penalty (market maker obligation)
            if not has_bid_quote and not has_ask_quote:
                nq_t -= nq_penalty_scale * reward_scale * 0.5

            # ── 6. RT_t: Round-Trip Bonus (Reward Shaping) ────────────────
            # Theoretical justification (Ng, Harada, Russell 1999):
            # Potential-based reward shaping preserves optimal policy.
            #
            # Data-driven scaling:
            # β = t_w × γ = expected penalty accumulation during characteristic hold time
            # This compensates for IP/ER penalties accumulated while holding position.
            #
            # Example: t_w=5, γ=0.1 → β=0.5 (bonus = 50% of spread captured)
            # Example: t_w=50, γ=0.1 → β=5.0 (bonus = 500% of spread captured)
            #
            # Slower markets (high t_w) need more bonus to offset longer hold penalties.
            rt_bonus_base = t_w * gamma_risk  # Data-driven from market characteristics
            rt_bonus_scale = getattr(self, '_rt_bonus_scale', None)
            if rt_bonus_scale is None:
                rt_bonus_scale = max(0.5, min(5.0, rt_bonus_base))  # Clamp to [0.5, 5.0]

            rt_t = 0.0
            if total_closed_qty > 0 and spread_captured > 0:
                # Bonus proportional to spread captured (profitable round-trips only)
                rt_t = rt_bonus_scale * spread_captured

            # ── iVPIN tracking for Lagrangian constraint in PPO ─────────
            fill_qty = sum(ex['quantity'] for ex in executions)
            ivpin_fast = self.market_processor.get_ivpin(
                self.market_config.fractal_windows[0], self._current_ts
            )
            self._last_ivpin_fast = ivpin_fast
            self._last_fill_qty = fill_qty

            # ── 7. Full Reward: R_t = PnL_t + IP_t + ER_t + NQ_t + RT_t ────
            # Theoretically grounded components:
            # - PnL_t: Wealth change (spread capture + MTM)
            # - IP_t:  Avellaneda-Stoikov inventory penalty (-γσ²q²Δt)
            # - ER_t:  Execution risk (stale orders)
            # - NQ_t:  Non-quotation penalty (Guéant-Lehalle opportunity cost)
            # - RT_t:  Round-trip bonus (reward shaping for spread capture)
            reward = pnl_t + ip_t + er_t + nq_t + rt_t

            # Log data-driven parameters on first call for verification
            if not hasattr(self, '_reward_params_logged'):
                logger.info(f"Reward params (data-driven): t_w={t_w:.1f}, λ_er={lambda_er:.3f}, "
                           f"rt_scale={rt_bonus_scale:.2f}, reward_scale=${reward_scale:.6f}, "
                           f"spread={spread_bps:.1f}bps, σ={sigma_bps:.1f}bps")
                self._reward_params_logged = True

            logger.debug(f"AS Reward: pnl={pnl_t:.6f} (real={realized_pnl:.6f}, "
                         f"mtm={mtm_unrealized:.6f}), ip={ip_t:.6f} (γ={gamma_risk:.3f}), "
                         f"er={er_t:.6f}, nq={nq_t:.6f}, rt={rt_t:.6f}, iVPIN={ivpin_fast:.3f}, "
                         f"fills={fill_qty:.1f}, closed={total_closed_qty:.1f}, "
                         f"total={reward:.6f}")

            return float(reward)

        except Exception as e:
            logger.error(f"Erreur lors du calcul de la récompense: {str(e)}")
            logger.error(f"Traceback complet: {traceback.format_exc()}")
            return 0.0
