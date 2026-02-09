"""Live Market Maker Environment for Binance Testnet deployment.

Replaces the deprecated reel_env.py with a 9D-compatible environment
that inherits from BaseMarketMakerEnv and uses real Binance data streams.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

from core.config import RLConfig, MarketConfig
from core.types import OrderSide, Trade
from data.processor import LOBFeatureProcessor, MarketFeatureProcessor
from data.binance.stream import BinanceStreamHandler
from environment.base_environment import BaseMarketMakerEnv, MarketMakerState
from execution.handler import LiveOrderExecutor
from execution.Position_manager import PositionManager

logger = logging.getLogger('market_maker.live_env')


class LiveMarketMakerEnv(BaseMarketMakerEnv):
    """Live environment connecting PPO agent to Binance Cross Margin via streams.

    Inherits calculate_quotes() and _get_observation() from BaseMarketMakerEnv
    (already 9D multi-level). Overrides reset() and step() for live trading.
    """

    def __init__(self,
                 rl_config: RLConfig,
                 market_config: MarketConfig,
                 lob_processor: LOBFeatureProcessor,
                 market_processor: MarketFeatureProcessor,
                 stream_handler: BinanceStreamHandler,
                 order_executor: LiveOrderExecutor,
                 position_manager: PositionManager,
                 stale_data_timeout: float = 5.0,
                 update_interval: float = 0.1,
                 shaping_config: Optional[Dict] = None):
        super().__init__(rl_config, market_config, lob_processor, market_processor)

        self.stream_handler = stream_handler
        self.order_executor = order_executor
        self.position_manager = position_manager
        self.stale_data_timeout = stale_data_timeout
        self.update_interval = update_interval

        # Reward shaping coefficients (0 = disabled, e.g. in deploy mode)
        sc = shaping_config or {}
        self.shaping_proximity = sc.get('shaping_proximity', 0.0)
        self.shaping_presence = sc.get('shaping_presence', 0.0)
        self.shaping_spread = sc.get('shaping_spread', 0.0)

        # Fill tracking
        self.last_bid_fill = 0.0
        self.last_ask_fill = 0.0
        self._last_trade_check_ms = int(time.time() * 1000)  # epoch ms for trade polling
        self._seen_trade_ids: set = set()  # dedup trade IDs across polls

        # Portfolio value tracking for PnL reward (aligned with sim_env)
        initial_cash = getattr(rl_config, 'initial_cash', 100.0)
        self.last_portfolio_value = initial_cash

        # Realized PnL tracking — avg entry price for spread capture
        self._avg_entry_price = 0.0
        self._last_realized_pnl = 0.0

        # Adverse selection (M3ORL) — λ_as synced from PPO Lagrangian
        self._lambda_as = 0.0
        self._last_ivpin_fast = 0.0
        self._last_fill_qty = 0.0

        # Mark-to-market: previous mid price for dense unrealized PnL signal
        self._prev_mid_price = 0.0

        if self.shaping_proximity or self.shaping_presence or self.shaping_spread:
            logger.info(f"Reward shaping enabled: proximity={self.shaping_proximity}, "
                        f"presence={self.shaping_presence}, spread={self.shaping_spread}")
        logger.info("LiveMarketMakerEnv initialized")

    async def reset(self) -> Dict:
        """Reset environment: cancel orders, sync position, get fresh orderbook."""
        self.episode_step = 0
        self.last_bid_fill = 0.0
        self.last_ask_fill = 0.0
        self._last_trade_check_ms = int(time.time() * 1000)
        self._seen_trade_ids.clear()
        self._avg_entry_price = 0.0
        self._last_realized_pnl = 0.0
        self._prev_mid_price = 0.0  # will be set after state init

        # Reset active quotes tracking
        self.active_quotes = {
            'bid_price': 0.0, 'bid_qty': 0.0,
            'ask_price': 0.0, 'ask_qty': 0.0,
            'bid_age': 0, 'ask_age': 0
        }

        # Cancel all outstanding orders
        await self.order_executor.cancel_all_orders()

        # Sync position from Binance (authoritative — queries exchange directly)
        await self.position_manager.sync_position()
        pos_info = self.position_manager.get_position_info()
        initial_cash = self.rl_config.initial_cash

        # Reset market feature processor temporal state
        self.market_processor.reset()

        # Reset portfolio value tracking (aligned with sim_env DSR)
        self.last_portfolio_value = initial_cash

        # Reset portfolio
        from environment.base_environment import Portfolio
        self.portfolio = Portfolio(initial_cash=initial_cash)

        # Get current orderbook from stream
        orderbook = self.stream_handler.orderbook
        if not orderbook.get('bids') or not orderbook.get('asks'):
            logger.warning("Waiting for orderbook data from stream...")
            for _ in range(50):  # Wait up to 5s
                await asyncio.sleep(0.1)
                orderbook = self.stream_handler.orderbook
                if orderbook.get('bids') and orderbook.get('asks'):
                    break
            else:
                raise RuntimeError("No orderbook data available from stream after timeout")

        # Initialize state from orderbook
        self.current_state = self._initialize_state(orderbook)

        # Set best_bid/best_ask/spread (aligned with sim_env reset)
        bids = np.array([[float(p), float(q)] for p, q in orderbook['bids']])
        asks = np.array([[float(p), float(q)] for p, q in orderbook['asks']])
        self.current_state.best_bid = bids[0][0]
        self.current_state.best_ask = asks[0][0]
        self.current_state.current_spread = asks[0][0] - bids[0][0]

        # Override with real position data
        self.current_state.inventory = pos_info.quantity
        self.current_state.cash = initial_cash  # Will be tracked locally

        self._prev_mid_price = self.current_state.mid_price or 0.0

        logger.info(f"Environment reset. Position: {pos_info.quantity}, "
                    f"Mid: {self.current_state.mid_price:.5f}")

        return self._get_observation()

    async def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step: get data, compute quotes, place orders, sync position."""
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        self.last_bid_fill = 0.0
        self.last_ask_fill = 0.0

        # 1. Get latest orderbook from stream
        orderbook = self.stream_handler.orderbook

        # 2. Check data freshness — reconnect if stale
        ob_timestamp = self.stream_handler.orderbook_timestamp
        if ob_timestamp:
            age = (datetime.now() - ob_timestamp).total_seconds()
            if age > self.stale_data_timeout:
                logger.warning(f"Stale orderbook data ({age:.1f}s old), reconnecting streams...")
                ok = await self.stream_handler.reconnect_streams()
                if ok:
                    # Propagate the new client to order executor & position manager
                    new_client = self.stream_handler.client
                    self.order_executor.client = new_client
                    self.position_manager.client = new_client
                    # Use fresh orderbook after reconnect
                    orderbook = self.stream_handler.orderbook
                    logger.info("Reconnected — resuming with fresh data")
                else:
                    logger.error("Reconnect failed, skipping step")
                    return self._get_observation(), 0.0, False, {'skipped': True, 'stale_age': age}

        # 3. Update market state (LOB + market features)
        self._update_market_state(orderbook)

        # 4. Calculate multi-level quotes (inherited from BaseMarketMakerEnv, 9D)
        market_state = {'orderbook': orderbook}
        quotes = self.calculate_quotes(action, market_state)

        # 5. Place/update orders via LiveOrderExecutor
        active_orders = await self.order_executor.update_multi_level_orders(quotes)

        # 6. Wait for update interval
        await asyncio.sleep(self.update_interval)

        # 7. Sync position from exchange — single source of truth
        prev_inventory = self.current_state.inventory
        fills = await self._sync_and_detect_fills()

        # 8. Update unrealized PnL
        mid_price = self.current_state.mid_price
        if mid_price and mid_price > 0:
            self.position_manager.update_unrealized_pnl(mid_price)
            self.current_state.unrealized_pnl = (
                self.current_state.inventory * mid_price + self.current_state.cash
                - self.rl_config.initial_cash
            )

        # 9. Calculate reward
        reward = self._calculate_reward(fills)

        # 10. Increment step
        self.episode_step += 1
        done = self.episode_step >= self.max_steps

        info = {
            'quotes': quotes,
            'fills': fills,
            'active_orders': active_orders,
            'inventory': self.current_state.inventory,
            'cash': self.current_state.cash,
            'mid_price': mid_price,
            'pnl': self.current_state.unrealized_pnl,
            'ivpin_fast': self._last_ivpin_fast,
            'fill_qty': self._last_fill_qty,
            'metrics': {
                'total_pnl': self.position_manager.realized_pnl + self.position_manager.unrealized_pnl,
                'realized_pnl': self.position_manager.realized_pnl,
                'unrealized_pnl': self.position_manager.unrealized_pnl,
                'inventory': self.current_state.inventory,
                'position_value': abs(self.current_state.inventory * mid_price) if mid_price else 0,
            }
        }

        return self._get_observation(), reward, done, info

    def _update_market_state(self, orderbook: Dict):
        """Update LOB and market features from fresh orderbook."""
        try:
            lob_features = self.lob_processor.process(orderbook)
            market_features = self.market_processor.process(orderbook)

            bids = np.array([[float(p), float(q)] for p, q in orderbook['bids']])
            asks = np.array([[float(p), float(q)] for p, q in orderbook['asks']])
            mid_price = (bids[0][0] + asks[0][0]) / 2

            best_bid = bids[0][0]
            best_ask = asks[0][0]

            self.current_state.lob_features = lob_features
            self.current_state.market_features = market_features
            self.current_state.mid_price = mid_price
            self.current_state.best_bid = best_bid
            self.current_state.best_ask = best_ask
            self.current_state.current_spread = best_ask - best_bid
        except Exception as e:
            logger.error(f"Error updating market state: {e}")

    async def _query_recent_trades(self) -> List[Dict]:
        """Query our recent account trades from Binance since last check."""
        mode = self.order_executor.mode
        symbol = self.market_config.symbol
        client = self.order_executor.client
        try:
            if mode == 'futures':
                trades = await client.futures_account_trades(
                    symbol=symbol, startTime=self._last_trade_check_ms)
            elif mode == 'spot':
                trades = await client.get_my_trades(
                    symbol=symbol, startTime=self._last_trade_check_ms)
            else:  # margin
                trades = await client.get_margin_trades(
                    symbol=symbol, startTime=self._last_trade_check_ms)
            return trades
        except Exception as e:
            logger.warning(f"Failed to query recent trades: {e}")
            return []

    async def _sync_and_detect_fills(self) -> List[Dict]:
        """Two simple queries to Binance:
          1. Position → inventory (portfolio, source of truth)
          2. Recent trades → fills (price, qty, commission)
        """
        fills = []

        # --- 1. Position: query portfolio directly ---
        try:
            await self.position_manager.sync_position()
            self.current_state.inventory = self.position_manager.current_position
        except Exception as e:
            logger.error(f"Position sync failed: {e}")

        # --- 2. Fills: query recent executed trades ---
        # Drain stream (best effort dedup)
        for sf in self.stream_handler.get_pending_fills():
            tid = sf.get('trade_id') or sf.get('order_id')
            if tid:
                self._seen_trade_ids.add(str(tid))

        try:
            raw_trades = await self._query_recent_trades()

            for t in raw_trades:
                trade_id = str(t.get('id', t.get('tradeId', '')))
                if trade_id in self._seen_trade_ids:
                    continue
                self._seen_trade_ids.add(trade_id)

                is_buyer = t.get('buyer', t.get('isBuyer', False))
                side = 'buy' if is_buyer else 'sell'
                price = float(t.get('price', 0))
                qty = float(t.get('qty', 0))
                commission = float(t.get('commission', 0))
                ts = t.get('time', None)
                timestamp = datetime.fromtimestamp(ts / 1000) if ts else datetime.now()

                fills.append({
                    'side': side,
                    'price': price,
                    'quantity': qty,
                    'commission': commission,
                    'timestamp': timestamp,
                    'order_id': str(t.get('orderId', '')),
                })

                # Update cash from real fill data
                if side == 'buy':
                    self.current_state.cash -= (price * qty + commission)
                    self.last_bid_fill = 1.0
                else:
                    self.current_state.cash += (price * qty - commission)
                    self.last_ask_fill = 1.0

                # Feed into market feature processor
                self.market_processor.update_trades({'p': price, 'q': qty, 'm': side == 'buy'})
                self.portfolio.add_trade(timestamp, price, qty, side, fee=commission)

                logger.info(f"Fill: {side} {qty} @ {price:.5f}, "
                            f"commission={commission:.6f}")

            # Advance timestamp
            if raw_trades:
                self._last_trade_check_ms = max(int(t.get('time', 0)) for t in raw_trades) + 1
            else:
                self._last_trade_check_ms = int(time.time() * 1000)

        except Exception as e:
            logger.error(f"Error querying trades: {e}")

        # Prune dedup set
        if len(self._seen_trade_ids) > 1000:
            self._seen_trade_ids = set(list(self._seen_trade_ids)[-500:])

        return fills

    def _calculate_reward(self, fills: List[Dict]) -> float:
        """Reward = realized_pnl + MTM_unrealized - inventory_penalty - AS_penalty.

        Dense reward combining:
        - Realized PnL: actual spread captured via avg entry price (FIFO).
        - MTM unrealized: (p_t - p_{t-1}) * z_t — dense signal for inventory value changes.
        - Inventory penalty: quadratic with time urgency (Avellaneda-Stoikov).
        - Adverse selection: iVPIN × fill_qty (M3ORL Lagrangian).
        Aligned with sim_env._calculate_reward().
        """
        mid = self.current_state.mid_price or 0
        if mid <= 0:
            return 0.0

        # Track portfolio value (for logging only, not reward)
        current_value = self.current_state.cash + self.current_state.inventory * mid
        pnl_change = current_value - self.last_portfolio_value
        self.last_portfolio_value = current_value

        # 1. Realized PnL from fills — actual spread captured via avg entry price
        realized_pnl = 0.0
        # Inventory before this step's fills (fills already applied in _sync_and_detect_fills)
        pre_inv = self.current_state.inventory
        for fill in fills:
            if fill['side'] == 'buy':
                pre_inv -= fill['quantity']
            else:
                pre_inv += fill['quantity']

        running_inv = pre_inv
        avg_price = self._avg_entry_price

        for fill in fills:
            price = fill['price']
            qty = fill['quantity']
            commission = fill.get('commission', 0)

            if fill['side'] == 'buy':
                if running_inv < 0:
                    close_qty = min(qty, abs(running_inv))
                    realized_pnl += (avg_price - price) * close_qty - commission
                    open_qty = qty - close_qty
                    if open_qty > 0:
                        avg_price = price
                    elif close_qty >= abs(running_inv):
                        avg_price = price if open_qty > 0 else 0.0
                else:
                    total_qty = running_inv + qty
                    if total_qty > 1e-9:
                        avg_price = (avg_price * running_inv + price * qty) / total_qty
                    else:
                        avg_price = price
                    realized_pnl -= commission
                running_inv += qty
            else:  # sell
                if running_inv > 0:
                    close_qty = min(qty, running_inv)
                    realized_pnl += (price - avg_price) * close_qty - commission
                    open_qty = qty - close_qty
                    if open_qty > 0:
                        avg_price = price
                    elif close_qty >= running_inv:
                        avg_price = price if open_qty > 0 else 0.0
                else:
                    total_qty = abs(running_inv) + qty
                    if total_qty > 1e-9:
                        avg_price = (avg_price * abs(running_inv) + price * qty) / total_qty
                    else:
                        avg_price = price
                    realized_pnl -= commission
                running_inv -= qty

        self._avg_entry_price = avg_price

        # 2. Mark-to-market unrealized PnL: (p_t - p_{t-1}) * z_t
        #    Dense signal — immediate feedback on inventory value changes.
        #    Uses pre-fill inventory to avoid crediting the agent's own price impact.
        mtm_weight = 0.5
        price_delta = mid - self._prev_mid_price if self._prev_mid_price > 0 else 0.0
        mtm_unrealized = price_delta * pre_inv
        self._prev_mid_price = mid

        # 3. Inventory penalty with time urgency
        current_inventory = self.current_state.inventory
        max_pos = float(getattr(self.market_config, 'max_position', 100.0))
        inv_ratio = current_inventory / max_pos

        progress = self.episode_step / max(1, self.max_steps)
        time_urgency = 1.0 + 2.0 * progress  # 1x at start, 3x at end

        zeta = self.rl_config.zeta
        inventory_penalty = zeta * (inv_ratio ** 2) * time_urgency

        # 3. Adverse selection penalty (M3ORL — λ_as synced from PPO)
        fill_qty = sum(f['quantity'] for f in fills)
        ivpin_fast = self.market_processor.get_ivpin(
            self.market_config.fractal_windows[0]  # 8s half-life
        )
        adverse_selection_penalty = ivpin_fast * fill_qty
        self._last_ivpin_fast = ivpin_fast
        self._last_fill_qty = fill_qty

        # 5. Reward = realized + MTM_unrealized - inventory_penalty - AS_penalty
        mtm_component = mtm_weight * mtm_unrealized
        reward = realized_pnl + mtm_component - inventory_penalty - self._lambda_as * adverse_selection_penalty

        # 6. Reward shaping (only active during fine-tuning, coefficients = 0 otherwise)
        shaping_proximity_r = 0.0
        shaping_presence_r = 0.0
        shaping_spread_r = 0.0

        if self.shaping_proximity or self.shaping_presence or self.shaping_spread:
            best_bid = self.current_state.best_bid or 0
            best_ask = self.current_state.best_ask or 0
            l1_bid = self.active_quotes.get('bid_price', 0.0)
            l1_ask = self.active_quotes.get('ask_price', 0.0)
            l1_bid_qty = self.active_quotes.get('bid_qty', 0.0)
            l1_ask_qty = self.active_quotes.get('ask_qty', 0.0)
            max_spread = self.rl_config.max_spread
            min_spread = self.rl_config.min_spread

            # 5a. Quote proximity — how close our quotes are to BBO
            # dist uses signed direction: bid below BBO = positive distance (bad),
            # bid above BBO (more competitive) = 0 distance (capped, we don't penalize).
            # Same logic for ask: above BBO = positive distance, below = capped at 0.
            if self.shaping_proximity and mid > 0 and l1_bid > 0 and l1_ask > 0 and best_bid > 0 and best_ask > 0:
                dist_bid = max(0.0, best_bid - l1_bid) / mid  # >0 when our bid is below BBO
                dist_ask = max(0.0, l1_ask - best_ask) / mid  # >0 when our ask is above BBO
                proximity = (max(0.0, 1.0 - dist_bid / max_spread)
                             + max(0.0, 1.0 - dist_ask / max_spread)) / 2.0
                shaping_proximity_r = self.shaping_proximity * proximity

            # 5b. Two-sided presence — quoting both sides
            if self.shaping_presence:
                has_bid = 1.0 if l1_bid_qty > 0 else 0.0
                has_ask = 1.0 if l1_ask_qty > 0 else 0.0
                shaping_presence_r = self.shaping_presence * (has_bid + has_ask) / 2.0

            # 5c. Spread tightness — tighter spread is better
            # Note: calculate_quotes uses max_spread/min_spread as HALF-spread per side,
            # so total quoted spread ranges from min_spread to ~2*max_spread.
            if self.shaping_spread and mid > 0 and l1_bid > 0 and l1_ask > 0:
                quoted_spread = (l1_ask - l1_bid) / mid
                total_min = min_spread          # best case: both sides at min_spread/2
                total_max = 2.0 * max_spread    # worst case: both sides at max_spread
                spread_range = total_max - total_min
                if spread_range > 0:
                    tightness = max(0.0, 1.0 - (quoted_spread - total_min) / spread_range)
                else:
                    tightness = 1.0
                shaping_spread_r = self.shaping_spread * tightness

            reward += shaping_proximity_r + shaping_presence_r + shaping_spread_r

        logger.debug(f"Reward: realized={realized_pnl:.6f}, mtm={mtm_component:.6f}, "
                     f"inv_pen={inventory_penalty:.6f}, "
                     f"as_pen={self._lambda_as * adverse_selection_penalty:.6f}, "
                     f"iVPIN={ivpin_fast:.3f}, fills={fill_qty:.2f}, "
                     f"total_mtm_pnl={pnl_change:.6f}, avg_entry={avg_price:.5f}, "
                     f"shaping=[prox={shaping_proximity_r:.6f}, pres={shaping_presence_r:.6f}, "
                     f"spr={shaping_spread_r:.6f}], total={reward:.6f}")

        return float(reward)
