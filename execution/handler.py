import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from binance import AsyncClient
from core.config import MarketConfig, RLConfig
from core.types import Order, OrderType, OrderSide, OrderStatus
from core.exceptions import OrderExecutionError, NetworkError
from environment.base_environment import BaseMarketMakerEnv
from data.processor import MarketFeatureProcessor
import vectorbt as vbt

logger = logging.getLogger(__name__)

class OrderExecutor:
    def __init__(self, 
                 client: Optional[AsyncClient],
                 market_config: MarketConfig,
                 rl_config: RLConfig,
                 market_env: BaseMarketMakerEnv,
                 market_feature_processor: Optional[MarketFeatureProcessor],
                 simulation: bool = True):
        self.client = client
        self.market_config = market_config
        self.rl_config = rl_config
        self.simulation = simulation
        self.market_env = market_env
        self.market_feature_processor = market_feature_processor
        self.current_orderbook = None
        self.current_state = None
        self.current_action = np.array([0.0, 0.0])
        self._active_orders: Dict[str, Order] = {}
        self._order_lock = asyncio.Lock()
        self.portfolio = None
        self.current_mid_price = None
        self._last_order_time = None
        self._order_count = 0

    async def place_quotes(self, quantity: float) -> Dict[str, Order]:
        try:
            logger.info("Placing bid and ask quotes with quantity: %s", quantity)
            if self.market_feature_processor.volatility is None:
                logger.warning("Not enough data to calculate volatility. Orders will not be placed.")
                return None
            bid_price = self.market_env.bid_price
            ask_price = self.market_env.ask_price
            logger.debug("Bid price: %s, Ask price: %s", bid_price, ask_price)

            bid_order = await self._place_limit_order(
                side=OrderSide.BUY,
                price=bid_price,
                quantity=quantity
            )
            
            ask_order = await self._place_limit_order(
                side=OrderSide.SELL,
                price=ask_price,
                quantity=quantity
            )
            
            return {'bid': bid_order, 'ask': ask_order}
            
        except Exception as e:
            logger.error("Error placing quotes: %s", e)
            raise OrderExecutionError(f"Failed to place quotes: {str(e)}")

    async def _place_limit_order(self, side: OrderSide, price: float, quantity: float) -> Order:
        logger.info("Placing %s limit order at price: %s with quantity: %s", side, price, quantity)
        if self.simulation:
            return await self._simulate_order_execution(side, price, quantity)
        
        try:
            price = self._round_price(price)
            quantity = self._round_quantity(quantity)
            logger.debug("Rounded price: %s, Rounded quantity: %s", price, quantity)
            response = await self.client.create_margin_order(
                symbol=self.market_config.symbol,
                side=side.value,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price
            )

            order = Order(
                symbol=self.market_config.symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=float(quantity),
                price=float(price),
                status=OrderStatus.NEW,
                timestamp=datetime.fromtimestamp(response['transactTime'] / 1000),
                order_id=response['orderId'],
                client_order_id=response['clientOrderId']
            )
            self._last_order_time = datetime.now()
            self._order_count += 1
            logger.info("Order placed: %s", order)
            return order
            
        except Exception as e:
            logger.error("Error placing limit order: %s", e)
            raise OrderExecutionError(f"Failed to place limit order: {str(e)}")

    async def _simulate_order_execution(self, side: OrderSide, price: float, quantity: float) -> Order:
        logger.info("Simulating order execution for %s at price %s with quantity %s", side, price, quantity)
        execution_probability = self._calculate_execution_probability(side)
        logger.debug("Calculated execution probability for %s: %s", side, execution_probability)

        if np.random.rand() < execution_probability:
            status = OrderStatus.FILLED
            executed_qty = quantity
            logger.info("Simulated order filled for %s with quantity %s", side, quantity)
        else:
            status = OrderStatus.NEW  
            executed_qty = 0.0
            logger.info("Simulated order not filled for %s", side)

        order = Order(
            symbol=self.market_config.symbol,
            side=side,
            order_type=OrderType.LIMIT, 
            quantity=quantity,
            price=price,
            status=status,
            timestamp=datetime.now(),
            order_id=f"SIM-{int(datetime.now().timestamp())}-{side.value}",
            client_order_id="SIM"
        )

        if status == OrderStatus.FILLED:
            direction = 1 if side == OrderSide.BUY else -1
            self._update_vectorbt_portfolio(price, direction * executed_qty)

        return order

    def _calculate_execution_probability(self, side: OrderSide) -> float:
        try:
            mid_price = self.market_feature_processor.mid_price
            volatility = max(self.market_feature_processor.volatility, 0.0001)
            price = self.market_env.bid_price if side == OrderSide.BUY else self.market_env.ask_price
            distance = abs(price - mid_price)
            poisson_lambda = distance / (volatility * mid_price)
            execution_probability = 1 - np.exp(-poisson_lambda)
            execution_probability = np.clip(execution_probability, 0, 1)
            
            logger.debug(
                "Execution probability calculated - Side: %s, Price: %.8f, Mid Price: %.8f, "
                "Distance: %.8f, Volatility: %.8f, Lambda: %.4f, Probability: %.4f",
                side, price, mid_price, distance, volatility, poisson_lambda, execution_probability
            )
            
            return execution_probability
            
        except Exception as e:
            logger.error("Error calculating execution probability: %s", e)
            raise
        
    def _update_vectorbt_portfolio(self, price: float, size: float):
        try:
            timestamp = datetime.now()
            price_series = pd.Series([price], index=[timestamp])
            size_series = pd.Series([size], index=[timestamp])

            if not hasattr(self, 'portfolio') or self.portfolio is None:
                logger.info("Initializing vectorbt portfolio with initial cash.")
                self.portfolio = vbt.Portfolio.from_orders(
                    close=price_series,
                    size=size,
                    init_cash=self.rl_config.initial_cash,
                    fees=self.market_config.fee_rate if hasattr(self.market_config, 'fee_rate') else 0.001,
                    freq='T'
                )
            else:
                logger.info("Updating existing vectorbt portfolio with new order data.")
                self.portfolio = vbt.Portfolio.from_orders(
                    close=pd.concat([self.portfolio.close, price_series]),
                    size=size,
                    init_cash=self.portfolio.init_cash,
                    fees=self.market_config.fee_rate if hasattr(self.market_config, 'fee_rate') else 0.001,
                    freq='T'
                )
        except Exception as e:
            logger.error("Error updating vectorbt portfolio: %s", e)
            raise

    async def cancel_all_orders(self):
        if self.simulation:
            logger.info("Cancelling all simulated orders.")
            self._active_orders.clear()
            return
            
        try:
            async with self._order_lock:
                open_orders = await self.client.get_open_margin_orders(symbol=self.market_config.symbol)
                for order in open_orders:
                    try:
                        await self.client.cancel_margin_order(symbol=self.market_config.symbol, orderId=order['orderId'])
                        logger.debug("Cancelled order %s", order['orderId'])
                    except Exception as e:
                        logger.warning("Error cancelling order %s: %s", order['orderId'], e)
                self._active_orders.clear()
        except Exception as e:
            logger.error("Error cancelling all orders: %s", e)
            raise OrderExecutionError(f"Failed to cancel orders: {str(e)}")

    def _validate_quotes(self, bid_price: float, ask_price: float, quantity: float) -> bool:
        try:
            spread = (ask_price - bid_price) / bid_price
            if spread < self.rl_config.min_spread or spread > self.rl_config.max_spread:
                logger.warning("Invalid spread: %s", spread)
                return False

            min_price = bid_price * 0.8
            max_price = ask_price * 1.2
            if not (min_price <= bid_price <= max_price and min_price <= ask_price <= max_price):
                logger.warning("Prices out of range: bid=%s, ask=%s", bid_price, ask_price)
                return False

            if quantity < self.market_config.min_qty:
                logger.warning("Quantity too small: %s", quantity)
                return False

            notional_value = min(bid_price, ask_price) * quantity
            if notional_value < self.market_config.min_notional:
                logger.warning("Notional value too small: %s", notional_value)
                return False

            logger.debug("Quotes validated for bid_price=%s, ask_price=%s, quantity=%s", bid_price, ask_price, quantity)
            return True

        except Exception as e:
            logger.error("Error validating quotes: %s", e)
            return False

    def _round_price(self, price: float) -> float:
        tick = self.market_config.tick_size
        return round(round(price / tick) * tick, 8)

    def _round_quantity(self, qty: float) -> float:
        return round(qty)


class LiveOrderExecutor:
    """Multi-level order executor for live Binance trading.

    Manages 3 bid + 3 ask limit orders with HOLD logic for queue priority.
    Supports futures, spot, and margin endpoints via `mode` parameter.
    """

    def __init__(self,
                 client: AsyncClient,
                 market_config: MarketConfig,
                 rl_config: RLConfig,
                 dry_run: bool = False,
                 use_spot: bool = False,
                 mode: str = 'futures'):
        self.client = client
        self.market_config = market_config
        self.rl_config = rl_config
        self.dry_run = dry_run
        self.mode = mode  # 'futures', 'spot', or 'margin'
        # Legacy compat
        if use_spot:
            self.mode = 'spot'

        # Track orders per level: 3 bids, 3 asks
        self._bid_orders: List[Optional[Order]] = [None, None, None]
        self._ask_orders: List[Optional[Order]] = [None, None, None]
        self._order_lock = asyncio.Lock()
        self._order_count = 0

        # Quantity formatting precision (derived from min_qty / lot_size)
        self._qty_precision = max(0, len(str(market_config.min_qty).rstrip('0').split('.')[-1])) if '.' in str(market_config.min_qty) else 0

    def _round_price(self, price: float) -> float:
        tick = self.market_config.tick_size
        return round(round(price / tick) * tick, 8)

    def _round_quantity(self, qty: float) -> float:
        step = self.market_config.min_qty  # lot_size / step_size
        return max(0, round(round(qty / step) * step, 8))

    def _format_qty(self, qty: float) -> str:
        """Format quantity string with correct precision."""
        if self._qty_precision == 0:
            return str(int(qty))
        return f"{qty:.{self._qty_precision}f}"

    async def update_multi_level_orders(self, quotes: Dict) -> Dict:
        """Update all 6 limit orders based on multi-level quotes from calculate_quotes().

        Args:
            quotes: Dict with 'levels' (list of 3 level dicts), 'flatten_qty', 'flatten_side'

        Returns:
            Dict with active order info.
        """
        async with self._order_lock:
            levels = quotes.get('levels', [])

            for i, level in enumerate(levels[:3]):
                # --- BID ---
                bid_qty = level.get('bid_qty', 0)
                bid_price = level.get('bid_price', 0)
                bid_age = level.get('bid_age', 0)

                if bid_qty > 0 and bid_price > 0:
                    self._bid_orders[i] = await self._handle_level_order(
                        OrderSide.BUY, bid_price, bid_qty, bid_age, self._bid_orders[i]
                    )
                else:
                    if self._bid_orders[i] is not None:
                        await self._cancel_order(self._bid_orders[i])
                        self._bid_orders[i] = None

                # --- ASK ---
                ask_qty = level.get('ask_qty', 0)
                ask_price = level.get('ask_price', 0)
                ask_age = level.get('ask_age', 0)

                if ask_qty > 0 and ask_price > 0:
                    self._ask_orders[i] = await self._handle_level_order(
                        OrderSide.SELL, ask_price, ask_qty, ask_age, self._ask_orders[i]
                    )
                else:
                    if self._ask_orders[i] is not None:
                        await self._cancel_order(self._ask_orders[i])
                        self._ask_orders[i] = None

            # --- FLATTEN (market order) ---
            flatten_qty = quotes.get('flatten_qty', 0)
            flatten_side = quotes.get('flatten_side')
            if flatten_qty > 0 and flatten_side:
                side = OrderSide.BUY if flatten_side == 'buy' else OrderSide.SELL
                await self.place_market_order(side, flatten_qty)

            return self._get_active_orders_info()

    async def _handle_level_order(self, side: OrderSide, price: float, qty: float,
                                   age: int, existing_order: Optional[Order]) -> Optional[Order]:
        """Handle a single level: HOLD if price unchanged, otherwise cancel+replace."""
        price = self._round_price(price)
        qty = self._round_quantity(qty)

        if qty < self.market_config.min_qty:
            if existing_order:
                await self._cancel_order(existing_order)
            return None

        # Validate min notional
        if price * qty < self.market_config.min_notional:
            if existing_order:
                await self._cancel_order(existing_order)
            return None

        # HOLD logic: if age > 0 and existing order price is close, keep it
        tick = self.market_config.tick_size
        if (age > 0
                and existing_order is not None
                and existing_order.status == OrderStatus.NEW
                and abs(existing_order.price - price) < tick * 1.5):
            logger.debug("HOLD %s order at %.5f (age=%d)", side.value, existing_order.price, age)
            return existing_order

        # Cancel existing and place new
        if existing_order is not None:
            await self._cancel_order(existing_order)

        return await self._place_limit_order(side, price, qty)

    async def _place_limit_order(self, side: OrderSide, price: float, qty: float) -> Optional[Order]:
        """Place a limit GTC order (futures, spot, or margin)."""
        price_str = f"{price:.{self._price_precision()}f}"
        qty_str = self._format_qty(qty)

        if self.dry_run:
            self._order_count += 1
            order = Order(
                symbol=self.market_config.symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=float(qty),
                price=float(price),
                status=OrderStatus.NEW,
                timestamp=datetime.now(),
                order_id=f"DRY-{self._order_count}-{side.value}",
                client_order_id=f"DRY-{self._order_count}"
            )
            logger.info("[DRY RUN] Would place %s LIMIT %s @ %s", side.value, qty_str, price_str)
            return order

        try:
            if self.mode == 'futures':
                response = await self.client.futures_create_order(
                    symbol=self.market_config.symbol,
                    side=side.value,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=qty_str,
                    price=price_str
                )
            elif self.mode == 'spot':
                response = await self.client.create_order(
                    symbol=self.market_config.symbol,
                    side=side.value,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=qty_str,
                    price=price_str
                )
            else:  # margin
                response = await self.client.create_margin_order(
                    symbol=self.market_config.symbol,
                    side=side.value,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=qty_str,
                    price=price_str
                )
            ts = response.get('transactTime') or response.get('updateTime', 0)
            order = Order(
                symbol=self.market_config.symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=float(qty),
                price=float(price),
                status=OrderStatus.NEW,
                timestamp=datetime.fromtimestamp(ts / 1000) if ts else datetime.now(),
                order_id=str(response['orderId']),
                client_order_id=response.get('clientOrderId', '')
            )
            self._order_count += 1
            logger.info("Placed %s LIMIT %s @ %s (id=%s)", side.value, qty_str, price_str, order.order_id)
            return order
        except Exception as e:
            logger.error("Failed to place %s limit order: %s", side.value, e)
            return None

    async def place_market_order(self, side: OrderSide, qty: float) -> Optional[Order]:
        """Place a market order for flatten/emergency."""
        qty = self._round_quantity(qty)
        if qty < self.market_config.min_qty:
            return None

        qty_str = self._format_qty(qty)

        if self.dry_run:
            logger.info("[DRY RUN] Would place %s MARKET %s", side.value, qty_str)
            return None

        try:
            if self.mode == 'futures':
                response = await self.client.futures_create_order(
                    symbol=self.market_config.symbol,
                    side=side.value,
                    type='MARKET',
                    quantity=qty_str
                )
            elif self.mode == 'spot':
                response = await self.client.create_order(
                    symbol=self.market_config.symbol,
                    side=side.value,
                    type='MARKET',
                    quantity=qty_str
                )
            else:  # margin
                response = await self.client.create_margin_order(
                    symbol=self.market_config.symbol,
                    side=side.value,
                    type='MARKET',
                    quantity=qty_str
                )
            avg_price = float(response.get('avgPrice', 0))
            if not avg_price:
                avg_price = float(response.get('fills', [{}])[0].get('price', 0))
            ts = response.get('transactTime') or response.get('updateTime', 0)
            order = Order(
                symbol=self.market_config.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=float(qty),
                price=avg_price,
                status=OrderStatus.FILLED,
                timestamp=datetime.fromtimestamp(ts / 1000) if ts else datetime.now(),
                order_id=str(response['orderId']),
                client_order_id=response.get('clientOrderId', '')
            )
            logger.info("Placed %s MARKET %s (id=%s)", side.value, qty_str, order.order_id)
            return order
        except Exception as e:
            logger.error("Failed to place %s market order: %s", side.value, e)
            return None

    async def cancel_all_orders(self):
        """Cancel all open orders and reset tracking."""
        async with self._order_lock:
            if not self.dry_run:
                try:
                    open_orders = await self.get_open_orders()
                    for order_data in open_orders:
                        try:
                            await self._cancel_order_by_id(order_data['orderId'])
                            logger.debug("Cancelled order %s", order_data['orderId'])
                        except Exception as e:
                            logger.warning("Error cancelling order %s: %s", order_data['orderId'], e)
                except Exception as e:
                    logger.error("Error fetching open orders for cancellation: %s", e)

            self._bid_orders = [None, None, None]
            self._ask_orders = [None, None, None]
            logger.info("All orders cancelled and tracking reset")

    async def _cancel_order_by_id(self, order_id):
        """Cancel a single order by ID using the correct endpoint."""
        if self.mode == 'futures':
            await self.client.futures_cancel_order(
                symbol=self.market_config.symbol, orderId=order_id)
        elif self.mode == 'spot':
            await self.client.cancel_order(
                symbol=self.market_config.symbol, orderId=order_id)
        else:
            await self.client.cancel_margin_order(
                symbol=self.market_config.symbol, orderId=order_id)

    async def _cancel_order(self, order: Order):
        """Cancel a single order."""
        if self.dry_run:
            logger.debug("[DRY RUN] Would cancel order %s", order.order_id)
            return

        try:
            await self._cancel_order_by_id(order.order_id)
            logger.debug("Cancelled order %s", order.order_id)
        except Exception as e:
            # Order may already be filled or cancelled
            logger.debug("Could not cancel order %s: %s", order.order_id, e)

    async def get_open_orders(self) -> List[Dict]:
        """Get currently open orders from Binance."""
        if self.dry_run:
            return []
        try:
            if self.mode == 'futures':
                return await self.client.futures_get_open_orders(
                    symbol=self.market_config.symbol)
            elif self.mode == 'spot':
                return await self.client.get_open_orders(
                    symbol=self.market_config.symbol)
            else:
                return await self.client.get_open_margin_orders(
                    symbol=self.market_config.symbol)
        except Exception as e:
            logger.error("Error fetching open orders: %s", e)
            return []

    def _get_active_orders_info(self) -> Dict:
        """Return summary of active orders."""
        active = {}
        for i, order in enumerate(self._bid_orders):
            if order:
                active[f'bid_L{i}'] = {'price': order.price, 'qty': order.quantity, 'id': order.order_id}
        for i, order in enumerate(self._ask_orders):
            if order:
                active[f'ask_L{i}'] = {'price': order.price, 'qty': order.quantity, 'id': order.order_id}
        return active

    def _price_precision(self) -> int:
        """Determine price precision from tick_size."""
        tick_str = f"{self.market_config.tick_size:.10f}".rstrip('0')
        if '.' in tick_str:
            return len(tick_str.split('.')[1])
        return 0
