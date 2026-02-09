import asyncio
import logging
from typing import Dict, Optional, Callable
from datetime import datetime
from collections import deque
from binance.async_client import AsyncClient
from binance.ws.streams import ThreadedWebsocketManager, BinanceSocketManager
import pandas as pd
from core.types import OrderBook, OrderBookLevel, Trade, OrderSide
from core.config import MarketConfig

logger = logging.getLogger(__name__)

class BinanceStreamHandler:
    """Gère les streams de données Binance en temps réel."""
    
    def __init__(self, market_config: MarketConfig, mode: str = 'margin'):
        self.config = market_config
        self.mode = mode  # 'futures', 'spot', or 'margin'
        self.client: Optional[AsyncClient] = None
        self.bm: Optional[BinanceSocketManager] = None

        # Buffers
        self.orderbook = {
            'bids': [], 'asks': [],
            'lastUpdateId': 0
        }
        self.orderbook_timestamp: Optional[datetime] = None
        self.trades_buffer = deque(maxlen=1000)
        self.price_buffer = deque(maxlen=1000)

        # User data stream: pending fills queue
        self.pending_fills: deque = deque(maxlen=100)
        self._listen_key: Optional[str] = None

        # Callbacks
        self.orderbook_callbacks = set()
        self.trade_callbacks = set()

        # State
        self._running = False
        self._lock = asyncio.Lock()

    async def connect(self, api_key: str, api_secret: str):
        """Établit la connexion à Binance."""
        try:
            self._api_key = api_key
            self._api_secret = api_secret
            self.client = await AsyncClient.create(api_key, api_secret)
            self.bm = BinanceSocketManager(self.client, max_queue_size=10000)
            await self.get_initial_state()
            self._running = True
            logger.info(f"Connected to Binance for {self.config.symbol}")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise
            
    async def get_initial_state(self):
        """Récupère l'état initial du carnet d'ordres."""
        if self.mode == 'futures':
            orderbook = await self.client.futures_order_book(
                symbol=self.config.symbol.upper(), limit=20)
        else:
            orderbook = await self.client.get_order_book(
                symbol=self.config.symbol.upper(), limit=20)

        self.orderbook = {
            'lastUpdateId': orderbook['lastUpdateId'],
            'bids': [[float(p), float(q)] for p, q in orderbook['bids'][:10]],
            'asks': [[float(p), float(q)] for p, q in orderbook['asks'][:10]]
        }

    async def start_streams(self):
        """Démarre tous les streams nécessaires."""
        try:
            if self.mode == 'futures':
                # Futures WebSocket streams
                symbol_lower = self.config.symbol.lower()
                depth_stream = self.bm.futures_depth_socket(self.config.symbol, depth=10)
                # Agg trades for trade data
                agg_stream = self.bm.aggtrade_futures_socket(self.config.symbol)
                asyncio.create_task(self._handle_depth_stream(depth_stream))
                asyncio.create_task(self._handle_futures_trade_stream(agg_stream))
            else:
                # Spot/margin WebSocket streams
                depth_stream = self.bm.depth_socket(self.config.symbol, depth=10)
                trades_stream = self.bm.trade_socket(self.config.symbol)
                asyncio.create_task(self._handle_depth_stream(depth_stream))
                asyncio.create_task(self._handle_trade_stream(trades_stream))

            logger.info(f"Started all streams for {self.config.symbol} (mode={self.mode})")
        except Exception as e:
            logger.error(f"Failed to start streams: {e}")
            raise
            
    async def _handle_depth_stream(self, stream):
        """Gère le stream du carnet d'ordres."""
        async with stream as tscm:
            while self._running:
                try:
                    msg = await tscm.recv()
                    # Multiplex streams wrap payload in {'stream': ..., 'data': {...}}
                    if 'data' in msg:
                        msg = msg['data']
                    await self._process_depth_update(msg)
                except Exception as e:
                    logger.error(f"Error in depth stream: {e}")
                    if not self._running:
                        break
                    await asyncio.sleep(1)
                    
    async def _process_depth_update(self, msg):
        """Traite une mise à jour du carnet d'ordres (@depth10 snapshots)."""
        async with self._lock:
            update_id = msg.get('u', msg.get('lastUpdateId', 0))
            if update_id <= self.orderbook['lastUpdateId']:
                logger.debug(f"Depth update skipped: stream u={update_id} <= stored {self.orderbook['lastUpdateId']}")
                return

            # @depth10 sends full snapshots — replace entirely
            self.orderbook['bids'] = [[float(p), float(q)] for p, q in msg['b']]
            self.orderbook['asks'] = [[float(p), float(q)] for p, q in msg['a']]
            self.orderbook['lastUpdateId'] = update_id
            event_time = msg.get('E', None)
            if event_time:
                self.orderbook_timestamp = datetime.fromtimestamp(event_time / 1000)
            else:
                self.orderbook_timestamp = datetime.now()

            # Créer l'objet OrderBook
            orderbook = OrderBook(
                bids=[
                    OrderBookLevel(price=float(bid[0]), quantity=float(bid[1]))
                    for bid in self.orderbook['bids']
                ],
                asks=[
                    OrderBookLevel(price=float(ask[0]), quantity=float(ask[1]))
                    for ask in self.orderbook['asks']
                ],
                timestamp=self.orderbook_timestamp
            )
            
            # Notifier les callbacks
            for callback in self.orderbook_callbacks:
                try:
                    await callback(orderbook)
                except Exception as e:
                    logger.error(f"Error in orderbook callback: {e}")
                    
    def _update_price_level(self, side: str, price: float, quantity: float):
        """Met à jour un niveau de prix dans le carnet d'ordres."""
        book = self.orderbook[side]
        price_idx = None
        
        for i, (p, _) in enumerate(book):
            if abs(p - price) < self.config.tick_size:
                price_idx = i
                break
                
        if quantity == 0 and price_idx is not None:
            book.pop(price_idx)
        elif quantity > 0:
            if price_idx is None:
                book.append([price, quantity])
                # Trier et garder les 10 meilleurs niveaux
                book.sort(reverse=(side=='bids'))
                book = book[:10]
            else:
                book[price_idx][1] = quantity
                
        self.orderbook[side] = book
                    
    async def _handle_trade_stream(self, stream):
        """Gère le stream des trades."""
        async with stream as tscm:
            while self._running:
                try:
                    msg = await tscm.recv()
                    await self._process_trade(msg)
                except Exception as e:
                    logger.error(f"Error in trade stream: {e}")
                    if not self._running:
                        break
                    await asyncio.sleep(1)
                    
    async def _process_trade(self, msg):
        """Traite un nouveau trade."""
        is_buyer_maker = msg['m']
        side = OrderSide.SELL if is_buyer_maker else OrderSide.BUY

        trade = Trade(
            price=float(msg['p']),
            quantity=float(msg['q']),
            timestamp=datetime.fromtimestamp(msg['T'] / 1000),
            side=side,
            is_buyer_maker=is_buyer_maker,
            commission=float(msg.get('n', 0)),
            commission_asset=msg.get('N', '')
        )
        
        self.trades_buffer.append(trade)
        
        # Notifier les callbacks
        for callback in self.trade_callbacks:
            try:
                await callback(trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
                
    async def _handle_futures_trade_stream(self, stream):
        """Handle futures aggregate trade stream."""
        async with stream as tscm:
            while self._running:
                try:
                    msg = await tscm.recv()
                    # Multiplex streams wrap payload in {'stream': ..., 'data': {...}}
                    if 'data' in msg:
                        msg = msg['data']
                    if msg.get('e') == 'aggTrade':
                        is_buyer_maker = msg.get('m', False)
                        side = OrderSide.SELL if is_buyer_maker else OrderSide.BUY
                        trade = Trade(
                            price=float(msg['p']),
                            quantity=float(msg['q']),
                            timestamp=datetime.fromtimestamp(msg['T'] / 1000),
                            side=side,
                            is_buyer_maker=is_buyer_maker,
                            commission=0.0,
                        )
                        self.trades_buffer.append(trade)
                        for callback in self.trade_callbacks:
                            try:
                                await callback(trade)
                            except Exception as e:
                                logger.error(f"Error in trade callback: {e}")
                except Exception as e:
                    logger.error(f"Error in futures trade stream: {e}")
                    if not self._running:
                        break
                    await asyncio.sleep(1)

    def add_orderbook_callback(self, callback: Callable):
        """Ajoute un callback pour les mises à jour du carnet d'ordres."""
        self.orderbook_callbacks.add(callback)

    def add_trade_callback(self, callback: Callable):
        """Ajoute un callback pour les trades."""
        self.trade_callbacks.add(callback)

    async def start_user_data_stream(self):
        """Start user data stream for fill detection (futures or margin)."""
        try:
            if self.mode == 'futures':
                # Futures listen key
                response = await self.client.futures_stream_get_listen_key()
                self._listen_key = response
                logger.info(f"Futures listen key created: {self._listen_key[:8]}...")
                user_stream = self.bm.futures_user_socket()
            else:
                # Margin listen key
                response = await self.client.margin_stream_get_listen_key()
                self._listen_key = response
                logger.info(f"Margin listen key created: {self._listen_key[:8]}...")
                user_stream = self.bm.margin_socket(self._listen_key)

            asyncio.create_task(self._handle_user_stream(user_stream))
            asyncio.create_task(self._keepalive_listen_key())
            logger.info("User data stream started")
        except Exception as e:
            logger.error(f"Failed to start user data stream: {e}")
            logger.warning("Falling back to order polling for fill detection")

    async def _handle_user_stream(self, stream):
        """Handle user data stream events (fills)."""
        async with stream as tscm:
            while self._running:
                try:
                    msg = await tscm.recv()
                    event = msg.get('e', '')
                    if event == 'executionReport':
                        # Spot/Margin fill
                        await self._process_execution_report(msg)
                    elif event == 'ORDER_TRADE_UPDATE':
                        # Futures fill
                        await self._process_futures_order_update(msg)
                except Exception as e:
                    logger.error(f"Error in user data stream: {e}")
                    if not self._running:
                        break
                    await asyncio.sleep(1)

    async def _process_execution_report(self, msg):
        """Process an execution report from spot/margin user data stream."""
        order_status = msg.get('X', '')
        exec_type = msg.get('x', '')

        if exec_type == 'TRADE':
            fill = {
                'order_id': msg.get('i'),
                'client_order_id': msg.get('c', ''),
                'symbol': msg.get('s', ''),
                'side': msg.get('S', ''),
                'price': float(msg.get('L', 0)),
                'quantity': float(msg.get('l', 0)),
                'commission': float(msg.get('n', 0)),
                'commission_asset': msg.get('N', ''),
                'order_status': order_status,
                'timestamp': datetime.fromtimestamp(msg.get('T', 0) / 1000),
            }
            self.pending_fills.append(fill)
            logger.info(f"Fill detected: {fill['side']} {fill['quantity']} @ {fill['price']} "
                       f"(order {fill['order_id']}, status={order_status})")

    async def _process_futures_order_update(self, msg):
        """Process a futures ORDER_TRADE_UPDATE event."""
        o = msg.get('o', {})
        exec_type = o.get('x', '')  # TRADE, NEW, CANCELED, etc.

        if exec_type == 'TRADE':
            fill = {
                'order_id': o.get('i'),
                'client_order_id': o.get('c', ''),
                'symbol': o.get('s', ''),
                'side': o.get('S', ''),
                'price': float(o.get('L', 0)),  # Last filled price
                'quantity': float(o.get('l', 0)),  # Last filled qty
                'commission': float(o.get('n', 0)),
                'commission_asset': o.get('N', ''),
                'order_status': o.get('X', ''),
                'timestamp': datetime.fromtimestamp(msg.get('T', 0) / 1000),
            }
            self.pending_fills.append(fill)
            logger.info(f"Futures fill: {fill['side']} {fill['quantity']} @ {fill['price']} "
                       f"(order {fill['order_id']})")

    def get_pending_fills(self):
        """Return and clear the queue of pending fills."""
        fills = list(self.pending_fills)
        self.pending_fills.clear()
        return fills

    async def _keepalive_listen_key(self):
        """Keep the listen key alive (must be refreshed every 30min)."""
        while self._running:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                if self._listen_key and self.client:
                    if self.mode == 'futures':
                        await self.client.futures_stream_keepalive(self._listen_key)
                        logger.debug("Futures listen key renewed")
                    else:
                        await self.client.margin_stream_keepalive(self._listen_key)
                        logger.debug("Margin listen key renewed")
            except Exception as e:
                logger.error(f"Error renewing listen key: {e}")

    async def reconnect_streams(self):
        """Kill all streams, re-fetch snapshot, and restart."""
        logger.warning("Reconnecting streams...")
        # 1. Signal current stream tasks to stop
        self._running = False
        await asyncio.sleep(0.5)  # Let stream loops exit

        # 2. Save old client config before closing
        old_api_url = getattr(self.client, 'API_URL', None) if self.client else None
        old_testnet = getattr(self.client, 'testnet', False) if self.client else False

        # 3. Close the old client connection (kills websockets)
        if self.client:
            try:
                await self.client.close_connection()
            except Exception:
                pass

        # 4. Re-create client + socket manager
        api_key = getattr(self, '_api_key', '')
        api_secret = getattr(self, '_api_secret', '')
        if not api_key:
            logger.error("Cannot reconnect: no API credentials available")
            return False

        try:
            self.client = await AsyncClient.create(api_key, api_secret, testnet=old_testnet)
            # Restore the exact API URL (futures testnet, spot testnet, etc.)
            if old_api_url:
                self.client.API_URL = old_api_url
            self.bm = BinanceSocketManager(self.client, max_queue_size=10000)

            # 5. Reset orderbook state so lastUpdateId doesn't filter new messages
            self.orderbook = {'bids': [], 'asks': [], 'lastUpdateId': 0}
            self.orderbook_timestamp = None

            # 6. Fresh REST snapshot + restart WS streams
            await self.get_initial_state()
            self._running = True
            await self.start_streams()

            # 7. Restart user data stream
            try:
                await self.start_user_data_stream()
            except Exception as e:
                logger.warning(f"User data stream not available after reconnect: {e}")

            logger.info("Streams reconnected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reconnect streams: {e}")
            self._running = False
            return False

    async def stop(self):
        """Arrête tous les streams proprement."""
        self._running = False
        # Close listen key if active
        if self._listen_key and self.client:
            try:
                if self.mode == 'futures':
                    await self.client.futures_stream_close(self._listen_key)
                else:
                    await self.client.margin_stream_close(self._listen_key)
            except Exception:
                pass
        if self.bm:
            # BinanceSocketManager doesn't have close(), streams are closed when client closes
            pass
        if self.client:
            await self.client.close_connection()

        logger.info("Stopped all streams")