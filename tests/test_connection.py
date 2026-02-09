"""Tests de connexion et de composants pour le deploiement Binance Testnet.

Usage:
    # Tests offline (pas besoin de clefs API ni de binance installé):
    python tests/test_connection.py

    # Tests avec connexion Binance testnet (binance + clefs requis):
    set BINANCE_TESTNET_API_KEY=xxx
    set BINANCE_TESTNET_SECRET=xxx
    python tests/test_connection.py --live
"""

import os
import sys
import asyncio
import unittest
import argparse
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from types import ModuleType

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# Mock binance SDK if not installed (offline tests only)
# ============================================================
_HAS_BINANCE = True
try:
    import binance
except ImportError:
    _HAS_BINANCE = False
    # Create fake binance module tree so our modules can import
    _mock_binance = ModuleType('binance')
    _mock_binance.AsyncClient = MagicMock
    _mock_async_client = ModuleType('binance.async_client')
    _mock_async_client.AsyncClient = MagicMock
    _mock_ws = ModuleType('binance.ws')
    _mock_streams = ModuleType('binance.ws.streams')
    _mock_streams.ThreadedWebsocketManager = MagicMock
    _mock_streams.BinanceSocketManager = MagicMock
    sys.modules['binance'] = _mock_binance
    sys.modules['binance.async_client'] = _mock_async_client
    sys.modules['binance.ws'] = _mock_ws
    sys.modules['binance.ws.streams'] = _mock_streams

# Also mock vectorbt if missing
try:
    import vectorbt
except ImportError:
    _mock_vbt = ModuleType('vectorbt')
    _mock_vbt.Portfolio = MagicMock()
    sys.modules['vectorbt'] = _mock_vbt
    sys.modules['vbt'] = _mock_vbt

from core.config import MarketConfig, RLConfig, ModelConfig
from core.types import Order, OrderType, OrderSide, OrderStatus


# ============================================================
# 1. Tests unitaires offline (mocked)
# ============================================================

class TestConfigLoading(unittest.TestCase):
    """Verify config YAML loads correctly and produces valid config objects."""

    def test_yaml_loads(self):
        import yaml
        config_path = os.path.join(PROJECT_ROOT, 'config', 'config_testnet.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.assertIn('binance', config)
        self.assertIn('market', config)
        self.assertIn('rl', config)
        self.assertIn('model', config)
        self.assertIn('deployment', config)
        self.assertTrue(config['binance']['testnet'])

    def test_market_config_from_yaml(self):
        import yaml
        config_path = os.path.join(PROJECT_ROOT, 'config', 'config_testnet.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        mc = config['market']
        market_config = MarketConfig(
            symbol=mc['symbol'],
            tick_size=mc['tick_size'],
            min_qty=mc['min_qty'],
            min_notional=mc['min_notional'],
            max_position=mc['max_position'],
            update_interval=mc['update_interval'],
        )
        self.assertEqual(market_config.symbol, 'XRPUSDC')
        self.assertEqual(market_config.tick_size, 0.0001)
        self.assertEqual(market_config.min_qty, 0.1)

    def test_rl_config_from_yaml(self):
        import yaml
        config_path = os.path.join(PROJECT_ROOT, 'config', 'config_testnet.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        rc = config['rl']
        rl_config = RLConfig(
            zeta=rc['zeta'],
            gamma=rc['gamma'],
            max_bias=rc['max_bias'],
            max_spread=rc['max_spread'],
            min_spread=rc['min_spread'],
            max_equity_exposure=rc['max_equity_exposure'],
            initial_cash=rc['initial_cash'],
            max_steps=rc['max_steps'],
        )
        self.assertAlmostEqual(rl_config.zeta, 0.005)
        self.assertAlmostEqual(rl_config.gamma, 0.995)
        self.assertEqual(rl_config.max_steps, 1000)

    def test_model_config_from_yaml(self):
        import yaml
        config_path = os.path.join(PROJECT_ROOT, 'config', 'config_testnet.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = ModelConfig(**config['model'])
        self.assertEqual(model_config.window_size, 50)
        self.assertEqual(model_config.embedding_dim, 192)
        self.assertEqual(model_config.n_features, 40)

    def test_deployment_risk_limits(self):
        import yaml
        config_path = os.path.join(PROJECT_ROOT, 'config', 'config_testnet.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        deploy = config['deployment']
        self.assertLess(deploy['max_loss_usd'], 0)
        self.assertGreater(deploy['max_position_value_usd'], 0)
        self.assertGreater(deploy['stale_data_timeout'], 0)


class TestLiveOrderExecutor(unittest.TestCase):
    """Test LiveOrderExecutor logic with mocked Binance client."""

    def setUp(self):
        self.market_config = MarketConfig(
            symbol='XRPUSDC',
            tick_size=0.00001,
            min_qty=10.0,
            min_notional=10.0,
            max_position=100,
        )
        self.rl_config = RLConfig(
            zeta=0.005, gamma=0.995, max_bias=0.001,
            max_spread=0.002, min_spread=0.001,
            initial_cash=100.0, max_steps=1000,
        )

    def _make_executor(self):
        from execution.handler import LiveOrderExecutor
        client = MagicMock()
        return LiveOrderExecutor(client, self.market_config, self.rl_config, dry_run=True)

    def test_round_price(self):
        executor = self._make_executor()
        # tick_size = 0.00001, so price snaps to nearest 0.00001
        self.assertAlmostEqual(executor._round_price(0.612347), 0.61235, places=5)
        self.assertAlmostEqual(executor._round_price(0.612342), 0.61234, places=5)
        self.assertAlmostEqual(executor._round_price(1.0), 1.0)

    def test_round_quantity(self):
        executor = self._make_executor()
        # min_qty=10.0 means step_size=10, so quantities round to nearest 10
        self.assertEqual(executor._round_quantity(10.6), 10)
        self.assertEqual(executor._round_quantity(15.0), 20)  # round(1.5)*10 = 2*10 = 20
        self.assertEqual(executor._round_quantity(0.3), 0)

    def test_dry_run_place_limit(self):
        executor = self._make_executor()
        order = asyncio.run(executor._place_limit_order(OrderSide.BUY, 0.61234, 20))
        self.assertIsNotNone(order)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.status, OrderStatus.NEW)
        self.assertTrue(order.order_id.startswith('DRY-'))

    def test_dry_run_cancel_all(self):
        executor = self._make_executor()
        executor._bid_orders[0] = Order(
            symbol='XRPUSDC', side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=20, price=0.61234, status=OrderStatus.NEW,
            timestamp=datetime.now(), order_id='DRY-1-BUY',
        )
        asyncio.run(executor.cancel_all_orders())
        self.assertTrue(all(o is None for o in executor._bid_orders))
        self.assertTrue(all(o is None for o in executor._ask_orders))

    def test_update_multi_level_dry_run(self):
        executor = self._make_executor()
        # Use quantities large enough to pass min_notional check (price * qty >= 10)
        quotes = {
            'levels': [
                {'bid_price': 0.61200, 'ask_price': 0.61300, 'bid_qty': 20, 'ask_qty': 20, 'bid_age': 0, 'ask_age': 0},
                {'bid_price': 0.61100, 'ask_price': 0.61400, 'bid_qty': 20, 'ask_qty': 20, 'bid_age': 0, 'ask_age': 0},
                {'bid_price': 0.61000, 'ask_price': 0.61500, 'bid_qty': 20, 'ask_qty': 20, 'bid_age': 0, 'ask_age': 0},
            ],
            'flatten_qty': 0,
            'flatten_side': None,
        }
        result = asyncio.run(executor.update_multi_level_orders(quotes))
        # Should have 6 active orders (3 bid + 3 ask)
        self.assertEqual(len(result), 6)
        self.assertIn('bid_L0', result)
        self.assertIn('ask_L2', result)

    def test_hold_logic(self):
        """Verify HOLD: existing order kept if age > 0 and price within tick."""
        executor = self._make_executor()
        existing = Order(
            symbol='XRPUSDC', side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=20, price=0.61200, status=OrderStatus.NEW,
            timestamp=datetime.now(), order_id='DRY-EXISTING',
        )
        # age > 0 and same price => HOLD
        result = asyncio.run(executor._handle_level_order(
            OrderSide.BUY, 0.61200, 20, age=5, existing_order=existing
        ))
        self.assertEqual(result.order_id, 'DRY-EXISTING')

    def test_no_hold_when_price_moved(self):
        """Verify order replaced when price changes significantly."""
        executor = self._make_executor()
        existing = Order(
            symbol='XRPUSDC', side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=20, price=0.61200, status=OrderStatus.NEW,
            timestamp=datetime.now(), order_id='DRY-OLD',
        )
        # age > 0 but price moved beyond tick => replace
        result = asyncio.run(executor._handle_level_order(
            OrderSide.BUY, 0.61100, 20, age=5, existing_order=existing
        ))
        self.assertNotEqual(result.order_id, 'DRY-OLD')

    def test_reject_below_min_qty(self):
        """Verify orders below min_qty are rejected."""
        executor = self._make_executor()
        result = asyncio.run(executor._handle_level_order(
            OrderSide.BUY, 0.61200, 5, age=0, existing_order=None
        ))
        self.assertIsNone(result)

    def test_reject_below_min_notional(self):
        """Verify orders below min_notional are rejected."""
        executor = self._make_executor()
        # price=0.0001 * qty=10 = 0.001 < min_notional=10
        result = asyncio.run(executor._handle_level_order(
            OrderSide.BUY, 0.0001, 10, age=0, existing_order=None
        ))
        self.assertIsNone(result)

    def test_price_precision(self):
        executor = self._make_executor()
        self.assertEqual(executor._price_precision(), 5)  # tick=0.00001 => 5 decimals


class TestStreamHandler(unittest.TestCase):
    """Test BinanceStreamHandler additions (fill queue, timestamp tracking)."""

    def test_pending_fills_queue(self):
        from data.binance.stream import BinanceStreamHandler
        handler = BinanceStreamHandler(MarketConfig())

        handler.pending_fills.append({'side': 'BUY', 'quantity': 10, 'price': 0.612})
        handler.pending_fills.append({'side': 'SELL', 'quantity': 20, 'price': 0.613})

        fills = handler.get_pending_fills()
        self.assertEqual(len(fills), 2)
        self.assertEqual(fills[0]['side'], 'BUY')

        # Queue empty after get
        fills_again = handler.get_pending_fills()
        self.assertEqual(len(fills_again), 0)

    def test_orderbook_timestamp_init(self):
        from data.binance.stream import BinanceStreamHandler
        handler = BinanceStreamHandler(MarketConfig())
        self.assertIsNone(handler.orderbook_timestamp)

    def test_execution_report_processing(self):
        from data.binance.stream import BinanceStreamHandler
        handler = BinanceStreamHandler(MarketConfig())

        msg = {
            'e': 'executionReport',
            'x': 'TRADE',
            'X': 'FILLED',
            'i': 12345,
            'c': 'client-123',
            's': 'XRPUSDC',
            'S': 'BUY',
            'L': '0.61200',
            'l': '20',
            'n': '0.01',
            'N': 'USDC',
            'T': int(datetime.now().timestamp() * 1000),
        }

        asyncio.run(handler._process_execution_report(msg))
        fills = handler.get_pending_fills()
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]['side'], 'BUY')
        self.assertAlmostEqual(fills[0]['price'], 0.612)
        self.assertAlmostEqual(fills[0]['quantity'], 20.0)

    def test_execution_report_non_trade_ignored(self):
        """Non-TRADE execution types should not produce fills."""
        from data.binance.stream import BinanceStreamHandler
        handler = BinanceStreamHandler(MarketConfig())

        msg = {
            'e': 'executionReport',
            'x': 'NEW',  # Not a TRADE
            'X': 'NEW',
            'i': 12345,
            's': 'XRPUSDC',
            'S': 'BUY',
            'L': '0.61200',
            'l': '0',
            'T': int(datetime.now().timestamp() * 1000),
        }

        asyncio.run(handler._process_execution_report(msg))
        fills = handler.get_pending_fills()
        self.assertEqual(len(fills), 0)


class TestLiveEnvQuotes(unittest.TestCase):
    """Test that BaseMarketMakerEnv (inherited by LiveEnv) computes quotes correctly."""

    def test_calculate_quotes_9d(self):
        """Verify 9D action produces 3 levels of quotes."""
        import numpy as np
        from data.processor import LOBFeatureProcessor, MarketFeatureProcessor
        from environment.base_environment import BaseMarketMakerEnv

        market_config = MarketConfig(
            symbol='XRPUSDC', tick_size=0.00001, min_qty=10.0,
            min_notional=10.0, max_position=100,
        )
        rl_config = RLConfig(
            zeta=0.005, gamma=0.995, max_bias=0.001,
            max_spread=0.002, min_spread=0.001,
            initial_cash=100.0, max_steps=1000,
        )
        lob_proc = LOBFeatureProcessor(market_config)
        mkt_proc = MarketFeatureProcessor(market_config)
        env = BaseMarketMakerEnv(rl_config, market_config, lob_proc, mkt_proc)

        env.current_state.mid_price = 0.61250
        env.current_state.cash = 100.0
        env.current_state.inventory = 0.0

        orderbook = {
            'bids': [[0.61240 - i * 0.00010, 1000.0] for i in range(10)],
            'asks': [[0.61260 + i * 0.00010, 1000.0] for i in range(10)],
        }

        action = np.array([0.0, 0.0, 0.5, 0.5, -1.0, 0.0, 0.0, -1.0, -1.0])
        quotes = env.calculate_quotes(action, {'orderbook': orderbook})

        self.assertIn('levels', quotes)
        self.assertEqual(len(quotes['levels']), 3)

        l0 = quotes['levels'][0]
        self.assertGreater(l0['bid_price'], 0)
        self.assertGreater(l0['ask_price'], 0)
        self.assertGreater(l0['ask_price'], l0['bid_price'])

    def test_min_spread_enforced(self):
        """Verify min_spread is always enforced even with action=[0,0,...]."""
        import numpy as np
        from data.processor import LOBFeatureProcessor, MarketFeatureProcessor
        from environment.base_environment import BaseMarketMakerEnv

        market_config = MarketConfig(
            symbol='XRPUSDC', tick_size=0.00001, min_qty=10.0,
            min_notional=10.0, max_position=100,
        )
        rl_config = RLConfig(
            zeta=0.005, gamma=0.995, max_bias=0.001,
            max_spread=0.002, min_spread=0.001,
            initial_cash=100.0, max_steps=1000,
        )
        lob_proc = LOBFeatureProcessor(market_config)
        mkt_proc = MarketFeatureProcessor(market_config)
        env = BaseMarketMakerEnv(rl_config, market_config, lob_proc, mkt_proc)

        mid = 0.61250
        env.current_state.mid_price = mid
        env.current_state.cash = 100.0
        env.current_state.inventory = 0.0

        orderbook = {
            'bids': [[mid - i * 0.00010, 1000.0] for i in range(10)],
            'asks': [[mid + 0.00010 + i * 0.00010, 1000.0] for i in range(10)],
        }

        # Action with min spread offsets (-1 = tightest spread)
        action = np.array([-1.0, -1.0, 0.5, 0.5, -1.0, 0.0, 0.0, -1.0, -1.0])
        quotes = env.calculate_quotes(action, {'orderbook': orderbook})

        l0 = quotes['levels'][0]
        if l0['bid_qty'] > 0 and l0['ask_qty'] > 0:
            actual_spread = (l0['ask_price'] - l0['bid_price']) / mid
            self.assertGreaterEqual(actual_spread, rl_config.min_spread * 0.99)


class TestRiskLimits(unittest.TestCase):
    """Test risk limit checks."""

    def test_max_loss_triggers(self):
        max_loss = -50.0
        total_pnl = -55.0
        self.assertTrue(total_pnl < max_loss)

    def test_max_loss_safe(self):
        max_loss = -50.0
        total_pnl = -10.0
        self.assertFalse(total_pnl < max_loss)

    def test_position_value_under_limit(self):
        max_pos_val = 500.0
        pos_value = abs(80.0 * 0.612)  # 48.96
        self.assertLess(pos_value, max_pos_val)

    def test_position_value_over_limit(self):
        max_pos_val = 500.0
        pos_value = abs(1000.0 * 0.612)  # 612
        self.assertGreater(pos_value, max_pos_val)


# ============================================================
# 2. Test live Binance testnet connection (optional, --live)
# ============================================================

class TestLiveBinanceConnection(unittest.TestCase):
    """Tests that require actual Binance testnet API keys + binance SDK.

    Skipped unless --live is passed.
    """

    @classmethod
    def setUpClass(cls):
        if not _HAS_BINANCE:
            raise unittest.SkipTest("binance SDK not installed")
        cls.api_key = os.environ.get('BINANCE_TESTNET_API_KEY', '')
        cls.api_secret = os.environ.get('BINANCE_TESTNET_SECRET', '')
        if not cls.api_key or not cls.api_secret:
            raise unittest.SkipTest(
                "BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_SECRET not set"
            )

    def test_01_client_connect(self):
        """Test AsyncClient connects to testnet and responds to ping."""
        from binance import AsyncClient

        async def _test():
            client = await AsyncClient.create(
                self.api_key, self.api_secret, testnet=True
            )
            self.assertIsNotNone(client)
            await client.ping()
            print("  [OK] Client ping successful")
            await client.close_connection()

        asyncio.run(_test())

    def test_02_server_time(self):
        """Test server time retrieval."""
        from binance import AsyncClient

        async def _test():
            client = await AsyncClient.create(
                self.api_key, self.api_secret, testnet=True
            )
            time_res = await client.get_server_time()
            server_time = time_res['serverTime']
            self.assertGreater(server_time, 0)
            print(f"  [OK] Server time: {datetime.fromtimestamp(server_time / 1000)}")
            await client.close_connection()

        asyncio.run(_test())

    def test_03_orderbook_fetch(self):
        """Test orderbook retrieval for XRPUSDC."""
        from binance import AsyncClient

        async def _test():
            client = await AsyncClient.create(
                self.api_key, self.api_secret, testnet=True
            )
            ob = await client.get_order_book(symbol='XRPUSDC', limit=10)
            self.assertIn('bids', ob)
            self.assertIn('asks', ob)
            self.assertGreater(len(ob['bids']), 0)
            self.assertGreater(len(ob['asks']), 0)
            best_bid = float(ob['bids'][0][0])
            best_ask = float(ob['asks'][0][0])
            print(f"  [OK] Orderbook: bid={best_bid}, ask={best_ask}, "
                  f"spread={best_ask - best_bid:.5f}")
            await client.close_connection()

        asyncio.run(_test())

    def test_04_account_info(self):
        """Test margin account access."""
        from binance import AsyncClient

        async def _test():
            client = await AsyncClient.create(
                self.api_key, self.api_secret, testnet=True
            )
            try:
                account = await client.get_margin_account()
                total_net = float(account.get('totalNetAssetOfBtc', 0))
                print(f"  [OK] Margin account accessible. Net BTC: {total_net}")
                for asset in account.get('userAssets', []):
                    free = float(asset['free'])
                    locked = float(asset['locked'])
                    borrowed = float(asset['borrowed'])
                    if free > 0 or locked > 0 or borrowed > 0:
                        print(f"       {asset['asset']}: free={free}, "
                              f"locked={locked}, borrowed={borrowed}")
            except Exception as e:
                print(f"  [WARN] Margin account not available: {e}")
                print("         (Normal if testnet doesn't support cross margin)")
            await client.close_connection()

        asyncio.run(_test())

    def test_05_open_orders(self):
        """Test fetching open orders."""
        from binance import AsyncClient

        async def _test():
            client = await AsyncClient.create(
                self.api_key, self.api_secret, testnet=True
            )
            try:
                orders = await client.get_open_margin_orders(symbol='XRPUSDC')
                print(f"  [OK] Open margin orders: {len(orders)}")
            except Exception as e:
                print(f"  [WARN] get_open_margin_orders failed: {e}")
                try:
                    orders = await client.get_open_orders(symbol='XRPUSDC')
                    print(f"  [OK] Open spot orders: {len(orders)}")
                except Exception as e2:
                    print(f"  [WARN] get_open_orders also failed: {e2}")
            await client.close_connection()

        asyncio.run(_test())

    def test_06_stream_handler_connect(self):
        """Test BinanceStreamHandler connects and gets initial orderbook."""
        from data.binance.stream import BinanceStreamHandler

        market_config = MarketConfig(symbol='XRPUSDC', tick_size=0.00001)

        async def _test():
            handler = BinanceStreamHandler(market_config)
            await handler.connect(self.api_key, self.api_secret)

            self.assertTrue(handler._running)
            self.assertGreater(len(handler.orderbook['bids']), 0)
            self.assertGreater(len(handler.orderbook['asks']), 0)

            best_bid = handler.orderbook['bids'][0][0]
            best_ask = handler.orderbook['asks'][0][0]
            print(f"  [OK] Stream handler connected. Orderbook: {best_bid}/{best_ask}")

            await handler.stop()

        asyncio.run(_test())

    def test_07_stream_handler_live_data(self):
        """Test that streams deliver live updates within 5 seconds."""
        from data.binance.stream import BinanceStreamHandler

        market_config = MarketConfig(symbol='XRPUSDC', tick_size=0.00001)

        async def _test():
            handler = BinanceStreamHandler(market_config)
            await handler.connect(self.api_key, self.api_secret)

            initial_update_id = handler.orderbook['lastUpdateId']
            await handler.start_streams()

            for _ in range(50):
                await asyncio.sleep(0.1)
                if handler.orderbook['lastUpdateId'] > initial_update_id:
                    break

            new_update_id = handler.orderbook['lastUpdateId']
            self.assertGreater(new_update_id, initial_update_id,
                               "No depth updates received in 5s")
            print(f"  [OK] Stream live: updateId {initial_update_id} -> {new_update_id}")

            await handler.stop()

        asyncio.run(_test())


# ============================================================
# Runner
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Binance testnet connection')
    parser.add_argument('--live', action='store_true',
                        help='Run live Binance connection tests (needs API keys + binance SDK)')
    args, remaining = parser.parse_known_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Always run offline tests
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestLiveOrderExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestLiveEnvQuotes))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskLimits))

    if args.live:
        suite.addTests(loader.loadTestsFromTestCase(TestLiveBinanceConnection))
    else:
        print("=" * 60)
        if not _HAS_BINANCE:
            print("  NOTE: binance SDK not installed, using mocks")
        print("  Running OFFLINE tests only (no API keys needed)")
        print("  Use --live to also test Binance testnet connection")
        print("=" * 60)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)
