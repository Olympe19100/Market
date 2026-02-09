"""Binance Testnet Deployment Script — Inference Only.

Deploys the trained PPO 9D agent on Binance Cross Margin testnet.
No training, no store_transition(), no update().

Usage:
    # Dry run (no real orders):
    python deploy.py --config config/config_testnet.yaml --dry-run

    # Live testnet:
    python deploy.py --config config/config_testnet.yaml

Environment variables:
    BINANCE_TESTNET_API_KEY
    BINANCE_TESTNET_SECRET
"""

import os
import sys
import signal
import asyncio
import argparse
import logging
from datetime import datetime
from typing import Optional

import yaml
import torch
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from binance import AsyncClient
from core.config import MarketConfig, RLConfig, ModelConfig
from data.processor import LOBFeatureProcessor, MarketFeatureProcessor
from data.binance.stream import BinanceStreamHandler
from execution.handler import LiveOrderExecutor
from execution.Position_manager import PositionManager
from environment.live_env import LiveMarketMakerEnv
from models.rl.PPO_agent import PPOAgent

logger = logging.getLogger('deploy')


class TestnetDeployer:
    """Manages the full lifecycle of testnet deployment."""

    def __init__(self, config_path: str, dry_run: bool = False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.dry_run = dry_run
        self.running = False

        # API keys from environment
        self.api_key = os.environ.get('BINANCE_TESTNET_API_KEY', '')
        self.api_secret = os.environ.get('BINANCE_TESTNET_SECRET', '')
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_SECRET environment variables"
            )

        # Build configs
        mc = self.config['market']
        self.market_config = MarketConfig(
            symbol=mc['symbol'],
            tick_size=mc['tick_size'],
            min_qty=mc['min_qty'],
            min_notional=mc['min_notional'],
            max_position=mc['max_position'],
            update_interval=mc['update_interval'],
        )

        rc = self.config['rl']
        self.rl_config = RLConfig(
            zeta=rc['zeta'],
            gamma=rc['gamma'],
            max_bias=rc['max_bias'],
            max_spread=rc['max_spread'],
            min_spread=rc['min_spread'],
            max_equity_exposure=rc['max_equity_exposure'],
            initial_cash=rc['initial_cash'],
            max_steps=rc['max_steps'],
        )

        self.model_config = ModelConfig(**self.config['model'])
        self.deploy_config = self.config['deployment']

        # Components (initialized in initialize())
        self.client: Optional[AsyncClient] = None
        self.stream_handler: Optional[BinanceStreamHandler] = None
        self.order_executor: Optional[LiveOrderExecutor] = None
        self.position_manager: Optional[PositionManager] = None
        self.env: Optional[LiveMarketMakerEnv] = None
        self.agent: Optional[PPOAgent] = None

        # Tracking
        self.total_pnl = 0.0
        self.step_count = 0
        self.episode_count = 0

    async def initialize(self):
        """Initialize all components: client, streams, executor, env, agent."""
        logger.info("=" * 60)
        logger.info("TESTNET DEPLOYER INITIALIZATION")
        logger.info(f"  Symbol: {self.market_config.symbol}")
        logger.info(f"  Dry run: {self.dry_run}")
        logger.info(f"  Checkpoint: {self.deploy_config['checkpoint_path']}")
        logger.info("=" * 60)

        # 1. Binance async client
        is_testnet = self.config.get('binance', {}).get('testnet', True)
        trading_mode = self.deploy_config.get('mode', 'futures')

        self.client = await AsyncClient.create(
            self.api_key,
            self.api_secret,
            testnet=is_testnet
        )

        # For futures testnet, override API_URL to futures testnet
        if is_testnet and trading_mode == 'futures':
            self.client.API_URL = self.client.FUTURES_TESTNET_URL
            logger.info(f"Using Futures testnet: {self.client.API_URL}")
        elif is_testnet:
            self.client.API_URL = self.client.API_TESTNET_URL

        logger.info(f"Binance client connected (testnet={is_testnet}, mode={trading_mode})")

        # 2. Stream handler
        self.stream_handler = BinanceStreamHandler(self.market_config, mode=trading_mode)
        self.stream_handler.client = self.client
        self.stream_handler._api_key = self.api_key
        self.stream_handler._api_secret = self.api_secret
        from binance.ws.streams import BinanceSocketManager
        self.stream_handler.bm = BinanceSocketManager(self.client, max_queue_size=10000)
        await self.stream_handler.get_initial_state()
        self.stream_handler._running = True
        await self.stream_handler.start_streams()

        # Start user data stream for fill detection
        try:
            await self.stream_handler.start_user_data_stream()
        except Exception as e:
            logger.warning(f"User data stream not available: {e}. Using polling fallback.")

        logger.info("Streams started")

        # 3. Feature processors
        lob_processor = LOBFeatureProcessor(self.market_config)
        market_processor = MarketFeatureProcessor(self.market_config)

        # 4. Order executor
        self.order_executor = LiveOrderExecutor(
            client=self.client,
            market_config=self.market_config,
            rl_config=self.rl_config,
            dry_run=self.dry_run,
            mode=trading_mode,
        )
        logger.info(f"Order executor mode: {trading_mode.upper()}")

        # 5. Position manager
        self.position_manager = PositionManager(self.client, self.market_config, mode=trading_mode)
        try:
            await self.position_manager.initialize()
        except Exception as e:
            logger.warning(f"Position manager initialization: {e}. Starting fresh.")

        # 6. Live environment
        stale_timeout = self.deploy_config.get('stale_data_timeout', 5.0)
        self.env = LiveMarketMakerEnv(
            rl_config=self.rl_config,
            market_config=self.market_config,
            lob_processor=lob_processor,
            market_processor=market_processor,
            stream_handler=self.stream_handler,
            order_executor=self.order_executor,
            position_manager=self.position_manager,
            stale_data_timeout=stale_timeout,
            update_interval=self.market_config.update_interval,
        )

        # 7. PPO Agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = PPOAgent(
            model_config=self.model_config,
            rl_config=self.rl_config,
            device=device,
        )

        # 8. Load checkpoint
        checkpoint_path = self.deploy_config['checkpoint_path']
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.agent.load_state_dict(checkpoint['agent_state'])
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        if 'episode' in checkpoint:
            logger.info(f"  Trained for {checkpoint['episode']} episodes")

        # 9. Eval mode (no dropout, no gradient tracking)
        self.agent.backbone.eval()
        self.agent.heads.eval()

        logger.info("Initialization complete. Agent in eval mode.")

    async def run(self):
        """Main deployment loop: inference only, no training."""
        self.running = True
        log_freq = self.deploy_config.get('log_frequency', 10)
        reconnect_delay = self.deploy_config.get('reconnect_delay', 5.0)

        logger.info("Starting deployment loop...")

        while self.running:
            try:
                # Reset episode
                state = await self.env.reset()
                self.agent.reset_episode()
                self.episode_count += 1
                episode_reward = 0.0
                episode_step = 0

                logger.info(f"=== Episode {self.episode_count} started ===")

                while self.running:
                    # Risk check
                    should_stop = self._check_risk_limits()
                    if should_stop:
                        logger.warning("Risk limit breached. Stopping.")
                        self.running = False
                        break

                    # Inference: select action (no grad, no noise storage needed)
                    with torch.no_grad():
                        action, _, _ = self.agent.select_action(state)

                    # Step environment
                    next_state, reward, done, info = await self.env.step(action)

                    # Skip handling
                    if info.get('skipped'):
                        continue

                    state = next_state
                    episode_reward += reward
                    episode_step += 1
                    self.step_count += 1

                    # Periodic logging
                    if episode_step % log_freq == 0:
                        metrics = info.get('metrics', {})
                        mid = info.get('mid_price', 0)
                        inv = metrics.get('inventory', 0)
                        total_pnl = metrics.get('total_pnl', 0)
                        pos_val = metrics.get('position_value', 0)

                        logger.info(
                            f"Step {self.step_count} | "
                            f"Mid: {mid:.5f} | "
                            f"Inv: {inv:.1f} | "
                            f"PnL: {total_pnl:.4f} | "
                            f"PosVal: ${pos_val:.2f} | "
                            f"EpReward: {episode_reward:.4f}"
                        )

                        # Log active orders
                        active = info.get('active_orders', {})
                        if active:
                            for key, val in active.items():
                                logger.debug(f"  {key}: {val['qty']}@{val['price']:.5f}")

                    # NO store_transition() — inference only
                    # NO agent.update() — inference only

                    if done:
                        break

                logger.info(f"=== Episode {self.episode_count} ended: "
                            f"steps={episode_step}, reward={episode_reward:.4f} ===")

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received")
                self.running = False
            except Exception as e:
                logger.error(f"Error in deployment loop: {e}", exc_info=True)
                if self.running:
                    logger.info(f"Reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)

    def _check_risk_limits(self) -> bool:
        """Check risk limits. Returns True if trading should stop."""
        max_loss = self.deploy_config.get('max_loss_usd', -50.0)
        max_pos_val = self.deploy_config.get('max_position_value_usd', 500.0)

        total_pnl = self.position_manager.realized_pnl + self.position_manager.unrealized_pnl
        mid = self.env.current_state.mid_price or 0
        pos_value = abs(self.env.current_state.inventory * mid) if mid > 0 else 0

        if total_pnl < max_loss:
            logger.critical(f"MAX LOSS BREACHED: PnL={total_pnl:.2f} < {max_loss}")
            return True

        if pos_value > max_pos_val:
            logger.warning(f"Position value ${pos_value:.2f} exceeds limit ${max_pos_val:.2f}")
            # Don't stop, but the agent should reduce position naturally
            # Only stop if way over limit
            if pos_value > max_pos_val * 2:
                logger.critical(f"Position value ${pos_value:.2f} far exceeds limit")
                return True

        return False

    async def shutdown(self):
        """Graceful shutdown: cancel all orders, stop streams, close client."""
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN")
        logger.info("=" * 60)

        self.running = False

        # Cancel all orders
        if self.order_executor:
            try:
                await self.order_executor.cancel_all_orders()
                logger.info("All orders cancelled")
            except Exception as e:
                logger.error(f"Error cancelling orders during shutdown: {e}")

        # Stop streams
        if self.stream_handler:
            try:
                await self.stream_handler.stop()
                logger.info("Streams stopped")
            except Exception as e:
                logger.error(f"Error stopping streams: {e}")

        # Close client
        if self.client:
            try:
                await self.client.close_connection()
                logger.info("Client connection closed")
            except Exception as e:
                logger.error(f"Error closing client: {e}")

        # Final metrics
        if self.position_manager:
            metrics = self.position_manager.get_metrics()
            logger.info("=== Final Metrics ===")
            logger.info(f"  Total PnL: {metrics.get('total_pnl', 0):.4f}")
            logger.info(f"  Realized PnL: {metrics.get('realized_pnl', 0):.4f}")
            logger.info(f"  Position: {metrics.get('current_position', 0):.1f}")
            logger.info(f"  Total Volume: {metrics.get('total_volume', 0):.2f}")
            logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
            logger.info(f"  Steps: {self.step_count}")
            logger.info(f"  Episodes: {self.episode_count}")

        logger.info("Shutdown complete.")


async def main():
    parser = argparse.ArgumentParser(description='Deploy PPO agent on Binance Testnet')
    parser.add_argument('--config', type=str, default='config/config_testnet.yaml',
                        help='Path to deployment config YAML')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without placing real orders')
    args = parser.parse_args()

    # Setup logging
    log_dir = 'logs/testnet'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ]
    )
    logger.info(f"Logging to {log_file}")

    deployer = TestnetDeployer(config_path=args.config, dry_run=args.dry_run)

    # Signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Signal received, initiating shutdown...")
        shutdown_event.set()
        deployer.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: _signal_handler())

    try:
        await deployer.initialize()
        # Run deployment in a task so we can wait for shutdown signal
        deploy_task = asyncio.create_task(deployer.run())

        # Wait for either the deploy task to finish or shutdown signal
        done, pending = await asyncio.wait(
            [deploy_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await deployer.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
