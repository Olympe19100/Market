"""Live Fine-Tuning on Binance Futures Testnet.

Loads the trained checkpoint, runs live inference, and fine-tunes
the PPO agent online using real market conditions.

Usage:
    python finetune_live.py --config config/config_testnet.yaml

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

logger = logging.getLogger('finetune_live')


class LiveFineTuner:
    """Fine-tunes a trained PPO agent on live Binance testnet data."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.running = False

        # API keys
        self.api_key = os.environ.get('BINANCE_TESTNET_API_KEY', '')
        self.api_secret = os.environ.get('BINANCE_TESTNET_SECRET', '')
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_SECRET")

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

        # Fine-tuning config
        ft = self.config.get('finetune', {})
        self.update_every_n_steps = ft.get('update_every_n_steps', 256)
        self.save_every_n_episodes = ft.get('save_every_n_episodes', 5)
        self.finetune_lr = ft.get('learning_rate', 1e-5)
        self.checkpoint_dir = ft.get('checkpoint_dir', 'checkpoints/finetune')

        # Reward shaping coefficients for live fine-tuning
        self.shaping_config = {
            'shaping_proximity': ft.get('shaping_proximity', 0.0),
            'shaping_presence': ft.get('shaping_presence', 0.0),
            'shaping_spread': ft.get('shaping_spread', 0.0),
        }

        # Components
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
        self.best_episode_reward = -float('inf')

    async def initialize(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("LIVE FINE-TUNING INITIALIZATION")
        logger.info(f"  Symbol: {self.market_config.symbol}")
        logger.info(f"  Checkpoint: {self.deploy_config['checkpoint_path']}")
        logger.info(f"  Fine-tune LR: {self.finetune_lr}")
        logger.info(f"  Update every: {self.update_every_n_steps} steps")
        logger.info(f"  Save every: {self.save_every_n_episodes} episodes")
        logger.info("=" * 60)

        # 1. Binance client
        is_testnet = self.config.get('binance', {}).get('testnet', True)
        trading_mode = self.deploy_config.get('mode', 'futures')

        self.client = await AsyncClient.create(
            self.api_key, self.api_secret, testnet=is_testnet
        )

        if is_testnet and trading_mode == 'futures':
            self.client.API_URL = self.client.FUTURES_TESTNET_URL
            logger.info(f"Using Futures testnet: {self.client.API_URL}")
        elif is_testnet:
            self.client.API_URL = self.client.API_TESTNET_URL

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

        try:
            await self.stream_handler.start_user_data_stream()
        except Exception as e:
            logger.warning(f"User data stream not available: {e}. Using polling fallback.")

        logger.info("Streams started")

        # 3. Feature processors
        lob_processor = LOBFeatureProcessor(self.market_config)
        market_processor = MarketFeatureProcessor(self.market_config)

        # 4. Order executor (LIVE — real orders on testnet)
        self.order_executor = LiveOrderExecutor(
            client=self.client,
            market_config=self.market_config,
            rl_config=self.rl_config,
            dry_run=False,
            mode=trading_mode,
        )
        logger.info(f"Order executor mode: {trading_mode.upper()} (LIVE)")

        # 5. Position manager
        self.position_manager = PositionManager(self.client, self.market_config, mode=trading_mode)
        try:
            await self.position_manager.initialize()
        except Exception as e:
            logger.warning(f"Position manager init: {e}. Starting fresh.")

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
            shaping_config=self.shaping_config,
        )

        # 7. PPO Agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = PPOAgent(
            model_config=self.model_config,
            rl_config=self.rl_config,
            device=device,
        )

        # 8. Load ALL weights from checkpoint
        checkpoint_path = self.deploy_config['checkpoint_path']
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.agent.load_state_dict(checkpoint['agent_state'])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"  Trained episodes: {checkpoint.get('episode', '?')}")
        logger.info(f"  Best reward: {checkpoint.get('best_reward', '?')}")
        logger.info(f"  obs_rms count: {self.agent.obs_rms.count:.0f}")
        logger.info(f"  reward_rms count: {self.agent.reward_rms.count:.0f}")

        # 9. Override learning rate for fine-tuning (lower to avoid catastrophic forgetting)
        for param_group in self.agent.optimizer.param_groups:
            param_group['lr'] = self.finetune_lr
        logger.info(f"Optimizer LR set to {self.finetune_lr} for fine-tuning")

        # 10. Create checkpoint dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logger.info("Initialization complete. Agent ready for fine-tuning.")

    async def run(self):
        """Main fine-tuning loop: inference + learning on live data."""
        self.running = True
        log_freq = self.deploy_config.get('log_frequency', 10)
        reconnect_delay = self.deploy_config.get('reconnect_delay', 5.0)
        steps_since_update = 0

        logger.info("Starting fine-tuning loop...")

        while self.running:
            try:
                # Reset episode
                state = await self.env.reset()
                self.agent.reset_episode()
                self.episode_count += 1
                episode_reward = 0.0
                episode_step = 0
                episode_fills = 0

                logger.info(f"=== Episode {self.episode_count} started ===")

                while self.running:
                    # Risk check
                    if self._check_risk_limits():
                        logger.warning("Risk limit breached. Stopping.")
                        self.running = False
                        break

                    # Select action (eval mode, no_grad — standard for PPO collection)
                    action, log_prob, value = self.agent.select_action(state)

                    # Step environment (LIVE orders)
                    next_state, reward, done, info = await self.env.step(action)

                    if info.get('skipped'):
                        continue

                    # Count fills
                    fills = info.get('fills', [])
                    episode_fills += len(fills)

                    # Track adverse selection for M3ORL Lagrangian
                    self.agent.track_adverse_selection(info)

                    # Store transition for PPO update
                    self.agent.store_transition(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        log_prob=log_prob,
                        value=value,
                        done=done,
                    )

                    state = next_state
                    episode_reward += reward
                    episode_step += 1
                    self.step_count += 1
                    steps_since_update += 1

                    # PPO update when enough steps collected
                    if steps_since_update >= self.update_every_n_steps:
                        update_metrics = self.agent.update()
                        # Sync λ_as from PPO Lagrangian back to env reward
                        self.env._lambda_as = self.agent.lambda_as
                        if update_metrics:
                            logger.info(
                                f"PPO UPDATE @ step {self.step_count} | "
                                f"actor_loss={update_metrics.get('actor_loss', 0):.6f} | "
                                f"critic_loss={update_metrics.get('critic_loss', 0):.6f} | "
                                f"entropy={update_metrics.get('entropy', 0):.6f} | "
                                f"approx_kl={update_metrics.get('approx_kl', 0):.6f}"
                            )
                        steps_since_update = 0

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
                            f"EpReward: {episode_reward:.4f} | "
                            f"Fills: {episode_fills} | "
                            f"Mem: {len(self.agent.memory)}"
                        )

                    if done:
                        break

                # End of episode — final update if data remains
                if steps_since_update > self.rl_config.batch_size:
                    update_metrics = self.agent.update()
                    if update_metrics:
                        logger.info(f"End-of-episode PPO UPDATE: {update_metrics}")
                    steps_since_update = 0

                logger.info(
                    f"=== Episode {self.episode_count} ended: "
                    f"steps={episode_step}, reward={episode_reward:.4f}, "
                    f"fills={episode_fills} ==="
                )

                # Save checkpoint periodically
                if self.episode_count % self.save_every_n_episodes == 0:
                    self._save_checkpoint(episode_reward)

                # Track best
                if episode_reward > self.best_episode_reward:
                    self.best_episode_reward = episode_reward
                    self._save_checkpoint(episode_reward, best=True)

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received")
                self.running = False
            except Exception as e:
                logger.error(f"Error in fine-tuning loop: {e}", exc_info=True)
                if self.running:
                    logger.info(f"Reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)

    def _save_checkpoint(self, episode_reward: float, best: bool = False):
        """Save agent checkpoint."""
        tag = "best_finetune" if best else f"finetune_ep{self.episode_count}"
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")

        checkpoint = {
            'agent_state': self.agent.state_dict(),
            'episode': self.episode_count,
            'step': self.step_count,
            'best_reward': self.best_episode_reward,
            'episode_reward': episode_reward,
            'timestamp': datetime.now().isoformat(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path} (reward={episode_reward:.4f})")

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

        if pos_value > max_pos_val * 2:
            logger.critical(f"Position value ${pos_value:.2f} far exceeds limit")
            return True

        return False

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN — saving final checkpoint")
        logger.info("=" * 60)

        self.running = False

        # Save final checkpoint
        try:
            self._save_checkpoint(0.0)
            logger.info("Final checkpoint saved")
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {e}")

        # Cancel all orders
        if self.order_executor:
            try:
                await self.order_executor.cancel_all_orders()
                logger.info("All orders cancelled")
            except Exception as e:
                logger.error(f"Error cancelling orders: {e}")

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
                logger.info("Client closed")
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
            logger.info(f"  Steps: {self.step_count}")
            logger.info(f"  Episodes: {self.episode_count}")
            logger.info(f"  Best episode reward: {self.best_episode_reward:.4f}")

        logger.info("Shutdown complete.")


async def main():
    parser = argparse.ArgumentParser(description='Fine-tune PPO agent on live Binance testnet')
    parser.add_argument('--config', type=str, default='config/config_testnet.yaml')
    args = parser.parse_args()

    # Logging
    log_dir = 'logs/finetune'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ]
    )
    logger.info(f"Logging to {log_file}")

    finetuner = LiveFineTuner(config_path=args.config)

    # Signal handlers
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Signal received, initiating shutdown...")
        shutdown_event.set()
        finetuner.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda s, f: _signal_handler())

    try:
        await finetuner.initialize()
        finetune_task = asyncio.create_task(finetuner.run())

        done, pending = await asyncio.wait(
            [finetune_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await finetuner.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
