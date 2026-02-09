# market_maker/models/rl/environment.py

from base_environment import BaseMarketMakerEnv, MarketMakerState
import numpy as np
from typing import Dict, Tuple, List
from core.config import RLConfig, MarketConfig
from core.exceptions import ModelError
import logging

# Configuration du logger pour suivre l'environnement
logger = logging.getLogger('market_maker.environment')

class MarketMakerEnv(BaseMarketMakerEnv):
    """Market making environment following the article."""

    def __init__(self, rl_config: RLConfig, market_config: MarketConfig):
        super().__init__(rl_config, market_config)
        logger.info("MarketMakerEnv initialized with RLConfig and MarketConfig.")

    def reset(self) -> Dict:
        """Réinitialise l'environnement au début d'un épisode."""
        self.episode_step = 0
        self.last_action = None

        # Réinitialiser les variables nécessaires
        self.execution_times.clear()
        self.price_history.clear()
        self.volume_history.clear()
        self.previous_unrealized_pnl = 0.0
        logger.debug("Environment reset: episode_step set to 0 and histories cleared.")

        # Obtenir le premier orderbook pour initialiser l'état
        orderbook = self._get_next_orderbook()
        if orderbook is None:
            logger.error("No orderbook available to initialize the environment.")
            raise ValueError("Aucun orderbook disponible pour initialiser l'environnement.")

        # Initialiser l'état
        self.current_state = self._initialize_state(orderbook)
        logger.info("Environment state initialized with first orderbook.")

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute an action in the environment."""
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        if not self.action_space.contains(action):
            logger.error(f"Invalid action: {action}")
            raise ModelError(f"Invalid action: {action}")

        self.last_action = action
        logger.debug(f"Action received: {action}")

        # Get market state
        market_state = self.get_market_state()
        logger.debug(f"Current market state retrieved: {market_state}")

        # Calculate quotes using the common method
        quotes = self.calculate_quotes(action, market_state)
        logger.debug(f"Quotes calculated: {quotes}")

        # Simulate order execution
        executions = self._simulate_order_execution(quotes)
        logger.debug(f"Order executions simulated: {executions}")

        # Update state
        self._update_state(executions)
        logger.debug("State updated with executions.")

        # Calculate reward
        reward = self._calculate_reward(executions)
        logger.debug(f"Reward calculated: {reward}")

        # Check end of episode
        self.episode_step += 1
        done = self.episode_step >= self.max_steps
        logger.debug(f"Step {self.episode_step} completed. Done: {done}")

        info = {
            'quotes': quotes,
            'executions': executions,
            'inventory': self.current_state.inventory,
            'pnl': self.current_state.unrealized_pnl,
            'spread': self.current_state.current_spread
        }

        return self._get_observation(), reward, done, info

    def _simulate_order_execution(self, quotes: Dict[str, float]) -> List[Dict]:
        """Simulate order execution based on probabilities."""
        executions = []

        # Execution probabilities
        bid_prob = np.exp(-self.rl_config.gamma * (self.current_state.mid_price - quotes['bid_price']))
        ask_prob = np.exp(-self.rl_config.gamma * (quotes['ask_price'] - self.current_state.mid_price))
        logger.debug(f"Execution probabilities - Bid: {bid_prob}, Ask: {ask_prob}")

        # Simulate executions
        if np.random.random() < bid_prob:
            executions.append({
                'side': 'buy',
                'price': quotes['bid_price'],
                'quantity': self.market_config.min_qty,
                'timestamp': None
            })
            logger.debug(f"Bid execution added at price {quotes['bid_price']} with quantity {self.market_config.min_qty}")

        if np.random.random() < ask_prob:
            executions.append({
                'side': 'sell',
                'price': quotes['ask_price'],
                'quantity': self.market_config.min_qty,
                'timestamp': None
            })
            logger.debug(f"Ask execution added at price {quotes['ask_price']} with quantity {self.market_config.min_qty}")

        return executions

    def _update_state(self, executions: List[Dict]):
        """Update the internal state based on executions."""
        for exec in executions:
            qty = exec['quantity'] * (1 if exec['side'] == 'buy' else -1)
            self.current_state.inventory += qty
            self.current_state.cash -= qty * exec['price']
            logger.debug(f"Updated inventory: {self.current_state.inventory}, Cash: {self.current_state.cash}")

        # Update unrealized PnL
        self.current_state.unrealized_pnl = (
            self.current_state.inventory * self.current_state.mid_price + self.current_state.cash
        )
        logger.debug(f"Unrealized PnL updated: {self.current_state.unrealized_pnl}")

    def _get_observation(self) -> Dict:
        """Convert the state into an observation for the agent."""
        observation = {
            'lob_features': self.current_state.lob_features,
            'market_features': np.array([
                self.current_state.market_features['osi_10s'],
                self.current_state.market_features['osi_60s'],
                self.current_state.market_features['osi_300s'],
                self.current_state.market_features['rv_5m'],
                self.current_state.market_features['rv_10m'],
                self.current_state.market_features['rv_30m'],
                self.current_state.market_features['rsi_5m'],
                self.current_state.market_features['rsi_10m'],
                self.current_state.market_features['rsi_30m']
            ], dtype=np.float32),
            'inventory': np.array([self.current_state.inventory], dtype=np.float32),
            'time_remaining': np.array([self.current_state.time_remaining], dtype=np.float32)
        }
        logger.debug(f"Observation generated: {observation}")
        return observation

    def get_market_state(self) -> Dict:
        """Retrieve the current market state."""
        market_state = {
            'orderbook': self.current_orderbook,
            'lob_features': self.current_state.lob_features,
            'time_remaining': self.current_state.time_remaining
        }
        logger.debug(f"Market state retrieved: {market_state}")
        return market_state
