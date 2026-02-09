from asyncio.log import logger
from typing import Dict, Optional
import numpy as np
from core.config import MarketConfig, RLConfig
from data.processor import LOBFeatureProcessor, MarketFeatureProcessor
from scipy.optimize import minimize

class MarketMakerCalculator:
    """Calculateur centralisé pour le Market Making avec properties."""
    
    def __init__(self, market_config: MarketConfig, rl_config: RLConfig):
        self.market_config = market_config
        self.rl_config = rl_config
        
        # Processeurs
        self.lob_processor = LOBFeatureProcessor(market_config)
        self.market_processor = MarketFeatureProcessor()
        
        # État interne
        self._current_orderbook = None
        self._current_features = None
        self._current_action = None
        self._current_inventory = 0.0
        
        # Cache
        self._cached_prices = None
        self._cached_hull = None
        self._cached_metrics = {}
        
    # === Properties pour l'état du marché ===
    @property
    def mid_price(self) -> float:
        """Prix mid actuel."""
        if self._current_orderbook is None:
            return None
        return (float(self._current_orderbook['bids'][0][0]) + 
                float(self._current_orderbook['asks'][0][0])) / 2
                
    @property 
    def market_imbalance(self) -> float:
        """Déséquilibre du carnet d'ordres."""
        if self._current_orderbook is None:
            return 0.0
            
        bids = np.array([float(level[1]) for level in self._current_orderbook['bids'][:5]])
        asks = np.array([float(level[1]) for level in self._current_orderbook['asks'][:5]])
        
        total_volume = np.sum(bids) + np.sum(asks)
        if total_volume == 0:
            return 0.0
            
        return (np.sum(bids) - np.sum(asks)) / total_volume
        
    # === Properties pour les indicateurs techniques ===
    @property
    def volatility(self) -> Dict[str, float]:
        """Volatilité calculée sur différentes périodes."""
        if 'volatility' not in self._cached_metrics:
            self._cached_metrics['volatility'] = {
                '10s': self.market_processor.get_rv(10),
                '1m': self.market_processor.get_rv(60),
                '5m': self.market_processor.get_rv(300)
            }
        return self._cached_metrics['volatility']
        
    @property
    def rsi(self) -> Dict[str, float]:
        """RSI calculé sur différentes périodes."""
        if 'rsi' not in self._cached_metrics:
            self._cached_metrics['rsi'] = {
                '10s': self.market_processor.get_rsi(10),
                '1m': self.market_processor.get_rsi(60),
                '5m': self.market_processor.get_rsi(300)
            }
        return self._cached_metrics['rsi']
        
    @property
    def osi(self) -> Dict[str, float]:
        """Order Strength Index sur différentes périodes."""
        if 'osi' not in self._cached_metrics:
            self._cached_metrics['osi'] = {
                '10s': self.market_processor.get_osi(10),
                '1m': self.market_processor.get_osi(60),
                '5m': self.market_processor.get_osi(300)
            }
        return self._cached_metrics['osi']
        
    # === Properties pour le pricing optimal ===
    @property
    def optimal_quotes(self) -> Dict[str, float]:
        """Prix optimaux via optimisation convexe avec ajustements de l'agent."""
        if self._cached_prices is None:
            self._cached_prices = self._calculate_optimal_quotes()
        return self._cached_prices
        
    def _calculate_optimal_quotes(self) -> Dict[str, float]:
        """Calcul interne des prix optimaux."""
        try:
            # Construction de l'espace des prix
            price_space = self._construct_price_space()
            
            # Optimisation convexe
            base_prices = self._solve_convex_optimization(price_space)
            
            # Ajustements de l'agent
            if self._current_action is not None:
                adjusted_prices = self._apply_agent_adjustments(
                    base_prices,
                    self._current_action,
                    self._current_inventory
                )
            else:
                adjusted_prices = base_prices
                
            # Validation finale
            return self._validate_prices(adjusted_prices)
            
        except Exception as e:
            logger.error(f"Erreur calcul prix optimaux: {e}")
            return self._get_fallback_prices()
            
    def _solve_convex_optimization(self, price_space: np.ndarray) -> Dict[str, float]:
        """Résolution du problème d'optimisation convexe."""
        def objective(x):
            entropy = -np.sum(x[x > 0] * np.log(x[x > 0]))
            expected_profit = np.dot(x, self._current_features)
            return -(expected_profit + self.rl_config.gamma * entropy)
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        result = minimize(
            objective,
            x0=np.ones(len(price_space))/len(price_space),
            constraints=constraints,
            method='SLSQP'
        )
        
        optimal_weights = result.x
        return {
            'bid_price': np.dot(optimal_weights, price_space[:len(price_space)//2]),
            'ask_price': np.dot(optimal_weights, price_space[len(price_space)//2:]),
            'mid_price': self.mid_price,
            'weights': optimal_weights
        }
        
    def _apply_agent_adjustments(self, 
                               base_prices: Dict[str, float],
                               action: np.ndarray,
                               inventory: float) -> Dict[str, float]:
        """Application des ajustements de l'agent."""
        bid_adj, ask_adj = action
        
        # Skew d'inventaire
        inventory_skew = (inventory / self.rl_config.max_position) * self.rl_config.skew_factor
        
        # Prix ajustés
        tick_size = self.market_config.tick_size
        adjusted_bid = round((base_prices['bid_price'] * (1 + bid_adj) + inventory_skew) / tick_size) * tick_size
        adjusted_ask = round((base_prices['ask_price'] * (1 + ask_adj) + inventory_skew) / tick_size) * tick_size
        
        return {**base_prices, 'bid_price': adjusted_bid, 'ask_price': adjusted_ask}
        
    # === Properties pour les probabilités d'exécution ===
    @property
    def execution_probabilities(self) -> Dict[str, float]:
        """Probabilités d'exécution pour bid et ask."""
        quotes = self.optimal_quotes
        vol = self.volatility['1m']
        
        return {
            'bid_prob': self._calculate_execution_probability(
                quotes['bid_price'],
                quotes['mid_price'],
                vol,
                'bid'
            ),
            'ask_prob': self._calculate_execution_probability(
                quotes['ask_price'],
                quotes['mid_price'],
                vol,
                'ask'
            )
        }
        
    # === Méthodes de mise à jour ===
    def update_state(self, 
                    orderbook: Dict,
                    action: Optional[np.ndarray] = None,
                    inventory: float = 0.0):
        """Mise à jour de l'état et reset du cache."""
        self._current_orderbook = orderbook
        self._current_action = action
        self._current_inventory = inventory
        
        # Reset cache
        self._cached_prices = None
        self._cached_metrics = {}
        
        # Mise à jour des processeurs
        self.lob_processor.process(orderbook)
        self.market_processor.process(orderbook)