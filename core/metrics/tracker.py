from typing import Dict, List, Optional
import numpy as np
from collections import deque
import logging
from .sharpe import SharpeRatio
from .drawdown import Drawdown

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Suit les métriques importantes du market maker."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = {
            'pnl': deque(maxlen=window_size),
            'rolling_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'total_pnl': 0.0,
            'inventory': 0,
            'rewards': deque(maxlen=window_size),
            'executed_orders': deque(maxlen=window_size),
            'spreads': deque(maxlen=window_size)
        }
        
        # Métriques supplémentaires pour le risk management
        self.daily_pnl = deque(maxlen=252)  # 1 an de trading
        self.hourly_metrics = {
            'trades': 0,
            'volume': 0.0,
            'pnl': 0.0,
            'max_inventory': 0,
        }
        self._last_reset = None
        
        # Calculators
        self.sharpe_calc = SharpeRatio()
        self.drawdown_calc = Drawdown()
        
        logger.debug(f"MetricsTracker initialized with window_size={window_size}")

    def update(self, metrics_dict: Dict):
        """Met à jour les métriques avec de nouvelles valeurs."""
        for key, value in metrics_dict.items():
            if key in self.metrics:
                if isinstance(self.metrics[key], deque):
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] = value
                logger.debug(f"Metric '{key}' updated with value: {value}")

        # Mettre à jour le PnL total
        previous_total_pnl = self.metrics['total_pnl']
        self.metrics['total_pnl'] = (
            self.metrics['rolling_pnl'] + 
            self.metrics['unrealized_pnl']
        )
        logger.debug(f"Total PnL updated from {previous_total_pnl} to {self.metrics['total_pnl']}")

        # Mise à jour des métriques horaires
        previous_max_inventory = self.hourly_metrics['max_inventory']
        self.hourly_metrics['max_inventory'] = max(
            previous_max_inventory,
            abs(self.metrics['inventory'])
        )
        logger.debug(f"Hourly max_inventory updated from {previous_max_inventory} to {self.hourly_metrics['max_inventory']}")

    def get_metrics_summary(self) -> Dict:
        """Retourne un résumé des métriques actuelles."""
        # Calcul du Sharpe Ratio
        if len(self.daily_pnl) > 1:
            returns = np.diff(list(self.daily_pnl))  # Calculate returns from actual PnL changes
            sharpe = self.sharpe_calc.calculate(returns)
            max_drawdown, drawdown_duration, current_drawdown = self.drawdown_calc.calculate(list(self.daily_pnl))
        else:
            sharpe = 0.0
            max_drawdown = 0.0
            drawdown_duration = 0
            current_drawdown = 0.0
            logger.warning("Pas assez de données pour calculer Sharpe Ratio et Drawdown")

        summary = {
            'total_pnl': self.metrics['total_pnl'],
            'rolling_pnl': self.metrics['rolling_pnl'],
            'unrealized_pnl': self.metrics['unrealized_pnl'],
            'inventory': self.metrics['inventory'],
            'avg_spread': np.mean(self.metrics['spreads']) if self.metrics['spreads'] else 0.0,
            'avg_reward': np.mean(self.metrics['rewards']) if self.metrics['rewards'] else 0.0,
            'num_trades': len(self.metrics['executed_orders']),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'current_drawdown': current_drawdown,
            'hourly_trades': self.hourly_metrics['trades'],
            'hourly_volume': self.hourly_metrics['volume'],
            'max_hourly_inventory': self.hourly_metrics['max_inventory']
        }
        logger.debug(f"Metrics summary generated: {summary}")
        return summary

    def reset_hourly_metrics(self):
        """Réinitialise les métriques horaires."""
        logger.info("Resetting hourly metrics")
        self.hourly_metrics = {
            'trades': 0,
            'volume': 0.0,
            'pnl': 0.0,
            'max_inventory': 0,
        }
        
    def update_daily_pnl(self):
        """Met à jour le PnL journalier."""
        self.daily_pnl.append(self.metrics['total_pnl'])
        logger.debug(f"Daily PnL updated. New daily_pnl list: {list(self.daily_pnl)}")
