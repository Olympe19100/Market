import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
from core.types import Trade, Order, Position
from core.config import MarketConfig, MonitoringConfig
from core.metrics import calculate_drawdown, calculate_sharpe

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Moniteur de performance qui suit en temps réel :
    - PnL (réalisé, non réalisé, total)
    - Inventaire et exposition
    - Statistiques des trades
    - Métriques de performance (Sharpe, drawdown, etc.)
    """
    
    def __init__(self, market_config: MarketConfig, monitoring_config: Optional[MonitoringConfig] = None):
        self.market_config = market_config
        self.monitoring_config = monitoring_config if monitoring_config else MonitoringConfig()
        
        # Buffers pour les métriques
        h_len = self.monitoring_config.history_maxlen
        m_len = self.monitoring_config.metrics_maxlen
        
        self.trades_history = deque(maxlen=h_len)
        self.orders_history = deque(maxlen=h_len)
        self.pnl_history = deque(maxlen=h_len)
        self.position_history = deque(maxlen=h_len)
        self.quotes_history = deque(maxlen=h_len)
        
        # Métriques actuelles
        self.metrics = {
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'total_pnl': 0.0,
            'current_position': 0.0,
            'avg_entry_price': 0.0,
            'total_volume': 0.0,
            'total_fees': 0.0,
            'trade_count': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_drawdown': 0.0,
            'max_position': 0.0,
            'sharpe_ratio': 0.0,
            'avg_trade_duration': 0.0,
            'win_rate': 0.0
        }
        
        # Métriques de l'agent RL
        self.agent_metrics = {
            'avg_reward': 0.0,
            'episode_rewards': deque(maxlen=m_len),
            'action_history': deque(maxlen=h_len),
            'value_predictions': deque(maxlen=h_len)
        }
        
        self._start_time = datetime.now()
        self._high_water_mark = 0.0
        self.min_position_value = self.monitoring_config.min_position_value
        
    def update_on_trade(self, trade: Trade):
        """Met à jour les métriques après un trade."""
        self.trades_history.append(trade)
        self._update_trade_metrics(trade)
        self._update_performance_metrics()
        self._log_trade(trade)
        
    def update_on_order(self, order: Order):
        """Met à jour les métriques après un changement d'ordre."""
        self.orders_history.append(order)
        self._update_order_metrics(order)
        self._log_order(order)
        
    def update_on_position(self, position: Position):
        """Met à jour les métriques de position."""
        self.position_history.append(position)
        self.metrics['current_position'] = position.quantity
        self.metrics['avg_entry_price'] = position.entry_price
        self.metrics['unrealized_pnl'] = position.unrealized_pnl
        
        # Mettre à jour max position
        self.metrics['max_position'] = max(
            self.metrics['max_position'],
            abs(position.quantity)
        )
        
    def update_agent_metrics(self, 
                           reward: float,
                           action: np.ndarray,
                           value: float):
        """Met à jour les métriques de l'agent RL."""
        self.agent_metrics['episode_rewards'].append(reward)
        self.agent_metrics['action_history'].append(action)
        self.agent_metrics['value_predictions'].append(value)
        
        # Mettre à jour moyenne mobile
        self.agent_metrics['avg_reward'] = np.mean(
            list(self.agent_metrics['episode_rewards'])
        )

    def _calculate_pnl_map(self) -> float:
        """Calcule le ratio PnL/MAP de façon robuste"""
        if not self.position_history:
            return 0.0
            
        # Calculer la position moyenne absolue (MAP)
        positions = [abs(pos.quantity) for pos in self.position_history]
        mean_absolute_position = np.mean(positions) if positions else self.min_position_value
        
        # Éviter la division par zéro
        mean_absolute_position = max(mean_absolute_position, self.min_position_value)
        
        return self.metrics['total_pnl'] / mean_absolute_position

    def get_current_metrics(self) -> Dict:
        """Retourne toutes les métriques actuelles."""
        metrics = self.metrics.copy()
        
        # Ajouter métriques calculées
        metrics.update({
            'runtime': (datetime.now() - self._start_time).total_seconds() / 3600,
            'trades_per_hour': self._calculate_trades_per_hour(),
            'avg_spread': self._calculate_average_spread(),
            'position_exposure': self._calculate_position_exposure(),
            'agent_avg_reward': self.agent_metrics['avg_reward'],
            'pnl_map': self._calculate_pnl_map(),
        })
        
        # SOTA: Unified metrics from core
        if self.pnl_history:
            pnl_curve = [x['total_pnl'] for x in self.pnl_history]
            max_dd, dd_dur, curr_dd = calculate_drawdown(pnl_curve)
            metrics.update({
                'max_drawdown': max_dd,
                'drawdown_duration': dd_dur,
                'current_drawdown': curr_dd
            })
        
        return metrics
        
    def get_trade_summary(self) -> pd.DataFrame:
        """Génère un résumé des trades sous forme de DataFrame."""
        if not self.trades_history:
            return pd.DataFrame()
            
        df = pd.DataFrame(list(self.trades_history))
        
        # Ajouter métriques par trade
        df['pnl'] = df.apply(self._calculate_trade_pnl, axis=1)
        df['duration'] = df.apply(self._calculate_trade_duration, axis=1)
        df['effective_spread'] = df.apply(self._calculate_effective_spread, axis=1)
        
        return df
        
    def _update_trade_metrics(self, trade: Trade):
        """Met à jour les métriques liées aux trades."""
        self.metrics['trade_count'] += 1
        self.metrics['total_volume'] += trade.quantity * trade.price
        
        if trade.commission:
            self.metrics['total_fees'] += trade.commission
            
        # Mettre à jour PnL
        pnl = self._calculate_trade_pnl(trade)
        if pnl != 0:
            self.metrics['realized_pnl'] += pnl
            self.metrics['total_pnl'] = (
                self.metrics['realized_pnl'] + 
                self.metrics['unrealized_pnl']
            )
            
            if pnl > 0:
                self.metrics['winning_trades'] += 1
            else:
                self.metrics['losing_trades'] += 1
                
            # Mettre à jour win rate
            total_closed = self.metrics['winning_trades'] + self.metrics['losing_trades']
            self.metrics['win_rate'] = self.metrics['winning_trades'] / total_closed if total_closed > 0 else 0
            
            # Stocker le PnL pour l'historique
            self.pnl_history.append({
                'timestamp': datetime.now(),
                'total_pnl': self.metrics['total_pnl']
            })
                
    def _update_performance_metrics(self):
        """Met à jour les métriques de performance."""
        if len(self.pnl_history) > 1:
            # Calculer Sharpe Ratio (unified with core)
            pnl_curve = [p['total_pnl'] for p in self.pnl_history]
            returns = np.diff(pnl_curve)
            
            # Annualisation factor
            STEPS_PER_YEAR = self.monitoring_config.steps_per_year
            self.metrics['sharpe_ratio'] = calculate_sharpe(returns, annualization_factor=STEPS_PER_YEAR)
                
        # Calculer durée moyenne des trades
        if self.trades_history:
            durations = [
                self._calculate_trade_duration(t)
                for t in self.trades_history
                if hasattr(t, 'exit_time')
            ]
            if durations:
                self.metrics['avg_trade_duration'] = np.mean(durations)
                
    def _calculate_trades_per_hour(self) -> float:
        """Calcule le nombre moyen de trades par heure."""
        runtime_hours = (datetime.now() - self._start_time).total_seconds() / 3600
        if runtime_hours > 0:
            return self.metrics['trade_count'] / runtime_hours
        return 0.0
        
    def _calculate_average_spread(self) -> float:
        """Calcule le spread moyen des quotes."""
        if not self.quotes_history:
            return 0.0
        spreads = [
            (q['ask_price'] - q['bid_price']) / q['mid_price']
            for q in self.quotes_history
        ]
        return np.mean(spreads) if spreads else 0.0
        
    def _calculate_position_exposure(self) -> float:
        """Calcule l'exposition actuelle en pourcentage."""
        if not hasattr(self, 'account_value'):
            return 0.0
        position_value = abs(
            self.metrics['current_position'] * 
            self.metrics['avg_entry_price']
        )
        return position_value / self.account_value
        
    @staticmethod
    def _calculate_trade_pnl(trade: Trade) -> float:
        """Calcule le PnL d'un trade."""
        if not hasattr(trade, 'entry_price') or not hasattr(trade, 'exit_price'):
            return 0.0
        
        pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        if trade.side == "SELL":
            pnl = -pnl
            
        if trade.commission:
            pnl -= trade.commission
            
        return pnl
        
    @staticmethod
    def _calculate_trade_duration(trade: Trade) -> float:
        """Calcule la durée d'un trade en secondes."""
        if not hasattr(trade, 'entry_time') or not hasattr(trade, 'exit_time'):
            return 0.0
        return (trade.exit_time - trade.entry_time).total_seconds()
        
    @staticmethod
    def _calculate_effective_spread(trade: Trade) -> float:
        """Calcule le spread effectif d'un trade."""
        if not hasattr(trade, 'mid_price'):
            return 0.0
        return abs(trade.price - trade.mid_price) / trade.mid_price
        
    def _log_trade(self, trade: Trade):
        """Log les informations d'un trade."""
        logger.info(
            f"Trade executed: {trade.side} {trade.quantity:.8f} {self.market_config.symbol} "
            f"@ {trade.price:.8f} | PnL: {self._calculate_trade_pnl(trade):.8f}"
        )
        
    def _log_order(self, order: Order):
        """Log les informations d'un ordre."""
        logger.debug(
            f"Order {order.order_id} {order.status.name}: {order.side.name} "
            f"{order.quantity:.8f} {self.market_config.symbol} @ {order.price:.8f}"
        )