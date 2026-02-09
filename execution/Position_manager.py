# market_maker/execution/position_manager.py

import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from binance import AsyncClient
from collections import deque
from core.types import Trade, Order, OrderSide, Position
from core.config import MarketConfig, MonitoringConfig
from core.exceptions import PositionError

logger = logging.getLogger(__name__)

class PositionManager:
    """
    Gère les positions et calcule le PnL en temps réel.
    Maintient l'historique des trades et calcule les métriques de performance.
    """
    
    def __init__(self, client: AsyncClient, market_config: MarketConfig, monitoring_config: Optional[MonitoringConfig] = None, mode: str = 'margin'):
        self.client = client
        self.market_config = market_config
        self.monitoring_config = monitoring_config if monitoring_config else MonitoringConfig()
        self.mode = mode  # 'futures', 'spot', or 'margin'
        
        # État des positions
        self.long_positions: List[Tuple[float, float, float]] = []  # [(qty, price, commission)]
        self.short_positions: List[Tuple[float, float, float]] = [] 
        self.current_position = 0.0
        self.avg_entry_price = 0.0
        
        # Historique des trades
        h_len = self.monitoring_config.history_maxlen
        self.trades_history = deque(maxlen=h_len)
        self.pnl_history = deque(maxlen=h_len)
        
        # Métriques
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_commission = 0.0
        self.total_volume = 0.0
        
        # État
        self._last_price = None
        self._initialized = False
        
    async def initialize(self):
        """Initialise l'état des positions depuis Binance."""
        try:
            symbol = self.market_config.symbol

            # Sync current position directly from exchange (reliable, no reconstruction needed)
            await self.sync_position()

            # Récupérer le prix actuel (mode-aware)
            if self.mode == 'futures':
                tickers = await self.client.futures_symbol_ticker(symbol=symbol)
                if isinstance(tickers, list):
                    self._last_price = float(tickers[0]['price'])
                else:
                    self._last_price = float(tickers['price'])
            elif self.mode == 'spot':
                ticker = await self.client.get_symbol_ticker(symbol=symbol)
                self._last_price = float(ticker['price'])
            else:  # margin
                ticker = await self.client.get_margin_price_index(symbol=symbol)
                self._last_price = float(ticker['price'])
            
            self._initialized = True
            logger.info(
                f"Position Manager initialized. Current position: {self.current_position}, "
                f"Entry price: {self.avg_entry_price:.8f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize position manager: {e}")
            raise PositionError(f"Initialization failed: {str(e)}")
            
    async def sync_position(self):
        """Query the actual current position directly from the exchange.

        Much more reliable than reconstructing from trade history — gives
        the authoritative position, entry price, and unrealized PnL in one call.
        """
        symbol = self.market_config.symbol
        try:
            if self.mode == 'futures':
                positions = await self.client.futures_position_information(symbol=symbol)
                for pos in positions:
                    if pos.get('symbol', '').upper() == symbol.upper():
                        qty = float(pos.get('positionAmt', 0))
                        entry = float(pos.get('entryPrice', 0))
                        unrealized = float(pos.get('unRealizedProfit', 0))

                        self.current_position = qty
                        self.avg_entry_price = entry
                        self.unrealized_pnl = unrealized

                        # Rebuild internal position lists to stay consistent
                        self.long_positions.clear()
                        self.short_positions.clear()
                        if qty > 0:
                            self.long_positions.append((qty, entry, 0))
                        elif qty < 0:
                            self.short_positions.append((abs(qty), entry, 0))

                        logger.info(f"Position synced (futures): qty={qty}, "
                                    f"entry={entry:.5f}, uPnL={unrealized:.4f}")
                        return

            elif self.mode == 'spot':
                # Spot: query account balances for the base asset
                account = await self.client.get_account()
                base_asset = symbol.replace('USDC', '').replace('USDT', '').replace('BUSD', '')
                for balance in account.get('balances', []):
                    if balance['asset'] == base_asset:
                        free = float(balance.get('free', 0))
                        locked = float(balance.get('locked', 0))
                        self.current_position = free + locked
                        self.long_positions.clear()
                        self.short_positions.clear()
                        if self.current_position > 0:
                            self.long_positions.append((self.current_position, 0, 0))
                        logger.info(f"Position synced (spot): {base_asset}={self.current_position}")
                        return

            else:  # margin
                account = await self.client.get_margin_account()
                base_asset = symbol.replace('USDC', '').replace('USDT', '').replace('BUSD', '')
                for asset_info in account.get('userAssets', []):
                    if asset_info['asset'] == base_asset:
                        net = float(asset_info.get('netAsset', 0))
                        self.current_position = net
                        self.long_positions.clear()
                        self.short_positions.clear()
                        if net > 0:
                            self.long_positions.append((net, 0, 0))
                        elif net < 0:
                            self.short_positions.append((abs(net), 0, 0))
                        logger.info(f"Position synced (margin): {base_asset}={net}")
                        return

            logger.warning(f"No position found for {symbol}")
        except Exception as e:
            logger.error(f"Failed to sync position: {e}")

    def _process_historical_trades(self, trades: List[Dict]):
        """Traite l'historique des trades pour reconstituer les positions."""
        df = pd.DataFrame(trades)
        if df.empty:
            return
            
        # Conversion des types
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['price'] = pd.to_numeric(df['price'])
        df['qty'] = pd.to_numeric(df['qty'])
        df['commission'] = pd.to_numeric(df['commission'])
        
        # Suivre les positions
        inventory_net = 0
        running_pnl = 0.0
        
        for idx, row in df.iterrows():
            price = row['price']
            qty = row['qty']
            commission = row['commission']
            is_buyer = row['isBuyer']
            
            if is_buyer:  # ACHAT
                if inventory_net < 0:  # Fermeture de short
                    close_qty = min(qty, abs(inventory_net))
                    if close_qty > 0:
                        avg_short_price = self._get_average_price(self.short_positions)
                        pnl = (avg_short_price - price) * close_qty - commission
                        running_pnl += pnl
                        
                        # Mise à jour positions short
                        remaining_short = abs(inventory_net) - close_qty
                        if remaining_short > 0:
                            self.short_positions = [(remaining_short, avg_short_price, 0)]
                        else:
                            self.short_positions.clear()
                            
                        # Nouvelle position long si reste
                        new_qty = qty - close_qty
                        if new_qty > 0:
                            self.long_positions.append(
                                (new_qty, price, commission * (new_qty/qty))
                            )
                else:
                    self.long_positions.append((qty, price, commission))
                    
                inventory_net += qty
                
            else:  # VENTE
                if inventory_net > 0:  # Fermeture de long
                    close_qty = min(qty, inventory_net)
                    if close_qty > 0:
                        avg_long_price = self._get_average_price(self.long_positions)
                        pnl = (price - avg_long_price) * close_qty - commission
                        running_pnl += pnl
                        
                        # Mise à jour positions long
                        remaining_long = inventory_net - close_qty
                        if remaining_long > 0:
                            self.long_positions = [(remaining_long, avg_long_price, 0)]
                        else:
                            self.long_positions.clear()
                            
                        # Nouvelle position short si reste
                        new_qty = qty - close_qty
                        if new_qty > 0:
                            self.short_positions.append(
                                (new_qty, price, commission * (new_qty/qty))
                            )
                else:
                    self.short_positions.append((qty, price, commission))
                    
                inventory_net -= qty
                
        # Mettre à jour l'état final
        self.current_position = inventory_net
        self.realized_pnl = running_pnl
        self.total_commission = df['commission'].sum()
        self.total_volume = (df['price'] * df['qty']).sum()
        
        if self.current_position > 0:
            self.avg_entry_price = self._get_average_price(self.long_positions)
        elif self.current_position < 0:
            self.avg_entry_price = self._get_average_price(self.short_positions)
        
    def update_on_trade(self, trade: Trade):
        """Met à jour les positions après un nouveau trade."""
        quantity = trade.quantity
        price = trade.price
        commission = trade.commission if trade.commission else 0
        
        if trade.side == OrderSide.BUY:
            if self.current_position < 0:  # Fermeture de short
                self._close_short_position(quantity, price, commission)
            else:  # Nouvelle position long
                self._open_long_position(quantity, price, commission)
                
        else:  # SELL
            if self.current_position > 0:  # Fermeture de long
                self._close_long_position(quantity, price, commission)
            else:  # Nouvelle position short
                self._open_short_position(quantity, price, commission)
                
        # Mettre à jour les métriques
        self._update_metrics(trade)
        self.trades_history.append(trade)
        
    def _close_short_position(self, quantity: float, price: float, commission: float):
        """Ferme une position short."""
        close_qty = min(quantity, abs(self.current_position))
        avg_short_price = self._get_average_price(self.short_positions)
        
        # Calculer PnL
        pnl = (avg_short_price - price) * close_qty - commission
        self.realized_pnl += pnl
        
        # Mettre à jour position
        self.current_position += close_qty

        # Mettre à jour short positions — current_position is already updated,
        # so abs(current_position) IS the remaining short qty
        remaining_qty = abs(self.current_position)
        if remaining_qty > 0:
            self.short_positions = [(remaining_qty, avg_short_price, 0)]
        else:
            self.short_positions.clear()
            
        # Ajouter nouvelle position long si reste
        remaining_buy = quantity - close_qty
        if remaining_buy > 0:
            self._open_long_position(remaining_buy, price, commission * (remaining_buy/quantity))
            
    def _close_long_position(self, quantity: float, price: float, commission: float):
        """Ferme une position long."""
        close_qty = min(quantity, self.current_position)
        avg_long_price = self._get_average_price(self.long_positions)
        
        # Calculer PnL
        pnl = (price - avg_long_price) * close_qty - commission
        self.realized_pnl += pnl
        
        # Mettre à jour position
        self.current_position -= close_qty

        # Mettre à jour long positions — current_position is already updated,
        # so current_position IS the remaining long qty
        remaining_qty = self.current_position
        if remaining_qty > 0:
            self.long_positions = [(remaining_qty, avg_long_price, 0)]
        else:
            self.long_positions.clear()
            
        # Ajouter nouvelle position short si reste
        remaining_sell = quantity - close_qty
        if remaining_sell > 0:
            self._open_short_position(remaining_sell, price, commission * (remaining_sell/quantity))
            
    def _open_long_position(self, quantity: float, price: float, commission: float):
        """Ouvre une nouvelle position long."""
        self.long_positions.append((quantity, price, commission))
        self.current_position += quantity
        self.avg_entry_price = self._get_average_price(self.long_positions)
        
    def _open_short_position(self, quantity: float, price: float, commission: float):
        """Ouvre une nouvelle position short."""
        self.short_positions.append((quantity, price, commission))
        self.current_position -= quantity
        self.avg_entry_price = self._get_average_price(self.short_positions)
        
    def update_unrealized_pnl(self, current_price: float):
        """Met à jour le PnL non réalisé avec le prix actuel."""
        self._last_price = current_price
        self.unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calcule le PnL non réalisé."""
        unrealized = 0.0
        
        # PnL des positions longues
        for qty, price, _ in self.long_positions:
            unrealized += (current_price - price) * qty
            
        # PnL des positions shorts
        for qty, price, _ in self.short_positions:
            unrealized += (price - current_price) * qty
            
        return unrealized
        
    def _get_average_price(self, positions: List[Tuple[float, float, float]]) -> float:
        """Calcule le prix moyen d'entrée pour une liste de positions."""
        if not positions:
            return 0.0
        total_qty = sum(qty for qty, _, _ in positions)
        if total_qty == 0:
            return 0.0
        return sum(qty * price for qty, price, _ in positions) / total_qty
        
    def _update_metrics(self, trade: Trade):
        """Met à jour les métriques après un trade."""
        self.total_volume += trade.price * trade.quantity
        if trade.commission:
            self.total_commission += trade.commission
            
        # Mettre à jour historique PnL
        self.pnl_history.append({
            'timestamp': trade.timestamp,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl
        })
        
    def get_position_info(self) -> Position:
        """Retourne l'information complète sur la position actuelle."""
        return Position(
            symbol=self.market_config.symbol,
            quantity=self.current_position,
            entry_price=self.avg_entry_price,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            timestamp=datetime.now()
        )
        
    def get_metrics(self) -> Dict:
        """Retourne les métriques de trading actuelles."""
        return {
            'current_position': self.current_position,
            'avg_entry_price': self.avg_entry_price,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'total_volume': self.total_volume,
            'total_commission': self.total_commission,
            'total_trades': len(self.trades_history),
            'last_price': self._last_price
        }