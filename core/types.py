# market_maker/core/types.py
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"

@dataclass
class OrderBookLevel:
    price: float
    quantity: float

@dataclass
class OrderBook:
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime
    
    @property
    def mid_price(self) -> float:
        """Simple Mid Price."""
        return (self.bids[0].price + self.asks[0].price) / 2
    
    @property
    def weighted_mid_price(self) -> float:
        """
        Micro-price (Weighted Mid Price) weighted by the opposite side's volume.
        Ref: SOTA high-frequency trading.
        """
        best_bid_vol = self.bids[0].quantity
        best_ask_vol = self.asks[0].quantity
        total_vol = best_bid_vol + best_ask_vol
        
        if total_vol == 0:
            return self.mid_price
            
        return (self.bids[0].price * best_ask_vol + self.asks[0].price * best_bid_vol) / total_vol

    @property
    def spread(self) -> float:
        return self.asks[0].price - self.bids[0].price
    
    @property
    def imbalance(self) -> float:
        bid_vol = sum(level.quantity for level in self.bids)
        ask_vol = sum(level.quantity for level in self.asks)
        total_vol = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

@dataclass
class Trade:
    price: float
    quantity: float
    timestamp: datetime
    side: OrderSide
    is_buyer_maker: bool
    commission: Optional[float] = None
    commission_asset: Optional[str] = None

@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    status: OrderStatus
    timestamp: datetime
    order_id: str
    client_order_id: Optional[str] = None
    executed_qty: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    unrealized_pnl: float
    timestamp: datetime
    realized_pnl: float = 0.0

@dataclass
class MarketState:
    orderbook: OrderBook
    position: Position
    active_orders: List[Order]
    timestamp: datetime