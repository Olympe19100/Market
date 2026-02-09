# market_maker/data/__init__.py
"""
Module de gestion des données et du streaming.
"""
from .processor import LOBFeatureProcessor, MarketFeatureProcessor

__all__ = [
    'BinanceStreamHandler',
    'LOBFeatureProcessor',
    'MarketFeatureProcessor'
]

def __getattr__(name):
    if name == 'BinanceStreamHandler':
        from data.binance.stream import BinanceStreamHandler
        return BinanceStreamHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
