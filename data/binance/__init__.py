# market_maker/data/binance/__init__.py
"""
Module pour l'intégration avec Binance.
"""

__all__ = ['BinanceStreamHandler']

def __getattr__(name):
    if name == 'BinanceStreamHandler':
        from .stream import BinanceStreamHandler
        return BinanceStreamHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")