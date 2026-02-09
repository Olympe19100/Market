from typing import List, Tuple
from .sharpe import SharpeRatio
from .drawdown import Drawdown
from .tracker import MetricsTracker

# Functions for backward compatibility
def calculate_sharpe(returns: List[float], annualization_factor: float = 252) -> float:
    """
    Backward compatibility function for Sharpe Ratio calculation.
    """
    return SharpeRatio(annualization_factor=annualization_factor).calculate(returns)

def calculate_drawdown(pnl_curve: List[float]) -> Tuple[float, int, float]:
    """
    Backward compatibility function for Drawdown calculation.
    """
    return Drawdown().calculate(pnl_curve)

__all__ = [
    'SharpeRatio',
    'Drawdown',
    'MetricsTracker',
    'calculate_sharpe',
    'calculate_drawdown'
]
