import numpy as np
from typing import List
from .base import Metric

class SharpeRatio(Metric):
    """
    Calculates the annualised Sharpe Ratio.
    """
    
    def __init__(self, annualization_factor: float = 252):
        self.annualization_factor = annualization_factor
        
    def calculate(self, returns: List[float]) -> float:
        """
        Calculates the annualised Sharpe Ratio from a list of returns.
        """
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        std = np.std(returns_array, ddof=1)
        
        if std < 1e-9:
            return 0.0
            
        sharpe = np.mean(returns_array) / std
        annualized_sharpe = np.sqrt(self.annualization_factor) * sharpe 
        return float(annualized_sharpe)
