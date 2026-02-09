import numpy as np
from typing import List, Tuple
from .base import Metric

class Drawdown(Metric):
    """
    Calculates maximum and current drawdown.
    """
    
    def calculate(self, pnl_curve: List[float]) -> Tuple[float, int, float]:
        """
        Calculates maximum drawdown, duration of max drawdown, and current drawdown.
        Returns: (max_drawdown, duration, current_drawdown)
        """
        if not pnl_curve or len(pnl_curve) < 1:
            return 0.0, 0, 0.0
                
        pnl_array = np.array(pnl_curve)
        peak = np.maximum.accumulate(pnl_array)
        drawdown = peak - pnl_array
        
        max_drawdown = np.max(drawdown)
        current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0.0
        
        # Simple duration calc: number of steps since last peak
        last_peak_idx = np.where(pnl_array == peak[-1])[0][-1]
        duration = len(pnl_array) - 1 - last_peak_idx
        
        return float(max_drawdown), int(duration), float(current_drawdown)
