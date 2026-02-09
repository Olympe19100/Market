import pytest
import numpy as np
from core.metrics.sharpe import SharpeRatio
from core.metrics.drawdown import Drawdown
from core.metrics.tracker import MetricsTracker
from core.metrics import calculate_sharpe, calculate_drawdown

def test_sharpe_ratio_calculation():
    # Test with constant returns (should be high but stable)
    returns = [0.01] * 10
    # Add a tiny bit of noise to avoid zero std
    returns[0] = 0.010001
    
    sr = SharpeRatio(annualization_factor=252)
    sharpe = sr.calculate(returns)
    assert isinstance(sharpe, float)
    assert sharpe > 0
    
    # Test with zero returns
    assert sr.calculate([0, 0, 0]) == 0.0
    
    # Test with single return
    assert sr.calculate([0.01]) == 0.0

def test_drawdown_calculation():
    # Test with clear drawdown
    pnl = [100, 110, 105, 102, 108, 115]
    dd = Drawdown()
    max_dd, duration, current_dd = dd.calculate(pnl)
    
    assert max_dd == 8.0  # Peak 110, trough 102
    assert current_dd == 0.0 # End is 115, which is a new peak
    assert duration == 0
    
    # Test current drawdown
    pnl_2 = [100, 110, 105]
    max_dd, duration, current_dd = dd.calculate(pnl_2)
    assert max_dd == 5.0
    assert current_dd == 5.0
    assert duration == 1 # 1 step since peak 110

def test_metrics_tracker():
    tracker = MetricsTracker(window_size=100)
    
    # Test update
    tracker.update({'inventory': 10, 'rolling_pnl': 50.0})
    assert tracker.metrics['inventory'] == 10
    assert tracker.metrics['total_pnl'] == 50.0 # unrealized is 0
    
    # Test daily update
    tracker.update_daily_pnl()
    assert len(tracker.daily_pnl) == 1
    
    # Test summary
    summary = tracker.get_metrics_summary()
    assert summary['inventory'] == 10
    assert summary['total_pnl'] == 50.0

def test_backward_compatibility():
    returns = [0.01, -0.01, 0.02]
    val1 = calculate_sharpe(returns)
    val2 = SharpeRatio().calculate(returns)
    assert val1 == val2
    
    pnl = [100, 90, 110]
    val3 = calculate_drawdown(pnl)
    val4 = Drawdown().calculate(pnl)
    assert val3 == val4

def test_zero_division_safety():
    # Sharpe with zero volatility
    sr = SharpeRatio()
    assert sr.calculate([0.01, 0.01, 0.01]) == 0.0
    
    # Empty list
    assert sr.calculate([]) == 0.0
    
    dd = Drawdown()
    assert dd.calculate([]) == (0.0, 0, 0.0)
