import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.metrics import calculate_sharpe, calculate_drawdown, SharpeRatio, Drawdown, MetricsTracker

def test_functions():
    print("Testing backward compatibility functions...")
    returns = [0.01, -0.005, 0.02, 0.015, -0.01]
    sharpe = calculate_sharpe(returns)
    print(f"Sharpe Ratio: {sharpe}")
    assert isinstance(sharpe, float)
    
    pnl = [100, 105, 102, 110, 108]
    max_dd, duration, curr_dd = calculate_drawdown(pnl)
    print(f"Drawdown: max={max_dd}, duration={duration}, current={curr_dd}")
    assert isinstance(max_dd, float)
    assert isinstance(duration, int)
    assert isinstance(curr_dd, float)
    print("Function tests passed!")

def test_classes():
    print("\nTesting new classes...")
    sr = SharpeRatio()
    returns = [0.01, -0.005, 0.02, 0.015, -0.01]
    sharpe = sr.calculate(returns)
    print(f"Sharpe Ratio (class): {sharpe}")
    
    dd = Drawdown()
    pnl = [100, 105, 102, 110, 108]
    results = dd.calculate(pnl)
    print(f"Drawdown (class): {results}")
    
    tracker = MetricsTracker(window_size=10)
    tracker.update({'spreads': 0.001, 'rewards': 1.0})
    tracker.update_daily_pnl()
    summary = tracker.get_metrics_summary()
    print(f"Tracker summary: {summary}")
    assert 'total_pnl' in summary
    print("Class tests passed!")

if __name__ == "__main__":
    try:
        test_functions()
        test_classes()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
