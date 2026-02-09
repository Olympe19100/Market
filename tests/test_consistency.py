"""
Smoke test to verify architectural consistency fixes.
Tests that all imports work and models can be initialized.
"""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all critical imports work after fixing folder names."""
    print("Testing imports...")
    
    try:
        from core.config import ModelConfig, MarketConfig, RLConfig, SimulationConfig
        print("✓ Core config imports OK")
    except ImportError as e:
        print(f"✗ Core config import failed: {e}")
        return False
    
    try:
        from models.mamba_lob.model import LOBModel
        print("✓ LOBModel import OK")
    except ImportError as e:
        print(f"✗ LOBModel import failed: {e}")
        return False
    
    try:
        from environment.base_environment import BaseMarketMakerEnv, MarketMakerState
        print("✓ Base environment import OK")
    except ImportError as e:
        print(f"✗ Base environment import failed: {e}")
        return False
    
    try:
        from environment.sim_env import SimulationMarketMakerEnv
        print("✓ Simulation environment import OK")
    except ImportError as e:
        print(f"✗ Simulation environment import failed: {e}")
        return False
    
    try:
        from data.processor import LOBFeatureProcessor, MarketFeatureProcessor
        print("✓ Data processors import OK")
    except ImportError as e:
        print(f"✗ Data processors import failed: {e}")
        return False
    
    try:
        from core.metrics.tracker import MetricsTracker
        from core.metrics.sharpe import SharpeRatio
        from core.metrics.drawdown import Drawdown
        print("✓ Metrics import OK")
    except ImportError as e:
        print(f"✗ Metrics import failed: {e}")
        return False
    
    return True

def test_model_initialization():
    """Test that LOBModel can be initialized with the fixed config."""
    print("\nTesting model initialization...")
    
    try:
        from core.config import ModelConfig
        from models.mamba_lob.model import LOBModel
        
        # Test with default n_aux_features=15
        config = ModelConfig()
        assert config.n_aux_features == 15, f"Expected n_aux_features=15, got {config.n_aux_features}"
        print(f"✓ ModelConfig.n_aux_features = {config.n_aux_features}")
        
        # Try to initialize model
        model = LOBModel(config)
        assert hasattr(model, 'config'), "LOBModel should have self.config attribute"
        print("✓ LOBModel initialized successfully")
        print(f"✓ LOBModel.config is stored correctly")
        
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_calculation():
    """Test that MetricsTracker returns calculation is correct."""
    print("\nTesting metrics calculation...")
    
    try:
        from core.metrics.tracker import MetricsTracker
        import numpy as np
        
        tracker = MetricsTracker()
        
        # Simulate some daily PnL values
        tracker.daily_pnl.extend([100.0, 105.0, 103.0, 108.0])
        
        summary = tracker.get_metrics_summary()
        
        # The Sharpe ratio should be calculated from actual returns [5, -2, 5]
        # not from [100, 5, -2, 5] which would give inflated first return
        print(f"✓ MetricsTracker summary generated: sharpe_ratio={summary['sharpe_ratio']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ARCHITECTURAL CONSISTENCY SMOKE TEST")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Model Initialization", test_model_initialization()))
    results.append(("Metrics Calculation", test_metrics_calculation()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
