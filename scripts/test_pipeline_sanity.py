# scripts/test_pipeline_sanity.py
import torch
import numpy as np
import logging
import sys
import os

# Ajout du path pour les imports
sys.path.append(os.getcwd())

from data.processor import LOBFeatureProcessor, MarketFeatureProcessor
from models.rl.PPO_agent import PPOAgent
from core.config import RLConfig, MarketConfig, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_processors():
    logger.info("Testing Feature Processors...")
    
    # Mock Orderbook
    orderbook = {
        'bids': [[100 - i*0.1, 10.0] for i in range(20)],
        'asks': [[100.1 + i*0.1, 10.0] for i in range(20)]
    }
    
    market_config = MarketConfig()
    lob_proc = LOBFeatureProcessor(market_config=market_config)
    market_proc = MarketFeatureProcessor(market_config=market_config)
    
    # Test LOB
    lob_features = lob_proc.process(orderbook)
    logger.info(f"LOB Features shape: {lob_features.shape} (Expected: (40,))")
    assert lob_features.shape == (40,), f"LOB features mismatch: {lob_features.shape}"
    
    # Test Market (needs history for some features)
    # Using explicit timestamps
    start_ts = datetime(2023, 1, 1, 12, 0, 0)
    for i in range(20):
        market_proc.update_orderbook(orderbook, timestamp=start_ts + timedelta(seconds=i))
    
    # Mock some trades
    trade = {'p': 100.05, 'q': 1.0, 'm': True}
    for i in range(50):
        market_proc.update_trades(trade, timestamp=start_ts + timedelta(seconds=i))
        
    market_features = market_proc.process(orderbook, timestamp=start_ts + timedelta(seconds=70))
    logger.info(f"Market Features shape: {market_features.shape} (Expected: (13,))")
    assert market_features.shape == (13,), f"Market features mismatch: {market_features.shape}"
    
    logger.info("✅ Feature Processors Test Passed!")
    return lob_features, market_features

def test_agent_forward():
    logger.info("Testing Agent Forward Pass...")
    rl_config = RLConfig()
    model_config = ModelConfig()
    agent = PPOAgent(model_config=model_config, rl_config=rl_config)
    
    # Mock state
    state = {
        'lob_features': np.random.randn(40).tolist(),
        'market_features': np.random.randn(13).tolist(),
        'inventory': 0.0,
        'time_remaining': 1.0
    }
    
    # First forward
    action1, log_prob1, value1 = agent.select_action(state)
    logger.info(f"Action 1: {action1}")
    
    # Second forward - check no reset
    # We can't easily check internal weights without manual comparison, 
    # but at least it shouldn't crash.
    action2, log_prob2, value2 = agent.select_action(state)
    logger.info(f"Action 2: {action2}")
    
    assert action1.shape == (4,), f"Action shape mismatch: {action1.shape}"
    
    logger.info("✅ Agent Forward Pass Test Passed!")

from datetime import datetime, timedelta

if __name__ == "__main__":
    try:
        test_feature_processors()
        test_agent_forward()
        logger.info("\n🏆 ALL SANITY TESTS PASSED!")
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
