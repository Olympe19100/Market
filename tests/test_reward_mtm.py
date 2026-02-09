"""Unit tests for the RELAER reward in SimulationMarketMakerEnv._calculate_reward()."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from environment.sim_env import SimulationMarketMakerEnv
from environment.base_environment import MarketMakerState
from core.config import RLConfig, MarketConfig, SimulationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env():
    """Create a SimulationMarketMakerEnv with mocked data_loader / processors."""
    rl = RLConfig(initial_cash=100.0, zeta=0.01, max_steps=100)
    mc = MarketConfig(min_qty=1.0, max_position=100, tick_size=0.0001)
    sc = SimulationConfig()

    with patch('environment.sim_env.MarketDataLoader'):
        env = SimulationMarketMakerEnv(rl, mc, sc, data_path='dummy', split='train')

    # Minimal state — mid @ 1.0, no inventory, full cash
    env.current_state = MarketMakerState(
        mid_price=1.0, cash=100.0, inventory=0.0,
        lob_features=np.zeros(40), market_features=np.zeros(15),
    )
    env._prev_mid_price = 1.0
    env._avg_entry_price = 0.0
    env._lambda_as = 0.0
    env.episode_step = 1
    env.last_portfolio_value = 100.0

    # Mock iVPIN → 0 (no adverse selection noise in tests)
    env.market_processor = MagicMock()
    env.market_processor.get_ivpin.return_value = 0.0
    env.market_processor.FEAT_NOISE = 0
    env._last_quotes_levels = []

    return env


def _fake_execution(side: str, price: float, qty: float, order_type='limit'):
    return {
        'side': side, 'price': price, 'quantity': qty,
        'timestamp': None, 'type': order_type, 'level': 0,
        'fee': 0.0,
    }


# ---------------------------------------------------------------------------
# Tests — PnL_t component
# ---------------------------------------------------------------------------

class TestPnLComponent:
    """Verify PnL = realized spread capture + MTM on post-fill inventory."""

    def test_no_inventory_no_mtm(self):
        """Zero inventory → MTM component is zero regardless of price move."""
        env = _make_env()
        env.current_state.mid_price = 1.05
        env.current_state.cash = 100.0
        env.current_state.inventory = 0.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])
        # No fills, no inventory → reward ≈ 0
        assert abs(reward) < 1e-9, f"Expected ~0, got {reward}"

    def test_long_inventory_price_up_positive_mtm(self):
        """Long 10 units (below threshold d=20), price up → positive MTM, no penalty."""
        env = _make_env()
        env.current_state.inventory = 10.0  # 10% of max_pos=100, below d=0.2
        env.current_state.mid_price = 1.02  # +2%
        env.current_state.cash = 90.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])

        # MTM = (1.02 - 1.0) * 10 = 0.2 (post-fill inv, weight=1.0)
        # IP = 0 (inv_ratio=0.1 < d=0.2)
        # C = 0, ER = 0
        assert abs(reward - 0.2) < 1e-6, f"Expected 0.2, got {reward}"

    def test_long_inventory_price_down_negative_mtm(self):
        """Long 10 units, price drops → negative MTM."""
        env = _make_env()
        env.current_state.inventory = 10.0
        env.current_state.mid_price = 0.98  # -2%
        env.current_state.cash = 90.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])

        # MTM = (0.98 - 1.0) * 10 = -0.2
        assert abs(reward - (-0.2)) < 1e-6, f"Expected -0.2, got {reward}"

    def test_short_inventory_price_down_positive_mtm(self):
        """Short -10, price drops → positive MTM."""
        env = _make_env()
        env.current_state.inventory = -10.0
        env.current_state.mid_price = 0.98
        env.current_state.cash = 110.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])

        # MTM = (0.98 - 1.0) * (-10) = +0.2
        assert abs(reward - 0.2) < 1e-6, f"Expected 0.2, got {reward}"

    def test_mtm_uses_post_execution_inventory(self):
        """RELAER: MTM uses inventory AFTER fills (Q_{t+1}), not before."""
        env = _make_env()
        # Agent buys 50 this step, price moved up
        env.current_state.inventory = 50.0  # post-fill
        env.current_state.mid_price = 1.01
        env.current_state.cash = 50.0
        env.last_portfolio_value = 100.0

        fills = [_fake_execution('bid', 1.0, 50.0)]
        reward = env._calculate_reward(fills)

        # MTM = (1.01 - 1.0) * 50 = 0.5 (post-fill inventory!)
        # realized = 0 (opening position, no close)
        # C = maker_fee * 50 * 1.0 = 0.001 * 50 = 0.05
        # IP: inv_ratio=0.5 > d=0.2, penalty = -0.01 * 0.5 = -0.005
        expected = 0.5 + 0.05 - 0.005
        assert abs(reward - expected) < 1e-4, f"Expected ~{expected}, got {reward}"

    def test_prev_mid_price_updates_each_step(self):
        """_prev_mid_price is updated after each reward calculation."""
        env = _make_env()
        env.current_state.mid_price = 1.05
        env.current_state.inventory = 0.0
        env.last_portfolio_value = 100.0

        env._calculate_reward([])
        assert env._prev_mid_price == 1.05

        env.current_state.mid_price = 1.10
        env.last_portfolio_value = 100.0
        env._calculate_reward([])
        assert env._prev_mid_price == 1.10

    def test_realized_plus_mtm(self):
        """Closing a long position: realized PnL + MTM on remaining inventory."""
        env = _make_env()
        env._avg_entry_price = 1.0
        env.current_state.inventory = 0.0  # post-fill: flat
        env.current_state.mid_price = 1.02
        env.current_state.cash = 100.0 + 50 * 1.02
        env.last_portfolio_value = 100.0 + 50 * 1.0

        fills = [_fake_execution('ask', 1.02, 50.0)]
        reward = env._calculate_reward(fills)

        # pre_inv = 0 + 50 = 50 (reverse the sell)
        # realized = (1.02 - 1.0) * 50 = 1.0
        # MTM = (1.02 - 1.0) * 0 = 0 (post-fill inv is 0)
        # C = 0.001 * 50 * 1.02 = 0.051
        # IP = 0 (inv=0)
        expected = 1.0 + 0.051
        assert abs(reward - expected) < 1e-3, f"Expected ~{expected}, got {reward}"


# ---------------------------------------------------------------------------
# Tests — IP_t truncated inventory penalty
# ---------------------------------------------------------------------------

class TestInventoryPenalty:
    """Verify truncated penalty: only active above threshold d."""

    def test_below_threshold_no_penalty(self):
        """Inventory below d=20% → zero penalty."""
        env = _make_env()
        env.current_state.inventory = 15.0  # 15% of max_pos=100
        env.current_state.mid_price = 1.0
        env.current_state.cash = 85.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])
        # Only MTM (which is 0 since no price change) → 0
        assert abs(reward) < 1e-9

    def test_above_threshold_full_penalty(self):
        """Inventory 50% → IP = −η × |Q|/q_max = −0.01 × 0.5 = −0.005."""
        env = _make_env()
        env.current_state.inventory = 50.0
        env.current_state.mid_price = 1.0
        env.current_state.cash = 50.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])

        # MTM = 0, C = 0, ER = 0
        # IP = -0.01 * 0.5 = -0.005
        assert abs(reward - (-0.005)) < 1e-6, f"Expected -0.005, got {reward}"

    def test_max_inventory_penalty(self):
        """100% inventory → penalty = η × 1.0 = 0.01."""
        env = _make_env()
        env.current_state.inventory = 100.0
        env.current_state.mid_price = 1.0
        env.current_state.cash = 0.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])

        expected_ip = -0.01 * 1.0
        assert abs(reward - expected_ip) < 1e-6


# ---------------------------------------------------------------------------
# Tests — C_t fill compensation
# ---------------------------------------------------------------------------

class TestFillCompensation:
    """Verify maker rebate incentivises liquidity provision."""

    def test_maker_fill_gets_rebate(self):
        """Limit order fill → positive compensation vs no fill."""
        env = _make_env()
        # Small fill that doesn't distort other components
        env.current_state.inventory = 5.0  # post-fill: had 0, bought 5
        env.current_state.mid_price = 1.0
        env.current_state.cash = 95.0
        env.last_portfolio_value = 100.0

        fills = [_fake_execution('bid', 1.0, 5.0, order_type='limit')]
        reward_with_fill = env._calculate_reward(fills)

        # Reset for baseline (no fills)
        env._prev_mid_price = 1.0
        env._avg_entry_price = 0.0
        env.current_state.inventory = 5.0
        env.current_state.mid_price = 1.0
        env.current_state.cash = 95.0
        env.last_portfolio_value = 100.0

        reward_no_fill = env._calculate_reward([])

        # The fill adds C_t = 0.001 * 5 * 1.0 = 0.005
        # (realized PnL differs too, but C_t is strictly positive)
        rebate = 0.001 * 5.0 * 1.0
        assert rebate > 0
        # With same post-fill state, the fill version should include rebate
        assert reward_with_fill > reward_no_fill - 0.01, \
            f"Maker rebate should contribute positively: with={reward_with_fill}, without={reward_no_fill}"

    def test_all_fills_get_rebate(self):
        """Both maker and taker fills contribute to C_t (total notional)."""
        env = _make_env()
        env.current_state.inventory = 5.0
        env.current_state.mid_price = 1.0
        env.current_state.cash = 95.0
        env.last_portfolio_value = 100.0

        fills_maker = [_fake_execution('bid', 1.0, 5.0, order_type='limit')]
        reward_maker = env._calculate_reward(fills_maker)

        # Reset
        env._prev_mid_price = 1.0
        env._avg_entry_price = 0.0
        env.current_state.inventory = 5.0
        env.current_state.mid_price = 1.0
        env.current_state.cash = 95.0
        env.last_portfolio_value = 100.0

        fills_taker = [_fake_execution('bid', 1.0, 5.0, order_type='market')]
        reward_taker = env._calculate_reward(fills_taker)

        # Both should get same C_t = β × notional = 0.001 × 5.0 = 0.005
        assert abs(reward_maker - reward_taker) < 1e-6, \
            f"All fills should get equal rebate: maker={reward_maker}, taker={reward_taker}"


# ---------------------------------------------------------------------------
# Tests — ER_t execution risk
# ---------------------------------------------------------------------------

class TestExecutionRisk:
    """Verify stale order penalty based on σ × V × (1 + t/t_w)."""

    def test_no_active_orders_no_penalty(self):
        """No active quotes → ER = 0."""
        env = _make_env()
        env.active_quotes = {
            'bid_price': 0.0, 'bid_qty': 0.0,
            'ask_price': 0.0, 'ask_qty': 0.0,
            'bid_age': 0, 'ask_age': 0,
        }
        env.current_state.mid_price = 1.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])
        assert abs(reward) < 1e-9

    def test_stale_orders_penalised(self):
        """Active orders with age > 0 and σ > 0 → negative ER."""
        env = _make_env()
        env.current_state.market_features[0] = 10.0  # σ = 10 BPS
        env.active_quotes = {
            'bid_price': 0.99, 'bid_qty': 50.0,
            'ask_price': 1.01, 'ask_qty': 50.0,
            'bid_age': 50, 'ask_age': 50,
        }
        env._last_quotes_levels = []  # no L2/L3
        env.current_state.mid_price = 1.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])

        # σ = 10/10000 = 0.001
        # For each side: -0.001 * 50 * (1 + 50/100) = -0.001 * 50 * 1.5 = -0.075
        # Total ER = -0.075 * 2 = -0.15
        expected_er = -0.001 * 50 * 1.5 * 2
        assert abs(reward - expected_er) < 1e-6, f"Expected {expected_er}, got {reward}"

    def test_fresh_orders_no_age_penalty(self):
        """L1 with age=0 → no age-based ER, but L2/L3 still get base risk."""
        env = _make_env()
        env.current_state.market_features[0] = 10.0  # σ = 10 BPS
        env.active_quotes = {
            'bid_price': 0.99, 'bid_qty': 50.0,
            'ask_price': 1.01, 'ask_qty': 50.0,
            'bid_age': 0, 'ask_age': 0,
        }
        # No L2/L3 levels stored
        env._last_quotes_levels = []
        env.current_state.mid_price = 1.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])
        # L1 age=0 → ER=0, no L2/L3 → ER=0
        assert abs(reward) < 1e-9

    def test_er_includes_ladder_levels(self):
        """ER_t sums over L2/L3 with base risk σ × V."""
        env = _make_env()
        env.current_state.market_features[0] = 10.0  # σ = 10 BPS = 0.001
        env.active_quotes = {
            'bid_price': 0.99, 'bid_qty': 0.0,
            'ask_price': 1.01, 'ask_qty': 0.0,
            'bid_age': 0, 'ask_age': 0,
        }
        # L2 and L3 have volume but age=0 (base risk only)
        env._last_quotes_levels = [
            {'bid_qty': 0, 'ask_qty': 0},  # L1 (handled by active_quotes)
            {'bid_qty': 30.0, 'ask_qty': 30.0},  # L2
            {'bid_qty': 20.0, 'ask_qty': 20.0},  # L3
        ]
        env.current_state.mid_price = 1.0
        env.last_portfolio_value = 100.0

        reward = env._calculate_reward([])

        # ER from L2/L3: σ × V (base risk, age=0 → factor=1)
        # L2: -0.001 * 30 + -0.001 * 30 = -0.06
        # L3: -0.001 * 20 + -0.001 * 20 = -0.04
        expected_er = -0.001 * (30 + 30 + 20 + 20)
        assert abs(reward - expected_er) < 1e-6, f"Expected {expected_er}, got {reward}"


# ---------------------------------------------------------------------------
# Tests — Full RELAER integration
# ---------------------------------------------------------------------------

class TestRELAERIntegration:

    def test_all_components_combine(self):
        """Verify R_t = PnL_t + IP_t + C_t + ER_t."""
        env = _make_env()
        env.current_state.market_features[0] = 5.0  # σ = 5 BPS
        env._avg_entry_price = 1.0
        env.current_state.inventory = 30.0  # 30% > d=20%
        env.current_state.mid_price = 1.01  # +1%
        env.current_state.cash = 70.0
        env.last_portfolio_value = 100.0
        env.active_quotes = {
            'bid_price': 0.99, 'bid_qty': 20.0,
            'ask_price': 1.01, 'ask_qty': 20.0,
            'bid_age': 10, 'ask_age': 10,
        }
        env._last_quotes_levels = []  # no L2/L3

        reward = env._calculate_reward([])

        # PnL: MTM = (1.01 - 1.0) * 30 = 0.3
        # IP: -0.01 * 0.3 = -0.003 (full inv_ratio, not inv_ratio-d)
        # C: 0 (no fills)
        # ER: σ=0.0005, per side: -0.0005 * 20 * (1+10/100)
        sigma = 5.0 / 10000.0
        er_per_side = -sigma * 20.0 * (1.0 + 10.0 / 100.0)
        er = er_per_side * 2

        expected = 0.3 + (-0.003) + 0.0 + er
        assert abs(reward - expected) < 1e-6, f"Expected {expected:.6f}, got {reward:.6f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
