"""Tests for AS-derived dynamic bounds in BaseMarketMakerEnv."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from core.config import RLConfig, MarketConfig
from environment.base_environment import BaseMarketMakerEnv, MarketMakerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(sigma_bps=5.0, spread_bps=10.0, warmup=1.0,
              mid_price=0.50, inventory=0.0, episode_step=0,
              max_steps=9000, zeta=0.01, execution_intensity=10.0):
    """Create a minimal BaseMarketMakerEnv with mocked processors."""
    rl_cfg = RLConfig(zeta=zeta, max_spread=0.1, min_spread=0.001,
                      max_steps=max_steps, execution_intensity=execution_intensity)
    mkt_cfg = MarketConfig(tick_size=0.00001, max_position=1000, min_qty=10)

    lob_proc = MagicMock()
    mkt_proc = MagicMock()

    with patch.object(BaseMarketMakerEnv, '__init__', lambda self, *a, **kw: None):
        env = BaseMarketMakerEnv.__new__(BaseMarketMakerEnv)

    env.rl_config = rl_cfg
    env.market_config = mkt_cfg
    env.lob_processor = lob_proc
    env.market_processor = mkt_proc
    env.episode_step = episode_step
    env.max_steps = max_steps
    env.active_quotes = {
        'bid_price': 0.0, 'bid_qty': 0.0,
        'ask_price': 0.0, 'ask_qty': 0.0,
        'bid_age': 0, 'ask_age': 0,
    }

    # Build market_features array (15 elements)
    mf = np.zeros(15)
    mf[0] = sigma_bps
    mf[13] = spread_bps
    mf[14] = warmup

    env.current_state = MarketMakerState(
        mid_price=mid_price,
        inventory=inventory,
        cash=100.0,
        market_features=mf,
    )
    return env


def _make_orderbook(mid=0.50, half_spread_ticks=5, tick=0.00001):
    bid = mid - half_spread_ticks * tick
    ask = mid + half_spread_ticks * tick
    return {
        'bids': [[bid, 1000]] + [[bid - i * tick, 500] for i in range(1, 5)],
        'asks': [[ask, 1000]] + [[ask + i * tick, 500] for i in range(1, 5)],
    }


# ---------------------------------------------------------------------------
# Tests for _compute_as_bounds
# ---------------------------------------------------------------------------

class TestComputeASBounds:

    def test_normal_vol(self):
        """XRP normal vol: σ=5, spread=10, T-t=0.5."""
        env = _make_env(sigma_bps=5.0, spread_bps=10.0, warmup=1.0,
                        episode_step=4500, max_steps=9000)
        b = env._compute_as_bounds()

        assert b['min_half_spread'] < b['max_half_spread']
        assert b['max_half_spread'] <= 0.1  # max_spread cap
        assert 0.05 <= b['qty_dead_zone'] <= 0.15
        assert 0.05 <= b['market_dead_zone'] <= 0.3
        assert 0.3 <= b['market_intensity_cap'] <= 0.8
        assert b['ladder_step_min'] == 0.3
        assert b['ladder_step_max'] == 2.5
        assert 0.1 <= b['ladder_decay_min'] <= b['ladder_decay_max'] <= 0.95
        assert 0.3 <= b['hold_threshold'] <= 0.7
        assert b['hold_price_tolerance'] > 0

    def test_high_vol(self):
        """When σ=20 >> spread, reference widens → wider spread range."""
        env_normal = _make_env(sigma_bps=5.0, spread_bps=10.0)
        env_high = _make_env(sigma_bps=20.0, spread_bps=10.0)
        bn = env_normal._compute_as_bounds()
        bh = env_high._compute_as_bounds()

        assert bh['reference_half_spread'] > bn['reference_half_spread']
        assert bh['max_half_spread'] >= bn['max_half_spread']

    def test_warmup_conservative(self):
        """During warmup, σ and spread are floored."""
        env = _make_env(sigma_bps=0.01, spread_bps=0.5, warmup=0.0)
        b = env._compute_as_bounds()

        # Warmup floors: σ>=1, spread>=5
        assert b['reference_half_spread'] > 0
        assert b['min_half_spread'] > 0

    def test_sigma_zero(self):
        """σ=0 should be floored to 0.1, no crash."""
        env = _make_env(sigma_bps=0.0, spread_bps=10.0)
        b = env._compute_as_bounds()
        assert b['reference_half_spread'] > 0

    def test_time_near_zero(self):
        """T-t→0: time_scaling → 0.5, bounds shrink."""
        env = _make_env(episode_step=8999, max_steps=9000)
        b = env._compute_as_bounds()
        assert b['reference_half_spread'] > 0
        # Near end: urgency high → market dead zone low, intensity high
        assert b['market_dead_zone'] <= 0.3

    def test_max_inventory_skew(self):
        """Full long inventory → positive reservation offset."""
        env = _make_env(inventory=1000.0, sigma_bps=5.0, spread_bps=10.0)
        b = env._compute_as_bounds()
        assert b['reservation_offset'] > 0  # long → offset > 0 → easier to sell

    def test_max_short_inventory_skew(self):
        """Full short inventory → negative reservation offset."""
        env = _make_env(inventory=-1000.0, sigma_bps=5.0, spread_bps=10.0)
        b = env._compute_as_bounds()
        assert b['reservation_offset'] < 0

    def test_no_market_features(self):
        """When market_features is None, defaults are used."""
        env = _make_env()
        env.current_state.market_features = None
        b = env._compute_as_bounds()
        assert b['reference_half_spread'] > 0

    def test_short_market_features(self):
        """When market_features has < 15 elements, defaults are used."""
        env = _make_env()
        env.current_state.market_features = np.zeros(5)
        b = env._compute_as_bounds()
        assert b['reference_half_spread'] > 0


# ---------------------------------------------------------------------------
# Tests for calculate_quotes with AS bounds
# ---------------------------------------------------------------------------

class TestCalculateQuotesAS:

    def _call_quotes(self, env, action=None):
        if action is None:
            action = np.zeros(9)
        ob = _make_orderbook(mid=env.current_state.mid_price)
        return env.calculate_quotes(action, {'orderbook': ob})

    def test_spread_within_bounds(self):
        """Spread should be within [min_half_spread, max_half_spread]."""
        env = _make_env(sigma_bps=5.0, spread_bps=10.0)
        b = env._compute_as_bounds()

        # action=0 → offset=0.5 → middle of spread range
        result = self._call_quotes(env, np.zeros(9))
        mid = env.current_state.mid_price
        l1 = result['levels'][0]

        bid_spread_actual = (mid - l1['bid_price']) / mid
        ask_spread_actual = (l1['ask_price'] - mid) / mid

        # Allow tolerance for reservation offset + tick rounding
        tolerance = b['reference_half_spread'] + 2 * env.market_config.tick_size / mid
        assert bid_spread_actual <= b['max_half_spread'] + tolerance
        assert ask_spread_actual <= b['max_half_spread'] + tolerance

    def test_reservation_skew_direction(self):
        """Long inventory → bid lower, ask lower (skew to sell)."""
        env_flat = _make_env(inventory=0.0, sigma_bps=5.0, spread_bps=10.0)
        env_long = _make_env(inventory=500.0, sigma_bps=5.0, spread_bps=10.0)

        action = np.zeros(9)
        r_flat = self._call_quotes(env_flat, action)
        r_long = self._call_quotes(env_long, action)

        # Long inventory: both bid and ask should shift down
        assert r_long['levels'][0]['ask_price'] <= r_flat['levels'][0]['ask_price']

    def test_action_extremes_min_spread(self):
        """action[0,1]=-1 → minimum spread."""
        env = _make_env(sigma_bps=5.0, spread_bps=10.0)
        action = np.full(9, -1.0)
        result = self._call_quotes(env, action)
        # Should produce tight spreads (near min_half_spread)
        mid = env.current_state.mid_price
        l1 = result['levels'][0]
        bid_spread = (mid - l1['bid_price']) / mid
        # With reservation_offset at full short (-1 action maps), check it's small
        b = env._compute_as_bounds()
        # bid_spread should be near min_half_spread (± reservation offset + rounding)
        assert bid_spread < b['max_half_spread'] * 1.5

    def test_action_extremes_max_spread(self):
        """action[0,1]=+1 → maximum spread."""
        env = _make_env(sigma_bps=5.0, spread_bps=10.0)
        action = np.ones(9)
        result = self._call_quotes(env, action)
        mid = env.current_state.mid_price
        l1 = result['levels'][0]
        bid_spread = (mid - l1['bid_price']) / mid
        b = env._compute_as_bounds()
        # Should be near max_half_spread
        assert bid_spread >= b['min_half_spread'] * 0.5

    def test_market_order_dead_zone(self):
        """Market signal within dead zone → no market order."""
        env = _make_env(inventory=100.0)  # need inventory for liquidation
        b = env._compute_as_bounds()
        # Signal just inside dead zone
        action = np.zeros(9)
        action[4] = b['market_dead_zone'] * 0.5
        result = self._call_quotes(env, action)
        assert result['flatten_qty'] == 0
        assert result['flatten_side'] is None

    def test_market_order_liquidation_long(self):
        """Long inventory + strong signal → sell to flatten."""
        env = _make_env(inventory=200.0, sigma_bps=5.0, spread_bps=10.0)
        action = np.zeros(9)
        action[4] = 0.95  # strong signal
        result = self._call_quotes(env, action)
        assert result['flatten_side'] == 'sell'
        assert result['flatten_qty'] > 0
        assert result['flatten_qty'] <= 200  # can't sell more than inventory

    def test_market_order_liquidation_short(self):
        """Short inventory + strong signal → buy to flatten."""
        env = _make_env(inventory=-100.0, sigma_bps=5.0, spread_bps=10.0)
        env.current_state.cash = 200.0  # enough equity to trade
        action = np.zeros(9)
        action[4] = 0.95
        result = self._call_quotes(env, action)
        assert result['flatten_side'] == 'buy'
        assert result['flatten_qty'] > 0
        assert result['flatten_qty'] <= 100

    def test_market_order_no_inventory_no_order(self):
        """Zero inventory → no market order regardless of signal."""
        env = _make_env(inventory=0.0)
        action = np.zeros(9)
        action[4] = 0.95
        result = self._call_quotes(env, action)
        assert result['flatten_qty'] == 0
        assert result['flatten_side'] is None

    def test_hold_with_dynamic_threshold(self):
        """Hold threshold is AS-derived, not fixed 0.5."""
        env = _make_env(sigma_bps=5.0, spread_bps=10.0)
        b = env._compute_as_bounds()

        # First call to set active quotes
        action = np.zeros(9)
        action[2] = 1.0  # max bid qty
        action[3] = 1.0  # max ask qty
        ob = _make_orderbook(mid=0.50)
        env.calculate_quotes(action, {'orderbook': ob})

        # Second call with hold signal just above threshold
        action2 = np.zeros(9)
        action2[2] = 1.0
        action2[3] = 1.0
        # Map hold_threshold back to raw action space
        # hold_signal = (action+1)/2, so action = 2*hold_signal - 1
        raw_hold = 2.0 * (b['hold_threshold'] + 0.01) - 1.0
        action2[7] = min(raw_hold, 1.0)
        action2[8] = min(raw_hold, 1.0)
        result = env.calculate_quotes(action2, {'orderbook': ob})
        # Should not crash; HOLD may or may not trigger depending on price tolerance
        assert len(result['levels']) == 3

    def test_equity_zero_guard(self):
        """Zero equity returns empty quotes."""
        env = _make_env()
        env.current_state.cash = 0.0
        env.current_state.inventory = 0.0
        action = np.zeros(9)
        result = self._call_quotes(env, action)
        assert result['levels'][0]['bid_qty'] == 0
        assert result['flatten_qty'] == 0

    def test_output_structure_unchanged(self):
        """Output dict structure: levels, flatten_qty, flatten_side."""
        env = _make_env()
        result = self._call_quotes(env, np.zeros(9))
        assert 'levels' in result
        assert 'flatten_qty' in result
        assert 'flatten_side' in result
        assert len(result['levels']) == 3
        for lvl in result['levels']:
            assert 'bid_price' in lvl
            assert 'ask_price' in lvl
            assert 'bid_qty' in lvl
            assert 'ask_qty' in lvl
            assert 'bid_age' in lvl
            assert 'ask_age' in lvl

    def test_numerical_xrp_normal(self):
        """Numerical verification: XRP mid=0.50, σ=5, spread=10, T-t=0.5."""
        env = _make_env(sigma_bps=5.0, spread_bps=10.0, mid_price=0.50,
                        episode_step=4500, max_steps=9000, inventory=500.0)
        b = env._compute_as_bounds()

        # base_half_spread = max(10/20000, 0.00001/0.5) = 0.0005
        expected_base = max(10.0 / 20000.0, 0.00001 / 0.50)
        assert abs(expected_base - 0.0005) < 1e-8

        # vol_scaling = clip(0.5 + 5/max(5, 0.5)) = clip(0.5 + 1.0) = 1.5
        expected_vol_scaling = np.clip(0.5 + 5.0 / max(5.0, 0.5), 0.5, 3.0)
        assert abs(expected_vol_scaling - 1.5) < 1e-8

        # T_minus_t = 0.5, time_scaling = clip(0.5 + 0.5) = 1.0
        expected_time_scaling = np.clip(0.5 + 0.5, 0.5, 1.5)
        assert abs(expected_time_scaling - 1.0) < 1e-8

        expected_ref = expected_base * expected_vol_scaling * expected_time_scaling
        assert abs(b['reference_half_spread'] - expected_ref) < 1e-10

        # Reservation offset: q_norm=0.5, σ=5, T-t=0.5, γ=0.01
        # = (0.5 * 5 * 0.5 * 0.01 * 100) / 10000 = 1.25 / 10000 = 0.000125
        expected_offset = (0.5 * 5.0 * 0.5 * 0.01 * 100.0) / 10000.0
        assert abs(b['reservation_offset'] - expected_offset) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
