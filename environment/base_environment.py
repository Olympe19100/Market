import traceback
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
import torch
from core.config import RLConfig, MarketConfig
from data.processor import LOBFeatureProcessor, MarketFeatureProcessor
from training import data_loader

# Lazy import vectorbt (heavy ~2s import)
if TYPE_CHECKING:
    import vectorbt as vbt

# Configuration du logger pour suivre l'environnement du market maker
logger = logging.getLogger('market_maker.environment')


# === Classe d'état du Market Maker ===
@dataclass
class MarketMakerState:
    """Classe représentant l'état du Market Maker avec les fonctionnalités de suivi."""
    lob_features: Optional[np.ndarray] = None
    market_features: Optional[np.ndarray] = None
    time_remaining: float = 1.0
    portfolio: Optional[Any] = None  # vbt.Portfolio (lazy imported)
    trades_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=[
        'timestamp', 'price', 'size', 'side', 'entry_price', 'exit_price', 'pnl', 'fees']))
    inventory: float = 0.0
    cash: float = 0.0
    total_pnl: float = 0.0
    mid_price: float = None
    best_bid: float = None
    best_ask: float = None
    current_spread: float = None
    unrealized_pnl: float = 0.0
    initial_cash: float = 100.0
    
    # SOTA: Active Order Tracking (Order Persistence)
    active_bid_price: float = 0.0
    active_bid_qty: float = 0.0
    active_ask_price: float = 0.0
    active_ask_qty: float = 0.0
    bid_order_age: int = 0  # Steps since order placed (Queue Priority)
    ask_order_age: int = 0

# === Classe du portefeuille pour la gestion des trades ===
class Portfolio:
    def __init__(self, initial_cash: float = 100.0):
        """Initialise le portefeuille avec les attributs d'inventaire, de PnL, et des frais."""
        self._inventory = 0.0
        self._cash = initial_cash  # Initialize with actual starting capital
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._total_pnl = 0.0
        self.trades_df = pd.DataFrame(columns=['timestamp', 'price', 'size', 'side', 'fees'])
        logger.debug(f"Portfolio initialized with cash={initial_cash}")

    @property
    def inventory(self) -> float:
        return self._inventory

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        return self._unrealized_pnl

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    def add_trade(self, timestamp: datetime, price: float, size: float, side: str, fee: float = 0.0):
        """Ajoute un trade de manière incrémentale pour la performance."""
        qty = size if side == 'buy' else -size
        
        # Realized PnL is NOT calculated here (too complex with partial FIFO)
        # We track cash and inventory directly for Sim speed.
        if side == 'buy':
            self._cash -= (price * size + fee)
            self._inventory += size
        else:
            self._cash += (price * size - fee)
            self._inventory -= size
            
        new_trade = pd.DataFrame({
            'timestamp': [timestamp],
            'price': [price],
            'size': [qty],
            'fee': [fee]
        })
        self.trades_df = pd.concat([new_trade, self.trades_df], ignore_index=True)
        # self._update_portfolio() # REMOVED: O(N^2) Bottleneck

    def _update_portfolio(self):
        """Met à jour le portfolio avec les nouveaux trades en utilisant vectorbt."""
        if self.trades_df.empty:
            logger.warning("Aucun trade pour mettre à jour le portfolio.")
            return
        
        self.portfolio = vbt.Portfolio.from_orders(
            close=self.trades_df['price'],
            size=self.trades_df['size'],
            fees=self.trades_df['fee'], # Fix column name too
            freq='T' # Standardized 'T' (1-minute)
        )
        logger.debug("Mise à jour du portfolio avec les nouveaux trades.")

    def get_metrics(self) -> Dict:
        """Retourne les métriques de trading calculées via vectorbt."""
        if self.portfolio is None:
            logger.warning("Portfolio est None, aucune métrique disponible.")
            return {}
        
        metrics = {
            'total_return': self.portfolio.total_return(),
            'sharpe_ratio': self.portfolio.sharpe_ratio(),
            'sortino_ratio': self.portfolio.sortino_ratio(),
            'max_drawdown': self.portfolio.max_drawdown(),
            'win_rate': self.portfolio.win_rate(),
            'profit_factor': self.portfolio.profit_factor(),
            'expectancy': self.portfolio.expectancy(),
            'trades_count': len(self.portfolio.trades),
            'inventory': self.inventory,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl
        }
        logger.debug(f"Métriques du portfolio: {metrics}")
        return metrics


# === Environnement de base pour le Market Maker ===
class BaseMarketMakerEnv:
    def __init__(self, rl_config: RLConfig, market_config: MarketConfig, lob_processor: LOBFeatureProcessor, market_processor: MarketFeatureProcessor):
        """Initialise l'environnement de base avec les configurations et processeurs."""
        self.rl_config = rl_config
        self.market_config = market_config
        self.lob_processor = lob_processor
        self.market_processor = market_processor
        initial_cash = getattr(rl_config, 'initial_cash', 100.0)
        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.current_state = MarketMakerState(cash=initial_cash, inventory=0.0)
        self.episode_step = 0
        self.data_loader = None  # Subclasses must initialize this
        self.max_steps = rl_config.max_steps
        
        # Order persistence tracking
        self.active_quotes = {
            'bid_price': 0.0, 'bid_qty': 0.0,
            'ask_price': 0.0, 'ask_qty': 0.0,
            'bid_age': 0, 'ask_age': 0
        }
        logger.info(f"BaseMarketMakerEnv initialisé avec initial_cash={initial_cash}")


    # === Méthode d'initialisation de l'état ===
    def _initialize_state(self, orderbook: Dict, timestamp=None) -> MarketMakerState:
        """
        Initialise l'état du Market Maker basé sur l'order book fourni.

        Args:
            orderbook (Dict): Données initiales de l'order book.
            timestamp: Optional datetime for deterministic feature computation.

        Returns:
            MarketMakerState: L'état initialisé pour le market making.
        """
        try:
            # Traitement des caractéristiques LOB et des caractéristiques du marché
            lob_features = self.lob_processor.process(orderbook)
            market_features = self.market_processor.process(orderbook, timestamp=timestamp)
            mid_price = self.market_processor._calculate_mid_price(orderbook)
            # Création de l'état initial du Market Maker
            initial_cash = getattr(self.rl_config, 'initial_cash', 100.0)
            initial_state = MarketMakerState(
                lob_features=lob_features,
                market_features=market_features,
                time_remaining=1.0,
                portfolio=self.portfolio,
                inventory=0.0,
                cash=initial_cash,  # Initialize cash properly
                total_pnl=0.0,
                mid_price=mid_price
            )
            
            logger.debug("État initialisé avec succès.")
            return initial_state
        
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'état: {str(e)}")
            logger.error(f"Traceback complet: {traceback.format_exc()}")
            raise

    # === Méthode pour récupérer le prochain order book ===
    def _get_next_orderbook(self) -> Optional[Dict]:
        """
        Récupère le prochain order book via MarketDataLoader.
        
        Returns:
            Dict: Données de l'order book ou None si fin des données.
        """
        try:
            # Utilisation de MarketDataLoader pour obtenir le prochain order book
            orderbook = self.data_loader.get_next_orderbook()
            if orderbook is not None:
                logger.debug("Orderbook chargé avec succès.")
            else:
                logger.warning("Aucun order book disponible. Fin des données.")
            return orderbook
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du prochain order book depuis MarketDataLoader: {str(e)}")
            logger.error(f"Traceback complet: {traceback.format_exc()}")
            return None

    # === AS-derived dynamic bounds for action mappings ===
    def _compute_as_bounds(self) -> Dict:
        """
        Compute Avellaneda-Stoikov derived bounds for all action mappings.

        Uses microstructure volatility (σ), observed spread, warmup flag,
        risk aversion (γ=zeta), execution intensity (κ), time remaining (T-t),
        and normalised inventory to derive adaptive bounds that replace
        the former hard-coded constants in calculate_quotes().
        """
        mid_price = self.current_state.mid_price or 1.0
        tick_size = self.market_config.tick_size
        max_spread = self.rl_config.max_spread
        max_pos = float(getattr(self.market_config, 'max_position', 1000.0))

        # --- Extract market features ---
        mf = self.current_state.market_features
        if mf is not None and len(mf) > 14:
            sigma_bps = float(mf[0])   # FEAT_NOISE (BPS)
            spread_bps = float(mf[13]) # FEAT_SPREAD_BPS
            warmup = float(mf[14])     # FEAT_WARMUP
        else:
            sigma_bps = 1.0
            spread_bps = 5.0
            warmup = 0.0

        # During warmup, use conservative defaults
        if warmup < 0.5:
            sigma_bps = max(sigma_bps, 1.0)
            spread_bps = max(spread_bps, 5.0)

        # Ensure sane floors
        sigma_bps = max(sigma_bps, 0.1)
        spread_bps = max(spread_bps, 1.0)

        # --- AS parameters ---
        gamma = self.rl_config.zeta                    # inventory risk aversion
        kappa = self.rl_config.execution_intensity     # order arrival intensity
        T_minus_t = max(1.0 - self.episode_step / self.max_steps, 0.01)

        # Normalised inventory
        current_inv = self.current_state.inventory
        q_normalized = current_inv / max_pos if max_pos > 0 else 0.0

        # --- Reference half-spread (anchored to observed market spread) ---
        base_half_spread = max(spread_bps / 20000.0, tick_size / mid_price)

        # Vol scaling: when vol exceeds half the spread, widen
        vol_scaling = np.clip(0.5 + sigma_bps / max(spread_bps / 2.0, 0.5), 0.5, 3.0)

        # Time scaling: more time remaining → slightly wider (inventory risk)
        time_scaling = np.clip(0.5 + T_minus_t, 0.5, 1.5)

        reference_half_spread = base_half_spread * vol_scaling * time_scaling

        # --- Spread bounds [action 0, 1] ---
        min_half_spread = reference_half_spread * 0.25
        max_half_spread = min(reference_half_spread * 4.0, max_spread)
        # Ensure min < max
        if min_half_spread >= max_half_spread:
            min_half_spread = max_half_spread * 0.25

        # --- Reservation price offset (AS core skew) ---
        reservation_offset = (q_normalized * sigma_bps * T_minus_t * gamma * 100.0) / 10000.0
        reservation_offset = np.clip(reservation_offset, -reference_half_spread, reference_half_spread)

        # --- Qty dead zone [action 2, 3] ---
        vol_over_spread = min(sigma_bps / spread_bps, 1.0)
        qty_dead_zone = 0.15 - 0.1 * vol_over_spread  # [0.05, 0.15]

        # --- Market order bounds [action 4] ---
        # urgency: high when inventory is extreme and time is running out
        inv_urgency = abs(q_normalized)
        time_urgency = 1.0 - T_minus_t  # higher near end
        urgency = np.clip(inv_urgency + time_urgency * 0.5, 0.0, 1.0)

        market_dead_zone = np.clip(0.3 - 0.25 * urgency, 0.05, 0.3)
        market_intensity_cap = np.clip(0.3 + 0.5 * urgency, 0.3, 0.8)

        # --- Ladder step bounds [action 5] ---
        ladder_step_min = 0.3
        ladder_step_max = 2.5

        # --- Ladder decay bounds [action 6] ---
        kappa_ref = kappa * reference_half_spread
        ladder_decay_min = np.clip(np.exp(-kappa_ref) * 0.5, 0.1, 0.95)
        ladder_decay_max = np.clip(np.exp(-kappa_ref) * 1.5, 0.1, 0.95)
        if ladder_decay_min >= ladder_decay_max:
            ladder_decay_max = min(ladder_decay_min + 0.1, 0.95)

        # --- Hold threshold [action 7, 8] ---
        tightness = min(sigma_bps / max(spread_bps, 1.0), 1.0)
        hold_threshold = np.clip(0.4 + 0.2 * (1.0 - tightness), 0.3, 0.7)

        # --- Hold price tolerance ---
        hold_price_tolerance = max(reference_half_spread * mid_price, 2.0 * tick_size)

        return {
            'reference_half_spread': reference_half_spread,
            'min_half_spread': min_half_spread,
            'max_half_spread': max_half_spread,
            'reservation_offset': reservation_offset,
            'qty_dead_zone': qty_dead_zone,
            'market_dead_zone': market_dead_zone,
            'market_intensity_cap': market_intensity_cap,
            'ladder_step_min': ladder_step_min,
            'ladder_step_max': ladder_step_max,
            'ladder_decay_min': ladder_decay_min,
            'ladder_decay_max': ladder_decay_max,
            'hold_threshold': hold_threshold,
            'hold_price_tolerance': hold_price_tolerance,
        }

    # === Calcul des prix bid/ask — Avellaneda-Stoikov style (9D continu) ===
    def calculate_quotes(self, action: np.ndarray, market_state: Dict) -> Dict:
        """
        Multi-level quoting with order persistence (HOLD).

        Action Space (9D continuous, all in [-1, 1] from Tanh):
        [0]: bid_spread_offset  — how far below mid to place bid (L1)
        [1]: ask_spread_offset  — how far above mid to place ask (L1)
        [2]: bid_qty_frac       — fraction of available room to buy (L1)
        [3]: ask_qty_frac       — fraction of available room to sell (L1)
        [4]: market_signal      — signed market order (-1=sell, +1=buy), dead zone ±0.2
        [5]: ladder_step        — [-1,1] → [0.5x, 2.0x] of L1 spread for level spacing
        [6]: ladder_decay       — [-1,1] → [0.3, 0.8] quantity decay per level
        [7]: hold_bid           — [-1,1] → if > 0.5 and conditions met → HOLD bid order
        [8]: hold_ask           — [-1,1] → if > 0.5 and conditions met → HOLD ask order
        """
        orderbook = market_state['orderbook']
        base_bid = float(orderbook['bids'][0][0])
        base_ask = float(orderbook['asks'][0][0])
        mid_price = self.current_state.mid_price or (base_bid + base_ask) / 2
        tick_size = self.market_config.tick_size

        # --- Equity guard: no trading when equity <= 0 ---
        equity = self.current_state.cash + (self.current_state.inventory * mid_price)
        if equity <= 0:
            self.active_quotes['bid_age'] = 0
            self.active_quotes['ask_age'] = 0
            return {
                'levels': [{'bid_price': 0.0, 'ask_price': 0.0,
                            'bid_qty': 0, 'ask_qty': 0,
                            'bid_age': 0, 'ask_age': 0}],
                'flatten_qty': 0, 'flatten_side': None
            }

        # --- Compute AS-derived dynamic bounds ---
        bounds = self._compute_as_bounds()

        # --- Parse continuous actions (all in [-1, 1] from Tanh mean) ---
        bid_spread_offset = np.clip((action[0] + 1) / 2, 0.0, 1.0)
        ask_spread_offset = np.clip((action[1] + 1) / 2, 0.0, 1.0)
        bid_qty_frac_raw = np.clip((action[2] + 1) / 2, 0.0, 1.0)
        ask_qty_frac_raw = np.clip((action[3] + 1) / 2, 0.0, 1.0)
        # Dead zone: AS-derived qty dead zone
        qty_dz = bounds['qty_dead_zone']
        bid_qty_frac = bid_qty_frac_raw if bid_qty_frac_raw >= qty_dz else 0.0
        ask_qty_frac = ask_qty_frac_raw if ask_qty_frac_raw >= qty_dz else 0.0

        # action[4]: signed market order signal (raw tanh [-1, 1])
        market_signal = float(action[4]) if len(action) > 4 else 0.0

        # action[5]: ladder_step — maps [-1,1] → [ladder_step_min, ladder_step_max]
        ls_min, ls_max = bounds['ladder_step_min'], bounds['ladder_step_max']
        ladder_step = ls_min + np.clip((action[5] + 1) / 2, 0.0, 1.0) * (ls_max - ls_min) if len(action) > 5 else 1.0
        # action[6]: ladder_decay — maps [-1,1] → [ladder_decay_min, ladder_decay_max]
        ld_min, ld_max = bounds['ladder_decay_min'], bounds['ladder_decay_max']
        ladder_decay = ld_min + np.clip((action[6] + 1) / 2, 0.0, 1.0) * (ld_max - ld_min) if len(action) > 6 else 0.5

        # action[7]: hold_bid signal
        hold_bid_signal = np.clip((action[7] + 1) / 2, 0.0, 1.0) if len(action) > 7 else 0.0
        # action[8]: hold_ask signal
        hold_ask_signal = np.clip((action[8] + 1) / 2, 0.0, 1.0) if len(action) > 8 else 0.0

        # --- Price computation with AS-derived spread bounds ---
        min_hs = bounds['min_half_spread']
        max_hs = bounds['max_half_spread']
        spread_range = max_hs - min_hs
        bid_spread = min_hs + bid_spread_offset * spread_range
        ask_spread = min_hs + ask_spread_offset * spread_range

        # AS reservation price skew: shift both sides by inventory-driven offset
        reservation_offset = bounds['reservation_offset']
        l1_bid_px = round((mid_price * (1 - bid_spread - reservation_offset)) / tick_size) * tick_size
        l1_ask_px = round((mid_price * (1 + ask_spread - reservation_offset)) / tick_size) * tick_size

        # --- Quantity computation (equity-based) ---
        max_pos = float(getattr(self.market_config, 'max_position', 1000.0))
        max_pos_qty = (equity * self.rl_config.max_equity_exposure) / mid_price

        current_inv = self.current_state.inventory
        room_to_buy = max(0.0, min(max_pos_qty - current_inv, max_pos - current_inv))
        room_to_sell = max(0.0, min(max_pos_qty + current_inv, max_pos + current_inv))

        # Round to lot_size (not integer!) for fractional assets like METH
        lot_size = self.market_config.min_qty  # min_qty = lot_size
        l1_bid_qty = round(bid_qty_frac * room_to_buy / lot_size) * lot_size
        l1_ask_qty = round(ask_qty_frac * room_to_sell / lot_size) * lot_size

        if l1_bid_qty < lot_size:
            l1_bid_qty = 0.0
        if l1_ask_qty < lot_size:
            l1_ask_qty = 0.0

        # --- FIX 3: Order persistence (HOLD) for Level 1 ---
        price_tolerance = bounds['hold_price_tolerance']
        hold_thresh = bounds['hold_threshold']

        # HOLD bid
        if (hold_bid_signal > hold_thresh
                and self.active_quotes['bid_price'] > 0
                and abs(l1_bid_px - self.active_quotes['bid_price']) < price_tolerance):
            l1_bid_px = self.active_quotes['bid_price']
            bid_age = min(self.active_quotes['bid_age'] + 1, 200)
        else:
            bid_age = 0

        # HOLD ask
        if (hold_ask_signal > hold_thresh
                and self.active_quotes['ask_price'] > 0
                and abs(l1_ask_px - self.active_quotes['ask_price']) < price_tolerance):
            l1_ask_px = self.active_quotes['ask_price']
            ask_age = min(self.active_quotes['ask_age'] + 1, 200)
        else:
            ask_age = 0

        # Update active quotes tracking
        self.active_quotes['bid_price'] = l1_bid_px
        self.active_quotes['bid_qty'] = l1_bid_qty
        self.active_quotes['ask_price'] = l1_ask_px
        self.active_quotes['ask_qty'] = l1_ask_qty
        self.active_quotes['bid_age'] = bid_age
        self.active_quotes['ask_age'] = ask_age

        # --- FIX 2: Multi-level quoting (3 levels per side) ---
        levels = []
        cumulative_bid_qty = 0
        cumulative_ask_qty = 0

        for lvl in range(3):
            if lvl == 0:
                lvl_bid_px = l1_bid_px
                lvl_ask_px = l1_ask_px
                lvl_bid_qty = l1_bid_qty
                lvl_ask_qty = l1_ask_qty
                lvl_bid_age = bid_age
                lvl_ask_age = ask_age
            else:
                # Spread widens by ladder_step per level
                lvl_bid_spread = bid_spread + lvl * bid_spread * ladder_step
                lvl_ask_spread = ask_spread + lvl * ask_spread * ladder_step
                # Cap to AS-derived max half-spread
                lvl_bid_spread = min(lvl_bid_spread, bounds['max_half_spread'])
                lvl_ask_spread = min(lvl_ask_spread, bounds['max_half_spread'])

                lvl_bid_px = round((mid_price * (1 - lvl_bid_spread - reservation_offset)) / tick_size) * tick_size
                lvl_ask_px = round((mid_price * (1 + lvl_ask_spread - reservation_offset)) / tick_size) * tick_size

                # Qty decays geometrically (round to lot_size)
                decay_factor = ladder_decay ** lvl
                lvl_bid_qty = round(l1_bid_qty * decay_factor / lot_size) * lot_size
                lvl_ask_qty = round(l1_ask_qty * decay_factor / lot_size) * lot_size

                if lvl_bid_qty < lot_size:
                    lvl_bid_qty = 0.0
                if lvl_ask_qty < lot_size:
                    lvl_ask_qty = 0.0

                # Levels 2+ are always fresh (no HOLD)
                lvl_bid_age = 0
                lvl_ask_age = 0

            # Cap cumulative quantity to room (round to lot_size)
            if cumulative_bid_qty + lvl_bid_qty > room_to_buy:
                lvl_bid_qty = max(0.0, round((room_to_buy - cumulative_bid_qty) / lot_size) * lot_size)
            if cumulative_ask_qty + lvl_ask_qty > room_to_sell:
                lvl_ask_qty = max(0.0, round((room_to_sell - cumulative_ask_qty) / lot_size) * lot_size)

            cumulative_bid_qty += lvl_bid_qty
            cumulative_ask_qty += lvl_ask_qty

            levels.append({
                'bid_price': lvl_bid_px,
                'ask_price': lvl_ask_px,
                'bid_qty': lvl_bid_qty,
                'ask_qty': lvl_ask_qty,
                'bid_age': lvl_bid_age,
                'ask_age': lvl_ask_age,
            })

        # --- Market order (liquidation only: reduces inventory) ---
        flatten_qty = 0
        flatten_side = None
        dead_zone = bounds['market_dead_zone']
        intensity_cap = bounds['market_intensity_cap']

        if abs(market_signal) > dead_zone and abs(current_inv) >= self.market_config.min_qty:
            intensity = (abs(market_signal) - dead_zone) / (1.0 - dead_zone)

            if current_inv > 0:  # Long → sell to flatten
                raw_qty = round(intensity * abs(current_inv) * intensity_cap)
                if raw_qty >= self.market_config.min_qty:
                    flatten_qty = min(raw_qty, round(abs(current_inv)))
                    flatten_side = 'sell'
            else:  # Short → buy to flatten
                raw_qty = round(intensity * abs(current_inv) * intensity_cap)
                if raw_qty >= self.market_config.min_qty:
                    flatten_qty = min(raw_qty, round(abs(current_inv)))
                    flatten_side = 'buy'

        return {
            'levels': levels,
            'flatten_qty': flatten_qty,
            'flatten_side': flatten_side
        }

    def _calculate_reward(self, executions: List[Dict]) -> float:
        """Calcule la récompense simple (overridden par sim_env avec DSR)."""
        try:
            # Simple reward: just inventory penalty
            # SimulationMarketMakerEnv overrides this with DSR
            inventory_penalty = self.rl_config.zeta * (self.current_state.inventory ** 2)
            return -inventory_penalty
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la récompense: {str(e)}")
            return 0.0



    # === Méthode pour la mise à jour après un trade ===
    def update_state(self, trade: Dict):
        """Met à jour l'état après un trade et met à jour le portfolio."""
        try:
            self.portfolio.add_trade(
                timestamp=trade['timestamp'],
                price=trade['price'],
                size=trade['size'],
                side=trade['side'],
                fees=trade.get('fees', 0)
            )
            logger.debug(f"Trade mis à jour: {trade}")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du trade: {str(e)}")
            logger.error(f"Traceback complet: {traceback.format_exc()}")
            raise

    # === Méthode pour obtenir les métriques ===
    def get_metrics(self) -> Dict:
        """Retourne les métriques actuelles du portefeuille."""
        return self.portfolio.get_metrics()

    # === Méthode pour générer l'observation pour l'agent ===
    def _get_observation(self) -> Dict:
        """Retourne l'observation pour l'agent à chaque étape."""
        initial_cash = getattr(self.rl_config, 'initial_cash', 100.0)
        max_pos = float(getattr(self.market_config, 'max_position', 1000.0))

        # Compute room_to_buy/sell so the agent sees its residual capacity
        mid_price = self.current_state.mid_price or 1.0
        equity = self.current_state.cash + self.current_state.inventory * mid_price
        max_pos_qty = (equity * self.rl_config.max_equity_exposure) / mid_price if mid_price > 0 else 0.0
        current_inv = self.current_state.inventory
        room_to_buy = max(0.0, min(max_pos_qty - current_inv, max_pos - current_inv))
        room_to_sell = max(0.0, min(max_pos_qty + current_inv, max_pos + current_inv))

        observation = {
            'lob_features': self.current_state.lob_features,
            'market_features': self.current_state.market_features,
            'inventory': self.current_state.inventory,
            'time_remaining': 1.0 - (self.episode_step / self.max_steps),
            # Normalized portfolio state
            'cash_normalized': self.current_state.cash / initial_cash,
            'inventory_normalized': self.current_state.inventory / max_pos,
            # Normalized residual capacity (0 = no room, 1 = full room)
            'room_to_buy': room_to_buy / max_pos if max_pos > 0 else 0.0,
            'room_to_sell': room_to_sell / max_pos if max_pos > 0 else 0.0,
            # Fill indicators from last step (set by sim_env)
            'last_bid_fill': getattr(self, 'last_bid_fill', 0.0),
            'last_ask_fill': getattr(self, 'last_ask_fill', 0.0),
            # Current spread in bps (direct signal for MM decisions)
            'spread_bps': ((self.current_state.current_spread or 0.0) / mid_price) * 10000.0 if mid_price > 0 else 0.0,
            # Order persistence ages (FIX 3)
            'bid_order_age': self.active_quotes.get('bid_age', 0) / 100.0,
            'ask_order_age': self.active_quotes.get('ask_age', 0) / 100.0,
        }
        logger.debug(f"Observation générée: {observation}")
        return observation
   
    def _get_current_timestamp(self):
        # Retourne l'heure actuelle, ou ajustez selon les besoins de simulation
        return datetime.now()