"""
Scanner Bybit pour trouver les paires viables pour le Market Making.
Analyse les spreads et volumes en temps réel.
"""

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

# === CONFIGURATION ===
# Bybit fees (sans VIP, spot)
MAKER_FEE_BPS = 10.0  # 0.10%
TAKER_FEE_BPS = 10.0  # 0.10%
ROUND_TRIP_BPS = MAKER_FEE_BPS + TAKER_FEE_BPS  # 20 bps

# Critères
MIN_SPREAD_BPS = 25.0       # Spread minimum pour être profitable
MIN_VOLUME_USDT = 50_000    # Volume 24h minimum
MAX_RESULTS = 300


@dataclass
class PairInfo:
    symbol: str
    base: str
    quote: str
    spread_bps: float
    margin_bps: float
    volume_usdt: float
    best_bid: float
    best_ask: float
    category: str  # spot, linear, inverse


def get_bybit_spot_symbols() -> List[dict]:
    """Get all spot trading pairs from Bybit."""
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "spot"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("retCode") == 0:
            return data["result"]["list"]
    except Exception as e:
        print(f"Error fetching spot symbols: {e}")
    return []


def get_bybit_linear_symbols() -> List[dict]:
    """Get all USDT perpetual pairs from Bybit."""
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "linear"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("retCode") == 0:
            return data["result"]["list"]
    except Exception as e:
        print(f"Error fetching linear symbols: {e}")
    return []


def get_bybit_tickers(category: str) -> dict:
    """Get 24h tickers for a category."""
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": category}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("retCode") == 0:
            return {t["symbol"]: t for t in data["result"]["list"]}
    except Exception as e:
        print(f"Error fetching {category} tickers: {e}")
    return {}


def get_bybit_orderbook(symbol: str, category: str, limit: int = 5) -> Optional[dict]:
    """Get orderbook for a symbol."""
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {"category": category, "symbol": symbol, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if data.get("retCode") == 0:
            return data["result"]
    except:
        pass
    return None


def analyze_pair(symbol: str, base: str, quote: str, category: str,
                 tickers: dict, usdt_prices: dict) -> Optional[PairInfo]:
    """Analyze a single pair."""
    try:
        ob = get_bybit_orderbook(symbol, category)
        if not ob or not ob.get("b") or not ob.get("a"):
            return None

        best_bid = float(ob["b"][0][0])
        best_ask = float(ob["a"][0][0])

        if best_bid <= 0 or best_ask <= 0:
            return None

        mid = (best_bid + best_ask) / 2
        spread_bps = (best_ask - best_bid) / mid * 10000

        # Get volume
        ticker = tickers.get(symbol, {})
        volume_24h = float(ticker.get("volume24h", 0))
        turnover_24h = float(ticker.get("turnover24h", 0))

        # Convert to USDT
        if quote == "USDT":
            volume_usdt = turnover_24h
        elif quote == "USDC":
            volume_usdt = turnover_24h
        elif quote in usdt_prices:
            volume_usdt = turnover_24h * usdt_prices[quote]
        else:
            volume_usdt = turnover_24h

        margin_bps = spread_bps - ROUND_TRIP_BPS

        return PairInfo(
            symbol=symbol,
            base=base,
            quote=quote,
            spread_bps=spread_bps,
            margin_bps=margin_bps,
            volume_usdt=volume_usdt,
            best_bid=best_bid,
            best_ask=best_ask,
            category=category
        )
    except Exception as e:
        return None


def scan_bybit() -> Tuple[List[PairInfo], List[PairInfo]]:
    """Scan Bybit for viable pairs. Returns (spot_pairs, perp_pairs)."""

    print("=" * 70)
    print("BYBIT MARKET MAKING PAIR SCANNER")
    print("=" * 70)
    print(f"\nFees Bybit (standard):")
    print(f"  Maker: {MAKER_FEE_BPS} bps (0.10%)")
    print(f"  Taker: {TAKER_FEE_BPS} bps (0.10%)")
    print(f"  Round-trip: {ROUND_TRIP_BPS} bps")
    print(f"\nCriteres:")
    print(f"  Spread minimum: {MIN_SPREAD_BPS} bps")
    print(f"  Volume minimum: ${MIN_VOLUME_USDT:,.0f}")
    print()

    # Get USDT prices for conversion
    usdt_prices = {}
    spot_tickers = get_bybit_tickers("spot")
    if "BTCUSDT" in spot_tickers:
        usdt_prices["BTC"] = float(spot_tickers["BTCUSDT"]["lastPrice"])
    if "ETHUSDT" in spot_tickers:
        usdt_prices["ETH"] = float(spot_tickers["ETHUSDT"]["lastPrice"])

    results_spot = []
    results_perp = []

    # === SCAN SPOT ===
    print("Fetching Spot symbols...")
    spot_symbols = get_bybit_spot_symbols()
    spot_symbols = [s for s in spot_symbols if s.get("status") == "Trading"]
    print(f"Found {len(spot_symbols)} active Spot pairs")

    print("Analyzing Spot spreads...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                analyze_pair,
                s["symbol"],
                s["baseCoin"],
                s["quoteCoin"],
                "spot",
                spot_tickers,
                usdt_prices
            ): s["symbol"]
            for s in spot_symbols
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  Spot: {done}/{len(futures)}...")
            result = future.result()
            if result and result.spread_bps >= MIN_SPREAD_BPS and result.volume_usdt >= MIN_VOLUME_USDT:
                results_spot.append(result)

    # === SCAN LINEAR (USDT PERP) ===
    print("\nFetching Linear (USDT Perp) symbols...")
    linear_symbols = get_bybit_linear_symbols()
    linear_symbols = [s for s in linear_symbols if s.get("status") == "Trading"
                      and s.get("settleCoin") == "USDT"]
    print(f"Found {len(linear_symbols)} active Linear pairs")

    linear_tickers = get_bybit_tickers("linear")

    print("Analyzing Linear spreads...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                analyze_pair,
                s["symbol"],
                s["baseCoin"],
                s["quoteCoin"],
                "linear",
                linear_tickers,
                usdt_prices
            ): s["symbol"]
            for s in linear_symbols
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  Linear: {done}/{len(futures)}...")
            result = future.result()
            if result and result.spread_bps >= MIN_SPREAD_BPS and result.volume_usdt >= MIN_VOLUME_USDT:
                results_perp.append(result)

    # Sort by margin
    results_spot.sort(key=lambda x: -x.margin_bps)
    results_perp.sort(key=lambda x: -x.margin_bps)

    return results_spot, results_perp


def print_results(spot_pairs: List[PairInfo], perp_pairs: List[PairInfo]):
    """Print results in a nice format."""

    print("\n" + "=" * 80)
    print("PAIRES SPOT VIABLES")
    print("=" * 80)
    print(f"{'#':<4} {'Symbol':<18} {'Spread':>10} {'Marge':>10} {'Volume 24h':>18}")
    print(f"{'':4} {'':18} {'(bps)':>10} {'(bps)':>10} {'(USDT)':>18}")
    print("-" * 80)

    for i, p in enumerate(spot_pairs[:MAX_RESULTS], 1):
        print(f"{i:<4} {p.symbol:<18} {p.spread_bps:>10.1f} {p.margin_bps:>+10.1f} {p.volume_usdt:>18,.0f}")

    print(f"\nTotal Spot viables: {len(spot_pairs)}")

    print("\n" + "=" * 80)
    print("PAIRES USDT PERPETUAL VIABLES")
    print("=" * 80)
    print(f"{'#':<4} {'Symbol':<18} {'Spread':>10} {'Marge':>10} {'Volume 24h':>18}")
    print(f"{'':4} {'':18} {'(bps)':>10} {'(bps)':>10} {'(USDT)':>18}")
    print("-" * 80)

    for i, p in enumerate(perp_pairs[:MAX_RESULTS], 1):
        print(f"{i:<4} {p.symbol:<18} {p.spread_bps:>10.1f} {p.margin_bps:>+10.1f} {p.volume_usdt:>18,.0f}")

    print(f"\nTotal Perp viables: {len(perp_pairs)}")


def save_results(spot_pairs: List[PairInfo], perp_pairs: List[PairInfo], output_path: str):
    """Save results to JSON for the downloader."""
    data = {
        "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "criteria": {
            "min_spread_bps": MIN_SPREAD_BPS,
            "min_volume_usdt": MIN_VOLUME_USDT,
            "round_trip_fee_bps": ROUND_TRIP_BPS
        },
        "spot": [
            {
                "symbol": p.symbol,
                "base": p.base,
                "quote": p.quote,
                "spread_bps": round(p.spread_bps, 2),
                "margin_bps": round(p.margin_bps, 2),
                "volume_usdt": round(p.volume_usdt, 0)
            }
            for p in spot_pairs
        ],
        "linear": [
            {
                "symbol": p.symbol,
                "base": p.base,
                "quote": p.quote,
                "spread_bps": round(p.spread_bps, 2),
                "margin_bps": round(p.margin_bps, 2),
                "volume_usdt": round(p.volume_usdt, 0)
            }
            for p in perp_pairs
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResultats sauvegardes dans: {output_path}")


def main():
    spot_pairs, perp_pairs = scan_bybit()
    print_results(spot_pairs, perp_pairs)

    # Save for downloader
    output_path = "viable_pairs_bybit.json"
    save_results(spot_pairs, perp_pairs, output_path)

    # Summary
    print("\n" + "=" * 80)
    print("RESUME")
    print("=" * 80)
    print(f"Paires Spot viables:  {len(spot_pairs)}")
    print(f"Paires Perp viables:  {len(perp_pairs)}")
    print(f"Total:                {len(spot_pairs) + len(perp_pairs)}")

    if spot_pairs:
        print(f"\nTop 5 Spot (par marge):")
        for p in spot_pairs[:5]:
            print(f"  {p.symbol:<15} Spread: {p.spread_bps:>6.1f} bps  Marge: {p.margin_bps:>+6.1f} bps  Vol: ${p.volume_usdt:>12,.0f}")

    if perp_pairs:
        print(f"\nTop 5 Perp (par marge):")
        for p in perp_pairs[:5]:
            print(f"  {p.symbol:<15} Spread: {p.spread_bps:>6.1f} bps  Marge: {p.margin_bps:>+6.1f} bps  Vol: ${p.volume_usdt:>12,.0f}")


if __name__ == "__main__":
    main()
