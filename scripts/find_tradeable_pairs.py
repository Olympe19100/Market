"""
Scanner de paires Binance pour Market Making

Trouve les paires où le spread est suffisant pour être profitable
après les fees (avec BNB discount).

Critères:
- Spread > round_trip_fees (15 bps pour Regular + BNB)
- Volume 24h suffisant (liquidité)
- Pas de paires exotiques/delisting
"""

import requests
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIGURATION ===
# Fees avec BNB discount (Regular user)
MAKER_FEE_BPS = 7.5
TAKER_FEE_BPS = 7.5
ROUND_TRIP_BPS = MAKER_FEE_BPS + TAKER_FEE_BPS  # 15 bps

# Critères de sélection
MIN_SPREAD_BPS = 20.0       # Spread minimum pour être profitable (marge de sécurité)
MIN_VOLUME_USDT = 100_000   # Volume 24h minimum en USDT
MAX_PAIRS_TO_SHOW = 50      # Nombre de paires à afficher

# Quote assets à scanner
QUOTE_ASSETS = ['USDC', 'USDT', 'BUSD', 'BTC', 'ETH']


@dataclass
class PairAnalysis:
    symbol: str
    base: str
    quote: str
    best_bid: float
    best_ask: float
    spread_bps: float
    volume_24h: float
    volume_usdt: float
    profitable: bool
    margin_bps: float  # spread - fees


def get_exchange_info() -> Dict:
    """Get all trading pairs from Binance."""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url, timeout=10)
    return response.json()


def get_ticker_24h() -> Dict[str, Dict]:
    """Get 24h ticker for all pairs."""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url, timeout=30)
    data = response.json()
    return {item['symbol']: item for item in data}


def get_orderbook(symbol: str, limit: int = 5) -> Dict:
    """Get order book for a symbol."""
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    try:
        response = requests.get(url, timeout=5)
        return response.json()
    except:
        return None


def get_usdt_price(symbol: str, tickers: Dict) -> float:
    """Get approximate USDT price for volume conversion."""
    # Direct USDT pair
    if symbol + 'USDT' in tickers:
        return float(tickers[symbol + 'USDT']['lastPrice'])
    # Via BTC
    if symbol + 'BTC' in tickers and 'BTCUSDT' in tickers:
        btc_price = float(tickers[symbol + 'BTC']['lastPrice'])
        btc_usdt = float(tickers['BTCUSDT']['lastPrice'])
        return btc_price * btc_usdt
    return 0


def analyze_pair(symbol: str, base: str, quote: str, tickers: Dict) -> PairAnalysis:
    """Analyze a single pair for market making viability."""
    try:
        # Get orderbook
        ob = get_orderbook(symbol)
        if not ob or 'bids' not in ob or 'asks' not in ob:
            return None
        if not ob['bids'] or not ob['asks']:
            return None

        best_bid = float(ob['bids'][0][0])
        best_ask = float(ob['asks'][0][0])

        if best_bid <= 0 or best_ask <= 0:
            return None

        mid_price = (best_bid + best_ask) / 2
        spread_bps = (best_ask - best_bid) / mid_price * 10000

        # Get volume
        ticker = tickers.get(symbol, {})
        volume_24h = float(ticker.get('volume', 0))
        quote_volume = float(ticker.get('quoteVolume', 0))

        # Convert to USDT equivalent
        if quote == 'USDT':
            volume_usdt = quote_volume
        elif quote == 'USDC':
            volume_usdt = quote_volume  # ~1:1
        elif quote == 'BTC':
            btc_usdt = float(tickers.get('BTCUSDT', {}).get('lastPrice', 0))
            volume_usdt = quote_volume * btc_usdt
        elif quote == 'ETH':
            eth_usdt = float(tickers.get('ETHUSDT', {}).get('lastPrice', 0))
            volume_usdt = quote_volume * eth_usdt
        else:
            volume_usdt = 0

        # Profitability analysis
        margin_bps = spread_bps - ROUND_TRIP_BPS
        profitable = margin_bps > 0

        return PairAnalysis(
            symbol=symbol,
            base=base,
            quote=quote,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            volume_24h=volume_24h,
            volume_usdt=volume_usdt,
            profitable=profitable,
            margin_bps=margin_bps
        )
    except Exception as e:
        return None


def main():
    print("=" * 70)
    print("BINANCE MARKET MAKING PAIR SCANNER")
    print("=" * 70)
    print(f"\nFees (avec BNB discount):")
    print(f"  Maker: {MAKER_FEE_BPS} bps")
    print(f"  Taker: {TAKER_FEE_BPS} bps")
    print(f"  Round-trip: {ROUND_TRIP_BPS} bps")
    print(f"\nCritères:")
    print(f"  Spread minimum: {MIN_SPREAD_BPS} bps")
    print(f"  Volume minimum: ${MIN_VOLUME_USDT:,.0f}")
    print()

    # Get exchange info
    print("Fetching exchange info...")
    exchange_info = get_exchange_info()

    # Filter active trading pairs
    symbols_info = {}
    for s in exchange_info['symbols']:
        if s['status'] == 'TRADING' and s['quoteAsset'] in QUOTE_ASSETS:
            symbols_info[s['symbol']] = {
                'base': s['baseAsset'],
                'quote': s['quoteAsset']
            }

    print(f"Found {len(symbols_info)} active pairs")

    # Get 24h tickers
    print("Fetching 24h tickers...")
    tickers = get_ticker_24h()

    # Analyze pairs in parallel
    print(f"Analyzing spreads (this may take a minute)...")
    results: List[PairAnalysis] = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                analyze_pair,
                symbol,
                info['base'],
                info['quote'],
                tickers
            ): symbol
            for symbol, info in symbols_info.items()
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 100 == 0:
                print(f"  Analyzed {done}/{len(futures)} pairs...")

            result = future.result()
            if result is not None:
                results.append(result)

    # Filter by criteria
    viable_pairs = [
        r for r in results
        if r.spread_bps >= MIN_SPREAD_BPS and r.volume_usdt >= MIN_VOLUME_USDT
    ]

    # Sort by margin (most profitable first)
    viable_pairs.sort(key=lambda x: x.margin_bps, reverse=True)

    # Display results
    print("\n" + "=" * 70)
    print("PAIRES VIABLES POUR MARKET MAKING")
    print("=" * 70)
    print(f"\n{'Symbol':<15} {'Spread':>10} {'Margin':>10} {'Volume 24h':>15} {'Quote':>8}")
    print(f"{'':15} {'(bps)':>10} {'(bps)':>10} {'(USDT)':>15} {'':>8}")
    print("-" * 70)

    if not viable_pairs:
        print("\n⚠️  Aucune paire trouvée avec les critères actuels!")
        print(f"   Essayez de réduire MIN_SPREAD_BPS (actuellement {MIN_SPREAD_BPS})")
    else:
        for i, pair in enumerate(viable_pairs[:MAX_PAIRS_TO_SHOW]):
            print(f"{pair.symbol:<15} {pair.spread_bps:>10.1f} {pair.margin_bps:>+10.1f} "
                  f"{pair.volume_usdt:>15,.0f} {pair.quote:>8}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("STATISTIQUES")
    print("=" * 70)
    print(f"Paires analysées: {len(results)}")
    print(f"Paires viables: {len(viable_pairs)}")

    if viable_pairs:
        avg_spread = sum(p.spread_bps for p in viable_pairs) / len(viable_pairs)
        avg_margin = sum(p.margin_bps for p in viable_pairs) / len(viable_pairs)
        total_volume = sum(p.volume_usdt for p in viable_pairs)
        print(f"Spread moyen: {avg_spread:.1f} bps")
        print(f"Marge moyenne: {avg_margin:+.1f} bps")
        print(f"Volume total: ${total_volume:,.0f}")

    # Show top candidates
    if viable_pairs:
        print("\n" + "=" * 70)
        print("TOP 5 RECOMMANDATIONS")
        print("=" * 70)
        for i, pair in enumerate(viable_pairs[:5], 1):
            print(f"\n{i}. {pair.symbol}")
            print(f"   Spread: {pair.spread_bps:.1f} bps | Marge nette: {pair.margin_bps:+.1f} bps")
            print(f"   Volume 24h: ${pair.volume_usdt:,.0f}")
            print(f"   Best Bid: {pair.best_bid} | Best Ask: {pair.best_ask}")

    # Also show pairs that are close to viable
    print("\n" + "=" * 70)
    print("PAIRES PRESQUE VIABLES (spread 10-20 bps, bon volume)")
    print("=" * 70)
    almost_viable = [
        r for r in results
        if 10 <= r.spread_bps < MIN_SPREAD_BPS and r.volume_usdt >= MIN_VOLUME_USDT
    ]
    almost_viable.sort(key=lambda x: x.spread_bps, reverse=True)

    for pair in almost_viable[:10]:
        status = "⚠️ VIP5+" if pair.spread_bps > 5 else "❌"
        print(f"{pair.symbol:<15} {pair.spread_bps:>8.1f} bps  Vol: ${pair.volume_usdt:>12,.0f}  {status}")


if __name__ == "__main__":
    main()
