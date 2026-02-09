"""
Téléchargeur massif de données Bybit
Télécharge 3 mois de données pour TOUTES les paires viables.

Données téléchargées:
- Trades historiques (depuis Bybit Public Data)
- Klines 1-minute (via API)
- Orderbook snapshots actuels

Pour les données L2 orderbook historiques complètes (snapshots + deltas),
vous aurez besoin de Tardis.dev ou d'enregistrer en temps réel.
"""

import os
import sys
import json
import time
import gzip
import shutil
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
import threading

# === CONFIGURATION ===
OUTPUT_DIR = "data/raw_bybit"
DAYS_OF_HISTORY = 90  # 3 mois
MAX_WORKERS = 5  # Parallel downloads
BYBIT_PUBLIC_DATA_URL = "https://public.bybit.com"

# Rate limiting
REQUEST_DELAY = 0.3  # seconds between requests


def ensure_dirs():
    """Create output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "trades"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "klines"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "orderbook"), exist_ok=True)


def download_file(url: str, output_path: str, timeout: int = 120) -> bool:
    """Download a file with progress indication."""
    if os.path.exists(output_path):
        return True  # Skip existing

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        elif response.status_code == 404:
            return False  # No data for this date
        else:
            print(f"    HTTP {response.status_code}: {url}")
            return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def download_trades_for_symbol(symbol: str, category: str, start_date: datetime,
                                end_date: datetime) -> Tuple[str, int, int]:
    """Download all trades for a symbol within date range."""

    symbol_dir = os.path.join(OUTPUT_DIR, "trades", symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    success_count = 0
    fail_count = 0
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        date_compact = current.strftime("%Y%m%d")

        # Bybit public data URL
        if category == "spot":
            url = f"{BYBIT_PUBLIC_DATA_URL}/spot/{symbol}/{symbol}{date_compact}.csv.gz"
        else:
            url = f"{BYBIT_PUBLIC_DATA_URL}/trading/{symbol}/{symbol}{date_compact}.csv.gz"

        output_file = os.path.join(symbol_dir, f"{date_str}_trades.csv.gz")

        if download_file(url, output_file):
            success_count += 1
        else:
            fail_count += 1

        current += timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    return symbol, success_count, fail_count


def download_klines_for_symbol(symbol: str, category: str,
                                start_date: datetime, end_date: datetime) -> Tuple[str, int]:
    """Download klines via API for a symbol."""

    url = "https://api.bybit.com/v5/market/kline"
    all_klines = []

    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    params = {
        "category": category,
        "symbol": symbol,
        "interval": "1",  # 1 minute
        "limit": 1000,
        "end": end_ts
    }

    while True:
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()

            if data.get("retCode") != 0:
                break

            klines = data["result"]["list"]
            if not klines:
                break

            all_klines.extend(klines)

            oldest_ts = int(klines[-1][0])
            if oldest_ts <= start_ts:
                break

            params["end"] = oldest_ts - 1
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"    Klines error {symbol}: {e}")
            break

    # Save klines
    if all_klines:
        output_file = os.path.join(OUTPUT_DIR, "klines", f"{symbol}_1m.json")
        with open(output_file, 'w') as f:
            json.dump(all_klines, f)

    return symbol, len(all_klines)


def download_orderbook_snapshot(symbol: str, category: str) -> Tuple[str, bool]:
    """Download current orderbook snapshot."""

    url = "https://api.bybit.com/v5/market/orderbook"
    params = {
        "category": category,
        "symbol": symbol,
        "limit": 200
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("retCode") == 0:
            output_file = os.path.join(OUTPUT_DIR, "orderbook", f"{symbol}_snapshot.json")
            with open(output_file, 'w') as f:
                json.dump(data["result"], f, indent=2)
            return symbol, True
    except Exception as e:
        print(f"    OB error {symbol}: {e}")

    return symbol, False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bybit Mass Data Downloader")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    print("=" * 70)
    print("BYBIT MASS DATA DOWNLOADER")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"History: {DAYS_OF_HISTORY} days (~3 months)")
    print()

    ensure_dirs()

    # Load viable pairs
    viable_file = "viable_pairs_bybit.json"
    if not os.path.exists(viable_file):
        print(f"Error: {viable_file} not found!")
        print("Run bybit_scanner.py first.")
        sys.exit(1)

    with open(viable_file, 'r') as f:
        data = json.load(f)

    spot_pairs = [(p["symbol"], "spot", p["margin_bps"], p["volume_usdt"])
                  for p in data.get("spot", [])]
    perp_pairs = [(p["symbol"], "linear", p["margin_bps"], p["volume_usdt"])
                  for p in data.get("linear", [])]

    all_pairs = spot_pairs + perp_pairs

    print(f"Paires Spot: {len(spot_pairs)}")
    print(f"Paires Perp: {len(perp_pairs)}")
    print(f"Total: {len(all_pairs)} paires")
    print()

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_OF_HISTORY)

    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()

    # Estimate
    total_downloads = len(all_pairs) * DAYS_OF_HISTORY
    print(f"Estimated downloads: {total_downloads} trade files")
    print(f"                   + {len(all_pairs)} kline series")
    print(f"                   + {len(all_pairs)} orderbook snapshots")
    print()

    if not args.yes:
        input("Press Enter to start (Ctrl+C to cancel)...")
    print()

    # === PHASE 1: ORDERBOOK SNAPSHOTS ===
    print("=" * 70)
    print("PHASE 1: Orderbook Snapshots")
    print("=" * 70)

    ob_success = 0
    for i, (symbol, category, margin, vol) in enumerate(all_pairs, 1):
        sym, ok = download_orderbook_snapshot(symbol, category)
        if ok:
            ob_success += 1
        print(f"[{i}/{len(all_pairs)}] {symbol}: {'OK' if ok else 'FAIL'}")
        time.sleep(REQUEST_DELAY)

    print(f"\nOrderbook snapshots: {ob_success}/{len(all_pairs)} success")

    # === PHASE 2: KLINES ===
    print()
    print("=" * 70)
    print("PHASE 2: Klines (1-minute bars)")
    print("=" * 70)

    klines_results = []
    for i, (symbol, category, margin, vol) in enumerate(all_pairs, 1):
        sym, count = download_klines_for_symbol(symbol, category, start_date, end_date)
        klines_results.append((sym, count))
        print(f"[{i}/{len(all_pairs)}] {symbol}: {count:,} klines")

    total_klines = sum(r[1] for r in klines_results)
    print(f"\nTotal klines downloaded: {total_klines:,}")

    # === PHASE 3: TRADES ===
    print()
    print("=" * 70)
    print("PHASE 3: Historical Trades")
    print("=" * 70)
    print("This will take a while...")
    print()

    trades_results = []

    for i, (symbol, category, margin, vol) in enumerate(all_pairs, 1):
        print(f"[{i}/{len(all_pairs)}] {symbol} ({category})...")
        sym, success, fail = download_trades_for_symbol(
            symbol, category, start_date, end_date
        )
        trades_results.append((sym, success, fail))
        print(f"    -> {success} files downloaded, {fail} missing/failed")

    total_success = sum(r[1] for r in trades_results)
    total_fail = sum(r[2] for r in trades_results)

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Orderbook snapshots: {ob_success}/{len(all_pairs)}")
    print(f"Klines downloaded:   {total_klines:,} bars")
    print(f"Trade files:         {total_success} success, {total_fail} missing")
    print()
    print(f"Data saved to: {OUTPUT_DIR}/")
    print()

    # Save summary
    summary = {
        "download_time": datetime.now().isoformat(),
        "date_range": {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "days": DAYS_OF_HISTORY
        },
        "pairs": {
            "spot": len(spot_pairs),
            "perp": len(perp_pairs),
            "total": len(all_pairs)
        },
        "results": {
            "orderbook_snapshots": ob_success,
            "klines_total": total_klines,
            "trade_files_success": total_success,
            "trade_files_missing": total_fail
        },
        "pair_details": [
            {
                "symbol": r[0],
                "trade_files_downloaded": r[1],
                "trade_files_missing": r[2]
            }
            for r in trades_results
        ]
    }

    summary_file = os.path.join(OUTPUT_DIR, "download_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")

    # Note about L2 data
    print()
    print("=" * 70)
    print("NOTE SUR LES DONNEES L2 ORDERBOOK")
    print("=" * 70)
    print("""
Les données trades et klines sont téléchargées.

Pour les données L2 ORDERBOOK HISTORIQUES (snapshots + deltas),
comme celles que vous aviez pour XRPUSDC, vous avez 2 options:

1. TARDIS.DEV (payant mais complet)
   - Données L2 complètes pour Bybit
   - Format identique à vos fichiers actuels
   - https://tardis.dev/

2. ENREGISTREMENT EN TEMPS REEL (gratuit, à partir de maintenant)
   Lancez: python scripts/record_l2_live.py --symbols ALL --duration 86400
   Cela enregistrera 24h de données L2 pour toutes les paires viables.

Les données trades + klines peuvent être utilisées pour l'entraînement
mais les données L2 sont meilleures pour le market making.
""")


if __name__ == "__main__":
    main()
