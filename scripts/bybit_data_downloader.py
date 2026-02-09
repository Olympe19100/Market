"""
Bybit Historical Data Downloader

Telecharge les donnees historiques (orderbook, trades) depuis Bybit
pour toutes les paires viables identifiees par le scanner.

Sources:
1. API Bybit officielle (klines, trades recents)
2. Bybit Public Data (historique complet)

Usage:
    python bybit_data_downloader.py                    # Telecharge toutes les paires viables
    python bybit_data_downloader.py --symbol BTCUSDT   # Telecharge une paire specifique
    python bybit_data_downloader.py --top 20           # Telecharge les 20 meilleures paires
"""

import os
import sys
import json
import time
import gzip
import argparse
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import zipfile
from io import BytesIO


# === CONFIGURATION ===
OUTPUT_DIR = "data/raw"
BYBIT_PUBLIC_DATA_URL = "https://public.bybit.com"


def ensure_output_dir():
    """Create output directory if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_bybit_public_trades(symbol: str, date: str, category: str = "spot") -> Optional[str]:
    """
    Download trades from Bybit public data repository.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        date: Date in YYYY-MM-DD format
        category: 'spot' or 'linear'

    Returns:
        Path to downloaded file or None
    """
    # Bybit public data URL format
    # Spot: https://public.bybit.com/spot/BTCUSDT/BTCUSDT2024-01-01.csv.gz
    # Linear: https://public.bybit.com/trading/BTCUSDT/BTCUSDT2024-01-01.csv.gz

    date_str = date.replace("-", "")
    if category == "spot":
        url = f"{BYBIT_PUBLIC_DATA_URL}/spot/{symbol}/{symbol}{date_str}.csv.gz"
    else:
        url = f"{BYBIT_PUBLIC_DATA_URL}/trading/{symbol}/{symbol}{date_str}.csv.gz"

    output_file = os.path.join(OUTPUT_DIR, f"{date}_{symbol}_trades.csv.gz")

    if os.path.exists(output_file):
        print(f"  [SKIP] {output_file} already exists")
        return output_file

    try:
        response = requests.get(url, timeout=60, stream=True)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"  [OK] {symbol} {date} - {size_mb:.1f} MB")
            return output_file
        elif response.status_code == 404:
            print(f"  [404] {symbol} {date} - No data available")
            return None
        else:
            print(f"  [ERR] {symbol} {date} - HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  [ERR] {symbol} {date} - {e}")
        return None


def download_klines_api(symbol: str, interval: str = "1",
                        start_date: str = None, end_date: str = None,
                        category: str = "spot") -> List[dict]:
    """
    Download klines (OHLCV) via Bybit API.

    Args:
        symbol: Trading pair
        interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        category: 'spot' or 'linear'
    """
    url = "https://api.bybit.com/v5/market/kline"

    all_klines = []
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000) if start_date else None
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000) if end_date else None

    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": 1000
    }

    if end_ts:
        params["end"] = end_ts

    while True:
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data.get("retCode") != 0:
                print(f"  [ERR] API error: {data.get('retMsg')}")
                break

            klines = data["result"]["list"]
            if not klines:
                break

            all_klines.extend(klines)

            # Klines are returned in reverse order (newest first)
            oldest_ts = int(klines[-1][0])
            if start_ts and oldest_ts <= start_ts:
                break

            params["end"] = oldest_ts - 1
            time.sleep(0.1)  # Rate limit

        except Exception as e:
            print(f"  [ERR] {e}")
            break

    return all_klines


def download_orderbook_snapshot(symbol: str, category: str = "spot", depth: int = 200) -> Optional[dict]:
    """Download current orderbook snapshot."""
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {
        "category": category,
        "symbol": symbol,
        "limit": depth
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("retCode") == 0:
            return data["result"]
    except Exception as e:
        print(f"  [ERR] Orderbook {symbol}: {e}")
    return None


def save_orderbook_stream(symbol: str, category: str, duration_seconds: int = 3600,
                          output_file: str = None) -> str:
    """
    Record orderbook updates via WebSocket for a duration.
    This creates data in the same format as Tardis.dev.
    """
    import websocket
    import threading

    if output_file is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(OUTPUT_DIR, f"{date_str}_{symbol}_ob200.data")

    messages = []
    stop_flag = threading.Event()

    def on_message(ws, message):
        data = json.loads(message)
        if "topic" in data:
            messages.append(json.dumps(data))

    def on_error(ws, error):
        print(f"  [WS ERR] {error}")

    def on_close(ws, close_status, close_msg):
        print(f"  [WS CLOSED] {symbol}")

    def on_open(ws):
        print(f"  [WS OPEN] {symbol}")
        # Subscribe to orderbook
        if category == "spot":
            sub = {"op": "subscribe", "args": [f"orderbook.200.{symbol}"]}
        else:
            sub = {"op": "subscribe", "args": [f"orderbook.200.{symbol}"]}
        ws.send(json.dumps(sub))

    # WebSocket URL
    if category == "spot":
        ws_url = "wss://stream.bybit.com/v5/public/spot"
    else:
        ws_url = "wss://stream.bybit.com/v5/public/linear"

    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )

    # Run in background thread
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

    print(f"  Recording {symbol} for {duration_seconds}s...")
    time.sleep(duration_seconds)

    ws.close()

    # Save to file
    with open(output_file, 'w') as f:
        for msg in messages:
            f.write(msg + "\n")

    # Compress
    zip_file = output_file + ".zip"
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, os.path.basename(output_file))
    os.remove(output_file)

    print(f"  [OK] Saved {len(messages)} messages to {zip_file}")
    return zip_file


def download_pair_data(symbol: str, category: str, days_back: int = 30,
                       include_trades: bool = True, include_klines: bool = True):
    """Download all available data for a pair."""

    print(f"\n{'='*60}")
    print(f"Downloading: {symbol} ({category})")
    print(f"{'='*60}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    downloaded_files = []

    # 1. Download trades (from public data)
    if include_trades:
        print(f"\n[Trades] Downloading {days_back} days...")
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            result = download_bybit_public_trades(symbol, date_str, category)
            if result:
                downloaded_files.append(result)
            current += timedelta(days=1)
            time.sleep(0.2)  # Rate limit

    # 2. Download klines
    if include_klines:
        print(f"\n[Klines] Downloading 1-minute bars...")
        klines = download_klines_api(
            symbol,
            interval="1",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            category=category
        )
        if klines:
            klines_file = os.path.join(OUTPUT_DIR, f"{symbol}_{category}_klines_1m.json")
            with open(klines_file, 'w') as f:
                json.dump(klines, f)
            print(f"  [OK] Saved {len(klines)} klines to {klines_file}")
            downloaded_files.append(klines_file)

    # 3. Save current orderbook snapshot
    print(f"\n[Orderbook] Saving current snapshot...")
    ob = download_orderbook_snapshot(symbol, category, depth=200)
    if ob:
        ob_file = os.path.join(OUTPUT_DIR, f"{symbol}_{category}_orderbook_snapshot.json")
        with open(ob_file, 'w') as f:
            json.dump(ob, f, indent=2)
        print(f"  [OK] Saved orderbook to {ob_file}")
        downloaded_files.append(ob_file)

    return downloaded_files


def main():
    parser = argparse.ArgumentParser(description="Bybit Historical Data Downloader")
    parser.add_argument("--symbol", type=str, help="Download specific symbol")
    parser.add_argument("--top", type=int, default=50, help="Download top N pairs by margin")
    parser.add_argument("--days", type=int, default=30, help="Days of history to download")
    parser.add_argument("--category", type=str, choices=["spot", "linear", "both"], default="both")
    parser.add_argument("--scan-first", action="store_true", help="Run scanner before downloading")
    parser.add_argument("--record-live", type=int, help="Record live orderbook for N seconds")
    args = parser.parse_args()

    ensure_output_dir()

    # Load or scan viable pairs
    viable_pairs_file = "viable_pairs_bybit.json"

    if args.scan_first or not os.path.exists(viable_pairs_file):
        print("Running Bybit scanner first...")
        import bybit_scanner
        bybit_scanner.main()

    if args.symbol:
        # Download specific symbol
        category = args.category if args.category != "both" else "spot"
        if args.record_live:
            save_orderbook_stream(args.symbol, category, args.record_live)
        else:
            download_pair_data(args.symbol, category, args.days)
    else:
        # Load viable pairs
        if not os.path.exists(viable_pairs_file):
            print(f"Error: {viable_pairs_file} not found. Run with --scan-first")
            sys.exit(1)

        with open(viable_pairs_file, 'r') as f:
            data = json.load(f)

        pairs_to_download = []

        if args.category in ["spot", "both"]:
            for p in data.get("spot", [])[:args.top]:
                pairs_to_download.append((p["symbol"], "spot", p["margin_bps"], p["volume_usdt"]))

        if args.category in ["linear", "both"]:
            for p in data.get("linear", [])[:args.top]:
                pairs_to_download.append((p["symbol"], "linear", p["margin_bps"], p["volume_usdt"]))

        # Sort by margin
        pairs_to_download.sort(key=lambda x: -x[2])
        pairs_to_download = pairs_to_download[:args.top]

        print(f"\n{'='*70}")
        print(f"BYBIT DATA DOWNLOADER")
        print(f"{'='*70}")
        print(f"Pairs to download: {len(pairs_to_download)}")
        print(f"Days of history:   {args.days}")
        print(f"Output directory:  {OUTPUT_DIR}")
        print()

        print("Pairs queued:")
        for sym, cat, margin, vol in pairs_to_download[:20]:
            print(f"  {sym:<20} {cat:<8} Marge: {margin:>+6.1f} bps  Vol: ${vol:>12,.0f}")
        if len(pairs_to_download) > 20:
            print(f"  ... and {len(pairs_to_download) - 20} more")

        input("\nPress Enter to start downloading (Ctrl+C to cancel)...")

        # Download each pair
        success_count = 0
        for i, (symbol, category, margin, volume) in enumerate(pairs_to_download, 1):
            print(f"\n[{i}/{len(pairs_to_download)}] {symbol}")
            try:
                files = download_pair_data(symbol, category, args.days,
                                           include_trades=True, include_klines=True)
                if files:
                    success_count += 1
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"  [ERR] {e}")

        print(f"\n{'='*70}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'='*70}")
        print(f"Successfully downloaded: {success_count}/{len(pairs_to_download)} pairs")
        print(f"Data saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
