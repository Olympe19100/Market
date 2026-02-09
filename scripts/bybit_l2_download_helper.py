"""
Bybit L2 Orderbook Download Helper

Bybit permet de télécharger les données L2 orderbook gratuitement depuis:
https://www.bybit.com/derivatives/en/history-data

Limitations:
- Maximum 5 paires à la fois
- Téléchargement manuel via l'interface web

Ce script:
1. Génère les groupes de 5 paires à télécharger
2. Crée les instructions pour chaque batch
3. Organise les fichiers téléchargés
4. Convertit au format compatible avec votre data_loader

Usage:
    python bybit_l2_download_helper.py --generate    # Génère les batches
    python bybit_l2_download_helper.py --organize    # Organise les fichiers téléchargés
    python bybit_l2_download_helper.py --convert     # Convertit au format compatible
"""

import os
import sys
import json
import glob
import zipfile
import shutil
from datetime import datetime, timedelta
from typing import List, Dict

# Configuration
VIABLE_PAIRS_FILE = "viable_pairs_bybit.json"
OUTPUT_DIR = "data/raw"
DOWNLOAD_DIR = "data/bybit_downloads"  # Où mettre les fichiers téléchargés manuellement
BATCH_SIZE = 5

# URL Bybit
BYBIT_DOWNLOAD_URL = "https://www.bybit.com/derivatives/en/history-data"


def load_viable_pairs() -> List[Dict]:
    """Load all viable pairs from scanner results."""
    if not os.path.exists(VIABLE_PAIRS_FILE):
        print(f"Error: {VIABLE_PAIRS_FILE} not found!")
        print("Run bybit_scanner.py first.")
        sys.exit(1)

    with open(VIABLE_PAIRS_FILE, 'r') as f:
        data = json.load(f)

    pairs = []
    for p in data.get("spot", []):
        pairs.append({
            "symbol": p["symbol"],
            "category": "spot",
            "margin_bps": p["margin_bps"],
            "volume_usdt": p["volume_usdt"]
        })
    for p in data.get("linear", []):
        pairs.append({
            "symbol": p["symbol"],
            "category": "linear",
            "margin_bps": p["margin_bps"],
            "volume_usdt": p["volume_usdt"]
        })

    # Sort by margin (best first)
    pairs.sort(key=lambda x: -x["margin_bps"])
    return pairs


def generate_batches(pairs: List[Dict]) -> List[List[Dict]]:
    """Split pairs into batches of 5."""
    batches = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batches.append(pairs[i:i + BATCH_SIZE])
    return batches


def generate_download_instructions():
    """Generate instructions for manual download."""
    pairs = load_viable_pairs()
    batches = generate_batches(pairs)

    # Date range (3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    print("=" * 70)
    print("BYBIT L2 ORDERBOOK DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print(f"URL: {BYBIT_DOWNLOAD_URL}")
    print()
    print(f"Total pairs: {len(pairs)}")
    print(f"Batches of 5: {len(batches)}")
    print()
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()
    print("=" * 70)
    print("BATCHES A TELECHARGER")
    print("=" * 70)

    instructions = []

    for i, batch in enumerate(batches, 1):
        print(f"\n--- BATCH {i}/{len(batches)} ---")
        symbols = [p["symbol"] for p in batch]
        print(f"Paires: {', '.join(symbols)}")

        for p in batch:
            print(f"  • {p['symbol']:<15} (marge: {p['margin_bps']:+.1f} bps, vol: ${p['volume_usdt']:,.0f})")

        instructions.append({
            "batch": i,
            "symbols": symbols,
            "details": batch
        })

    # Save instructions to file
    instructions_file = "download_instructions.json"
    with open(instructions_file, 'w') as f:
        json.dump({
            "url": BYBIT_DOWNLOAD_URL,
            "date_range": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            },
            "total_pairs": len(pairs),
            "total_batches": len(batches),
            "batches": instructions
        }, f, indent=2)

    print()
    print("=" * 70)
    print("INSTRUCTIONS")
    print("=" * 70)
    print(f"""
1. Allez sur: {BYBIT_DOWNLOAD_URL}

2. Sélectionnez "Spot" puis "OrderBook"

3. Pour chaque batch ci-dessus:
   a. Sélectionnez les 5 paires du batch
   b. Définissez la date de début: {start_date.strftime('%Y-%m-%d')}
   c. Définissez la date de fin: {end_date.strftime('%Y-%m-%d')}
   d. Cliquez "Confirm" pour télécharger
   e. Sauvegardez le fichier dans: {DOWNLOAD_DIR}/

4. Une fois tous les fichiers téléchargés, lancez:
   python bybit_l2_download_helper.py --organize

Instructions sauvegardées dans: {instructions_file}
""")

    # Create download directory
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"\nDossier de téléchargement créé: {DOWNLOAD_DIR}/")

    # Create a checklist file
    checklist_file = os.path.join(DOWNLOAD_DIR, "CHECKLIST.txt")
    with open(checklist_file, 'w') as f:
        f.write("BYBIT L2 ORDERBOOK DOWNLOAD CHECKLIST\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"URL: {BYBIT_DOWNLOAD_URL}\n")
        f.write(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n")

        for i, batch in enumerate(batches, 1):
            symbols = [p["symbol"] for p in batch]
            f.write(f"[ ] Batch {i}: {', '.join(symbols)}\n")

    print(f"Checklist créée: {checklist_file}")


def organize_downloads():
    """Organize downloaded files into the correct structure."""
    print("=" * 70)
    print("ORGANIZING DOWNLOADED FILES")
    print("=" * 70)

    if not os.path.exists(DOWNLOAD_DIR):
        print(f"Error: Download directory not found: {DOWNLOAD_DIR}")
        print("Download files first, then run --organize")
        return

    # Find all downloaded files
    files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.zip")) + \
            glob.glob(os.path.join(DOWNLOAD_DIR, "*.csv")) + \
            glob.glob(os.path.join(DOWNLOAD_DIR, "*.gz"))

    if not files:
        print(f"No files found in {DOWNLOAD_DIR}")
        print("Download the files from Bybit first.")
        return

    print(f"Found {len(files)} files")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each file
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")

        # Try to extract symbol from filename
        # Bybit filenames are typically like: BTCUSDT_orderbook_2024-01-01.csv.gz
        # or similar patterns

        # Copy to output directory for now
        dest = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(dest):
            shutil.copy2(filepath, dest)
            print(f"  -> Copied to {OUTPUT_DIR}/")
        else:
            print(f"  -> Already exists, skipped")

    print()
    print(f"Files organized in: {OUTPUT_DIR}/")


def convert_to_compatible_format():
    """Convert Bybit orderbook format to the format expected by data_loader."""
    print("=" * 70)
    print("CONVERTING TO COMPATIBLE FORMAT")
    print("=" * 70)

    # Find orderbook files
    files = glob.glob(os.path.join(OUTPUT_DIR, "*orderbook*.csv")) + \
            glob.glob(os.path.join(OUTPUT_DIR, "*orderbook*.csv.gz")) + \
            glob.glob(os.path.join(OUTPUT_DIR, "*_ob*.data"))

    if not files:
        print("No orderbook files found to convert.")
        print("Run --organize first, or check file patterns.")
        return

    print(f"Found {len(files)} orderbook files")

    # The data_loader expects files like:
    # 2025-12-06_XRPUSDC_ob200.data
    # With JSON lines containing:
    # {"topic":"orderbook.200.SYMBOL","ts":...,"type":"snapshot/delta","data":{"s":"SYMBOL","b":[...],"a":[...]}}

    for filepath in files:
        print(f"Converting: {os.path.basename(filepath)}")
        # Conversion logic depends on actual Bybit file format
        # TODO: Implement actual conversion once we see the file format

    print()
    print("Note: Run a test download first to see the exact file format.")
    print("Then I can implement the specific conversion logic.")


def print_summary():
    """Print summary of downloaded data."""
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    if os.path.exists(OUTPUT_DIR):
        files = os.listdir(OUTPUT_DIR)
        total_size = sum(
            os.path.getsize(os.path.join(OUTPUT_DIR, f))
            for f in files if os.path.isfile(os.path.join(OUTPUT_DIR, f))
        )
        print(f"Files in {OUTPUT_DIR}: {len(files)}")
        print(f"Total size: {total_size / (1024*1024*1024):.2f} GB")
    else:
        print("No data downloaded yet.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bybit L2 Orderbook Download Helper")
    parser.add_argument("--generate", action="store_true", help="Generate download instructions")
    parser.add_argument("--organize", action="store_true", help="Organize downloaded files")
    parser.add_argument("--convert", action="store_true", help="Convert to compatible format")
    parser.add_argument("--summary", action="store_true", help="Show download summary")
    args = parser.parse_args()

    if args.generate:
        generate_download_instructions()
    elif args.organize:
        organize_downloads()
    elif args.convert:
        convert_to_compatible_format()
    elif args.summary:
        print_summary()
    else:
        # Default: generate instructions
        generate_download_instructions()


if __name__ == "__main__":
    main()
