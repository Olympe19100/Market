"""
Bybit L2 Orderbook Auto-Downloader

Automatise le téléchargement des données L2 orderbook depuis:
https://www.bybit.com/derivatives/en/history-data

Utilise Selenium pour contrôler le navigateur.

Usage:
    python bybit_auto_download.py --batch 1      # Télécharge le batch 1
    python bybit_auto_download.py --all          # Télécharge tous les batchs
    python bybit_auto_download.py --symbols BTCUSDT,ETHUSDT  # Symboles spécifiques
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.microsoft import EdgeChromiumDriverManager

# Configuration
BYBIT_URL = "https://www.bybit.com/derivatives/en/history-data"
DOWNLOAD_DIR = Path("C:/Users/ANTEC MSI/Desktop/Market-main/data/bybit_downloads")
INSTRUCTIONS_FILE = Path("C:/Users/ANTEC MSI/Desktop/Market-main/download_instructions.json")

# Date range (3 months)
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=90)


def load_instructions():
    """Load download instructions from JSON file."""
    if not INSTRUCTIONS_FILE.exists():
        print(f"Error: {INSTRUCTIONS_FILE} not found!")
        print("Run: python scripts/bybit_l2_download_helper.py --generate")
        sys.exit(1)

    with open(INSTRUCTIONS_FILE, 'r') as f:
        return json.load(f)


def setup_driver():
    """Setup Edge driver with download preferences."""
    edge_options = Options()

    # Set download directory
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    prefs = {
        "download.default_directory": str(DOWNLOAD_DIR),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    edge_options.add_experimental_option("prefs", prefs)

    # Don't run headless - we need to see the page for debugging
    # edge_options.add_argument("--headless")

    # Disable automation flags to avoid detection
    edge_options.add_argument("--disable-blink-features=AutomationControlled")
    edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    edge_options.add_experimental_option('useAutomationExtension', False)

    # Install/use Edge driver
    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=edge_options)

    # Set window size
    driver.set_window_size(1400, 900)

    return driver


def wait_for_element(driver, by, value, timeout=30):
    """Wait for element to be clickable."""
    wait = WebDriverWait(driver, timeout)
    return wait.until(EC.element_to_be_clickable((by, value)))


def wait_for_elements(driver, by, value, timeout=30):
    """Wait for elements to be present."""
    wait = WebDriverWait(driver, timeout)
    return wait.until(EC.presence_of_all_elements_located((by, value)))


def download_batch(driver, symbols, start_date, end_date, batch_num, total_batches):
    """Download L2 orderbook data for a batch of symbols."""
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num}/{total_batches}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*60}")

    try:
        # Navigate to Bybit history data page
        print("Navigating to Bybit...")
        driver.get(BYBIT_URL)
        time.sleep(3)  # Wait for page load

        # Handle cookie consent if present
        try:
            cookie_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Accept')]")
            cookie_btn.click()
            time.sleep(1)
        except NoSuchElementException:
            pass

        # Wait for page to fully load
        time.sleep(2)

        # Take screenshot for debugging
        driver.save_screenshot(str(DOWNLOAD_DIR / f"batch_{batch_num}_start.png"))

        # === STEP 1: Select Data Type (OrderBook) ===
        print("1. Selecting OrderBook data type...")

        # Find and click on data type dropdown/selector
        # Note: The exact selectors depend on Bybit's page structure
        # You may need to inspect the page and adjust these

        try:
            # Try to find OrderBook option
            orderbook_options = driver.find_elements(By.XPATH, "//*[contains(text(), 'OrderBook') or contains(text(), 'orderbook') or contains(text(), 'Order Book')]")
            if orderbook_options:
                orderbook_options[0].click()
                time.sleep(1)
                print("   OrderBook selected")
            else:
                print("   WARNING: Could not find OrderBook option")
        except Exception as e:
            print(f"   Error selecting OrderBook: {e}")

        # === STEP 2: Select Symbols ===
        print(f"2. Selecting {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                # Find symbol input/search
                symbol_input = driver.find_elements(By.XPATH, "//input[@placeholder='Search' or @placeholder='search' or contains(@class, 'search')]")
                if symbol_input:
                    symbol_input[0].clear()
                    symbol_input[0].send_keys(symbol)
                    time.sleep(0.5)

                # Click on the symbol in dropdown
                symbol_elem = driver.find_elements(By.XPATH, f"//*[contains(text(), '{symbol}')]")
                if symbol_elem:
                    symbol_elem[0].click()
                    print(f"   + {symbol}")
                    time.sleep(0.3)
                else:
                    print(f"   - {symbol} not found")
            except Exception as e:
                print(f"   Error selecting {symbol}: {e}")

        # === STEP 3: Set Date Range ===
        print(f"3. Setting date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        try:
            # Find date inputs
            date_inputs = driver.find_elements(By.XPATH, "//input[@type='date' or contains(@class, 'date')]")
            if len(date_inputs) >= 2:
                date_inputs[0].clear()
                date_inputs[0].send_keys(start_date.strftime('%Y-%m-%d'))
                date_inputs[1].clear()
                date_inputs[1].send_keys(end_date.strftime('%Y-%m-%d'))
                print("   Date range set")
            else:
                print("   WARNING: Could not find date inputs")
        except Exception as e:
            print(f"   Error setting dates: {e}")

        # Take screenshot before download
        driver.save_screenshot(str(DOWNLOAD_DIR / f"batch_{batch_num}_before_download.png"))

        # === STEP 4: Click Download/Confirm ===
        print("4. Starting download...")

        try:
            # Find download/confirm button
            download_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'Confirm') or contains(text(), 'Download') or contains(text(), 'confirm') or contains(text(), 'download')]")
            if download_btns:
                download_btns[0].click()
                print("   Download started!")
                time.sleep(2)
            else:
                print("   WARNING: Could not find download button")
        except Exception as e:
            print(f"   Error clicking download: {e}")

        # Wait for download to start/complete
        print("5. Waiting for download...")
        time.sleep(10)  # Adjust based on typical download time

        # Take final screenshot
        driver.save_screenshot(str(DOWNLOAD_DIR / f"batch_{batch_num}_after_download.png"))

        print(f"Batch {batch_num} complete!")
        return True

    except Exception as e:
        print(f"Error in batch {batch_num}: {e}")
        driver.save_screenshot(str(DOWNLOAD_DIR / f"batch_{batch_num}_error.png"))
        return False


def interactive_download(driver, symbols, start_date, end_date):
    """
    Semi-automated download: Opens the page and lets user do manual steps.
    Useful when full automation doesn't work due to page structure changes.
    """
    print("\n" + "="*60)
    print("INTERACTIVE DOWNLOAD MODE")
    print("="*60)
    print(f"\nSymbols to download: {', '.join(symbols)}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"\nDownload folder: {DOWNLOAD_DIR}")
    print("\n" + "-"*60)
    print("The browser will open Bybit's page.")
    print("Please manually:")
    print("  1. Select 'Spot' category")
    print("  2. Select 'OrderBook' data type")
    print("  3. Select the symbols listed above")
    print("  4. Set the date range")
    print("  5. Click 'Confirm' to download")
    print("-"*60)

    # Open the page
    driver.get(BYBIT_URL)

    # Copy symbols to clipboard for easy pasting
    symbols_text = ", ".join(symbols)
    print(f"\nSymbols: {symbols_text}")

    input("\nPress Enter when download is complete (or Ctrl+C to stop)...")
    return True


def main():
    parser = argparse.ArgumentParser(description="Bybit L2 Auto-Downloader")
    parser.add_argument("--batch", type=int, help="Download specific batch number")
    parser.add_argument("--all", action="store_true", help="Download all batches")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols to download")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (manual steps)")
    parser.add_argument("--start-batch", type=int, default=1, help="Start from batch N")
    args = parser.parse_args()

    # Load instructions
    instructions = load_instructions()
    batches = instructions["batches"]

    print("="*60)
    print("BYBIT L2 ORDERBOOK AUTO-DOWNLOADER")
    print("="*60)
    print(f"Total batches: {len(batches)}")
    print(f"Download directory: {DOWNLOAD_DIR}")
    print(f"Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print()

    # Setup browser
    print("Starting Chrome browser...")
    driver = setup_driver()

    try:
        if args.symbols:
            # Download specific symbols
            symbols = [s.strip() for s in args.symbols.split(",")]
            if args.interactive:
                interactive_download(driver, symbols, START_DATE, END_DATE)
            else:
                download_batch(driver, symbols, START_DATE, END_DATE, 1, 1)

        elif args.batch:
            # Download specific batch
            if args.batch < 1 or args.batch > len(batches):
                print(f"Invalid batch number. Must be 1-{len(batches)}")
                return

            batch = batches[args.batch - 1]
            symbols = batch["symbols"]

            if args.interactive:
                interactive_download(driver, symbols, START_DATE, END_DATE)
            else:
                download_batch(driver, symbols, START_DATE, END_DATE, args.batch, len(batches))

        elif args.all:
            # Download all batches
            start_from = args.start_batch - 1
            for i, batch in enumerate(batches[start_from:], start=start_from + 1):
                symbols = batch["symbols"]

                if args.interactive:
                    print(f"\n>>> BATCH {i}/{len(batches)} <<<")
                    interactive_download(driver, symbols, START_DATE, END_DATE)
                else:
                    success = download_batch(driver, symbols, START_DATE, END_DATE, i, len(batches))
                    if not success:
                        print(f"Batch {i} failed. Continue anyway? (y/n)")
                        if input().lower() != 'y':
                            break

                # Pause between batches
                if i < len(batches):
                    print(f"\nWaiting 5 seconds before next batch...")
                    time.sleep(5)

            print("\n" + "="*60)
            print("ALL BATCHES COMPLETE!")
            print("="*60)

        else:
            # Default: interactive mode for first batch
            print("No batch specified. Running interactive mode for batch 1.")
            print("Use --all for all batches, or --batch N for specific batch.")
            batch = batches[0]
            interactive_download(driver, batch["symbols"], START_DATE, END_DATE)

    finally:
        print("\nClosing browser...")
        driver.quit()


if __name__ == "__main__":
    main()
