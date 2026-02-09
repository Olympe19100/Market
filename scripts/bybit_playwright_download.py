"""
Bybit L2 Orderbook Auto-Downloader (Playwright version)

Automatise le téléchargement des données L2 orderbook depuis:
https://www.bybit.com/derivatives/en/history-data
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from playwright.async_api import async_playwright

# Configuration
BYBIT_URL = "https://www.bybit.com/derivatives/en/history-data"
DOWNLOAD_DIR = Path("C:/Users/ANTEC MSI/Desktop/Market-main/data/bybit_downloads")
INSTRUCTIONS_FILE = Path("C:/Users/ANTEC MSI/Desktop/Market-main/download_instructions.json")

# Date range
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=90)


def load_instructions():
    """Load download instructions from JSON file."""
    if not INSTRUCTIONS_FILE.exists():
        print(f"Error: {INSTRUCTIONS_FILE} not found!")
        sys.exit(1)
    with open(INSTRUCTIONS_FILE, 'r') as f:
        return json.load(f)


async def download_batch(page, symbols, batch_num, total_batches):
    """Download L2 orderbook data for a batch of symbols."""
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num}/{total_batches}: {', '.join(symbols)}")
    print(f"{'='*60}")

    # Navigate to page
    print("1. Loading Bybit page...")
    await page.goto(BYBIT_URL, wait_until="networkidle", timeout=60000)
    await page.wait_for_timeout(3000)

    # Screenshot for debugging
    await page.screenshot(path=str(DOWNLOAD_DIR / f"batch_{batch_num}_01_loaded.png"))

    # Handle any popups/cookies
    try:
        accept_btn = page.locator("button:has-text('Accept'), button:has-text('Got it'), button:has-text('OK')")
        if await accept_btn.count() > 0:
            await accept_btn.first.click()
            await page.wait_for_timeout(1000)
    except:
        pass

    # === Select Spot category ===
    print("2. Selecting Spot category...")
    try:
        spot_tab = page.locator("text=Spot").first
        await spot_tab.click()
        await page.wait_for_timeout(2000)
        print("   Spot selected")
    except Exception as e:
        print(f"   Could not find Spot tab: {e}")

    await page.screenshot(path=str(DOWNLOAD_DIR / f"batch_{batch_num}_02_spot.png"))

    # === Select OrderBook data type ===
    print("3. Selecting OrderBook...")
    try:
        # Look for OrderBook option
        orderbook = page.locator("text=OrderBook, text=Order Book, text=orderbook").first
        await orderbook.click()
        await page.wait_for_timeout(1000)
        print("   OrderBook selected")
    except Exception as e:
        print(f"   Could not find OrderBook: {e}")

    await page.screenshot(path=str(DOWNLOAD_DIR / f"batch_{batch_num}_03_orderbook.png"))

    # === Select symbols ===
    print(f"4. Selecting {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            # Click on search/input field
            search_input = page.locator("input[placeholder*='Search'], input[placeholder*='search'], input[type='text']").first
            await search_input.fill("")
            await search_input.fill(symbol)
            await page.wait_for_timeout(500)

            # Click on the symbol in dropdown
            symbol_option = page.locator(f"text={symbol}").first
            await symbol_option.click()
            await page.wait_for_timeout(300)
            print(f"   + {symbol}")
        except Exception as e:
            print(f"   - {symbol}: {e}")

    await page.screenshot(path=str(DOWNLOAD_DIR / f"batch_{batch_num}_04_symbols.png"))

    # === Set date range ===
    print(f"5. Setting dates: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    try:
        date_inputs = page.locator("input[type='date'], input[placeholder*='date'], input.date-input")
        if await date_inputs.count() >= 2:
            await date_inputs.nth(0).fill(START_DATE.strftime('%Y-%m-%d'))
            await date_inputs.nth(1).fill(END_DATE.strftime('%Y-%m-%d'))
            print("   Dates set")
    except Exception as e:
        print(f"   Could not set dates: {e}")

    await page.screenshot(path=str(DOWNLOAD_DIR / f"batch_{batch_num}_05_dates.png"))

    # === Click Confirm/Download ===
    print("6. Starting download...")
    try:
        confirm_btn = page.locator("button:has-text('Confirm'), button:has-text('Download'), button:has-text('Submit')").first

        # Setup download handler
        async with page.expect_download(timeout=120000) as download_info:
            await confirm_btn.click()
            download = await download_info.value

            # Save to our directory
            save_path = DOWNLOAD_DIR / download.suggested_filename
            await download.save_as(str(save_path))
            print(f"   Downloaded: {download.suggested_filename}")

    except Exception as e:
        print(f"   Download error: {e}")
        # Try clicking anyway
        try:
            await page.locator("button:has-text('Confirm')").first.click()
        except:
            pass

    await page.screenshot(path=str(DOWNLOAD_DIR / f"batch_{batch_num}_06_done.png"))

    # Wait for any background downloads
    await page.wait_for_timeout(5000)

    print(f"Batch {batch_num} complete!")
    return True


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bybit L2 Auto-Downloader (Playwright)")
    parser.add_argument("--batch", type=int, help="Download specific batch")
    parser.add_argument("--all", action="store_true", help="Download all batches")
    parser.add_argument("--start-batch", type=int, default=1, help="Start from batch N")
    parser.add_argument("--headless", action="store_true", help="Run headless (no UI)")
    args = parser.parse_args()

    # Load instructions
    instructions = load_instructions()
    batches = instructions["batches"]

    print("="*60)
    print("BYBIT L2 ORDERBOOK AUTO-DOWNLOADER (Playwright)")
    print("="*60)
    print(f"Total batches: {len(batches)}")
    print(f"Download dir: {DOWNLOAD_DIR}")
    print()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # Launch browser
        print("Launching browser...")
        browser = await p.chromium.launch(
            headless=args.headless,
            downloads_path=str(DOWNLOAD_DIR)
        )

        context = await browser.new_context(
            accept_downloads=True,
            viewport={"width": 1400, "height": 900}
        )

        page = await context.new_page()

        try:
            if args.batch:
                # Single batch
                batch = batches[args.batch - 1]
                await download_batch(page, batch["symbols"], args.batch, len(batches))

            elif args.all:
                # All batches
                for i in range(args.start_batch - 1, len(batches)):
                    batch = batches[i]
                    await download_batch(page, batch["symbols"], i + 1, len(batches))

                    if i < len(batches) - 1:
                        print(f"\nWaiting 5s before next batch...")
                        await asyncio.sleep(5)

                print("\n" + "="*60)
                print("ALL BATCHES COMPLETE!")
                print("="*60)

            else:
                # Default: first batch
                batch = batches[0]
                await download_batch(page, batch["symbols"], 1, len(batches))

        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
