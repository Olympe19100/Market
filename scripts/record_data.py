
import asyncio
import logging
import os
import signal
import sys
import json
from pathlib import Path
from aiohttp import web
import aiohttp

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import MarketConfig, DashboardConfig
from data.binance.stream import BinanceStreamHandler
from data.recorder import MarketDataRecorder
from data.processor import MarketFeatureProcessor
from models.mamba_lob.predictor import MambaLOBPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataRecorder")

# --- HTML TEMPLATE FOR DASHBOARD ---
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MM Terminal | Antigravity</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        :root {
            --bg-dark: #09090b;
            --bg-panel: #18181b;
            --text-primary: #e4e4e7;
            --text-secondary: #71717a;
            --accent: #3b82f6;
            --bid-color: #22c55e;
            --ask-color: #ef4444;
            --toxic-color: #f97316;
            --border-color: #27272a;
        }
        body { margin: 0; background: var(--bg-dark); color: var(--text-primary); font-family: 'JetBrains Mono', monospace; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
        
        /* Layout */
        header { 
            display: flex; justify-content: space-between; align-items: center; 
            padding: 10px 20px; background: var(--bg-panel); border-bottom: 1px solid var(--border-color);
            height: 40px;
        }
        .grid {
            display: grid;
            grid-template-columns: 280px 1fr 320px;
            grid-template-rows: auto 1fr;
            flex: 1;
            gap: 1px;
            background: var(--border-color);
        }
        .panel { background: var(--bg-dark); overflow: hidden; display: flex; flex-direction: column; }
        .panel-header {
            padding: 8px 12px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
            display: flex; justify-content: space-between;
        }

        /* Metrics */
        .kpi-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1px; background: var(--border-color); grid-column: 1 / -1; height: 80px; }
        .kpi-card { background: var(--bg-panel); padding: 10px 15px; display: flex; flex-direction: column; justify-content: center; }
        .kpi-label { font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 4px; }
        .kpi-value { font-size: 1.25rem; font-weight: 700; }
        .kpi-tag { font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; margin-left: 8px; font-weight: 700; }
        .tag-safe { background: rgba(34, 197, 94, 0.2); color: var(--bid-color); }
        .tag-toxic { background: rgba(239, 68, 68, 0.2); color: var(--ask-color); }
        .tag-neutral { background: rgba(59, 130, 246, 0.2); color: var(--accent); }

        /* Order Book */
        .ob-row { display: flex; justify-content: space-between; padding: 2px 8px; font-size: 0.8rem; position: relative; cursor: crosshair; }
        .ob-row:hover { background: #27272a; }
        .price { z-index: 2; font-weight: 600; }
        .qty { z-index: 2; color: var(--text-secondary); }
        .depth-bar { position: absolute; top: 0; bottom: 0; right: 0; opacity: 0.15; z-index: 1; transition: width 0.2s; }
        .bids .depth-bar { background: var(--bid-color); }
        .asks .depth-bar { background: var(--ask-color); }

        /* Tape */
        .trade-row { 
            display: flex; justify-content: space-between; padding: 3px 12px; 
            font-size: 0.8rem; border-bottom: 1px solid #18181b; 
        }
        .trade-buy { color: var(--bid-color); }
        .trade-sell { color: var(--ask-color); }
        .large-trade { background: rgba(255, 255, 255, 0.05); }

        /* Chart */
        #chart-container { flex: 1; position: relative; }
        .chart-legend { position: absolute; top: 10px; left: 10px; z-index: 20; display: flex; gap: 15px; font-size: 0.8rem; background: rgba(0,0,0,0.5); padding: 5px; border-radius: 4px; }
        
        /* Utils */
        .text-up { color: var(--bid-color); }
        .text-down { color: var(--ask-color); }
    </style>
</head>
<body>
    <header>
        <div style="display:flex; align-items:center; gap:15px">
            <div style="font-weight:800; color: #fff; letter-spacing:-0.5px">ANTIGRAVITY <span style="color:var(--accent)">MM</span></div>
            <div id="symbol-display" style="font-size:0.9rem; font-weight:700; color:var(--text-secondary)">LOADING...</div>
        </div>
        <div style="font-size:0.75rem; color:var(--text-secondary)">
            LATENCY: <span id="latency" style="color:var(--bid-color)">14ms</span> | 
            STATUS: <span style="color:var(--bid-color)">LIVE RECORDING</span>
        </div>
    </header>

    <div class="grid">
        <!-- Top Row KPIs -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">VPIN (TOXICITY)</div>
                <div style="display:flex; align-items:baseline">
                    <span class="kpi-value" id="val-vpin">0.00</span>
                    <span class="kpi-tag tag-neutral" id="tag-vpin">Wait</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">OFI (FLOW)</div>
                <div style="display:flex; align-items:baseline">
                    <span class="kpi-value" id="val-ofi">0.00</span>
                    <span class="kpi-tag tag-neutral" id="tag-ofi">Neutral</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">EFF. SPREAD</div>
                <div style="display:flex; align-items:baseline">
                    <span class="kpi-value" id="val-spread">0.0000</span>
                    <span class="kpi-tag tag-neutral" id="tag-spread">Normal</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">NOISE / VOL</div>
                <div style="display:flex; align-items:baseline">
                    <span class="kpi-value" id="val-noise">0.00</span>
                    <span class="kpi-label" style="margin-left:5px">bps</span>
                </div>
            </div>
        </div>

        <!-- Left Panel: Order Book -->
        <div class="panel" style="grid-row: 2;">
            <div class="panel-header">Depth of Market</div>
            <div style="flex:1; overflow-y:auto; display:flex; flex-direction:column">
                <div id="asks-container" class="asks" style="display:flex; flex-direction:column-reverse; padding-bottom:5px; border-bottom:1px dashed #333"></div>
                <div style="text-align:center; padding:5px; font-weight:800; font-size:1.1rem" id="mid-price">0.0000</div>
                <div id="bids-container" class="bids" style="padding-top:5px"></div>
            </div>
            <div style="padding:10px; font-size:0.7rem; color:var(--text-secondary); border-top:1px solid var(--border-color)">
                <div style="display:flex; justify-content:space-between"><span>IMBALANCE</span> <span id="val-imb">0.00%</span></div>
                <div style="display:flex; justify-content:space-between"><span>SLOPE</span> <span id="val-slope">0.00</span></div>
            </div>
        </div>

        <!-- Center Panel: Chart -->
        <div class="panel" style="grid-row: 2;">
            <div class="panel-header">Real-time Price & VWAP</div>
            <div id="chart-container">
                <div class="chart-legend">
                    <span style="color:#22c55e">Last Price</span>
                    <span style="color:#3b82f6">VWAP (Session)</span>
                </div>
            </div>
        </div>

        <!-- Right Panel: Tape -->
        <div class="panel" style="grid-row: 2;">
            <div class="panel-header">Time & Sales</div>
            <div id="tape-container" style="flex:1; overflow-y:auto;"></div>
            <div style="padding:8px; border-top:1px solid var(--border-color); font-size:0.75rem">
                <div class="kpi-label">TFI (TRADE IMBALANCE)</div>
                <div id="val-tfi" style="font-weight:700">0.00</div>
            </div>
        </div>
    </div>

    <script>
        // --- 3. UI Updates ---
        const els = {
            vpin: document.getElementById('val-vpin'),
            tagVpin: document.getElementById('tag-vpin'),
            ofi: document.getElementById('val-ofi'),
            tagOfi: document.getElementById('tag-ofi'),
            spread: document.getElementById('val-spread'),
            tagSpread: document.getElementById('tag-spread'),
            noise: document.getElementById('val-noise'),
            bids: document.getElementById('bids-container'),
            asks: document.getElementById('asks-container'),
            mid: document.getElementById('mid-price'),
            tape: document.getElementById('tape-container'),
            imb: document.getElementById('val-imb'),
            slope: document.getElementById('val-slope'),
            tfi: document.getElementById('val-tfi'),
            symbol: document.getElementById('symbol-display'),
            latency: document.getElementById('latency')
        };
        
        // --- 1. WebSocket & Core Logic ---
        const ws = new WebSocket(`ws://${location.host}/ws`);
        
        ws.onopen = () => {
            console.log("Connected to WS");
            els.symbol.innerText = "WAITING FOR DATA...";
            els.latency.innerText = "CONNECTED";
        };
        
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            
            if (msg.type === 'snapshot') {
                renderOrderBook(msg.data);
                els.symbol.innerText = msg.symbol;
                
                // Update Mid Price
                if(msg.data.bids[0] && msg.data.asks[0]) {
                    const mid = (msg.data.bids[0][0] + msg.data.asks[0][0]) / 2;
                    els.mid.innerText = mid.toFixed(4);
                    // Update Chart if available
                    if (window.lineSeries) {
                        try { window.lineSeries.update({ time: Date.now() / 1000, value: mid }); } catch(e){}
                    }
                }

                if (msg.features) renderFeatures(msg.features);
                
            } else if (msg.type === 'trade') {
                addTrade(msg.data);
            }
        };

        // --- 2. Lightweight Chart Setup (Safe Mode) ---
        try {
            const chartContainer = document.getElementById('chart-container');
            const chart = LightweightCharts.createChart(chartContainer, {
                layout: { background: { color: '#09090b' }, textColor: '#71717a' },
                grid: { vertLines: { color: '#18181b' }, horzLines: { color: '#18181b' } },
                timeScale: { timeVisible: true, secondsVisible: true },
            });
            window.lineSeries = chart.addLineSeries({ color: '#22c55e', lineWidth: 2 });
            
            // Auto-resize
            new ResizeObserver(entries => {
                if (entries.length === 0 || entries[0].target !== chartContainer) { return; }
                const newRect = entries[0].contentRect;
                chart.applyOptions({ width: newRect.width, height: newRect.height });
            }).observe(chartContainer);
        } catch(e) {
            console.error("Chart init failed (CDN blocked?):", e);
            document.getElementById('chart-container').innerHTML = "<div style='padding:20px; color:#ef4444'>CHART ERROR: " + e.message + "</div>";
        }

        function renderFeatures(f) {
            // VPIN
            els.vpin.innerText = f.vpin.toFixed(2);
            if (f.vpin > 0.8) { els.tagVpin.innerText = "TOXIC"; els.tagVpin.className = "kpi-tag tag-toxic"; }
            else if (f.vpin < 0.3) { els.tagVpin.innerText = "SAFE"; els.tagVpin.className = "kpi-tag tag-safe"; }
            else { els.tagVpin.innerText = "NORMAL"; els.tagVpin.className = "kpi-tag tag-neutral"; }
            
            // Update VPIN Progress Bar if it exists, else create it
            let prog = document.getElementById('vpin-progress');
            if(!prog && f.vpin_progress !== undefined) {
                 // Inject progress bar into VPIN card dynamically if missing
                 const container = els.vpin.parentElement.parentElement;
                 const bar = document.createElement('div');
                 bar.style.marginTop = '8px';
                 bar.style.height = '4px';
                 bar.style.background = '#333';
                 bar.style.borderRadius = '2px';
                 bar.innerHTML = `<div id="vpin-progress" style="height:100%; width:0%; background:#3b82f6; transition:width 0.5s"></div>`;
                 container.appendChild(bar);
                 // Add label
                 const lbl = document.createElement('div');
                 lbl.id = "vpin-prog-lbl";
                 lbl.style.fontSize = '0.65rem';
                 lbl.style.color = '#71717a';
                 lbl.style.textAlign = 'right';
                 lbl.style.marginTop = '2px';
                 lbl.innerText = 'Bucket Fill: 0%';
                 container.appendChild(lbl);
                 prog = document.getElementById('vpin-progress');
            }
            if(prog && f.vpin_progress !== undefined) {
                prog.style.width = (f.vpin_progress * 100) + '%';
                const lbl = document.getElementById('vpin-prog-lbl');
                if(lbl) lbl.innerText = `Bucket Fill: ${(f.vpin_progress * 100).toFixed(0)}%`;
            }

            // OFI (Flow)
            els.ofi.innerText = f.ofi.toFixed(0);
            if(f.ofi > 500) { els.tagOfi.innerText = "BUYING"; els.tagOfi.className = "kpi-tag tag-safe"; }
            else if(f.ofi < -500) { els.tagOfi.innerText = "SELLING"; els.tagOfi.className = "kpi-tag tag-toxic"; }
            else { els.tagOfi.innerText = "BALANCED"; els.tagOfi.className = "kpi-tag tag-neutral"; }

            // Effective Spread
            els.spread.innerText = f.effective_spread ? f.effective_spread.toFixed(5) : "0.0000";
            
            // Other Metrics
            els.noise.innerText = f.noise.toFixed(2);
            els.tfi.innerText = f.tfi.toFixed(2);
            els.tfi.className = f.tfi > 0 ? "text-up" : "text-down";
            
            if(f.order_imbalance) els.imb.innerText = (f.order_imbalance * 100).toFixed(1) + "%";
            if(f.slope) els.slope.innerText = (f.slope / 1000000).toFixed(1) + "M";
        }

        function renderOrderBook(ob) {
            // Find max volume for bars
            const maxVol = Math.max(
                ...ob.bids.slice(0, 15).map(b => b[1]), 
                ...ob.asks.slice(0, 15).map(a => a[1])
            );

            // Bids
            els.bids.innerHTML = ob.bids.slice(0, 15).map(b => `
                <div class="ob-row">
                    <span class="qty">${b[1].toFixed(1)}</span>
                    <span class="price text-up">${b[0].toFixed(4)}</span>
                    <div class="depth-bar" style="width:${(b[1]/maxVol)*100}%"></div>
                </div>
            `).join('');

            // Asks
            els.asks.innerHTML = ob.asks.slice(0, 15).map(a => `
                <div class="ob-row">
                    <span class="qty">${a[1].toFixed(1)}</span>
                    <span class="price text-down">${a[0].toFixed(4)}</span>
                    <div class="depth-bar" style="width:${(a[1]/maxVol)*100}%"></div>
                </div>
            `).join('');
        }

        function addTrade(t) {
            const row = document.createElement('div');
            const isBuy = !t.m; // m=False means Buyer Maker = False -> Taker was Buyer -> Buy Trade
            const size = parseFloat(t.q);
            const price = parseFloat(t.p);
            
            const largeThreshold = {{LARGE_TRADE_THRESHOLD}};
            row.className = 'trade-row' + (size > largeThreshold ? ' large-trade' : '');
            
            row.innerHTML = `
                <span style="opacity:0.6">${new Date(t.T).toLocaleTimeString().split(' ')[0]}</span>
                <span class="${isBuy ? 'text-up' : 'text-down'}">${price.toFixed(4)}</span>
                <span>${size.toFixed(1)}</span>
            `;
            
            els.tape.prepend(row);
            const tapeLimit = {{TAPE_LIMIT}};
            if (els.tape.children.length > tapeLimit) els.tape.removeChild(els.tape.lastChild);
        }
    </script>
</body>
</html>
"""

async def handle_index(request):
    dash_config = request.app['config'].dashboard
    html = HTML_DASHBOARD.replace('{{TAPE_LIMIT}}', str(dash_config.tape_limit))
    html = html.replace('{{LARGE_TRADE_THRESHOLD}}', str(dash_config.large_trade_threshold))
    return web.Response(text=html, content_type='text/html')

async def recorder_websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app['websockets'].add(ws)
    try:
        async for msg in ws: pass
    finally:
        request.app['websockets'].discard(ws)
    return ws

async def broadcast_loop(app, stream_manager, processor=None, recorder=None, ai_predictor=None):
    """Periodically broadcasts state and analytics, and logs features."""
    dash_config = app['config'].dashboard
    while True:
        try:
            await asyncio.sleep(dash_config.update_freq)
            if not app['websockets'] and not recorder:
                continue
                
            # Get Snapshot
            ob = stream_manager.orderbook
            if not ob or not ob['bids'] or not ob['asks']:
                continue

            clean_ob = {
                'bids': [[float(p), float(q)] for p, q in ob['bids'][:20]],
                'asks': [[float(p), float(q)] for p, q in ob['asks'][:20]]
            }
            
            # Get Features
            features = {k: 0.0 for k in ['vpin', 'ofi', 'tfi', 'vol', 'noise', 'amihud', 'slope', 'entropy', 'effective_spread', 'order_imbalance']}
            attn_features = []
            
            # AI Actions
            ai_bias, ai_spread, ai_ready = 0.0, 0.0, False
            if ai_predictor:
                ai_bias, ai_spread, ai_ready = ai_predictor.get_action(clean_ob)
            
            if processor:
                try:
                    # Update processor with snapshot data + current timestamp from stream
                    ts = None
                    if 'local_timestamp' in ob:
                        ts = datetime.fromtimestamp(ob['local_timestamp'] / 1000.0)
                    elif 'T' in ob:
                        ts = datetime.fromtimestamp(ob['T'] / 1000.0)
                        
                    processor.update_orderbook(clean_ob, timestamp=ts)
                    
                    # 1. Attn-LOB Features (New 40-dim vector)
                    attn_features = processor.get_attn_lob_features(clean_ob)
                    # Log to DB if recorder available
                    if recorder and attn_features:
                        recorder.log_features(attn_features)

                    # 2. Helper to safely get dashboard stats
                    def get_f(func, *args):
                        try:
                            return float(func(*args))
                        except:
                            return 0.0

                     features = {
                        'vpin': get_f(processor.get_ivpin, processor.windows[2], ts), # iVPIN slow for dashboard
                        'ofi': get_f(processor.get_ofi, processor.windows[2]),
                        'tfi': get_f(processor.get_tfi, processor.windows[1]),
                        'vol': get_f(lambda: processor.volatility),
                        'noise': get_f(processor.get_microstructure_noise),
                        'amihud': get_f(processor.get_amihud_illiquidity),
                        'slope': get_f(processor.get_orderbook_slope),
                        'entropy': get_f(processor.get_trade_entropy),
                        'effective_spread': get_f(processor.get_effective_spread, ts),
                        'order_imbalance': get_f(processor.get_order_imbalance),
                        'vpin_progress': 0.0,  # iVPIN has no bucket progress
                        # Add AI Signals
                        'ai_bias': ai_bias,
                        'ai_spread': ai_spread,
                        'ai_ready': ai_ready
                    }
                except Exception as e:
                    logger.error(f"Feature calc error: {e}")
                    # Provide defaults to prevent JS crash
                    features = {k: 0.0 for k in ['vpin', 'ofi', 'tfi', 'vol', 'noise', 'amihud', 'slope', 'entropy', 'effective_spread', 'order_imbalance', 'vpin_progress', 'ai_bias', 'ai_spread', 'ai_ready']}

            payload = json.dumps({
                'type': 'snapshot',
                'symbol': stream_manager.config.symbol,
                'data': clean_ob,
                'features': features
            })
            
            for ws in set(app['websockets']):
                await ws.send_str(payload)
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Broadcast error: {e}")

async def trade_relay_callback(trade, app, processor=None):
    """Relay trades to frontend and update processor."""
    # Update Processor
    if processor:
        try:
            ts = getattr(trade, 'timestamp', datetime.now())
            processor.update_trades(trade, timestamp=ts)
        except Exception as e:
            logger.error(f"Processor trade update error: {e}")

    if not app['websockets']:
        return
        
    t_dict = {
        'p': trade.price,
        'q': trade.quantity,
        'T': trade.timestamp.timestamp() * 1000,
        'm': trade.is_buyer_maker
    }
    
    payload = json.dumps({
        'type': 'trade',
        'data': t_dict
    })
    
    for ws in set(app['websockets']):
        try:
            await ws.send_str(payload)
        except:
            pass

async def get_24h_volume(symbol: str) -> float:
    """Fetch 24h volume for the symbol to calibrate VPIN buckets."""
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    vol = float(data['volume'])
                    logger.info(f"Fetched 24h Volume for {symbol}: {vol:,.0f}")
                    return vol
    except Exception as e:
        logger.error(f"Failed to fetch 24h volume: {e}")
    return 1000.0 * 1440 # Lower fallback for safety

async def record_data(symbol: str):
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    
    market_config = MarketConfig(
        symbol=symbol,
        tick_size=0.0001, 
        min_qty=1.0,
        update_interval=0.1 # default
    )
    dash_config = DashboardConfig()
    
    recorder = MarketDataRecorder(db_path=f"data/db/market_data_{symbol.lower()}.db")
    stream_manager = BinanceStreamHandler(market_config)
    
    # Initialize Processor (iVPIN mode — no bucket calibration needed)
    processor = MarketFeatureProcessor(market_config=market_config)
    
    # Initialize MambaLOB AI Agent (using invariant normalization)
    ai_predictor = MambaLOBPredictor(
        model_path="models.train/mamba_lob.pth",
        device="cpu" # Force CPU for recording script to be safe
    )
    
    logger.info(f"Initializing recording for {symbol}...")
    
    # Setup Web App
    app = web.Application()
    app['websockets'] = set()
    app['config'] = type('AppConfig', (), {'dashboard': dash_config})()
    app.router.add_get('/', handle_index)
    app.router.add_get('/ws', recorder_websocket_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', dash_config.port)
    await site.start()
    
    logger.info(f"DASHBOARD AVAILABLE AT: http://localhost:{dash_config.port}")
    
    import webbrowser
    webbrowser.open(f"http://localhost:{dash_config.port}")
    
    await stream_manager.connect(api_key, api_secret)
    recorder.start()
    
    stream_manager.add_orderbook_callback(recorder.log_orderbook_snapshot)
    stream_manager.add_trade_callback(recorder.log_trade)
    
    # Frontend/Processor Callbacks
    async def on_trade(trade):
        try:
            # logger.info(f"Callback received trade: {trade.price}")
            await trade_relay_callback(trade, app, processor)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    stream_manager.add_trade_callback(on_trade)

    await stream_manager.start_streams()
    
    # Start Broadcast Loop with Processor, Recorder and AI
    asyncio.create_task(broadcast_loop(app, stream_manager, processor, recorder, ai_predictor))
    
    logger.info(f"Recording and Dashboard started! Press Ctrl+C to stop.")
    
    try:
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Recording cancelled.")
    finally:
        logger.info("Stopping...")
        await stream_manager.stop()
        recorder.stop()
        await runner.cleanup()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = "XRPUSDT"
        
    try:
        asyncio.run(record_data(symbol))
    except KeyboardInterrupt:
        pass
