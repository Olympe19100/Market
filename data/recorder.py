
import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import queue
import threading
import time

logger = logging.getLogger(__name__)

class MarketDataRecorder:
    """
    Enregistre les données de marché (L2 updates, snapshots, trades) dans une base de données SQLite.
    Conçu pour gérer le haut débit via une queue et un thread d'écriture dédié.
    """
    
    def __init__(self, db_path: str = "market_data.db", market_config: Optional[MarketConfig] = None):
        self.db_path = db_path
        self.market_config = market_config
        self.queue = queue.Queue()
        self.running = False
        self.writer_thread = None
        self._init_db()
        
    def _init_db(self):
        """Initialise le schéma de la base de données."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table pour les snapshots du carnet (LOB complet)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        symbol TEXT,
                        bids TEXT, -- JSON array [[price, qty], ...]
                        asks TEXT  -- JSON array
                    )
                ''')
                
                # Table pour les mises à jour incrémentales (Diff Depth)
                # C'est ce qui se rapproche le plus du L3 accessible (flux continu de changements)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS depth_updates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        symbol TEXT,
                        first_update_id INTEGER,
                        final_update_id INTEGER,
                        bids TEXT, -- JSON changes
                        asks TEXT
                    )
                ''')
                
                # Table pour les trades
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        symbol TEXT,
                        price REAL,
                        quantity REAL,
                        is_buyer_maker BOOLEAN,
                        trade_time REAL
                    )
                ''')
                
                # Table pour les features du modèle (Attn-LOB 40-dim vector)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        symbol TEXT,
                        feature_vector TEXT -- JSON array of 40 floats
                    )
                ''')
                conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to init database: {e}")
            raise

    def start(self):
        """Démarre le thread d'écriture."""
        self.running = True
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        logger.info("MarketDataRecorder started")

    def stop(self):
        """Arrête l'enregistrement."""
        self.running = False
        if self.writer_thread:
            self.writer_thread.join()
        logger.info("MarketDataRecorder stopped")

    async def log_trade(self, trade):
        """Met un trade en queue pour enregistrement."""
        # trade object or dict
        try:
            # Extract attributes safely handling both object and dict types
            price = float(trade.price) if hasattr(trade, 'price') else float(trade['p'])
            qty = float(trade.quantity) if hasattr(trade, 'quantity') else float(trade['q'])
            
            is_buyer_maker = False
            if hasattr(trade, 'is_buyer_maker'):
                is_buyer_maker = trade.is_buyer_maker
            elif isinstance(trade, dict):
                is_buyer_maker = trade.get('m', False)
                
            trade_timestamp = getattr(trade, 'timestamp', datetime.now()).timestamp()
            # If it's a dict and has 'T', use it
            if isinstance(trade, dict) and 'T' in trade:
                trade_timestamp = trade['T'] / 1000.0

            data = {
                'type': 'trade',
                'ts': datetime.now().timestamp(),
                'symbol': getattr(trade, 'symbol', 'UNKNOWN'), 
                'price': price,
                'qty': qty,
                'buyer_maker': is_buyer_maker,
                'trade_time': trade_timestamp
            }
            self.queue.put(data)
        except Exception as e:
            logger.error(f"Error queuing trade: {e}")

    async def log_orderbook_snapshot(self, orderbook):
        """Enregistre un snapshot complet."""
        try:
            depth_limit = self.market_config.record_depth if self.market_config else 20
            
            # Handle list of OrderBookLevel objects
            bids_data = []
            for level in orderbook.bids[:depth_limit]:
                if hasattr(level, 'price'):
                    bids_data.append([float(level.price), float(level.quantity)])
                else:
                    bids_data.append([float(level[0]), float(level[1])]) # Fallback for tuples

            asks_data = []
            for level in orderbook.asks[:depth_limit]:
                if hasattr(level, 'price'):
                    asks_data.append([float(level.price), float(level.quantity)])
                else:
                    asks_data.append([float(level[0]), float(level[1])])

            data = {
                'type': 'snapshot',
                'ts': datetime.now().timestamp(),
                'symbol': getattr(orderbook, 'symbol', 'UNKNOWN'), 
                'bids': json.dumps(bids_data),
                'asks': json.dumps(asks_data)
            }
            self.queue.put(data)
        except Exception as e:
            logger.error(f"Error queuing snapshot: {e}")

    def log_features(self, feature_vector: List[float]):
        """Enregistre le vecteur de features calculé."""
        try:
            symbol = self.market_config.symbol if self.market_config else 'UNKNOWN'
            data = {
                'type': 'features',
                'ts': datetime.now().timestamp(),
                'symbol': symbol, 
                'vector': json.dumps(feature_vector)
            }
            self.queue.put(data)
        except Exception as e:
            logger.error(f"Error queuing features: {e}")

    def _writer_loop(self):
        """Boucle principale d'écriture en batch."""
        conn = sqlite3.connect(self.db_path)
        while self.running or not self.queue.empty():
            try:
                batch = []
                batch_size = self.market_config.record_batch_size if self.market_config else 1000
                # Récupérer jusqu'à batch_size items ou timeout
                try:
                    while len(batch) < batch_size:
                        item = self.queue.get(timeout=1.0)
                        batch.append(item)
                except queue.Empty:
                    pass
                
                if batch:
                    self._write_batch(conn, batch)
                    
            except Exception as e:
                logger.error(f"Error in writer loop: {e}")
                time.sleep(1)
        
        conn.close()

    def _write_batch(self, conn, batch):
        """Écrit un lot de données dans la base."""
        cursor = conn.cursor()
        trades = []
        snapshots = []
        features = []
        
        for item in batch:
            if item['type'] == 'trade':
                trades.append((
                    item['ts'], item['symbol'], item['price'], 
                    item['qty'], item['buyer_maker'], item['trade_time']
                ))
            elif item['type'] == 'snapshot':
                snapshots.append((
                   item['ts'], item['symbol'], item['bids'], item['asks']
                ))
            elif item['type'] == 'features':
                features.append((
                    item['ts'], item['symbol'], item['vector']
                ))
                
        if trades:
            cursor.executemany('''
                INSERT INTO trades (timestamp, symbol, price, quantity, is_buyer_maker, trade_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', trades)
            
        if snapshots:
            cursor.executemany('''
                INSERT INTO orderbook_snapshots (timestamp, symbol, bids, asks)
                VALUES (?, ?, ?, ?)
            ''', snapshots)

        if features:
            cursor.executemany('''
                INSERT INTO model_features (timestamp, symbol, feature_vector)
                VALUES (?, ?, ?)
            ''', features)
            
        conn.commit()
