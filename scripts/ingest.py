
import os
import sys
import json
import sqlite3
import numpy as np
import zipfile
import logging
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from data.processor import MarketFeatureProcessor
from core.config import MarketConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ingest")

def main():
    data_dir = "."
    db_name = "training_data.db"
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            lob_vector TEXT,
            aux_vector TEXT
        )
    ''')
    conn.commit()
    
    zip_files = [f for f in os.listdir(data_dir) if f.endswith('_ob200.data.zip')]
    zip_files.sort()
    
    if not zip_files:
        logger.error("No files found")
        return

    logger.info(f"Found {len(zip_files)} files")
    
    # Initialize Processor (Single Processor handles all)
    market_config = MarketConfig()
    processor = MarketFeatureProcessor(market_config=market_config)
    
    total = 0
    for zf in zip_files:
        logger.info(f"Processing {zf}")
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                fname = z.namelist()[0]
                with z.open(fname) as f:
                    batch = []
                    for line in f:
                        try:
                            s = json.loads(line.decode('utf-8'))
                            
                            # Features
                            # MarketFeatureProcessor has get_attn_lob_features AND process
                            if hasattr(processor, 'get_attn_lob_features'):
                                lv = processor.get_attn_lob_features(s)
                            else:
                                lv = [0.0] * 40  # Fixed LOB feature dimension
                                
                            av = processor.process(s)
                            ts = s.get('ts', 0)
                            
                            batch.append((ts, json.dumps(list(lv)), json.dumps(list(av))))
                            
                            batch_size = market_config.record_batch_size
                            if len(batch) >= batch_size:
                                cursor.executemany("INSERT INTO training_features (timestamp, lob_vector, aux_vector) VALUES (?, ?, ?)", batch)
                                conn.commit()
                                total += len(batch)
                                batch = []
                        except: continue
                    
                    if batch:
                        cursor.executemany("INSERT INTO training_features (timestamp, lob_vector, aux_vector) VALUES (?, ?, ?)", batch)
                        conn.commit()
                        total += len(batch)
        except Exception as e:
            logger.error(f"Error {zf}: {e}")
            
    logger.info(f"Done. Total: {total}")
    conn.close()

if __name__ == "__main__":
    main()
