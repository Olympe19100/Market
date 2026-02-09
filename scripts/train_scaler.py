
import os, sys, json, zipfile, logging, pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScalerTrainer")

def train_scalers():
    files = sorted([f for f in os.listdir(".") if f.endswith("_ob200.data.zip")])
    if not files:
        logger.error("No data found")
        return
        
    prices = []
    vols = []
    
    # Collect 50k samples
    limit = 50000
    count = 0
    
    # Use first, middle, last file for variety
    targets = [files[0], files[len(files)//2], files[-1]]
    
    for zf in targets:
        logger.info(f"Sampling {zf}...")
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                with z.open(z.namelist()[0]) as f:
                    for l in f:
                        if count >= limit: break
                        try:
                            d = json.loads(l)
                            b = np.array(d['bids'][:10], dtype=float)
                            a = np.array(d['asks'][:10], dtype=float)
                            if len(b)<10 or len(a)<10: continue
                            
                            prices.extend(b[:,0])
                            prices.extend(a[:,0])
                            vols.extend(b[:,1])
                            vols.extend(a[:,1])
                            count += 1
                        except Exception as e: 
                            if count < 5: logger.error(f"Row Error: {e}")
                            pass
        except Exception as e: logger.error(f"File Error: {e}")
        if count >= limit: break
        
    logger.info(f"Collected {len(prices)} prices and {len(vols)} volumes.")
    
    ps = StandardScaler()
    vs = StandardScaler()
    
    ps.fit(np.array(prices).reshape(-1,1))
    vs.fit(np.array(vols).reshape(-1,1))
    
    os.makedirs("models.train", exist_ok=True)
    with open("models.train/price_scaler.pkl", "wb") as f: pickle.dump(ps, f)
    with open("models.train/qty_scaler.pkl", "wb") as f: pickle.dump(vs, f)
    
    logger.info("Scalers saved.")

if __name__ == "__main__":
    train_scalers()
