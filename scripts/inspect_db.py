import sqlite3
import json
import numpy as np

conn = sqlite3.connect('training_data.db')
cursor = conn.cursor()

cursor.execute("SELECT timestamp, lob_vector, aux_vector FROM training_features LIMIT 1")
row = cursor.fetchone()

if row:
    ts, lob_json, aux_json = row
    lob = json.loads(lob_json)
    aux = json.loads(aux_json)
    
    print(f"Timestamp: {ts}")
    print(f"LOB Vector (Len {len(lob)}):")
    print(f"  [BidP, BidV ... AskP, AskV ... Spread, Imb]")
    print(f"  Sample: {lob[:5]} ... {lob[-5:]}")
    
    print(f"Aux Vector (Len {len(aux)}):")
    print(f"  Sample: {aux}")
else:
    print("No data found.")

conn.close()
