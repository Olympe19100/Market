import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
print("PYTHONPATH:", sys.path)

print("Attempting to import data.processor...")
try:
    import data.processor
    print("data.processor module loaded:", data.processor)
except Exception as e:
    print("Failed to load module:", e)

try:
    from data.processor import MarketFeatureProcessor
    print("MarketFeatureProcessor SUCCESS")
except Exception as e:
    print("MarketFeatureProcessor FAIL:", e)

try:
    from data.processor import LOBFeatureProcessor
    print("LOBFeatureProcessor SUCCESS")
except Exception as e:
    print("LOBFeatureProcessor FAIL (Expected):", e)
