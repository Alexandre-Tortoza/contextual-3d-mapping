import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "contracts", ROOT / "datasets", ROOT / "adapters" / "datasets"):
    sys.path.insert(0, str(path))
