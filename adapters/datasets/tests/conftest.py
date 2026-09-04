# Adiciona os pacotes irmãos (contracts, datasets, adapters/datasets) ao
# sys.path para os testes deste adapter, já que eles não são instalados
# como dependências formais durante o desenvolvimento local.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "contracts", ROOT / "datasets", ROOT / "adapters" / "datasets"):
    sys.path.insert(0, str(path))
