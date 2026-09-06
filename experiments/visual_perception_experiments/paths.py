"""Localização das raízes do repositório usadas pelos experimentos.

Existe para que cada script não recalcule a raiz do repositório a partir do
próprio caminho, e para que a inclusão das raízes de import fique em um
único lugar: nenhum dos pacotes envolvidos é instalado durante o
desenvolvimento local.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Raiz do repositório, derivada da posição fixa deste arquivo.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Raízes de import necessárias para compor módulo, contracts, datasets e
#: avaliação em um mesmo processo de experimento.
IMPORT_ROOTS = (
    REPOSITORY_ROOT / "contracts",
    REPOSITORY_ROOT / "datasets",
    REPOSITORY_ROOT / "adapters" / "datasets",
    REPOSITORY_ROOT / "evaluation",
    REPOSITORY_ROOT / "modules" / "visual-perception" / "src",
)

#: Onde o manifest do conjunto de referência anotado é versionado.
REFERENCE_MANIFEST = REPOSITORY_ROOT / "datasets" / "manifests" / "corridor-02-visual-reference.json"

#: Onde os frames extraídos do conjunto de referência ficam (dados brutos,
#: fora do controle de versão).
REFERENCE_FRAMES = REPOSITORY_ROOT / "datasets" / "raw" / "corridor-02" / "visual-reference"

#: Onde os artifacts de experimento (predições, reports) são gravados.
RESULTS_ROOT = REPOSITORY_ROOT / "experiments" / "results"


# Insere as raízes de import no ``sys.path`` uma única vez. Existe porque os
# scripts deste pacote podem ser executados diretamente, fora de uma sessão
# de pytest que já tenha o ``pythonpath`` configurado.
def ensure_import_roots() -> None:
    """Garante que módulo, contracts, datasets e avaliação sejam importáveis."""
    for root in IMPORT_ROOTS:
        path = str(root)
        if path not in sys.path:
            sys.path.insert(0, path)
