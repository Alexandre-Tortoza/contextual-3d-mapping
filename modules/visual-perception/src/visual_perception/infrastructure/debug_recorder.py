"""Gravação de valores de calibração em cada etapa do pipeline para auditoria.

Quando debug está ativado (via config ou variável de ambiente), cada etapa
grava artefatos locais em `DEBUG/` sob o diretório de resultados, permitindo
inspecionar a distribuição de labels, verdicts de superfícies, claims
suprimidas, etc. sem precisar re-rodar todo o pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from visual_perception.domain.contextual_evidence import ContextualEvidenceVerdict
from visual_perception.domain.semantics import SemanticClaim

__all__ = ["DebugRecorder", "record_partition_stage"]

logger = logging.getLogger(__name__)


@dataclass
class VerdictCounts:
    """Contagem de resultados de partition_observation por tipo de verdict."""

    context_bearing: int = 0
    generic_structural_surface: int = 0
    uninterpreted: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class SuppressionRecord:
    """Um exemplo de região suprimida com seu motivo e conceito."""

    region_id: str
    reason: str
    concept: str


class DebugRecorder:
    """Grava artefatos de calibração em diretório local."""

    def __init__(self, root: Path | str | None = None):
        """Inicializa o recorder.

        Argumentos:
            root: raiz do diretório DEBUG. Se None, não grava nada.
        """
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def record_partition(
        self,
        frame_id: str,
        verdicts: dict[str, ContextualEvidenceVerdict],
        suppressed: list[SuppressionRecord],
    ) -> None:
        """Grava resultado de partition_observation de um frame.

        Argumentos:
            frame_id: identificador do frame (para nomes de arquivo).
            verdicts: mapeamento {region_id -> verdict} produzido por
                contextual_evidence_verdict().
            suppressed: lista de SuppressedRegion, com reason e concept.
        """
        if not self.root:
            return

        # Conta os verdicts.
        counts = VerdictCounts()
        for verdict in verdicts.values():
            if verdict is ContextualEvidenceVerdict.CONTEXT_BEARING:
                counts.context_bearing += 1
            elif verdict is ContextualEvidenceVerdict.GENERIC_STRUCTURAL_SURFACE:
                counts.generic_structural_surface += 1
            elif verdict is ContextualEvidenceVerdict.UNINTERPRETED:
                counts.uninterpreted += 1

        # Prepara amostra de suprimidas (até 10 exemplos).
        suppressed_sample = [asdict(r) for r in suppressed[:10]]

        data = {
            "frame_id": frame_id,
            "verdict_counts": counts.to_dict(),
            "suppressed_count": len(suppressed),
            "suppressed_sample": suppressed_sample,
        }

        path = self.root / f"{frame_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Debug recorded partition: {path}")


def record_partition_stage(
    root: Path | str | None,
    frame_id: str,
    verdicts: dict[str, ContextualEvidenceVerdict],
    suppressed_regions: list[tuple[str, str, str]],
) -> None:
    """Conveniência: grava partition de um frame sem manter um recorder.

    Argumentos:
        root: caminho do DEBUG root.
        frame_id: id do frame.
        verdicts: {region_id -> verdict}.
        suppressed_regions: lista de (region_id, reason, concept).
    """
    if not root:
        return

    suppressed = [
        SuppressionRecord(region_id=rid, reason=reason, concept=concept)
        for rid, reason, concept in suppressed_regions
    ]
    recorder = DebugRecorder(root)
    recorder.record_partition(frame_id, verdicts, suppressed)
