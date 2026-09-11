"""Partição da observação entre evidência publicada e contexto estrutural.

Este é o último estágio semântico do pipeline canônico, e o único que decide o
que o módulo **publica**. Ele roda depois de todos os estágios contextuais
porque todos eles precisam das superfícies estruturais: a reconciliação agrupa
fragmentos de parede, as relações usam o conceito reconciliado dos dois lados de
um par, e o audit mede a coerência entre conceito e natureza declarada.

O que ele produz, e o que deliberadamente **não** produz:

```text
produz                                   não produz
------                                   ----------
partição publicado / contexto            remoção de região
claim host_surface derivada              reescrita de label
registro do motivo de cada supressão     substituição de hipótese
```

Uma região suprimida continua inteira na observação, em
``VisualObservation.structural_context``: ela mantém máscara, box, claims,
embeddings e evidência, continua alcançável por ``region_by_id``, e continua
sendo alvo válido de relação e de grupo de entidade. Supressão aqui significa
"isto não é a evidência que o módulo promete entregar", nunca "isto não foi
observado".

A regra em si mora em ``domain/contextual_evidence.py``, como Specification
pura. Este módulo é a aplicação dela: particiona, deriva o host surface quando a
identidade afirmada o nomeia, e registra o motivo para que o descarte seja
auditável — a mesma promessa que ``proposal_filtering`` já faz para a geometria.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

from visual_perception.application.support import fingerprint_of
from visual_perception.config import ContextualPublicationConfig
from visual_perception.domain.contextual_evidence import (
    ContextualEvidenceVerdict,
    RegionSuppressionReason,
    contextual_evidence_verdict,
    host_surface_concept,
)
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion, primary_label_claim
from visual_perception.domain.semantics import ClaimKind, Evidence, SemanticClaim
from visual_perception.infrastructure.debug_recorder import record_partition_stage

#: Nome do estágio e do produtor em ``ModelProvenance``. Uma claim de host
#: surface nunca se apresenta como se o VLM a tivesse escrito.
STAGE = "contextual_publication"
PRODUCER = "contextual_publication"

__all__ = [
    "PRODUCER",
    "STAGE",
    "PublicationResult",
    "SuppressedRegion",
    "partition_observation",
]


# Preserva a proveniência de uma região não publicada. Existe pelo mesmo motivo
# que ``RejectedProposal``: suprimir em silêncio destruiria a auditoria que este
# módulo promete, e o artifact precisa poder responder *o que* saiu do output
# público, *por quê*, e *com que conceito*.
@dataclass(frozen=True)
class SuppressedRegion:
    """Uma região mantida fora do output público, com o motivo e o conceito.

    Argumentos:
        region_id: identidade da região suprimida.
        reason: motivo da supressão.
        concept: o conceito afirmado que disparou a regra, preservado para que
            o limiar da política seja auditável sem reabrir os claims.
    """

    region_id: str
    reason: RegionSuppressionReason
    concept: str


# Agrupa o resultado do estágio: as duas metades da partição e o registro por
# região suprimida. Existe para que o pipeline receba um objeto só, no mesmo
# formato dos demais estágios.
@dataclass(frozen=True)
class PublicationResult:
    """As regiões publicadas, as de contexto estrutural, e os motivos."""

    published: tuple[ObservedRegion, ...]
    structural_context: tuple[ObservedRegion, ...] = field(default_factory=tuple)
    records: tuple[SuppressedRegion, ...] = field(default_factory=tuple)


# Ponto de entrada público do estágio: separa evidência contextual de contexto
# estrutural. Chamada pelo pipeline canônico depois das relações semânticas e
# antes da construção da VisualObservation, que é o contract público.
def partition_observation(
    regions: tuple[ObservedRegion, ...],
    config: ContextualPublicationConfig,
    observation_id: str | None = None,
    debug_root: Path | str | None = None,
) -> PublicationResult:
    """Particiona as regiões do frame sem alterar nenhuma geometria.

    Com o estágio desligado, tudo continua publicado e o contexto estrutural
    fica vazio — o comportamento anterior à política, preservado para ablação.

    Argumentos:
        regions: as regiões já interpretadas, calibradas, refinadas e
            reconciliadas.
        config: se a partição roda e quais núcleos estruturais extras a
            composição declarou.
        observation_id: id do frame (para nomear artefatos de debug).
        debug_root: raiz do diretório DEBUG. Se None, não grava nada.
    Retorna:
        o :class:`PublicationResult` com as duas metades e os motivos.
    """
    if not config.enabled:
        return PublicationResult(published=regions)

    extra = frozenset(config.extra_structural_head_nouns)
    provenance = ModelProvenance(
        stage=STAGE, producer=PRODUCER, config_fingerprint=fingerprint_of(config)
    )
    published: list[ObservedRegion] = []
    structural: list[ObservedRegion] = []
    records: list[SuppressedRegion] = []
    verdicts: dict[str, ContextualEvidenceVerdict] = {}
    suppressed_for_debug: list[tuple[str, str, str]] = []

    for region in regions:
        verdict = contextual_evidence_verdict(region, extra_head_nouns=extra)
        verdicts[region.region_id] = verdict
        if verdict is ContextualEvidenceVerdict.GENERIC_STRUCTURAL_SURFACE:
            structural.append(region)
            concept = _asserted_concept(region)
            records.append(
                SuppressedRegion(
                    region_id=region.region_id,
                    reason=RegionSuppressionReason.GENERIC_STRUCTURAL_SURFACE,
                    concept=concept,
                )
            )
            suppressed_for_debug.append(
                (region.region_id, "GENERIC_STRUCTURAL_SURFACE", concept)
            )
            continue
        published.append(_with_host_surface(region, provenance, extra))

    # Grava debug se solicitado.
    if debug_root and observation_id:
        debug_path = Path(debug_root) / "visual-perception"
        debug_path.mkdir(parents=True, exist_ok=True)
        record_partition_stage(debug_path, observation_id, verdicts, suppressed_for_debug)

    return PublicationResult(
        published=tuple(published),
        structural_context=tuple(structural),
        records=tuple(records),
    )


# Anexa a claim de superfície hospedeira, quando a identidade afirmada a nomeia.
# Existe separada para que "publicar" e "derivar o host" sejam duas decisões
# legíveis: a região é publicada de qualquer forma, e a ausência da claim
# significa host desconhecido — nunca um valor de preenchimento.
def _with_host_surface(
    region: ObservedRegion, provenance: ModelProvenance, extra: frozenset[str]
) -> ObservedRegion:
    """Retorna a região com a claim ``host_surface``, ou ela mesma sem host."""
    host = host_surface_concept(region, extra_head_nouns=extra)
    if host is None:
        return region
    claim = SemanticClaim(
        ClaimKind.HOST_SURFACE,
        host,
        None,
        (
            Evidence(
                description=(
                    f"host surface named by the region's own asserted identity "
                    f"{_asserted_concept(region)!r}"
                )
            ),
        ),
        provenance,
    )
    return dataclasses.replace(region, claims=region.claims + (claim,))


# Recupera o conceito que o produtor afirmou para a região. Existe para que o
# registro de supressão e a evidência do host surface citem exatamente o mesmo
# texto, em vez de duas leituras que podem divergir.
def _asserted_concept(region: ObservedRegion) -> str:
    """Retorna o label primário da região, ou string vazia sem identidade."""
    claim = primary_label_claim(region)
    return "" if claim is None else claim.value
