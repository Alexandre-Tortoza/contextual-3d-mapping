"""Casamento entre as regiões de um frame e o que a observação anterior afirmou.

O módulo é puro por frame por construção, e continua sendo: nada aqui guarda
estado entre chamadas. Quem sabe que um frame precede outro é a composição, que
deriva um :class:`ScenePrior` do resultado de um frame e o entrega ao seguinte.
Este arquivo é só a regra de **qual** região herda **qual** afirmação anterior.

O casamento é geométrico, por sobreposição de caixa, e não por similaridade de
embedding. A razão é medida e está registrada em ``application/reconciliation.py``:
o cosseno DINOv2 separa "mesmo label" de "label diferente" com acurácia
balanceada de 0,660, e por isso o módulo o usa como corroboração e nunca como
gate. Em amostragem temporal densa a câmera mal se desloca entre frames
vizinhos, e a sobreposição de caixa é o sinal forte nesse regime; o embedding
entra apenas para desempatar candidatos que já se sobrepõem.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.embeddings import VisualEmbedding
from visual_perception.domain.region_reasoning import (
    PriorHypothesis,
    PriorRegion,
    ScenePrior,
    TemporalPriorMode,
    select_region_prior,
)
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import ClaimKind, HypothesisRole, SemanticClaim
from visual_perception.domain.visual_observation import VisualObservation


# Registra que uma região recebeu uma afirmação anterior, preservando a
# evidência do casamento. Existe pelo mesmo motivo que ``SuppressedRegion``
# existe para a publicação: sem registro, não haveria como distinguir depois um
# prior bem ancorado de um casamento de sorte, nem medir se o modelo copiou a
# sugestão. Consumido pelo diagnóstico e pelos artifacts de frame.
@dataclass(frozen=True)
class PriorAssignment:
    """A afirmação anterior que uma região do frame atual herdou."""

    region_id: str
    prior: PriorHypothesis


# Devolve a claim que representa o que o frame concluiu sobre a identidade de
# uma região. A reconciliação intra-frame, quando roda, é a autoridade: ela
# existe justamente para arbitrar entre hipóteses primárias concorrentes. Sem
# ela, vale a primária. Uma ``ALTERNATIVE`` nunca entra — ela é a dúvida que o
# produtor registrou na mesma resposta, e promovê-la a prior transformaria
# incerteza declarada em evidência para o frame seguinte.
def _concluded_identity(region: ObservedRegion) -> SemanticClaim | None:
    """Retorna a claim de identidade que o frame concluiu para a região."""
    labels = [claim for claim in region.claims if claim.kind is ClaimKind.LABEL]
    reconciled = next((claim for claim in labels if claim.role is HypothesisRole.RECONCILED), None)
    if reconciled is not None:
        return reconciled
    return next((claim for claim in labels if claim.role is HypothesisRole.PRIMARY), None)


# Deriva o prior que o próximo frame receberá a partir da observação deste.
# Inclui **todas** as regiões observadas, publicadas e contexto estrutural, e
# não só as contextuais: um prior que carregasse apenas evidência publicável
# sugeriria algo interessante para toda região casada, que é exatamente o viés
# que este experimento precisa não introduzir. Saber que uma área era parede é
# tão informativo quanto saber que era um pallet.
def prior_from(observation: VisualObservation, embeddings: tuple[VisualEmbedding, ...] = ()) -> ScenePrior:
    """Converte uma observação concluída no prior consumível pelo frame seguinte.

    Argumentos:
        observation: a observação canônica já concluída de um frame.
        embeddings: os embeddings visuais desse frame, usados no desempate.
    Retorna:
        o prior correspondente; vazio quando nenhuma região concluiu identidade.
    """
    vector_by_ref = {embedding.embedding_id: embedding.vector for embedding in embeddings}
    regions: list[PriorRegion] = []
    for region in observation.all_regions:
        claim = _concluded_identity(region)
        if claim is None or not claim.value.strip():
            continue
        regions.append(
            PriorRegion(
                region_id=region.region_id,
                box=region.box,
                concept=claim.value,
                category=claim.category,
                embedding=tuple(vector_by_ref.get(region.visual_embedding_ref or "", ())),
            )
        )
    return ScenePrior(regions=tuple(regions))


# Casa cada região do frame atual com a afirmação anterior da mesma área.
# Chamada pelo pipeline antes do raciocínio de região, para que a sugestão
# chegue no request e não depois dele. Devolve apenas os casamentos que
# aconteceram: uma região sem contrapartida anterior é a saída normal, e não uma
# falha.
def match_scene_prior(
    regions: tuple[ObservedRegion, ...],
    prior: ScenePrior | None,
    embeddings: tuple[VisualEmbedding, ...],
    config: MultimodalReasoningConfig,
) -> tuple[PriorAssignment, ...]:
    """Resolve qual região herda qual afirmação anterior.

    Argumentos:
        regions: as regiões canônicas do frame atual.
        prior: o que a observação anterior afirmou, ou ``None``.
        embeddings: os embeddings visuais do frame atual, para o desempate.
        config: a configuração de raciocínio, que declara modo e sobreposição
            mínima.
    Retorna:
        um registro por região casada, na ordem das regiões.
    """
    mode = TemporalPriorMode(config.temporal_prior_mode)
    if mode is TemporalPriorMode.DISABLED or prior is None or not prior.regions:
        return ()
    vector_by_ref = {embedding.embedding_id: embedding.vector for embedding in embeddings}
    assignments: list[PriorAssignment] = []
    for region in regions:
        matched = select_region_prior(
            region.box,
            prior,
            mode=mode,
            min_overlap=config.temporal_prior_min_overlap,
            embedding=tuple(vector_by_ref.get(region.visual_embedding_ref or "", ())),
        )
        if matched is not None:
            assignments.append(PriorAssignment(region_id=region.region_id, prior=matched))
    return tuple(assignments)
