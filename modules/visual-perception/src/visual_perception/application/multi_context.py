"""Extração de evidência multi-contexto por região.

Issue: #194 (implementa o contract da #193).

Esta etapa é a dona única da evidência de região: ela produz o embedding
visual de foreground (pooling mask-aware), os embeddings alinhados a
linguagem do crop justo e do crop com contexto, e a evidência global de
cena, registrando cada um como um
:class:`~visual_perception.domain.region_evidence.RegionEvidenceSlot` com
sua própria geometria, pré-processamento e identidade de espaço.

Duas garantias estruturam o código:

- **isolamento por slot**: uma falha ao produzir contexto opcional marca
  aquele slot como ``failed`` com um motivo, e é reportada ao chamador, sem
  invalidar os slots que deram certo na mesma região;
- **geometria preservada**: contexto global nunca altera a geometria da
  região. O crop contextual tem a sua própria caixa e o seu próprio
  transform; a máscara e o box canônicos da região não são tocados.

Slots desabilitados por configuração são registrados como ``missing`` em vez
de omitidos, para que um report de ablation (#200) mostre a diferença entre
"não configurado" e "falhou".
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import numpy as np

from visual_perception.application.pooling import PooledRegion, pool_region_evidence, pooling_method_for
from visual_perception.config import ModuleConfig
from visual_perception.domain.embeddings import (
    EmbeddingModality,
    EmbeddingSpace,
    LanguageEmbedding,
    VisualEmbedding,
)
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.geometry import BoundingBox, CoordinateTransform
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_evidence import (
    EvidenceSlot,
    EvidenceState,
    RegionEvidenceSlot,
)
from visual_perception.domain.regions import ObservedRegion
from visual_perception.ports.language_embedding import LanguageAlignedEncoder

#: Identidade do artifact de evidência global de cena. É único por imagem:
#: todas as regiões referenciam o mesmo vetor, porque a evidência de cena
#: não depende da região.
SCENE_EVIDENCE_REF = "language-scene"


# Registra a falha de um slot específico de uma região. Existe para que o
# chamador veja o que não foi produzido sem precisar varrer todos os slots,
# espelhando o formato ``(resultados, falhas)`` que a interpretação de
# região (#165) já usa.
@dataclass(frozen=True)
class EvidenceExtractionFailure:
    """Uma falha isolada ao produzir um slot de evidência de uma região."""

    region_id: str
    slot: EvidenceSlot
    reason: str


# Agrega custo e cobertura observados por slot em um frame. Existe para que
# perfis e benchmarks distingam slot desabilitado, falha e evidência real,
# além de contabilizar chamadas ao encoder compartilhado (#209).
@dataclass(frozen=True)
class EvidenceSlotMetrics:
    """Métricas operacionais agregadas de um slot em um frame."""

    slot: EvidenceSlot
    available: int
    missing: int
    failed: int
    model_calls: int
    latency_s: float


# Agrupa tudo que a etapa de evidência produz: as regiões já com seus slots,
# os embeddings referenciados por eles e as falhas isoladas. Existe para que
# o pipeline receba um único objeto em vez de quatro retornos posicionais.
@dataclass(frozen=True)
class MultiContextResult:
    """As regiões com evidência anexada, os embeddings produzidos e as falhas isoladas."""

    regions: tuple[ObservedRegion, ...]
    visual_embeddings: tuple[VisualEmbedding, ...]
    language_embeddings: tuple[LanguageEmbedding, ...]
    failures: tuple[EvidenceExtractionFailure, ...]
    metrics: tuple[EvidenceSlotMetrics, ...] = ()


# Ponto de entrada público da etapa: produz todos os slots de evidência
# configurados para cada região e devolve as regiões atualizadas junto dos
# embeddings e das falhas. Chamada pelo pipeline canônico logo após o merge
# de regiões, no lugar das chamadas separadas de pooling e de embedding.
def extract_region_evidence(
    regions: tuple[ObservedRegion, ...],
    image: ImagePayload,
    config: ModuleConfig,
    encoder: LanguageAlignedEncoder,
    *,
    feature_map: FeatureMap | None = None,
) -> MultiContextResult:
    """Produz a evidência multi-contexto de cada região, isolando falhas por slot.

    Argumentos:
        regions: as regiões canônicas já mescladas.
        image: o payload de pixels da imagem completa.
        config: a configuração do módulo (slots habilitados, backends).
        encoder: o encoder alinhado a linguagem usado pelos slots de crop.
        feature_map: o mapa denso; ``None`` desabilita o slot de foreground.
    Retorna:
        um :class:`MultiContextResult` com regiões, embeddings e falhas.
    """
    settings = config.multi_context
    visual_embeddings: list[VisualEmbedding] = []
    language_embeddings: list[LanguageEmbedding] = []
    failures: list[EvidenceExtractionFailure] = []
    updated: list[ObservedRegion] = []
    latency = dict.fromkeys(EvidenceSlot, 0.0)
    model_calls = dict.fromkeys(EvidenceSlot, 0)

    scene_slot_template = None
    if settings.scene_conditioned_enabled and regions:
        started = time.monotonic()
        scene_slot_template = _scene_evidence(
            image, config, encoder, language_embeddings, failures
        )
        latency[EvidenceSlot.SCENE_CONDITIONED] += time.monotonic() - started
        model_calls[EvidenceSlot.SCENE_CONDITIONED] += 1

    for region in regions:
        slots: list[RegionEvidenceSlot] = []
        visual_ref: str | None = None
        language_ref: str | None = None

        if not settings.foreground_enabled or feature_map is None:
            slots.append(_missing(region, EvidenceSlot.FOREGROUND_DENSE, _why_disabled(feature_map)))
        else:
            started = time.monotonic()
            dense_slot, visual = _foreground_evidence(region, feature_map, config, failures)
            latency[EvidenceSlot.FOREGROUND_DENSE] += time.monotonic() - started
            slots.append(dense_slot)
            if visual is not None:
                visual_embeddings.append(visual)
                visual_ref = visual.embedding_id

        for slot_kind, enabled in (
            (EvidenceSlot.TIGHT_CROP, settings.tight_crop_enabled),
            (EvidenceSlot.CONTEXTUAL_CROP, settings.contextual_crop_enabled),
        ):
            if not enabled:
                slots.append(_missing(region, slot_kind, "slot disabled by configuration"))
                continue
            started = time.monotonic()
            crop_slot, language = _crop_evidence(region, image, config, encoder, slot_kind, failures)
            latency[slot_kind] += time.monotonic() - started
            model_calls[slot_kind] += 1
            slots.append(crop_slot)
            if language is not None:
                language_embeddings.append(language)
                if slot_kind is EvidenceSlot.TIGHT_CROP:
                    language_ref = language.embedding_id

        if scene_slot_template is None:
            slots.append(
                _missing(region, EvidenceSlot.SCENE_CONDITIONED, "slot disabled by configuration")
            )
        else:
            slots.append(dataclasses.replace(scene_slot_template, region_id=region.region_id))

        updated.append(
            dataclasses.replace(
                region,
                evidence=tuple(slots),
                visual_embedding_ref=visual_ref,
                language_embedding_ref=language_ref,
            )
        )

    state_counts = {
        slot: {state: 0 for state in EvidenceState}
        for slot in EvidenceSlot
    }
    for region in updated:
        for evidence_slot in region.evidence:
            state_counts[evidence_slot.slot][evidence_slot.state] += 1
    metrics = tuple(
        EvidenceSlotMetrics(
            slot=slot,
            available=state_counts[slot][EvidenceState.AVAILABLE],
            missing=state_counts[slot][EvidenceState.MISSING],
            failed=state_counts[slot][EvidenceState.FAILED],
            model_calls=model_calls[slot],
            latency_s=latency[slot],
        )
        for slot in EvidenceSlot
    )
    return MultiContextResult(
        regions=tuple(updated),
        visual_embeddings=tuple(visual_embeddings),
        language_embeddings=tuple(language_embeddings),
        failures=tuple(failures),
        metrics=metrics,
    )


# Produz o slot de foreground denso: pooling mask-aware sobre o mapa denso,
# sem nenhum pixel de fora da máscara. Helper de extract_region_evidence,
# separado para que a falha de pooling de uma região não escape do try.
def _foreground_evidence(
    region: ObservedRegion,
    feature_map: FeatureMap,
    config: ModuleConfig,
    failures: list[EvidenceExtractionFailure],
) -> tuple[RegionEvidenceSlot, VisualEmbedding | None]:
    """Agrega as features densas sob a máscara da região em um slot de foreground."""
    method = pooling_method_for(config.feature_extraction.upsampling)
    try:
        pooled = pool_region_evidence(region.mask, feature_map, method)
    except ValueError as error:
        failures.append(
            EvidenceExtractionFailure(region.region_id, EvidenceSlot.FOREGROUND_DENSE, str(error))
        )
        return _failed(region, EvidenceSlot.FOREGROUND_DENSE, str(error)), None

    embedding = _visual_embedding(region, pooled, feature_map, method)
    slot = RegionEvidenceSlot(
        slot=EvidenceSlot.FOREGROUND_DENSE,
        region_id=region.region_id,
        state=EvidenceState.AVAILABLE,
        crop_box=region.box,
        transform=feature_map.transform,
        preprocessing=f"mask_aware_pooling:{method}",
        artifact_ref=embedding.embedding_id,
        space=_visual_space(feature_map, config, pooled),
        mask_ref=f"mask-{region.region_id}",
        support_ratio=pooled.support_ratio,
    )
    return slot, embedding


# Produz um slot de crop alinhado a linguagem (justo ou com contexto),
# recortando a imagem e chamando o encoder. Helper de
# extract_region_evidence, compartilhado pelos dois slots de crop porque
# eles só diferem na caixa recortada.
def _crop_evidence(
    region: ObservedRegion,
    image: ImagePayload,
    config: ModuleConfig,
    encoder: LanguageAlignedEncoder,
    slot_kind: EvidenceSlot,
    failures: list[EvidenceExtractionFailure],
) -> tuple[RegionEvidenceSlot, LanguageEmbedding | None]:
    """Codifica o crop pedido da região no espaço alinhado a linguagem."""
    settings = config.multi_context
    expansion = settings.context_expansion if slot_kind is EvidenceSlot.CONTEXTUAL_CROP else 0.0
    masked = settings.masked_tight_crop and slot_kind is EvidenceSlot.TIGHT_CROP
    try:
        box = _expanded_box(region.box, expansion, image.width, image.height)
        crop = _crop_payload(image, region, box, masked=masked)
        vector = encoder.encode_image(crop, config.language_embedding)
        embedding = LanguageEmbedding(
            embedding_id=_language_ref(slot_kind, region.region_id),
            region_id=region.region_id,
            vector=vector,
            dimension=len(vector),
            model_id=config.language_embedding.backend,
            checkpoint=config.language_embedding.checkpoint,
            normalized=config.language_embedding.normalize,
        )
    except (ValueError, TypeError, KeyError) as error:
        failures.append(EvidenceExtractionFailure(region.region_id, slot_kind, str(error)))
        return _failed(region, slot_kind, str(error)), None

    preprocessing = "masked_crop" if masked else "crop"
    slot = RegionEvidenceSlot(
        slot=slot_kind,
        region_id=region.region_id,
        state=EvidenceState.AVAILABLE,
        crop_box=box,
        transform=CoordinateTransform(1.0, 1.0, box.x_min, box.y_min),
        preprocessing=f"{preprocessing}:expansion={expansion}",
        artifact_ref=embedding.embedding_id,
        space=_language_space(config),
        mask_ref=f"mask-{region.region_id}" if masked else None,
    )
    return slot, embedding


# Produz o slot de evidência global de cena, computado uma única vez por
# imagem e referenciado por todas as regiões. Helper de
# extract_region_evidence: a evidência de cena não depende da região, e
# recomputá-la por região seria custo puro.
def _scene_evidence(
    image: ImagePayload,
    config: ModuleConfig,
    encoder: LanguageAlignedEncoder,
    language_embeddings: list[LanguageEmbedding],
    failures: list[EvidenceExtractionFailure],
) -> RegionEvidenceSlot | None:
    """Codifica a imagem completa como evidência global compartilhada entre as regiões.

    Retorna um slot modelo, cujo ``region_id`` é substituído por região, ou
    ``None`` quando a codificação falhou (a falha é reportada uma vez, não
    por região).
    """
    try:
        vector = encoder.encode_image(image, config.language_embedding)
        language_embeddings.append(
            LanguageEmbedding(
                embedding_id=SCENE_EVIDENCE_REF,
                region_id="scene",
                vector=vector,
                dimension=len(vector),
                model_id=config.language_embedding.backend,
                checkpoint=config.language_embedding.checkpoint,
                normalized=config.language_embedding.normalize,
            )
        )
    except (ValueError, TypeError, KeyError) as error:
        failures.append(EvidenceExtractionFailure("scene", EvidenceSlot.SCENE_CONDITIONED, str(error)))
        return None

    return RegionEvidenceSlot(
        slot=EvidenceSlot.SCENE_CONDITIONED,
        region_id="scene",
        state=EvidenceState.AVAILABLE,
        # A caixa é a imagem inteira: contexto global é anexado à região sem
        # jamais alterar a geometria dela (critério de aceitação da #194).
        crop_box=BoundingBox(0.0, 0.0, float(image.width), float(image.height)),
        transform=CoordinateTransform.identity(),
        preprocessing="full_image",
        artifact_ref=SCENE_EVIDENCE_REF,
        space=_language_space(config),
    )


# Constrói o VisualEmbedding do slot de foreground, mantendo o id canônico
# ``visual-<region_id>`` que o contract de região (#154) já expõe.
def _visual_embedding(
    region: ObservedRegion, pooled: PooledRegion, feature_map: FeatureMap, method: str
) -> VisualEmbedding:
    """Empacota o vetor agregado da região em um :class:`VisualEmbedding` canônico."""
    return VisualEmbedding(
        embedding_id=f"visual-{region.region_id}",
        region_id=region.region_id,
        vector=pooled.vector,
        dimension=len(pooled.vector),
        pooling_method=method,
        feature_resolution=f"{feature_map.grid_width}x{feature_map.grid_height}",
        model_id=feature_map.model_id,
        normalized=True,
    )


# Deriva a identidade do espaço visual a partir do mapa denso e da config.
# Existe para que dois slots de foreground produzidos por backbones ou
# checkpoints diferentes nunca sejam comparados como se fossem o mesmo.
def _visual_space(feature_map: FeatureMap, config: ModuleConfig, pooled: PooledRegion) -> EmbeddingSpace:
    """Retorna o :class:`EmbeddingSpace` do vetor visual agregado."""
    return EmbeddingSpace(
        model_id=feature_map.model_id,
        checkpoint=feature_map.checkpoint or config.feature_extraction.checkpoint,
        dimension=len(pooled.vector),
        modality=EmbeddingModality.VISUAL_DENSE,
    )


# Deriva a identidade do espaço alinhado a linguagem a partir da config.
# Existe pela mesma razão que _visual_space, do lado do encoder de texto.
def _language_space(config: ModuleConfig) -> EmbeddingSpace:
    """Retorna o :class:`EmbeddingSpace` dos vetores alinhados a linguagem."""
    return EmbeddingSpace(
        model_id=config.language_embedding.backend,
        checkpoint=config.language_embedding.checkpoint,
        dimension=config.language_embedding.dimension,
        modality=EmbeddingModality.LANGUAGE_ALIGNED,
        normalized=config.language_embedding.normalize,
    )


# Expande a caixa de uma região por uma fração das suas próprias dimensões,
# recortando aos limites da imagem. Existe para que o crop contextual seja
# proporcional ao objeto (uma margem fixa em pixels engoliria um objeto
# pequeno e mal tocaria um grande).
def _expanded_box(box: BoundingBox, expansion: float, width: int, height: int) -> BoundingBox:
    """Retorna ``box`` expandida por ``expansion`` e recortada à imagem."""
    if expansion <= 0.0:
        return box.clipped_to(width=width, height=height)
    margin_x = box.width * expansion / 2.0
    margin_y = box.height * expansion / 2.0
    return BoundingBox(
        x_min=box.x_min - margin_x,
        y_min=box.y_min - margin_y,
        x_max=box.x_max + margin_x,
        y_max=box.y_max + margin_y,
    ).clipped_to(width=width, height=height)


# Recorta os pixels da caixa pedida, opcionalmente zerando tudo que está
# fora da máscara da região. Existe para que o crop mascarado seja uma
# opção de configuração e não uma variante duplicada desta etapa.
def _crop_payload(
    image: ImagePayload, region: ObservedRegion, box: BoundingBox, *, masked: bool
) -> ImagePayload:
    """Recorta ``box`` da imagem, zerando o fundo quando ``masked``."""
    x_min, y_min = int(box.x_min), int(box.y_min)
    x_max, y_max = int(box.x_max), int(box.y_max)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Region {region.region_id!r} has a degenerate crop box after clipping.")
    crop = image.crop(x_min, y_min, x_max, y_max)
    if not masked:
        return crop
    window = region.mask.data[y_min:y_max, x_min:x_max]
    pixels = np.where(window[:, :, None], crop.pixels, 0).astype(crop.pixels.dtype)
    return ImagePayload(pixels, width=crop.width, height=crop.height)


# Nomeia o artifact de embedding de cada slot de crop, mantendo o id
# canônico ``language-<region_id>`` para o crop justo.
def _language_ref(slot_kind: EvidenceSlot, region_id: str) -> str:
    """Retorna o identificador de artifact do embedding de ``slot_kind``."""
    if slot_kind is EvidenceSlot.TIGHT_CROP:
        return f"language-{region_id}"
    return f"language-context-{region_id}"


# Constrói um slot ausente com o motivo explícito. Existe para que "não
# configurado" apareça na saída em vez de virar um buraco silencioso.
def _missing(region: ObservedRegion, slot_kind: EvidenceSlot, reason: str) -> RegionEvidenceSlot:
    """Retorna um slot ``missing`` explicando por que a evidência não existe."""
    return RegionEvidenceSlot(
        slot=slot_kind, region_id=region.region_id, state=EvidenceState.MISSING, reason=reason
    )


# Constrói um slot que falhou, preservando a mensagem do erro original.
# Existe para que a falha fique auditável na própria região, além de ser
# reportada ao chamador.
def _failed(region: ObservedRegion, slot_kind: EvidenceSlot, reason: str) -> RegionEvidenceSlot:
    """Retorna um slot ``failed`` carregando o motivo da falha."""
    return RegionEvidenceSlot(
        slot=slot_kind, region_id=region.region_id, state=EvidenceState.FAILED, reason=reason
    )


# Explica por que o slot de foreground não foi produzido, distinguindo
# "desabilitado" de "sem mapa denso disponível".
def _why_disabled(feature_map: FeatureMap | None) -> str:
    """Retorna o motivo pelo qual o slot de foreground não foi produzido."""
    return "no dense feature map available" if feature_map is None else "slot disabled by configuration"
