"""Consolidação determinística de evidência contextual entre runs publicadas."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any

from semantic_fusion import SemanticContribution, fuse_point_contributions, measure_spatial_support, LabelledPoint

from contextual_mapping_contracts import FrameId, MapId

CONTEXT_ARTIFACT_TYPE = "contextual_rgb_lidar_slice"
CONSOLIDATED_ARTIFACT_TYPE = "consolidated_contextual_map"
#: O mapa consolidado reusa a mesma forma de documento das runs de origem
#: (observations, regions, debug_manifest); não é um schema à parte com
#: numeração própria, então ele acompanha o schema_version da run contextual
#: em vez de manter uma versão independente que nunca é bumpada junto.
CONSOLIDATED_SCHEMA_VERSION = 3


# Normaliza um label para a comparação textual entre runs. Existe porque
# variações de caixa e espaços não devem criar votos semanticamente distintos.
def _normalise_label(value: str) -> str:
    """Normaliza espaços e caixa de um label aberto."""
    return " ".join(value.casefold().split())


# Verifica se uma sequência de tokens aparece inteira em outra. A regra é a
# mesma usada pelo viewer para reunir ``pallet`` e ``wooden pallet``.
def _contains_tokens(tokens: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    """Retorna se ``needle`` é uma subsequência contígua própria de ``tokens``."""
    if len(needle) >= len(tokens):
        return False
    return any(tokens[start:start + len(needle)] == needle for start in range(len(tokens) - len(needle) + 1))


# Deriva famílias canônicas de labels a partir de todos os votos publicados.
# O mapa semântico possui essa decisão porque ela altera a fusão, não apenas a
# paleta visual do consumidor.
def _label_families(labels: Iterable[str]) -> dict[str, str]:
    """Agrupa variantes textuais em um label canônico determinístico."""
    counts = Counter(_normalise_label(label) for label in labels if _normalise_label(label))
    ordered = sorted(counts, key=lambda label: (-counts[label], label))
    parents: dict[str, str] = {}
    for label in ordered:
        tokens = tuple(label.split())
        parent = next(
            (candidate for candidate in ordered if candidate != label and _contains_tokens(tokens, tuple(candidate.split()))),
            None,
        )
        parents[label] = parent or label
    families: dict[str, str] = {}
    for label in ordered:
        root = label
        seen: set[str] = set()
        while parents[root] != root and root not in seen:
            seen.add(root)
            root = parents[root]
        families[label] = root
    return families


# Identifica a nuvem de origem de um artifact contextual. Runs de trechos diferentes
# carregam recortes diferentes da mesma nuvem; o que as torna fundíveis é a origem
# comum (PCD, mapa e frame), e não o conjunto exato de pontos exportados.
def geometry_fingerprint(payload: Mapping[str, Any]) -> str:
    """Calcula o SHA-256 canônico da geometria de origem de um artifact.

    Argumentos:
        payload: artifact com ``map_id``, ``map_frame`` e ``source.sha256``.
    Retorna:
        digest hexadecimal que identifica a nuvem de origem.
    Levanta:
        ValueError: se faltar identidade de mapa, frame ou digest da origem.
    """
    source = payload.get("source")
    if not isinstance(payload.get("map_id"), str) or not isinstance(payload.get("map_frame"), str):
        raise ValueError("context run must declare map_id and map_frame.")
    try:
        map_id, map_frame = MapId(payload["map_id"]), FrameId(payload["map_frame"])
    except ValueError as error:
        raise ValueError("context run must declare a valid map_id and map_frame.") from error
    if not isinstance(source, Mapping) or not isinstance(source.get("sha256"), str) or not source["sha256"]:
        raise ValueError("context run must declare source.sha256 of its point cloud.")
    canonical = json.dumps(
        {"map_id": str(map_id), "map_frame": str(map_frame), "source_sha256": source["sha256"],
         "source_point_count": source.get("point_count")},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


# Valida e indexa os pontos de um artifact por identidade. Identidades duplicadas,
# coordenadas não finitas ou fora de 3D tornam a fusão ponto a ponto ambígua.
def _points_by_geometry_id(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Retorna os pontos do artifact indexados por ``geometry_id``."""
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError("context run must declare points.")
    indexed: dict[str, Mapping[str, Any]] = {}
    for point in points:
        if not isinstance(point, Mapping) or not isinstance(point.get("geometry_id"), str):
            raise ValueError("every point must declare geometry_id.")
        coordinates = point.get("coordinates_m")
        if point["geometry_id"] in indexed or not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3:
            raise ValueError("geometry ids must be unique and coordinates_m must be three-dimensional.")
        try:
            numbers = tuple(float(value) for value in coordinates)
        except (TypeError, ValueError) as error:
            raise ValueError("coordinates_m must be numeric.") from error
        if not all(number == number and abs(number) != float("inf") for number in numbers):
            raise ValueError("coordinates_m must be finite.")
        indexed[point["geometry_id"]] = point
    return indexed


# Copia só a geometria de um ponto; o contexto consolidado é recalculado.
def _geometry_record(point: Mapping[str, Any]) -> dict[str, Any]:
    """Retorna o ponto sem o contexto de nenhuma run específica."""
    return {key: deepcopy(value) for key, value in point.items() if key in {"geometry_id", "coordinates_m", "intensity", "display_color_rgb"}}


# Representa uma run já copiada e validada pelo publisher. A aplicação entrega
# URLs públicas, enquanto o módulo mantém a decisão de como fundir conteúdo.
@dataclass(frozen=True)
class PublishedContextRun:
    """Run contextual imutável disponível no catálogo estático.

    Argumentos:
        run_id: identidade estável da publicação de origem.
        label: nome legível apresentado no catálogo.
        artifact_url: URL pública do ``context.json`` de origem.
        artifact_sha256: hash do artifact copiado para a run.
        payload: conteúdo contextual validado pela aplicação.
    """

    run_id: str
    label: str
    artifact_url: str
    artifact_sha256: str
    payload: Mapping[str, Any]
    map_id: MapId = field(init=False)
    map_frame: FrameId = field(init=False)

    # Rejeita uma identidade incompleta antes que ela seja namespace de
    # observações e regiões no resultado global.
    def __post_init__(self) -> None:
        """Valida a identidade pública da run publicada e sua origem de mapa/frame."""
        if not self.run_id.strip() or not self.label.strip() or not self.artifact_url.startswith("/"):
            raise ValueError("published context runs require id, label and absolute artifact URL.")
        if self.payload.get("artifact_type") != CONTEXT_ARTIFACT_TYPE:
            raise ValueError("published run must contain a contextual RGB-LiDAR artifact.")
        if not isinstance(self.payload.get("map_id"), str) or not isinstance(self.payload.get("map_frame"), str):
            raise ValueError("published run must declare map_id and map_frame.")
        object.__setattr__(self, "map_id", MapId(self.payload["map_id"]))
        object.__setattr__(self, "map_frame", FrameId(self.payload["map_frame"]))


# Publica a saída do módulo sem expor detalhes do algoritmo ao publisher. O
# payload permanece serializável para a persistência estática do map-explorer.
@dataclass(frozen=True)
class ConsolidatedContextMap:
    """Artifact global derivado de runs sobre a mesma nuvem de origem.

    Argumentos:
        geometry_fingerprint: identidade da geometria compartilhada.
        payload: documento público pronto para serialização JSON.
    """

    geometry_fingerprint: str
    payload: Mapping[str, Any]

    # Copia o payload na fronteira para que o chamador não possa alterar o
    # estado interno depois da consolidação determinística.
    def to_payload(self) -> dict[str, Any]:
        """Retorna uma cópia serializável do artifact consolidado."""
        return deepcopy(dict(self.payload))


# Namespaceia uma identidade local de run. Isso evita colisões porque duas
# execuções podem ter processado o mesmo frame e atribuído o mesmo region_id.
def _namespaced(run_id: str, local_id: str) -> str:
    """Cria uma identidade global rastreável até uma run publicada."""
    return f"{run_id}::{local_id}"


# Converte a URL relativa do artifact de origem em URL pública estável dentro
# da pasta imutável da run. O resultado permite ao inspector abrir previews de
# qualquer contribuição sem duplicar as imagens no mapa consolidado.
def _asset_url(run_id: str, uri: Any) -> Any:
    """Resolve uma preview de run para uma URL pública absoluta."""
    if not isinstance(uri, str) or not uri:
        return uri
    return uri if uri.startswith("/") else f"/runs/{run_id}/{uri.lstrip('/')}"


# Reescreve toda URI relativa dentro de um ``debug_manifest`` para a URL
# pública da run. Existe porque o manifest é uma árvore de profundidade
# variável (por etapa, com listas como ``discovery_tiles``/``region_views``),
# e antes desta função só ``raw_image_uri``/``overlay_image_uri`` eram
# reescritas: os links de debug do mapa consolidado já quebravam hoje,
# silenciosamente, porque ``debug_assets``/``debug_manifest`` nunca passava
# por ``_asset_url``.
def _rewrite_debug_manifest_urls(run_id: str, value: Any) -> Any:
    """Reescreve recursivamente as URIs de um nó de ``debug_manifest``.

    Argumentos:
        run_id: identidade da run de origem, dona das URIs relativas.
        value: nó do manifest — ``dict``, lista ou string de URI relativa.
    Retorna:
        o mesmo formato de ``value``, com toda folha string resolvida por
        ``_asset_url``.
    """
    if isinstance(value, str):
        return _asset_url(run_id, value)
    if isinstance(value, Mapping):
        return {key: _rewrite_debug_manifest_urls(run_id, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_debug_manifest_urls(run_id, item) for item in value]
    return value


# Extrai a única evidência que uma run oferece para um ponto. A fusão interna
# da run já escolheu esse claim; reutilizá-lo impede que runs longas tenham mais
# peso apenas por terem mais keyframes.
def _run_vote(run: PublishedContextRun, point: Mapping[str, Any], regions: Mapping[str, Mapping[str, Any]], observations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Converte o contexto forte de um ponto em um voto de nível de run."""
    context = point.get("context")
    if not isinstance(context, Mapping) or not isinstance(context.get("label"), str):
        return None
    region_id = context.get("region_id")
    observation_id = context.get("observation_id")
    if not isinstance(region_id, str) or not isinstance(observation_id, str):
        return None
    region = regions.get(region_id, {})
    observation = observations.get(observation_id, {})
    confidence = context.get("confidence")
    if confidence is None and isinstance(region.get("confidence"), Mapping):
        confidence = region["confidence"].get("value")
    return {
        "source_run_id": run.run_id,
        "source_observation_id": observation_id,
        "source_region_id": region_id,
        "observation_id": _namespaced(run.run_id, observation_id),
        "region_id": _namespaced(run.run_id, region_id),
        "raw_label": context["label"],
        "confidence": confidence,
        "calibrated_confidence": region.get("calibrated_confidence"),
        "visual_support": region.get("visual_support"),
        "region_quality": region.get("region_quality"),
        "timestamp_ns": int(observation.get("timestamp_ns", 0)),
        "pixel": context.get("pixel"),
        "semantic_status": context.get("semantic_status"),
        "surface_evidence": context.get("surface_evidence"),
    }


# Converte votos para a API de ranking já validada por semantic-fusion. É usada
# somente quando não há maioria, mantendo uma única regra de qualidade no repo.
def _ranked_vote(vote: Mapping[str, Any]) -> SemanticContribution:
    """Materializa um voto no contract de ranking de semantic-fusion."""
    return SemanticContribution(
        observation_id=str(vote["observation_id"]),
        timestamp_ns=int(vote["timestamp_ns"]),
        region_id=str(vote["region_id"]),
        label=str(vote["canonical_label"]),
        confidence=vote.get("confidence"),
        calibrated_confidence=vote.get("calibrated_confidence"),
        visual_support=vote.get("visual_support"),
        region_quality=vote.get("region_quality"),
    )


# Monta a geometria do mapa geral. Com um fundo global, ela é o fundo mais os pontos
# com contexto das runs (os recortes densos inteiros multiplicariam o tamanho sem
# acrescentar contexto); sem fundo, é a união dos pontos das runs. Uma identidade
# repetida precisa ter as mesmas coordenadas em todas as fontes.
def _output_geometry(
    material: list[PublishedContextRun],
    points_by_run: Mapping[str, Mapping[str, Mapping[str, Any]]],
    backdrop: Mapping[str, Any] | None,
    fingerprint: str,
) -> list[dict[str, Any]]:
    """Retorna os pontos geométricos do mapa consolidado, sem contexto."""
    records: dict[str, dict[str, Any]] = {}
    if backdrop is not None:
        if backdrop.get("artifact_type") != "geometric_pcd_slice" or geometry_fingerprint(backdrop) != fingerprint:
            raise ValueError("backdrop must be a geometric slice of the same geometry source.")
        records = {geometry_id: _geometry_record(point) for geometry_id, point in _points_by_geometry_id(backdrop).items()}
    for run in material:
        for geometry_id, point in points_by_run[run.run_id].items():
            contextual = isinstance(point.get("context"), Mapping) and point["context"].get("label")
            existing = records.get(geometry_id)
            if existing is not None:
                if [float(v) for v in existing["coordinates_m"]] != [float(v) for v in point["coordinates_m"]]:
                    raise ValueError(f"point {geometry_id} has different coordinates across sources.")
            elif backdrop is None or contextual:
                records[geometry_id] = _geometry_record(point)
    return list(records.values())


# Consolida runs que compartilham a mesma nuvem de origem. A função é a fronteira
# pública do módulo usada pelo publisher depois que cada run se tornou imutável.
def consolidate_context_runs(
    runs: Iterable[PublishedContextRun], backdrop: Mapping[str, Any] | None = None
) -> ConsolidatedContextMap:
    """Funde contexto semântico de runs publicadas sobre a mesma nuvem de origem.

    Argumentos:
        runs: runs publicadas com artifacts contextuais completos.
        backdrop: slice geométrico global da mesma origem; quando presente, é a
            geometria do mapa geral, e das runs entram só os pontos com contexto.
    Retorna:
        artifact global com evidências namespaceadas e decisão por ponto.
    Levanta:
        ValueError: se não houver runs, se a origem divergir ou se uma mesma
            identidade de ponto tiver coordenadas diferentes.
    """
    material = sorted(runs, key=lambda run: run.run_id)
    if not material:
        raise ValueError("consolidation requires at least one published context run.")
    fingerprints = {geometry_fingerprint(run.payload) for run in material}
    if len(fingerprints) != 1:
        raise ValueError("context runs must share the same geometry source.")
    fingerprint = fingerprints.pop()
    first = material[0].payload
    map_id, map_frame = material[0].map_id, material[0].map_frame
    if any(run.map_id != map_id or run.map_frame != map_frame for run in material):
        raise ValueError("context runs must share map_id and map_frame.")

    all_labels = [
        str(point.get("context", {}).get("label"))
        for run in material for point in run.payload.get("points", [])
        if isinstance(point, Mapping) and isinstance(point.get("context"), Mapping) and point["context"].get("label")
    ]
    families = _label_families(all_labels)
    observations_out: list[dict[str, Any]] = []
    regions_out: list[dict[str, Any]] = []
    observations_by_run: dict[str, dict[str, Mapping[str, Any]]] = {}
    regions_by_run: dict[str, dict[str, Mapping[str, Any]]] = {}
    points_by_run: dict[str, dict[str, Mapping[str, Any]]] = {}
    for run in material:
        observations = {str(item.get("observation_id")): item for item in run.payload.get("observations", []) if isinstance(item, Mapping) and item.get("observation_id")}
        regions = {str(item.get("region_id")): item for item in run.payload.get("regions", []) if isinstance(item, Mapping) and item.get("region_id")}
        observations_by_run[run.run_id] = observations
        regions_by_run[run.run_id] = regions
        points_by_run[run.run_id] = _points_by_geometry_id(run.payload)
        for local_id, observation in observations.items():
            record = deepcopy(dict(observation))
            record.update({"observation_id": _namespaced(run.run_id, local_id), "source_run_id": run.run_id, "source_observation_id": local_id})
            record["raw_image_uri"] = _asset_url(run.run_id, record.get("raw_image_uri"))
            record["overlay_image_uri"] = _asset_url(run.run_id, record.get("overlay_image_uri"))
            if "debug_manifest" in record:
                record["debug_manifest"] = _rewrite_debug_manifest_urls(run.run_id, record["debug_manifest"])
            observations_out.append(record)
        for local_id, region in regions.items():
            record = deepcopy(dict(region))
            record.update({"region_id": _namespaced(run.run_id, local_id), "source_run_id": run.run_id, "source_region_id": local_id})
            regions_out.append(record)

    output_points = _output_geometry(material, points_by_run, backdrop, fingerprint)
    labelled: list[LabelledPoint] = []
    for point in output_points:
        geometry_id = str(point["geometry_id"])
        votes = [
            vote for run in material
            if (source := points_by_run[run.run_id].get(geometry_id)) is not None
            if (vote := _run_vote(run, source, regions_by_run[run.run_id], observations_by_run[run.run_id])) is not None
        ]
        for vote in votes:
            vote["canonical_label"] = families[_normalise_label(str(vote["raw_label"]))]
        if not votes:
            continue
        counts = Counter(str(vote["canonical_label"]) for vote in votes)
        majority = next((label for label, count in counts.items() if count > len(votes) / 2), None)
        if majority is not None:
            candidates = [vote for vote in votes if vote["canonical_label"] == majority]
            selected = next(vote for vote in candidates if _ranked_vote(vote) == fuse_point_contributions(map(_ranked_vote, candidates)).contributions[0])
            resolution = "majority"
        else:
            fused = fuse_point_contributions(map(_ranked_vote, votes))
            selected = next(vote for vote in votes if vote["observation_id"] == fused.observation_id and vote["region_id"] == fused.region_id)
            majority = str(selected["canonical_label"])
            resolution = "single_run" if len(votes) == 1 else "quality_tiebreak"
        agreement = counts[majority] / len(votes)
        context = {
            "status": "associated",
            "observation_id": selected["observation_id"],
            "region_id": selected["region_id"],
            "label": majority,
            "confidence": selected.get("calibrated_confidence") if selected.get("calibrated_confidence") is not None else selected.get("confidence"),
            "pixel": selected.get("pixel"),
            "semantic_status": selected.get("semantic_status"),
            "surface_evidence": selected.get("surface_evidence"),
            "observation_count": len(votes),
            "agreement": agreement,
            "contributions": votes,
            "consolidation": {"resolution": resolution, "observed_run_count": len(votes), "winning_vote_count": counts[majority]},
        }
        if resolution == "quality_tiebreak":
            context["support_state"] = "weak"
        point["context"] = context
        labelled.append(LabelledPoint(geometry_id, tuple(float(value) for value in point["coordinates_m"]), majority))
    spatial_support = measure_spatial_support(labelled)
    for point in output_points:
        context = point.get("context")
        if isinstance(context, dict) and context.get("label") and spatial_support.get(str(point["geometry_id"])) is not None:
            context["spatial_support"] = spatial_support[str(point["geometry_id"])]

    source_runs = [
        {"run_id": run.run_id, "label": run.label, "artifact_url": run.artifact_url, "artifact_sha256": run.artifact_sha256}
        for run in material
    ]
    # Uma run individual carrega uma única string em ``debug_manifest.composition``;
    # o consolidado funde várias runs, então a mesma chave vira uma entrada por
    # run de origem que tiver composição, e não uma string só.
    composition_entries = [
        {"run_id": run.run_id, "uri": _asset_url(run.run_id, composition_uri)}
        for run in material
        if isinstance(run.payload.get("debug_manifest"), Mapping)
        and isinstance((composition_uri := run.payload["debug_manifest"].get("composition")), str)
    ]
    # Mesma lógica de ``composition``: cada run declara os próprios backends
    # (não é uma URI, então não passa por ``_asset_url``), e o consolidado
    # precisa preservar de qual run cada configuração veio — uma comparação
    # entre backends só faz sentido sabendo qual run usou qual.
    backends_entries = [
        {"run_id": run.run_id, "backends": run.payload["debug_manifest"]["pipeline_backends"]}
        for run in material
        if isinstance(run.payload.get("debug_manifest"), Mapping)
        and isinstance(run.payload["debug_manifest"].get("pipeline_backends"), Mapping)
    ]
    payload = {
        "schema_version": CONSOLIDATED_SCHEMA_VERSION,
        "artifact_type": CONSOLIDATED_ARTIFACT_TYPE,
        "map_id": str(map_id),
        "map_frame": str(map_frame),
        "source": deepcopy((backdrop or first).get("source")),
        "points": output_points,
        "observations": observations_out,
        "regions": regions_out,
        "source_runs": source_runs,
        "consolidation": {
            "geometry_fingerprint": fingerprint,
            "source_run_count": len(material),
            "backdrop_point_count": None if backdrop is None else len(backdrop["points"]),
            "algorithm": "one_vote_per_run:textual_family:strict_majority_or_quality_tiebreak",
        },
        "context_summary": {
            "contextual_point_count": sum(1 for point in output_points if (point.get("context") or {}).get("label")),
            "source_run_count": len(material),
        },
    }
    if composition_entries or backends_entries:
        debug_manifest: dict[str, Any] = {}
        if composition_entries:
            debug_manifest["composition"] = composition_entries
        if backends_entries:
            debug_manifest["pipeline_backends"] = backends_entries
        payload["debug_manifest"] = debug_manifest
    return ConsolidatedContextMap(fingerprint, payload)
