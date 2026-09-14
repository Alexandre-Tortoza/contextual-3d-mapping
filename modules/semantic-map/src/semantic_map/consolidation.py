"""Consolidação determinística de evidência contextual entre runs publicadas."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from semantic_fusion import SemanticContribution, fuse_point_contributions, measure_spatial_support, LabelledPoint

CONTEXT_ARTIFACT_TYPE = "contextual_rgb_lidar_slice"
CONSOLIDATED_ARTIFACT_TYPE = "consolidated_contextual_map"


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


# Materializa uma identidade de geometria independente da ordem do JSON. Ela
# protege a fusão ponto a ponto contra runs que compartilham nome de mapa mas
# não representam exatamente a mesma nuvem persistente.
def geometry_fingerprint(payload: Mapping[str, Any]) -> str:
    """Calcula o SHA-256 canônico da geometria de um artifact contextual.

    Argumentos:
        payload: artifact contextual com ``map_id``, ``map_frame`` e pontos.
    Retorna:
        digest hexadecimal que identifica a geometria completa.
    Levanta:
        ValueError: se a geometria não possuir identidades, coordenadas ou
            valores finitos válidos.
    """
    points = payload.get("points")
    if not isinstance(payload.get("map_id"), str) or not isinstance(payload.get("map_frame"), str):
        raise ValueError("context run must declare map_id and map_frame.")
    if not isinstance(points, list):
        raise ValueError("context run must declare points.")
    records = []
    seen: set[str] = set()
    for point in points:
        if not isinstance(point, Mapping) or not isinstance(point.get("geometry_id"), str):
            raise ValueError("every point must declare geometry_id.")
        geometry_id = point["geometry_id"]
        coordinates = point.get("coordinates_m")
        if geometry_id in seen or not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3:
            raise ValueError("geometry ids must be unique and coordinates_m must be three-dimensional.")
        try:
            numbers = tuple(float(value) for value in coordinates)
        except (TypeError, ValueError) as error:
            raise ValueError("coordinates_m must be numeric.") from error
        if not all(number == number and abs(number) != float("inf") for number in numbers):
            raise ValueError("coordinates_m must be finite.")
        seen.add(geometry_id)
        records.append((geometry_id, numbers))
    canonical = json.dumps(
        {"map_id": payload["map_id"], "map_frame": payload["map_frame"], "points": sorted(records)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


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

    # Rejeita uma identidade incompleta antes que ela seja namespace de
    # observações e regiões no resultado global.
    def __post_init__(self) -> None:
        """Valida a identidade pública da run publicada."""
        if not self.run_id.strip() or not self.label.strip() or not self.artifact_url.startswith("/"):
            raise ValueError("published context runs require id, label and absolute artifact URL.")
        if self.payload.get("artifact_type") != CONTEXT_ARTIFACT_TYPE:
            raise ValueError("published run must contain a contextual RGB-LiDAR artifact.")


# Publica a saída do módulo sem expor detalhes do algoritmo ao publisher. O
# payload permanece serializável para a persistência estática do map-explorer.
@dataclass(frozen=True)
class ConsolidatedContextMap:
    """Artifact global derivado de um grupo de runs geometricamente idênticas.

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


# Consolida runs que compartilham exatamente a geometria. A função é a fronteira
# pública do módulo usada pelo publisher depois que cada run se tornou imutável.
def consolidate_context_runs(runs: Iterable[PublishedContextRun]) -> ConsolidatedContextMap:
    """Funde contexto semântico de runs publicadas sobre a mesma geometria.

    Argumentos:
        runs: runs publicadas com artifacts contextuais completos.
    Retorna:
        artifact global com evidências namespaceadas e decisão por ponto.
    Levanta:
        ValueError: se não houver runs ou se a geometria não for idêntica.
    """
    material = sorted(runs, key=lambda run: run.run_id)
    if not material:
        raise ValueError("consolidation requires at least one published context run.")
    fingerprints = {geometry_fingerprint(run.payload) for run in material}
    if len(fingerprints) != 1:
        raise ValueError("context runs must share exactly the same geometry.")
    fingerprint = fingerprints.pop()
    first = material[0].payload
    map_id, map_frame = first["map_id"], first["map_frame"]
    if any(run.payload.get("map_id") != map_id or run.payload.get("map_frame") != map_frame for run in material):
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
        points_by_run[run.run_id] = {str(item.get("geometry_id")): item for item in run.payload.get("points", []) if isinstance(item, Mapping)}
        for local_id, observation in observations.items():
            record = deepcopy(dict(observation))
            record.update({"observation_id": _namespaced(run.run_id, local_id), "source_run_id": run.run_id, "source_observation_id": local_id})
            record["raw_image_uri"] = _asset_url(run.run_id, record.get("raw_image_uri"))
            record["overlay_image_uri"] = _asset_url(run.run_id, record.get("overlay_image_uri"))
            observations_out.append(record)
        for local_id, region in regions.items():
            record = deepcopy(dict(region))
            record.update({"region_id": _namespaced(run.run_id, local_id), "source_run_id": run.run_id, "source_region_id": local_id})
            regions_out.append(record)

    output_points = deepcopy(list(first["points"]))
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
    payload = {
        "schema_version": 1,
        "artifact_type": CONSOLIDATED_ARTIFACT_TYPE,
        "map_id": map_id,
        "map_frame": map_frame,
        "source": deepcopy(first.get("source")),
        "points": output_points,
        "observations": observations_out,
        "regions": regions_out,
        "source_runs": source_runs,
        "consolidation": {
            "geometry_fingerprint": fingerprint,
            "source_run_count": len(material),
            "algorithm": "one_vote_per_run:textual_family:strict_majority_or_quality_tiebreak",
        },
        "context_summary": {
            "contextual_point_count": sum(1 for point in output_points if (point.get("context") or {}).get("label")),
            "source_run_count": len(material),
        },
    }
    return ConsolidatedContextMap(fingerprint, payload)
