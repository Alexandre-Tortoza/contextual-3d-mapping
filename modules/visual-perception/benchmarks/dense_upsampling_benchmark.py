"""Benchmark dos caminhos de evidência densa para regiões pequenas e distantes.

Issue: #192 (sobre o contract da #191).

Compara três caminhos de leitura do mesmo feature map denso — a grade de
patches (baseline), a amostragem pixel-aligned por vizinho mais próximo e a
bilinear — sobre as mesmas imagens e exatamente as mesmas regiões.

O feature map é extraído **uma vez por frame** e reaproveitado pelos três
caminhos: eles diferem em como o mapa é *lido*, não em como é produzido.
Isso é o que garante que a diferença medida seja do caminho de resolução, e
não de duas execuções distintas do backbone.

## O que é medido, e por que sem anotação

Este repositório ainda não tem o conjunto de referência anotado revisado
(#197), então nenhuma métrica de acerto semântico é possível aqui. As
quatro medidas abaixo não dependem de anotação e respondem diretamente à
pergunta da issue — se a alta resolução melhora a evidência de regiões
pequenas:

- **representabilidade**: fração das regiões que o caminho consegue
  representar. A grade de patches rejeita qualquer região menor que uma
  célula, o que é uma falha dura e contável;
- **cobertura de suporte**: fração dos pixels da máscara com suporte de
  feature válido;
- **consistência intra-região**: quanto o vetor agregado se parece com as
  features dos próprios pixels da região. Um vetor que representa mal seus
  pixels é evidência ruim, independentemente do label;
- **separação inter-região**: quanto vetores de regiões *vizinhas* diferem
  entre si. É a medida que mais importa para regiões pequenas: se duas
  regiões adjacentes recebem o mesmo vetor, a evidência não as distingue.

Custo (latência, pico de VRAM, bytes de materialização) é medido junto,
porque um ganho de qualidade só é decidível sabendo o que custou.

Uso (a partir de ``modules/visual-perception``, com o extra ``ml``
instalado e o conjunto de referência preparado):

    python benchmarks/dense_upsampling_benchmark.py --limit 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _MODULE_ROOT.parents[1]
for _relative in ("src", "../../contracts", "../../datasets"):
    sys.path.insert(0, str((_MODULE_ROOT / _relative).resolve()))

import numpy as np  # noqa: E402

from visual_perception.application.dense_evidence import (  # noqa: E402
    DEFAULT_MAX_UPSAMPLED_BYTES,
    upsample_feature_map,
    upsampled_bytes,
)
from visual_perception.application.pooling import (  # noqa: E402
    BASELINE,
    HIGH_RESOLUTION,
    PIXEL_BILINEAR,
    pool_region_evidence,
)
from visual_perception.application.region_merge import merge_regions  # noqa: E402
from visual_perception.application.tiling import build_tiles, remap_to_global  # noqa: E402
from visual_perception.config import ModuleConfig  # noqa: E402
from visual_perception.domain.feature_map import (  # noqa: E402
    FeatureMap,
    SamplingRule,
    feature_map_spec_to_dict,
    sample_feature_map,
)
from visual_perception.domain.geometry import Mask  # noqa: E402
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.domain.regions import ObservedRegion  # noqa: E402

#: Caminhos comparados, na ordem em que aparecem no relatório. O primeiro é
#: o baseline reproduzível contra o qual os demais são lidos.
PATHS: tuple[tuple[str, str], ...] = (
    ("patch_grid", BASELINE),
    ("nearest", HIGH_RESOLUTION),
    ("bilinear", PIXEL_BILINEAR),
)

#: Limites de área (em pixels) que separam os estratos de tamanho. O
#: tamanho projetado de uma região é o proxy disponível para distância: uma
#: mesma superfície ocupa menos pixels quanto mais longe está.
SMALL_REGION_MAX_AREA = 1024
MEDIUM_REGION_MAX_AREA = 8192

#: Quantos pixels da máscara são amostrados para medir a consistência
#: intra-região. Amostrar limita o custo em regiões grandes sem enviesar a
#: medida, já que a amostragem é uniforme e com semente fixa.
INTRA_REGION_SAMPLE = 256

RESULTS_DIR = _MODULE_ROOT / "benchmarks" / "results"
REFERENCE_FRAMES = _REPOSITORY_ROOT / "datasets" / "raw" / "corridor-02" / "visual-reference"


# Reúne as medidas de um caminho de resolução em uma única região. Existe
# para que a agregação por estrato use exatamente as mesmas observações que
# a agregação global.
@dataclass(frozen=True)
class RegionMeasurement:
    """As medidas de um caminho de resolução sobre uma região."""

    frame: str
    region_id: str
    area: int
    stratum: str
    representable: bool
    support_ratio: float | None = None
    intra_region_consistency: float | None = None
    baseline_agreement: float | None = None
    reason: str | None = None


# Reúne o resultado agregado de um caminho de resolução. Existe como a
# linha do relatório: cada campo aqui é uma coluna comparável entre
# caminhos.
@dataclass(frozen=True)
class PathResult:
    """O resultado agregado de um caminho de resolução."""

    name: str
    pooling_method: str
    regions: int
    representable_regions: int
    pool_latency_s: float
    peak_vram_bytes: int | None
    materialized_bytes: int | None
    materialize_latency_s: float | None
    materialize_refused: str | None
    strata: dict[str, dict[str, float | int]] = field(default_factory=dict)
    inter_region_separation: float | None = None
    measurements: tuple[RegionMeasurement, ...] = ()
    vectors: dict[str, tuple[float, ...]] = field(default_factory=dict)

    # Enumera as regiões que este caminho conseguiu representar. Existe para
    # que a comparação entre caminhos possa ser restrita ao conjunto que
    # *todos* representam, que é a única forma de comparar sobre as mesmas
    # identidades de região (critério de aceitação da #192).
    def representable_keys(self) -> frozenset[str]:
        """Retorna as chaves ``frame/region_id`` que este caminho representou."""
        return frozenset(
            f"{item.frame}/{item.region_id}" for item in self.measurements if item.representable
        )


# Classifica a região em um estrato de tamanho projetado. Existe para que o
# relatório separe região pequena de região comum sem que o chamador
# reimplemente os limiares.
def stratum_of(area: int) -> str:
    """Retorna o estrato de tamanho projetado de uma região."""
    if area <= SMALL_REGION_MAX_AREA:
        return "small"
    if area <= MEDIUM_REGION_MAX_AREA:
        return "medium"
    return "large"


# Mede quanto o vetor agregado representa as features dos próprios pixels da
# região. Existe porque um vetor que discorda dos seus pixels é evidência
# ruim, e isso é verificável sem nenhuma anotação.
def intra_region_consistency(
    mask: Mask, feature_map: FeatureMap, pooled: tuple[float, ...], rule: SamplingRule, *, seed: int = 0
) -> float | None:
    """Retorna a similaridade média entre o vetor agregado e as features dos seus pixels.

    Argumentos:
        mask: a máscara da região.
        feature_map: o mapa denso amostrado.
        pooled: o vetor agregado da região, já normalizado.
        rule: a regra de amostragem do caminho avaliado.
        seed: semente da amostragem de pixels, fixada para reprodutibilidade.
    Retorna:
        a similaridade média em ``[-1, 1]``, ou ``None`` sem pixel suportado.
    """
    ys, xs = np.where(mask.data)
    if ys.size == 0:
        return None
    rng = np.random.default_rng(seed)
    if ys.size > INTRA_REGION_SAMPLE:
        chosen = rng.choice(ys.size, size=INTRA_REGION_SAMPLE, replace=False)
        ys, xs = ys[chosen], xs[chosen]
    values, valid = sample_feature_map(
        feature_map, xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5, interpolation=rule
    )
    if not valid.any():
        return None
    supported = values[valid]
    norms = np.linalg.norm(supported, axis=1)
    usable = norms > 0
    if not usable.any():
        return None
    unit = supported[usable] / norms[usable][:, None]
    return float(np.mean(unit @ np.asarray(pooled, dtype=np.float64)))


# Mede quanto os vetores de regiões vizinhas diferem entre si. Existe porque
# é a propriedade que decide o caso da issue: se duas regiões pequenas e
# adjacentes recebem vetores quase idênticos, a evidência densa não as
# distingue, por mais fina que seja a grade.
def inter_region_separation(
    vectors: dict[str, tuple[float, ...]], regions: tuple[ObservedRegion, ...]
) -> float | None:
    """Retorna a dissimilaridade média entre vetores de regiões espacialmente vizinhas."""
    distances: list[float] = []
    for index, subject in enumerate(regions):
        for target in regions[index + 1 :]:
            if subject.region_id not in vectors or target.region_id not in vectors:
                continue
            if not _are_neighbours(subject, target):
                continue
            first = np.asarray(vectors[subject.region_id], dtype=np.float64)
            second = np.asarray(vectors[target.region_id], dtype=np.float64)
            distances.append(1.0 - float(first @ second))
    return float(np.mean(distances)) if distances else None


# Decide se duas regiões são vizinhas espaciais, por proximidade de box.
# Helper de inter_region_separation: comparar regiões distantes mediria
# diversidade de cena, e não poder de discriminação local.
def _are_neighbours(subject: ObservedRegion, target: ObservedRegion, *, margin: float = 8.0) -> bool:
    """Indica se os boxes de duas regiões estão a menos de ``margin`` pixels."""
    a, b = subject.box, target.box
    gap_x = max(a.x_min - b.x_max, b.x_min - a.x_max, 0.0)
    gap_y = max(a.y_min - b.y_max, b.y_min - a.y_max, 0.0)
    return gap_x <= margin and gap_y <= margin


# Mede um caminho de resolução sobre as regiões de um conjunto de frames já
# processados. Ponto de entrada reutilizável do benchmark, separado do I/O
# para ser testável com dados sintéticos.
def measure_path(
    name: str,
    pooling_method: str,
    frames: tuple[tuple[str, ImagePayload, FeatureMap, tuple[ObservedRegion, ...]], ...],
    *,
    baseline_vectors: dict[str, tuple[float, ...]] | None = None,
    max_bytes: int = DEFAULT_MAX_UPSAMPLED_BYTES,
) -> PathResult:
    """Mede qualidade e custo de um caminho de resolução sobre os frames dados.

    Argumentos:
        name: nome do caminho (``patch_grid``, ``nearest``, ``bilinear``).
        pooling_method: método de pooling correspondente.
        frames: tuplas ``(nome, payload, feature_map, regiões)`` já preparadas.
        baseline_vectors: vetores do baseline, para medir concordância.
        max_bytes: teto de memória para materializar o mapa pixel-aligned.
    Retorna:
        o :class:`PathResult` agregado.
    """
    rule = SamplingRule.BILINEAR if name == "bilinear" else SamplingRule.NEAREST
    measurements: list[RegionMeasurement] = []
    vectors: dict[str, tuple[float, ...]] = {}
    separations: list[float] = []
    _reset_vram()
    started = time.monotonic()

    for frame_name, _payload, feature_map, regions in frames:
        frame_vectors: dict[str, tuple[float, ...]] = {}
        for region in regions:
            area = region.mask.area()
            key = f"{frame_name}/{region.region_id}"
            try:
                pooled = pool_region_evidence(region.mask, feature_map, pooling_method)
            except ValueError as error:
                measurements.append(
                    RegionMeasurement(
                        frame_name, region.region_id, area, stratum_of(area), False, reason=str(error)
                    )
                )
                continue
            frame_vectors[region.region_id] = pooled.vector
            vectors[key] = pooled.vector
            agreement = None
            if baseline_vectors is not None and key in baseline_vectors:
                agreement = float(
                    np.asarray(pooled.vector) @ np.asarray(baseline_vectors[key], dtype=np.float64)
                )
            measurements.append(
                RegionMeasurement(
                    frame=frame_name,
                    region_id=region.region_id,
                    area=area,
                    stratum=stratum_of(area),
                    representable=True,
                    support_ratio=pooled.support_ratio,
                    intra_region_consistency=intra_region_consistency(
                        region.mask, feature_map, pooled.vector, rule
                    ),
                    baseline_agreement=agreement,
                )
            )
        separation = inter_region_separation(frame_vectors, regions)
        if separation is not None:
            separations.append(separation)

    pool_latency = time.monotonic() - started
    materialized, materialize_latency, refused = _materialization_cost(name, frames, max_bytes)

    return PathResult(
        name=name,
        pooling_method=pooling_method,
        regions=len(measurements),
        representable_regions=sum(1 for item in measurements if item.representable),
        pool_latency_s=pool_latency,
        peak_vram_bytes=_peak_vram(),
        materialized_bytes=materialized,
        materialize_latency_s=materialize_latency,
        materialize_refused=refused,
        strata=_strata(measurements),
        inter_region_separation=float(np.mean(separations)) if separations else None,
        measurements=tuple(measurements),
        vectors=vectors,
    )


# Recalcula as medidas de um caminho restritas a um conjunto comum de
# regiões. Existe porque a grade de patches rejeita regiões que os caminhos
# pixel-aligned representam: comparar as médias sobre conjuntos diferentes
# creditaria ao baseline exatamente as regiões em que ele falha.
def restrict_to_common(
    result: PathResult,
    common: frozenset[str],
    frames: tuple[tuple[str, ImagePayload, FeatureMap, tuple[ObservedRegion, ...]], ...],
) -> dict[str, Any]:
    """Reagrega as medidas de ``result`` apenas sobre as regiões de ``common``.

    Argumentos:
        result: o resultado completo do caminho.
        common: chaves ``frame/region_id`` representáveis por todos os caminhos.
        frames: os frames processados, usados para a vizinhança espacial.
    Retorna:
        um dict com estratos e separação inter-região sobre o conjunto comum.
    """
    subset = [
        item
        for item in result.measurements
        if f"{item.frame}/{item.region_id}" in common and item.representable
    ]
    separations: list[float] = []
    for frame_name, _payload, _feature_map, regions in frames:
        comparable = tuple(
            region for region in regions if f"{frame_name}/{region.region_id}" in common
        )
        vectors = {
            region.region_id: result.vectors[f"{frame_name}/{region.region_id}"]
            for region in comparable
            if f"{frame_name}/{region.region_id}" in result.vectors
        }
        separation = inter_region_separation(vectors, comparable)
        if separation is not None:
            separations.append(separation)
    return {
        "regions": len(subset),
        "strata": {**_strata(subset), "__all__": _combined(subset)},
        "inter_region_separation": float(np.mean(separations)) if separations else None,
    }


# Agrega as medidas de um conjunto de regiões em uma única linha. Existe
# porque a tabela de topo do relatório e os estratos precisam da mesma
# agregação, e duplicá-la faria as duas divergirem.
def _combined(measurements: list[RegionMeasurement]) -> dict[str, float | int]:
    """Agrega as medidas de todas as regiões informadas em uma linha só."""
    representable = [item for item in measurements if item.representable]
    entry: dict[str, float | int] = {
        "regions": len(measurements),
        "representable": len(representable),
    }
    for field_name in ("support_ratio", "intra_region_consistency", "baseline_agreement"):
        values = [
            getattr(item, field_name) for item in representable if getattr(item, field_name) is not None
        ]
        if values:
            entry[f"mean_{field_name}"] = float(np.mean(values))
    return entry


# Mede o custo de materializar o mapa pixel-aligned completo do primeiro
# frame. Existe porque a issue exige registrar o custo do caminho, e porque
# a recusa por budget de memória é ela própria um resultado.
def _materialization_cost(
    name: str,
    frames: tuple[tuple[str, ImagePayload, FeatureMap, tuple[ObservedRegion, ...]], ...],
    max_bytes: int,
) -> tuple[int | None, float | None, str | None]:
    """Mede tempo e memória de materializar o mapa completo, ou registra a recusa."""
    if name == "patch_grid" or not frames:
        return None, None, None
    _, payload, feature_map, _ = frames[0]
    required = upsampled_bytes(feature_map, payload.width, payload.height)
    if required > max_bytes:
        return required, None, (
            f"materializing {payload.width}x{payload.height}x{feature_map.dimension} needs "
            f"{required / 1e9:.2f} GB, above the {max_bytes / 1e9:.2f} GB budget"
        )
    started = time.monotonic()
    upsample_feature_map(
        feature_map, method=name, width=payload.width, height=payload.height, max_bytes=max_bytes
    )
    return required, time.monotonic() - started, None


# Agrega as medidas por estrato de tamanho. Existe porque a conclusão da
# issue depende justamente do recorte: um ganho médio pode esconder uma
# perda em regiões grandes e um ganho grande em regiões pequenas.
def _strata(measurements: list[RegionMeasurement]) -> dict[str, dict[str, float | int]]:
    """Agrega as medidas por estrato de tamanho projetado."""
    grouped: dict[str, list[RegionMeasurement]] = {}
    for item in measurements:
        grouped.setdefault(item.stratum, []).append(item)
    summary: dict[str, dict[str, float | int]] = {}
    for stratum, items in sorted(grouped.items()):
        representable = [item for item in items if item.representable]
        entry: dict[str, float | int] = {
            "regions": len(items),
            "representable": len(representable),
            "representable_fraction": len(representable) / len(items),
        }
        for field_name in ("support_ratio", "intra_region_consistency", "baseline_agreement"):
            values = [
                getattr(item, field_name) for item in representable if getattr(item, field_name) is not None
            ]
            if values:
                entry[f"mean_{field_name}"] = float(np.mean(values))
        summary[stratum] = entry
    return summary


# Prepara os frames: descobre regiões uma única vez e extrai o feature map
# uma única vez por frame, para que os três caminhos leiam exatamente a
# mesma evidência sobre exatamente as mesmas regiões.
def prepare_frames(
    paths: tuple[Path, ...], config: ModuleConfig, ports: Any
) -> tuple[tuple[str, ImagePayload, FeatureMap, tuple[ObservedRegion, ...]], ...]:
    """Descobre regiões e extrai o feature map de cada frame, uma vez só."""
    from PIL import Image

    prepared = []
    for path in paths:
        pixels = np.array(Image.open(path).convert("RGB"))
        payload = ImagePayload(pixels, width=int(pixels.shape[1]), height=int(pixels.shape[0]))
        proposals = []
        for tile in build_tiles(payload, config.tiling):
            for local in ports.region_discoverer.discover(tile.payload, config.region_discovery):
                proposals.append(
                    remap_to_global(local, tile, image_width=payload.width, image_height=payload.height)
                )
        regions = merge_regions(path.stem, tuple(proposals), config.merge)
        feature_map = ports.feature_extractor.extract(payload, config.feature_extraction)
        prepared.append((path.stem, payload, feature_map, regions))
        grid = f"{feature_map.grid_width}x{feature_map.grid_height}"
        print(f"  {path.stem}: {len(regions)} regions, grid {grid}")
    return tuple(prepared)


# Zera o pico de VRAM entre caminhos, para que cada um reporte o próprio.
def _reset_vram() -> None:
    """Zera o pico de VRAM acumulado, quando há CUDA."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# Lê o pico de VRAM, ou None sem GPU.
def _peak_vram() -> int | None:
    """Retorna o pico de VRAM alocado, ou ``None`` sem CUDA."""
    try:
        import torch
    except ImportError:
        return None
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


# Retorna a revisão de código, sem a qual o resultado não é reproduzível.
def _git_revision() -> str:
    """Retorna o hash curto do commit atual, ou ``unknown``."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_MODULE_ROOT)
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# Descreve a GPU da execução, para que o custo medido seja interpretável.
def _describe_hardware() -> str:
    """Descreve a GPU usada, ou a ausência dela."""
    try:
        import torch
    except ImportError:
        return "unknown (torch unavailable)"
    if not torch.cuda.is_available():
        return "CPU (no CUDA device available)"
    properties = torch.cuda.get_device_properties(0)
    return f"{properties.name} ({properties.total_memory / 1e9:.1f} GB)"


# Renderiza o resumo em Markdown do relatório. Existe para que o resultado
# seja legível sem abrir o JSON, mantendo exatamente os mesmos números.
#
# A tabela de qualidade usa o *conjunto comum* — as regiões que os três
# caminhos conseguem representar. Comparar as médias sobre os conjuntos
# completos creditaria ao baseline justamente as regiões pequenas em que ele
# falha, e a comparação deixaria de ser sobre as mesmas identidades.
def render_summary(document: dict[str, Any]) -> str:
    """Renderiza o relatório do benchmark em Markdown."""
    lines = [
        "# Benchmark de evidência densa em alta resolução (#192)",
        "",
        f"- revisão de código: `{document['code_revision']}`",
        f"- hardware: {document['hardware']}",
        f"- backbone: `{document['backbone']}` / checkpoint `{document['checkpoint']}`",
        f"- fingerprint de configuração: `{document['config_fingerprint']}`",
        f"- frames: {document['frames']} — regiões descobertas: {document['regions_per_path']}",
        f"- resolução de entrada: {document['input_resolution']}",
        f"- grade de features: {document['feature_grid']}",
        f"- regiões representáveis por todos os caminhos: {document['comparable_regions']}",
        "",
        "## Representabilidade (conjunto completo)",
        "",
        "Quantas regiões cada caminho consegue representar. A grade de patches",
        "rejeita qualquer região sem centro de célula dentro da máscara.",
        "",
        "| caminho | representáveis | pequenas | médias | grandes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in document["paths"]:
        strata = result["strata"]
        cells = [
            f"{strata.get(name, {}).get('representable', 0)}/{strata.get(name, {}).get('regions', 0)}"
            for name in ("small", "medium", "large")
        ]
        lines.append(
            f"| `{result['name']}` | {result['representable_regions']}/{result['regions']} "
            f"| {cells[0]} | {cells[1]} | {cells[2]} |"
        )

    lines += [
        "",
        "## Qualidade de evidência (conjunto comum)",
        "",
        "| caminho | consistência intra | separação inter | suporte médio | concordância baseline |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in document["paths"]:
        overall = result["comparable"]["strata"].get("__all__", {})
        lines.append(
            f"| `{result['name']}` "
            f"| {_format(overall.get('mean_intra_region_consistency'))} "
            f"| {_format(result['comparable']['inter_region_separation'])} "
            f"| {_format(overall.get('mean_support_ratio'))} "
            f"| {_format(overall.get('mean_baseline_agreement'))} |"
        )

    lines += [
        "",
        "## Por estrato de tamanho projetado (conjunto comum)",
        "",
        "| caminho | estrato | regiões | consistência intra | suporte médio | concordância baseline |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in document["paths"]:
        for stratum, entry in result["comparable"]["strata"].items():
            if stratum == "__all__":
                continue
            lines.append(
                f"| `{result['name']}` | {stratum} | {entry['regions']} "
                f"| {_format(entry.get('mean_intra_region_consistency'))} "
                f"| {_format(entry.get('mean_support_ratio'))} "
                f"| {_format(entry.get('mean_baseline_agreement'))} |"
            )

    lines += ["", "## Custo", "", "| caminho | pooling (s) | pico de VRAM (GB) |", "| --- | --- | --- |"]
    for result in document["paths"]:
        vram = result["peak_vram_bytes"]
        lines.append(
            f"| `{result['name']}` | {result['pool_latency_s']:.3f} "
            f"| {'não medido' if vram is None else f'{vram / 1e9:.2f}'} |"
        )

    lines += ["", "### Materialização do mapa pixel-aligned completo", ""]
    for result in document["paths"]:
        if result["materialized_bytes"] is None:
            continue
        if result["materialize_refused"]:
            lines.append(f"- `{result['name']}`: recusado — {result['materialize_refused']}")
        else:
            lines.append(
                f"- `{result['name']}`: {result['materialized_bytes'] / 1e9:.2f} GB em "
                f"{result['materialize_latency_s']:.3f} s"
            )
    return "\n".join(lines) + "\n"


# Formata um número opcional para o relatório, distinguindo ausência.
def _format(value: float | None) -> str:
    """Formata um valor opcional, marcando ausência como não medida."""
    return "não medida" if value is None else f"{value:.4f}"


# Interface de linha de comando do benchmark.
def main() -> None:
    """Executa o benchmark de evidência densa e grava relatório JSON e Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=REFERENCE_FRAMES)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--fake-backends", action="store_true")
    arguments = parser.parse_args()

    from visual_perception.application.execution_profile import research_quality_config
    from visual_perception.application.lifecycle import ModelLifecycleManager
    from visual_perception.infrastructure.adapters.factory import create_perception_ports

    real = not arguments.fake_backends
    config = research_quality_config(multi_scale_justified=False, real_backends=real)
    if real:
        ports = create_perception_ports(config, ModelLifecycleManager())
    else:
        from visual_perception.infrastructure.fakes.fake_feature_extractor import (
            FakeDenseFeatureExtractor,
        )
        from visual_perception.infrastructure.fakes.fake_region_discoverer import FakeRegionDiscoverer

        class _Ports:
            region_discoverer = FakeRegionDiscoverer()
            feature_extractor = FakeDenseFeatureExtractor()

        ports = _Ports()

    frame_paths = tuple(sorted(arguments.frames_dir.glob("*.png"))[: arguments.limit])
    if not frame_paths:
        raise SystemExit(f"No frames found in {arguments.frames_dir}.")

    print(f"Preparing {len(frame_paths)} frames...")
    frames = prepare_frames(frame_paths, config, ports)

    results: list[PathResult] = []
    baseline_vectors: dict[str, tuple[float, ...]] | None = None
    for name, method in PATHS:
        print(f"Measuring {name}...")
        result = measure_path(name, method, frames, baseline_vectors=baseline_vectors)
        if name == "patch_grid":
            baseline_vectors = dict(result.vectors)
        results.append(result)

    common = frozenset.intersection(*(result.representable_keys() for result in results))
    print(f"Regions representable by every path: {len(common)}")
    document = _document(config, frames, results, common)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"benchmark-192-dense-upsampling-{run_id}.json"
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = json_path.with_suffix(".md")
    summary_path.write_text(render_summary(document), encoding="utf-8")
    print(f"\nWrote {json_path}\nWrote {summary_path}")
    print(render_summary(document))


# Monta o documento final do relatório, incluindo toda a proveniência
# exigida pela issue.
def _document(
    config: ModuleConfig,
    frames: tuple[tuple[str, ImagePayload, FeatureMap, tuple[ObservedRegion, ...]], ...],
    results: list[PathResult],
    common: frozenset[str],
) -> dict[str, Any]:
    """Monta o documento JSON do relatório, com proveniência completa."""
    _, payload, feature_map, _ = frames[0]
    serialized = []
    for result in results:
        entry = asdict(result)
        entry.pop("measurements")
        entry.pop("vectors")
        entry["comparable"] = restrict_to_common(result, common, frames)
        entry["strata"] = {
            **_strata(list(result.measurements)),
            "__all__": _combined(list(result.measurements)),
        }
        serialized.append(entry)
    return {
        "issue": 192,
        "code_revision": _git_revision(),
        "hardware": _describe_hardware(),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backbone": config.feature_extraction.backend,
        "checkpoint": config.feature_extraction.checkpoint,
        "config_fingerprint": config.fingerprint(),
        "frames": len(frames),
        "regions_per_path": results[0].regions if results else 0,
        "input_resolution": f"{payload.width}x{payload.height}",
        "feature_grid": f"{feature_map.grid_width}x{feature_map.grid_height}",
        "feature_spec": feature_map_spec_to_dict(feature_map),
        "small_region_max_area": SMALL_REGION_MAX_AREA,
        "medium_region_max_area": MEDIUM_REGION_MAX_AREA,
        "comparable_regions": len(common),
        "paths": serialized,
    }


if __name__ == "__main__":  # pragma: no cover - ponto de entrada de CLI
    main()
