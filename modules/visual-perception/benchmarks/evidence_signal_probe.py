"""Mede se os canais de evidência já pagos carregam sinal independente do VLM.

Esta sonda existe porque a revisão de contexto visual precisava decidir, **antes**
de escrever código, duas coisas que só a medição responde:

1. o alinhamento language-aligned (CLIP) consegue arbitrar entre a hipótese
   primária e as alternativas que o próprio reasoner registrou?
2. a coerência de features densas (DINOv2) identifica que duas regiões vizinhas
   descrevem a mesma superfície?

Ela roda **sobre artifacts de um run já versionado**: reusa as máscaras e os
labels daquele run e reconstrói as views com o mesmo ``build_region_views`` do
módulo, de modo que os pixels medidos aqui são exatamente os pixels que o
pipeline mostrou aos modelos. Nenhum modelo é reexecutado para produzir labels,
e nenhuma métrica de acurácia é calculada: sem ground truth humano revisado
(#210), o que se mede é discriminação e concordância, nomeadas como tal.

Uso, a partir de ``modules/visual-perception``:

    python benchmarks/evidence_signal_probe.py \
      --run benchmarks/results/samples/20260908T131207Z

O relatório vai para stdout e, com ``--output``, para um JSON versionável.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MODULE_ROOT = Path(__file__).resolve().parents[1]
for relative in ("src", "../../contracts"):
    sys.path.insert(0, str((_MODULE_ROOT / relative).resolve()))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from visual_perception.application.pooling import HIGH_RESOLUTION, pool_region_evidence  # noqa: E402
from visual_perception.application.region_views import build_region_views  # noqa: E402
from visual_perception.config import (  # noqa: E402
    FeatureExtractionConfig,
    LanguageEmbeddingConfig,
    ModuleConfig,
    MultiContextConfig,
)
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.domain.region_evidence import EvidenceSlot  # noqa: E402
from visual_perception.domain.regions import (  # noqa: E402
    ObservedRegion,
    alternative_label_claims,
    primary_label_claim,
)
from visual_perception.domain.semantics import normalize_claim_value  # noqa: E402
from visual_perception.infrastructure.adapters import (  # noqa: E402
    RealDenseFeatureExtractionAdapter,
    RealLanguageAlignedEncoderAdapter,
)
from visual_perception.infrastructure.serialization import deserialize_observation  # noqa: E402

#: Abaixo desta diferença de cosseno, o alinhamento não distingue duas hipóteses.
#: Não é um limiar de decisão: é o piso abaixo do qual a sonda se recusa a
#: reportar um vencedor, porque a margem mediana medida vive nessa ordem de
#: grandeza (ver docs/visual-context-sota-review.md, §3.1).
INDISTINGUISHABLE_MARGIN = 0.01

#: Quão perto dois bounding boxes precisam estar para as regiões serem tratadas
#: como em contato. Espelha a margem de adjacência de ``relation_generation``.
ADJACENCY_MARGIN_PX = 5.0

#: Slots cuja view textual é comparada contra as hipóteses. ``foreground_dense``
#: fica de fora: ele alimenta o extractor denso, não o espaço de linguagem.
PROBED_SLOTS = (
    EvidenceSlot.MASKED_SUBJECT,
    EvidenceSlot.TIGHT_CROP,
    EvidenceSlot.CONTEXTUAL_CROP,
)


# Acumula, por slot de evidência, quantas vezes o alinhamento concordou com a
# escolha do reasoner, discordou, ou não conseguiu distinguir. Existe para que o
# relatório reporte os três desfechos separadamente: colapsar "indistinguível"
# em "discorda" inventaria um veredito que a margem medida não sustenta.
@dataclass
class ArbiterTally:
    """Placar do alinhamento como árbitro entre primary e alternatives."""

    agrees: int = 0
    indistinguishable: int = 0
    disagrees: int = 0
    margins: list[float] = field(default_factory=list)

    # Registra um par (primária, melhor alternativa) já pontuado. Chamada uma
    # vez por região que tenha ao menos uma alternativa.
    def record(self, primary_score: float, best_alternative_score: float) -> None:
        """Classifica uma comparação em concorda, indistinguível ou discorda."""
        delta = primary_score - best_alternative_score
        self.margins.append(delta)
        if abs(delta) < INDISTINGUISHABLE_MARGIN:
            self.indistinguishable += 1
        elif delta > 0.0:
            self.agrees += 1
        else:
            self.disagrees += 1

    # Converte o placar no dict serializado pelo relatório.
    def to_dict(self) -> dict[str, Any]:
        """Retorna o placar e a magnitude mediana da margem observada."""
        total = self.agrees + self.indistinguishable + self.disagrees
        absolute = [abs(margin) for margin in self.margins]
        return {
            "total": total,
            "agrees": self.agrees,
            "indistinguishable": self.indistinguishable,
            "disagrees": self.disagrees,
            "median_absolute_margin": None if not absolute else float(np.median(absolute)),
        }


# Acumula os cossenos densos de pares adjacentes, separados por concordarem ou
# não no label primário. Existe para que a sonda responda se a coerência densa
# **sozinha** decide a pergunta — e o resultado medido é que não decide.
@dataclass
class CoherenceTally:
    """Cossenos DINOv2 entre pares de regiões adjacentes do mesmo frame."""

    same_label: list[float] = field(default_factory=list)
    different_label: list[float] = field(default_factory=list)

    # Descreve uma distribuição por percentis, ou ``None`` quando vazia.
    @staticmethod
    def _describe(values: list[float]) -> dict[str, Any]:
        """Retorna contagem e percentis de uma amostra de cossenos."""
        if not values:
            return {"count": 0}
        array = np.asarray(values)
        return {
            "count": int(array.size),
            "p10": float(np.percentile(array, 10)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
        }

    # Varre limiares e reporta o melhor equilíbrio possível entre reter pares de
    # mesmo label e rejeitar pares de label diferente. É a medida que decide se
    # a similaridade densa pode ou não ser o gate da reconciliação.
    def separability(self) -> dict[str, Any]:
        """Retorna a melhor acurácia balanceada alcançável por um limiar único."""
        if not self.same_label or not self.different_label:
            return {"best_balanced_accuracy": None, "threshold": None, "curve": []}
        same = np.asarray(self.same_label)
        different = np.asarray(self.different_label)
        curve = []
        best = (0.0, 0.0)
        for threshold in np.arange(0.30, 0.96, 0.05):
            recall = float((same >= threshold).mean())
            rejection = float((different < threshold).mean())
            balanced = (recall + rejection) / 2.0
            curve.append(
                {
                    "threshold": round(float(threshold), 2),
                    "recall_same_label": recall,
                    "rejection_different_label": rejection,
                    "balanced_accuracy": balanced,
                }
            )
            if balanced > best[0]:
                best = (balanced, float(threshold))
        return {
            "best_balanced_accuracy": best[0],
            "threshold": round(best[1], 2),
            "curve": curve,
        }

    # Converte as duas distribuições e a separabilidade no dict do relatório.
    def to_dict(self) -> dict[str, Any]:
        """Retorna as distribuições e o que um limiar único conseguiria separar."""
        return {
            "same_label": self._describe(self.same_label),
            "different_label": self._describe(self.different_label),
            "separability": self.separability(),
        }


# Reconstrói a configuração de views usada pelo run auditado. Existe para que a
# sonda recorte exatamente os mesmos pixels que o pipeline recortou: uma sonda
# que recortasse diferente mediria outra coisa e atribuiria o resultado a esta.
def probe_config() -> ModuleConfig:
    """Retorna a configuração de views e backends usada pela sonda."""
    return ModuleConfig(
        multi_context=MultiContextConfig(
            foreground_enabled=True,
            masked_subject_enabled=True,
            tight_crop_enabled=True,
            contextual_crop_enabled=True,
            scene_conditioned_enabled=False,
            context_expansion=0.25,
        ),
        feature_extraction=FeatureExtractionConfig(
            backend="dinov2", checkpoint="facebook/dinov2-base", input_resolution=448
        ),
        language_embedding=LanguageEmbeddingConfig(
            backend="clip", checkpoint="openai/clip-vit-large-patch14", dimension=768
        ),
    )


# Responde se duas regiões estão em contato no plano da imagem, pela mesma regra
# que ``relation_generation`` usa para ``near``. Existe aqui, e não importada de
# lá, porque aquela função devolve a proximidade normalizada e esta pergunta é
# binária; duplicar a constante seria pior que duplicar a comparação.
def regions_touch(first: ObservedRegion, second: ObservedRegion) -> bool:
    """Indica se as duas regiões se sobrepõem ou têm caixas encostadas."""
    if first.mask.iou(second.mask) > 0.0:
        return True
    a, b = first.box, second.box
    gap_x = max(a.x_min - b.x_max, b.x_min - a.x_max, 0.0)
    gap_y = max(a.y_min - b.y_max, b.y_min - a.y_max, 0.0)
    return max(gap_x, gap_y) <= ADJACENCY_MARGIN_PX


# Mede a hipótese A num frame: para cada região com alternativa, compara o
# cosseno texto-imagem da primária contra a melhor alternativa, em cada slot.
# Chamada por ``probe_run`` uma vez por frame.
def probe_alignment(
    regions: tuple[ObservedRegion, ...],
    payload: ImagePayload,
    config: ModuleConfig,
    encoder: RealLanguageAlignedEncoderAdapter,
) -> dict[str, ArbiterTally]:
    """Arbitra primary contra alternatives por alinhamento, slot a slot."""
    views = build_region_views(regions, payload, config)
    text_cache: dict[str, np.ndarray] = {}

    # Codifica um texto uma vez por frame. O cache existe porque ``wall`` é a
    # hipótese de dezenas de regiões do mesmo frame e recodificá-la seria custo
    # puro, sem nenhuma diferença no resultado.
    def text_vector(value: str) -> np.ndarray:
        """Retorna o embedding de texto de ``value``, memoizado por frame."""
        if value not in text_cache:
            vector = encoder.encode_text(f"a photo of {value}", config.language_embedding)
            text_cache[value] = np.asarray(vector)
        return text_cache[value]

    tallies = {slot.value: ArbiterTally() for slot in PROBED_SLOTS}
    for region in regions:
        primary = primary_label_claim(region)
        if primary is None:
            continue
        alternatives = [normalize_claim_value(claim.value) for claim in alternative_label_claims(region)]
        if not alternatives:
            continue
        primary_text = text_vector(normalize_claim_value(primary.value))
        alternative_texts = [text_vector(value) for value in alternatives]
        by_slot = {view.slot: view for view in views.get(region.region_id, ())}
        for slot in PROBED_SLOTS:
            view = by_slot.get(slot)
            if view is None:
                continue
            image_vector = np.asarray(encoder.encode_image(view.payload, config.language_embedding))
            tallies[slot.value].record(
                float(image_vector @ primary_text),
                max(float(image_vector @ text) for text in alternative_texts),
            )
    return tallies


# Mede a hipótese B num frame: entre pares de regiões que se tocam, compara o
# cosseno das features densas de quem compartilha o label com o de quem não
# compartilha. Chamada por ``probe_run`` uma vez por frame.
def probe_coherence(
    regions: tuple[ObservedRegion, ...],
    payload: ImagePayload,
    config: ModuleConfig,
    extractor: RealDenseFeatureExtractionAdapter,
) -> tuple[CoherenceTally, str]:
    """Mede a coerência densa entre pares adjacentes, por igualdade de label."""
    feature_map = extractor.extract(payload, config.feature_extraction)
    vectors: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    for region in regions:
        try:
            pooled = pool_region_evidence(region.mask, feature_map, HIGH_RESOLUTION)
        except ValueError:
            continue
        vectors[region.region_id] = np.asarray(pooled.vector)
        claim = primary_label_claim(region)
        labels[region.region_id] = "?" if claim is None else normalize_claim_value(claim.value)

    by_id = {region.region_id: region for region in regions}
    tally = CoherenceTally()
    for first_id, second_id in itertools.combinations(sorted(vectors), 2):
        if not regions_touch(by_id[first_id], by_id[second_id]):
            continue
        similarity = float(vectors[first_id] @ vectors[second_id])
        if labels[first_id] == labels[second_id]:
            tally.same_label.append(similarity)
        else:
            tally.different_label.append(similarity)
    grid = f"{feature_map.grid_width}x{feature_map.grid_height}"
    return tally, grid


# Ponto de entrada da sonda: percorre os frames de um run versionado e agrega os
# dois experimentos. Chamada por ``main``.
def probe_run(run_dir: Path, frame_ids: tuple[str, ...]) -> dict[str, Any]:
    """Executa as duas sondas sobre os frames de um run já persistido."""
    config = probe_config()
    encoder = RealLanguageAlignedEncoderAdapter()
    extractor = RealDenseFeatureExtractionAdapter()

    totals = {slot.value: ArbiterTally() for slot in PROBED_SLOTS}
    total_coherence = CoherenceTally()
    frames: list[dict[str, Any]] = []

    for frame_dir in sorted((run_dir / "frames").iterdir()):
        if frame_ids and frame_dir.name not in frame_ids:
            continue
        observation = deserialize_observation(json.loads((frame_dir / "observation.json").read_text()))
        pixels = np.asarray(Image.open(frame_dir / "raw.png").convert("RGB"))
        payload = ImagePayload(pixels, width=int(pixels.shape[1]), height=int(pixels.shape[0]))

        alignment = probe_alignment(observation.regions, payload, config, encoder)
        coherence, grid = probe_coherence(observation.regions, payload, config, extractor)

        for slot_value, tally in alignment.items():
            totals[slot_value].agrees += tally.agrees
            totals[slot_value].indistinguishable += tally.indistinguishable
            totals[slot_value].disagrees += tally.disagrees
            totals[slot_value].margins.extend(tally.margins)
        total_coherence.same_label.extend(coherence.same_label)
        total_coherence.different_label.extend(coherence.different_label)

        frames.append(
            {
                "frame_id": frame_dir.name,
                "region_count": len(observation.regions),
                "dense_grid": grid,
                "alignment_arbiter": {name: tally.to_dict() for name, tally in alignment.items()},
                "adjacent_pair_coherence": coherence.to_dict(),
            }
        )

    return {
        "run": str(run_dir),
        "indistinguishable_margin": INDISTINGUISHABLE_MARGIN,
        "frames": frames,
        "totals": {
            "alignment_arbiter": {name: tally.to_dict() for name, tally in totals.items()},
            "adjacent_pair_coherence": total_coherence.to_dict(),
        },
    }


# Imprime o relatório em texto, no formato usado por
# docs/visual-context-sota-review.md. Existe separado da coleta para que o JSON
# continue sendo a fonte da verdade e o texto, só uma leitura dele.
def render_report(report: dict[str, Any]) -> str:
    """Formata o relatório da sonda para leitura humana."""
    lines = [f"Sonda de sinal de evidência sobre {report['run']}", ""]
    for frame in report["frames"]:
        lines.append(f"{frame['frame_id']}  regiões={frame['region_count']}  grade densa={frame['dense_grid']}")
        for slot, tally in frame["alignment_arbiter"].items():
            margin = tally["median_absolute_margin"]
            rendered = "n/a" if margin is None else f"{margin:.4f}"
            lines.append(
                f"    árbitro {slot:17s} concorda={tally['agrees']:3d} "
                f"indistinguível={tally['indistinguishable']:3d} discorda={tally['disagrees']:3d} "
                f"de {tally['total']:3d}  |margem| mediana={rendered}"
            )
        coherence = frame["adjacent_pair_coherence"]
        for name in ("same_label", "different_label"):
            stats = coherence[name]
            if stats["count"]:
                lines.append(
                    f"    pares adjacentes {name:16s} n={stats['count']:4d} "
                    f"p10={stats['p10']:.3f} med={stats['median']:.3f} p90={stats['p90']:.3f}"
                )
        lines.append("")

    totals = report["totals"]
    lines.append("TOTAL")
    for slot, tally in totals["alignment_arbiter"].items():
        margin = tally["median_absolute_margin"]
        rendered = "n/a" if margin is None else f"{margin:.4f}"
        lines.append(
            f"    árbitro {slot:17s} concorda={tally['agrees']:3d} "
            f"indistinguível={tally['indistinguishable']:3d} discorda={tally['disagrees']:3d} "
            f"de {tally['total']:3d}  |margem| mediana={rendered}"
        )
    separability = totals["adjacent_pair_coherence"]["separability"]
    if separability["best_balanced_accuracy"] is not None:
        lines.append(
            f"    melhor separação por cosseno denso entre pares adjacentes: "
            f"{separability['best_balanced_accuracy']:.3f} em t={separability['threshold']}"
        )
    return "\n".join(lines)


# Compõe a CLI da sonda. Mantém a seleção de frames explícita para que uma
# medição possa ser repetida exatamente.
def main(argv: list[str] | None = None) -> int:
    """Executa a sonda sobre um run versionado e escreve o relatório."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="diretório de um run em results/samples")
    parser.add_argument("--frame-id", action="append", default=[], help="frame específico; repetível")
    parser.add_argument("--output", type=Path, default=None, help="onde gravar o relatório JSON")
    args = parser.parse_args(argv)

    report = probe_run(args.run, tuple(args.frame_id))
    print(render_report(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nRelatório JSON: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrypoint de linha de comando
    raise SystemExit(main())
