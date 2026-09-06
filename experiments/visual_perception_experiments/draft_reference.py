"""Converte predições reais em rascunhos de anotação para revisão humana (#210).

O rascunho reduz trabalho mecânico de máscara e transcrição, mas nunca muda
``review_state`` para ``reviewed``. Apenas uma pessoa, seguindo a política
versionada, pode aceitar, corrigir, ignorar ou rejeitar cada proposta.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from contextual_mapping_datasets import (
    AnnotationCertainty,
    AnnotationProvenance,
    MaskAnnotation,
    ReferenceManifest,
    RegionAnnotation,
    ReviewState,
    SampleAnnotation,
    load_reference_manifest,
    save_reference_manifest,
    validate_reference,
)
from visual_perception_evaluation.prediction import (
    PredictedRegion,
    PredictionSet,
    ScoredValue,
    load_predictions,
)

from .paths import REFERENCE_MANIFEST


# Preserva valores previstos distintos na ordem original. Existe para que
# alternativas do modelo virem sugestões de ambiguidade, sem duplicatas.
def _values(items: tuple[ScoredValue, ...]) -> tuple[str, ...]:
    """Retorna os campos ``value`` distintos de uma sequência prevista."""
    values: list[str] = []
    for item in items:
        value = str(item.value).strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


# Converte uma região prevista em sugestão editável, preservando máscara e
# semântica, mas marcando em notes que nada foi revisado por humano.
def draft_region(region: PredictedRegion) -> RegionAnnotation:
    """Converte uma região prevista em rascunho de anotação não revisado."""
    labels = _values(region.labels)
    certainty = (
        AnnotationCertainty.UNKNOWN
        if not labels
        else AnnotationCertainty.AMBIGUOUS
        if len(labels) > 1
        else AnnotationCertainty.CERTAIN
    )
    conditions = _values(region.conditions)
    materials = _values(region.materials)
    return RegionAnnotation(
        annotation_id=f"draft-{region.region_id}",
        mask=MaskAnnotation(region.width, region.height, region.runs),
        labels=labels,
        certainty=certainty,
        attributes=_values(region.attributes),
        condition=conditions[0] if conditions else None,
        material=materials[0] if materials else None,
        hazards=_values(region.hazards),
        ignored=False,
        notes="Rascunho gerado pelo pipeline; confirmar máscara, semântica e ignored na revisão humana.",
    )


# Anexa rascunhos somente a amostras com predição correspondente e mantém
# split, digest e identidade do manifest original imutáveis.
def build_draft_manifest(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    *,
    policy_version: str = "annotation-policy/1",
) -> ReferenceManifest:
    """Retorna um novo manifest pending_review preenchido por predições.

    Argumentos:
        manifest: estrutura e particionamento de referência existentes.
        predictions: predições do pipeline sobre as mesmas amostras.
        policy_version: versão da política que orientará a revisão.
    Retorna:
        manifest estruturalmente válido, ainda não liberado para métricas.
    Levanta:
        ValueError: quando uma predição tem resolução incompatível.
    """
    by_sample = predictions.by_sample_id()
    samples: list[SampleAnnotation] = []
    for sample in manifest.samples:
        prediction = by_sample.get(sample.sample_id)
        if prediction is None or prediction.failed:
            regions: tuple[RegionAnnotation, ...] = ()
        else:
            if (prediction.width, prediction.height) != (sample.width, sample.height):
                raise ValueError(
                    f"Prediction {sample.sample_id!r} has resolution "
                    f"{prediction.width}x{prediction.height}, expected {sample.width}x{sample.height}."
                )
            regions = tuple(draft_region(region) for region in prediction.regions)
        samples.append(
            dataclasses.replace(
                sample,
                regions=regions,
                relations=(),
                review_state=ReviewState.PENDING_REVIEW,
                provenance=AnnotationProvenance(
                    annotator=f"pipeline:{predictions.run_id}",
                    method="model_assisted_draft",
                    policy_version=policy_version,
                    resolution="aguardando revisão humana; nenhuma região está validada",
                ),
            )
        )
    draft = dataclasses.replace(manifest, samples=tuple(samples))
    validate_reference(draft, require_reviewed=False)
    return draft


# Interface de linha de comando do gerador de rascunhos.
def main(argv: list[str] | None = None) -> None:
    """Gera um manifest de rascunho sem sobrescrever a revisão canônica."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    draft = build_draft_manifest(
        load_reference_manifest(arguments.manifest),
        load_predictions(arguments.predictions),
    )
    save_reference_manifest(draft, arguments.out)
    print(
        f"Wrote {arguments.out}: {sum(len(sample.regions) for sample in draft.samples)} "
        "regiões em pending_review"
    )


if __name__ == "__main__":  # pragma: no cover - entrada de CLI
    main()
