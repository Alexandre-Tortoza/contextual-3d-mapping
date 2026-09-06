"""Leitura e escrita do manifest JSON do conjunto de referência anotado.

Issue: #197.

Mantido separado de ``annotation_manifest.py`` pela mesma razão que o
manifest de dataset separa schema de I/O: as invariantes pertencem aos
dataclasses do domínio, e este módulo só traduz entre JSON e esses tipos,
aplicando as regras de fronteira compartilhadas em ``_document.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._document import boolean, mapping, optional_string, positive_integer, relative_artifact_uri
from ._document import sequence as json_sequence
from ._document import string as json_string
from .annotation_manifest import (
    REFERENCE_SCHEMA_VERSION,
    AnnotationCertainty,
    AnnotationProvenance,
    MaskAnnotation,
    ReferenceManifest,
    RegionAnnotation,
    RelationAnnotation,
    ReviewState,
    SampleAnnotation,
    SampleArtifact,
    Split,
)


# Carrega e valida um manifest de referência rastreado no repositório.
# Existe para que experimentos e avaliação recebam um objeto tipado, com
# falha acionável na fronteira, em vez de um dict solto.
def load_reference_manifest(path: Path) -> ReferenceManifest:
    """Carrega um :class:`ReferenceManifest` a partir de um arquivo JSON.

    Argumentos:
        path: arquivo JSON versionado do conjunto de referência.
    Retorna:
        o manifest tipado e validado.
    Levanta:
        FileNotFoundError: se o arquivo não existir.
        ValueError: se o JSON ou sua estrutura não forem válidos.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON reference manifest {path}: {error.msg}.") from error
    return reference_manifest_from_mapping(document, source=str(path))


# Grava um manifest de referência de forma determinística. Existe para que a
# ferramenta de preparação (#197) e a de revisão produzam byte a byte o mesmo
# arquivo para a mesma entrada, o que é o que torna o conjunto versionável.
def save_reference_manifest(manifest: ReferenceManifest, path: Path) -> None:
    """Grava ``manifest`` como JSON determinístico e legível.

    Argumentos:
        manifest: o conjunto de referência a persistir.
        path: destino do arquivo JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    document = reference_manifest_to_mapping(manifest)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Converte o documento já desserializado na representação tipada. É pública
# para que ferramentas validem um manifest antes de gravá-lo, sem duplicar as
# regras de parsing do loader de arquivo.
def reference_manifest_from_mapping(document: object, *, source: str = "manifest") -> ReferenceManifest:
    """Converte um mapeamento JSON em um manifest de referência validado.

    Argumentos:
        document: objeto desserializado de um documento JSON.
        source: identificação usada nas mensagens de validação.
    Retorna:
        o manifest tipado e validado.
    Levanta:
        ValueError: se campos obrigatórios ou referências locais forem inválidos.
    """
    root = mapping(document, source)
    schema_version = json_string(root, "schema_version", source, default=REFERENCE_SCHEMA_VERSION)
    samples = tuple(
        _sample(item, f"{source}.samples[{index}]")
        for index, item in enumerate(json_sequence(root, "samples", source))
    )
    return ReferenceManifest(
        reference_id=json_string(root, "reference_id", source),
        dataset_id=json_string(root, "dataset_id", source),
        policy_uri=json_string(root, "policy_uri", source),
        samples=samples,
        schema_version=schema_version,
        created_at=optional_string(root, "created_at", source),
    )


# Converte um manifest tipado de volta em um documento JSON. Existe como o
# inverso exato de reference_manifest_from_mapping, para que gravar e reler
# um manifest seja um round-trip sem perda.
def reference_manifest_to_mapping(manifest: ReferenceManifest) -> dict[str, Any]:
    """Converte um :class:`ReferenceManifest` em um documento JSON serializável."""
    return {
        "schema_version": manifest.schema_version,
        "reference_id": manifest.reference_id,
        "dataset_id": manifest.dataset_id,
        "policy_uri": manifest.policy_uri,
        "created_at": manifest.created_at,
        "samples": [_sample_to_mapping(sample) for sample in manifest.samples],
    }


# Constrói uma amostra anotada a partir de um item do documento. A validação
# de identidade e integridade referencial permanece no dataclass do domínio,
# que é a fonte de verdade dessas invariantes.
def _sample(value: object, source: str) -> SampleAnnotation:
    """Converte um item ``samples[i]`` do documento em :class:`SampleAnnotation`."""
    document = mapping(value, source)
    return SampleAnnotation(
        sample_id=json_string(document, "sample_id", source),
        artifact=_artifact(document.get("artifact"), f"{source}.artifact"),
        width=positive_integer(document, "width", source),
        height=positive_integer(document, "height", source),
        split=Split(json_string(document, "split", source)),
        provenance=_provenance(document.get("provenance"), f"{source}.provenance"),
        source_frame_id=optional_string(document, "source_frame_id", source),
        capture_conditions=_strings(document, "capture_conditions", source),
        strata=_strings(document, "strata", source),
        regions=tuple(
            _region(item, f"{source}.regions[{index}]")
            for index, item in enumerate(json_sequence(document, "regions", source, default=()))
        ),
        relations=tuple(
            _relation(item, f"{source}.relations[{index}]")
            for index, item in enumerate(json_sequence(document, "relations", source, default=()))
        ),
        review_state=ReviewState(
            json_string(document, "review_state", source, default=ReviewState.PENDING_REVIEW.value)
        ),
    )


# Constrói a referência de artifact da amostra, restringindo a uri a um
# caminho relativo pela mesma regra do manifest de dataset bruto.
def _artifact(value: object, source: str) -> SampleArtifact:
    """Converte o objeto ``artifact`` em :class:`SampleArtifact`."""
    document = mapping(value, source)
    return SampleArtifact(
        uri=relative_artifact_uri(json_string(document, "uri", source), source),
        media_type=json_string(document, "media_type", source),
        digest=json_string(document, "digest", source),
    )


# Constrói a proveniência de anotação declarada pela amostra.
def _provenance(value: object, source: str) -> AnnotationProvenance:
    """Converte o objeto ``provenance`` em :class:`AnnotationProvenance`."""
    document = mapping(value, source)
    return AnnotationProvenance(
        annotator=json_string(document, "annotator", source),
        method=json_string(document, "method", source),
        policy_version=json_string(document, "policy_version", source),
        annotated_at=optional_string(document, "annotated_at", source),
        resolution=optional_string(document, "resolution", source),
    )


# Constrói uma região anotada, incluindo sua máscara RLE.
def _region(value: object, source: str) -> RegionAnnotation:
    """Converte um item ``regions[i]`` em :class:`RegionAnnotation`."""
    document = mapping(value, source)
    return RegionAnnotation(
        annotation_id=json_string(document, "annotation_id", source),
        mask=_mask(document.get("mask"), f"{source}.mask"),
        labels=_strings(document, "labels", source),
        certainty=AnnotationCertainty(
            json_string(document, "certainty", source, default=AnnotationCertainty.CERTAIN.value)
        ),
        attributes=_strings(document, "attributes", source),
        condition=optional_string(document, "condition", source),
        material=optional_string(document, "material", source),
        hazards=_strings(document, "hazards", source),
        visibility=optional_string(document, "visibility", source),
        ignored=boolean(document, "ignored", source, default=False),
        notes=optional_string(document, "notes", source),
    )


# Constrói a máscara RLE de uma região anotada.
def _mask(value: object, source: str) -> MaskAnnotation:
    """Converte o objeto ``mask`` em :class:`MaskAnnotation`."""
    document = mapping(value, source)
    runs = json_sequence(document, "runs", source)
    if any(type(run) is not int for run in runs):
        raise ValueError(f"{source}.runs must contain only integers.")
    return MaskAnnotation(
        width=positive_integer(document, "width", source),
        height=positive_integer(document, "height", source),
        runs=tuple(int(run) for run in runs),
    )


# Constrói uma relação anotada entre duas regiões da amostra.
def _relation(value: object, source: str) -> RelationAnnotation:
    """Converte um item ``relations[i]`` em :class:`RelationAnnotation`."""
    document = mapping(value, source)
    return RelationAnnotation(
        relation_id=json_string(document, "relation_id", source),
        subject_annotation_id=json_string(document, "subject_annotation_id", source),
        predicate=json_string(document, "predicate", source),
        object_annotation_id=json_string(document, "object_annotation_id", source),
        certainty=AnnotationCertainty(
            json_string(document, "certainty", source, default=AnnotationCertainty.CERTAIN.value)
        ),
    )


# Lê uma lista de strings opcional, recusando entradas vazias. Existe porque
# labels, atributos, hazards, condições de captura e estratos compartilham
# exatamente a mesma regra de vocabulário aberto.
def _strings(document: Mapping[str, Any], field: str, source: str) -> tuple[str, ...]:
    """Converte uma lista JSON de strings não vazias em tupla."""
    values = json_sequence(document, field, source, default=())
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}.{field} must contain only non-empty strings.")
    return tuple(str(value) for value in values)


# Converte uma amostra tipada de volta em um documento JSON, mantendo a
# ordem de campos estável para que o arquivo gravado seja determinístico.
def _sample_to_mapping(sample: SampleAnnotation) -> dict[str, Any]:
    """Converte uma :class:`SampleAnnotation` em um dict serializável."""
    return {
        "sample_id": sample.sample_id,
        "artifact": {
            "uri": sample.artifact.uri,
            "media_type": sample.artifact.media_type,
            "digest": sample.artifact.digest,
        },
        "width": sample.width,
        "height": sample.height,
        "split": sample.split.value,
        "source_frame_id": sample.source_frame_id,
        "capture_conditions": list(sample.capture_conditions),
        "strata": list(sample.strata),
        "review_state": sample.review_state.value,
        "provenance": {
            "annotator": sample.provenance.annotator,
            "method": sample.provenance.method,
            "policy_version": sample.provenance.policy_version,
            "annotated_at": sample.provenance.annotated_at,
            "resolution": sample.provenance.resolution,
        },
        "regions": [
            {
                "annotation_id": region.annotation_id,
                "mask": {
                    "width": region.mask.width,
                    "height": region.mask.height,
                    "runs": list(region.mask.runs),
                },
                "labels": list(region.labels),
                "certainty": region.certainty.value,
                "attributes": list(region.attributes),
                "condition": region.condition,
                "material": region.material,
                "hazards": list(region.hazards),
                "visibility": region.visibility,
                "ignored": region.ignored,
                "notes": region.notes,
            }
            for region in sample.regions
        ],
        "relations": [
            {
                "relation_id": relation.relation_id,
                "subject_annotation_id": relation.subject_annotation_id,
                "predicate": relation.predicate,
                "object_annotation_id": relation.object_annotation_id,
                "certainty": relation.certainty.value,
            }
            for relation in sample.relations
        ],
    }
