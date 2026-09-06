"""Testes do manifest do conjunto de referência anotado (#197)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextual_mapping_datasets import (
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
    load_and_validate_reference_split,
    load_reference_manifest,
    load_reference_split,
    reference_manifest_from_mapping,
    reference_manifest_to_mapping,
    save_reference_manifest,
    validate_reference,
    validate_reference_split,
)

_POLICY = AnnotationProvenance(
    annotator="pending", method="tool_generated", policy_version="annotation-policy/1"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# Constrói uma máscara RLE que cobre um retângulo de uma imagem 8x8, para
# que os testes tenham anotações de geometria válidas sem depender de numpy.
def _mask(occupied: int = 8) -> MaskAnnotation:
    return MaskAnnotation(width=8, height=8, runs=(4, occupied, 64 - 4 - occupied))


# Constrói um artifact imutável com um digest determinístico derivado do
# nome, para que o teste de vazamento de split possa controlar a identidade.
def _artifact(name: str) -> SampleArtifact:
    import hashlib

    return SampleArtifact(
        uri=f"frames/{name}.png",
        media_type="image/png",
        digest=hashlib.sha256(name.encode()).hexdigest(),
    )


# Constrói uma amostra anotada mínima e válida.
def _sample(
    sample_id: str,
    split: Split = Split.DEVELOPMENT,
    *,
    artifact_name: str | None = None,
    regions: tuple[RegionAnnotation, ...] | None = None,
    relations: tuple[RelationAnnotation, ...] = (),
    source_frame_id: str | None = None,
    review_state: ReviewState = ReviewState.PENDING_REVIEW,
) -> SampleAnnotation:
    return SampleAnnotation(
        sample_id=sample_id,
        artifact=_artifact(artifact_name or sample_id),
        width=8,
        height=8,
        split=split,
        provenance=_POLICY,
        source_frame_id=source_frame_id,
        capture_conditions=("indoor", "low_light"),
        strata=("small_region",),
        regions=regions if regions is not None else (RegionAnnotation("ann-1", _mask(), ("door",)),),
        relations=relations,
        review_state=review_state,
    )


# Constrói um manifest com um split de cada, que é o mínimo aceito pela
# validação de avaliação.
def _manifest(samples: tuple[SampleAnnotation, ...] | None = None) -> ReferenceManifest:
    return ReferenceManifest(
        reference_id="corridor-02-reference-1",
        dataset_id="corridor-02",
        policy_uri="modules/visual-perception/docs/annotation-policy.md",
        samples=samples
        or (
            _sample("sample-dev"),
            _sample("sample-cal", Split.CALIBRATION),
            _sample("sample-test", Split.TEST),
        ),
    )


def test_every_sample_has_stable_identity_provenance_and_explicit_split() -> None:
    manifest = _manifest()

    sample = manifest.samples_in(Split.TEST)[0]

    assert sample.sample_id == "sample-test"
    assert sample.split is Split.TEST
    assert len(sample.artifact.digest) == 64
    assert sample.provenance.policy_version == "annotation-policy/1"


def test_duplicate_sample_identities_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate sample ids"):
        _manifest((_sample("same"), _sample("same", Split.TEST, artifact_name="other")))


def test_duplicate_annotation_and_relation_identities_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate annotation ids"):
        _sample(
            "sample",
            regions=(
                RegionAnnotation("ann-1", _mask(), ("door",)),
                RegionAnnotation("ann-1", _mask(), ("wall",)),
            ),
        )
    regions = (RegionAnnotation("ann-1", _mask(), ("door",)), RegionAnnotation("ann-2", _mask(), ("wall",)))
    with pytest.raises(ValueError, match="duplicate relation ids"):
        _sample(
            "sample",
            regions=regions,
            relations=(
                RelationAnnotation("rel-1", "ann-1", "near", "ann-2"),
                RelationAnnotation("rel-1", "ann-2", "near", "ann-1"),
            ),
        )


def test_dangling_annotation_references_are_rejected() -> None:
    with pytest.raises(ValueError, match="references unknown annotation"):
        _sample(
            "sample",
            regions=(RegionAnnotation("ann-1", _mask(), ("door",)),),
            relations=(RelationAnnotation("rel-1", "ann-1", "near", "ann-missing"),),
        )


def test_the_same_artifact_in_two_splits_is_reported_as_split_leakage() -> None:
    with pytest.raises(ValueError, match="split leakage: the same artifact digest"):
        _manifest(
            (
                _sample("sample-dev", Split.DEVELOPMENT, artifact_name="shared"),
                _sample("sample-test", Split.TEST, artifact_name="shared"),
            )
        )


def test_the_same_source_frame_in_two_splits_is_reported_as_split_leakage() -> None:
    with pytest.raises(ValueError, match="split leakage: the same source frame"):
        _manifest(
            (
                _sample("sample-dev", Split.DEVELOPMENT, source_frame_id="frame-000012"),
                _sample("sample-test", Split.TEST, source_frame_id="frame-000012"),
            )
        )


def test_ambiguity_and_unknown_labels_are_expressible_without_a_closed_taxonomy() -> None:
    ambiguous = RegionAnnotation(
        "ann-1", _mask(), ("door", "cabinet"), certainty=AnnotationCertainty.AMBIGUOUS
    )
    unknown = RegionAnnotation("ann-2", _mask(), (), certainty=AnnotationCertainty.UNKNOWN)

    assert ambiguous.labels == ("door", "cabinet")
    assert unknown.labels == ()
    with pytest.raises(ValueError, match="is 'unknown' but declares labels"):
        RegionAnnotation("ann-3", _mask(), ("door",), certainty=AnnotationCertainty.UNKNOWN)
    with pytest.raises(ValueError, match="has no label"):
        RegionAnnotation("ann-4", _mask(), ())


def test_ignored_regions_leave_the_metric_without_becoming_background() -> None:
    sample = _sample(
        "sample",
        regions=(
            RegionAnnotation("ann-1", _mask(), ("door",)),
            RegionAnnotation("ann-2", _mask(), (), ignored=True, notes="motion blur"),
        ),
    )

    assert len(sample.regions) == 2
    assert [region.annotation_id for region in sample.scored_regions] == ["ann-1"]


def test_a_mask_that_does_not_cover_the_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="must cover exactly 64 pixels"):
        MaskAnnotation(width=8, height=8, runs=(4, 8))
    assert _mask(8).area == 8


def test_a_mask_resolution_must_match_its_sample() -> None:
    with pytest.raises(ValueError, match="mask resolution does not match"):
        SampleAnnotation(
            sample_id="sample",
            artifact=_artifact("sample"),
            width=16,
            height=16,
            split=Split.TEST,
            provenance=_POLICY,
            regions=(RegionAnnotation("ann-1", _mask(), ("door",)),),
        )


def test_release_validation_refuses_an_unreviewed_reference_set() -> None:
    manifest = _manifest()

    validate_reference(manifest)  # em anotação: estruturalmente válido
    with pytest.raises(ValueError, match="have not been reviewed"):
        validate_reference(manifest, require_reviewed=True)


def test_release_validation_refuses_a_reference_set_missing_a_split() -> None:
    manifest = _manifest((_sample("only-dev"),))

    with pytest.raises(ValueError, match="no samples for splits"):
        validate_reference(manifest)


def test_the_manifest_round_trips_through_json(tmp_path: Path) -> None:
    manifest = _manifest(
        (
            _sample(
                "sample-dev",
                regions=(
                    RegionAnnotation(
                        "ann-1",
                        _mask(),
                        ("door",),
                        attributes=("closed",),
                        condition="intact",
                        material="wood",
                        hazards=(),
                        visibility="clear",
                    ),
                    RegionAnnotation("ann-2", _mask(12), ("floor",)),
                ),
                relations=(RelationAnnotation("rel-1", "ann-1", "near", "ann-2"),),
            ),
            _sample("sample-cal", Split.CALIBRATION),
            _sample("sample-test", Split.TEST),
        )
    )
    path = tmp_path / "reference.json"

    save_reference_manifest(manifest, path)
    reloaded = load_reference_manifest(path)

    assert reloaded == manifest
    assert reference_manifest_from_mapping(reference_manifest_to_mapping(manifest)) == manifest


def test_writing_the_manifest_is_deterministic(tmp_path: Path) -> None:
    manifest = _manifest()
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    save_reference_manifest(manifest, first)
    save_reference_manifest(manifest, second)

    assert first.read_bytes() == second.read_bytes()


def test_an_artifact_uri_may_not_escape_the_dataset_directory(tmp_path: Path) -> None:
    document = reference_manifest_to_mapping(_manifest())
    document["samples"][0]["artifact"]["uri"] = "../../etc/passwd"
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="path traversal"):
        load_reference_manifest(path)


def test_an_unsupported_schema_version_is_refused(tmp_path: Path) -> None:
    document = reference_manifest_to_mapping(_manifest())
    document["schema_version"] = "visual-reference/99"
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_reference_manifest(path)


# Confirma que o artifact versionado cobre exatamente o manifest corrente e
# preserva três blocos temporais exclusivos e ordenados.
def test_versioned_reference_split_matches_the_source_manifest() -> None:
    """Valida digest, cobertura, pertença e ordem temporal dos splits reais."""
    partition = load_and_validate_reference_split(
        _REPOSITORY_ROOT / "datasets/splits/corridor-02-visual-reference-1.json",
        _REPOSITORY_ROOT / "datasets/manifests/corridor-02-visual-reference.json",
    )

    assert {split: len(items) for split, items in partition.splits.items()} == {
        Split.DEVELOPMENT: 12,
        Split.CALIBRATION: 12,
        Split.TEST: 12,
    }


# Simula vazamento explícito entre calibration e development para garantir
# que um mesmo frame nunca possa alimentar ajuste e outra partição.
def test_reference_split_rejects_cross_split_sample_leakage(tmp_path: Path) -> None:
    """Rejeita um ID declarado em dois splits mesmo com manifest íntegro."""
    split_path = _REPOSITORY_ROOT / "datasets/splits/corridor-02-visual-reference-1.json"
    manifest_path = _REPOSITORY_ROOT / "datasets/manifests/corridor-02-visual-reference.json"
    document = json.loads(split_path.read_text(encoding="utf-8"))
    document["splits"]["calibration"].append(document["splits"]["development"][0])
    tampered_path = tmp_path / "split.json"
    tampered_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="more than one split"):
        validate_reference_split(
            load_reference_split(tampered_path),
            load_reference_manifest(manifest_path),
            manifest_path,
        )


# Altera apenas os bytes do manifest para provar que o vínculo criptográfico
# detecta regeneração ou edição não acompanhada por uma nova partição.
def test_reference_split_rejects_source_manifest_digest_drift(tmp_path: Path) -> None:
    """Rejeita um manifest semanticamente legível cujo SHA-256 mudou."""
    split_path = _REPOSITORY_ROOT / "datasets/splits/corridor-02-visual-reference-1.json"
    manifest_path = _REPOSITORY_ROOT / "datasets/manifests/corridor-02-visual-reference.json"
    changed_manifest = tmp_path / "manifest.json"
    changed_manifest.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="digest does not match"):
        load_and_validate_reference_split(split_path, changed_manifest)
