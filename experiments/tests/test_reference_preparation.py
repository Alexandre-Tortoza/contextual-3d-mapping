"""Testes da preparação determinística do conjunto de referência (#197)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from contextual_mapping_datasets import (
    ReviewState,
    Split,
    load_reference_manifest,
    save_reference_manifest,
    validate_reference,
)
from visual_perception_evaluation.prediction import (
    PredictedRegion,
    PredictionSet,
    SamplePrediction,
    ScoredValue,
)
from visual_perception_experiments.draft_reference import build_draft_manifest
from visual_perception_experiments.fixture_reference import (
    FIXTURE_REFERENCE_ID,
    build_fixture_manifest,
    fixture_shapes,
    render_frame,
    write_fixture_frames,
)
from visual_perception_experiments.frame_conditions import laplacian_variance, measure_conditions
from visual_perception_experiments.prepare_reference import (
    EGO_VEHICLE_ROW_START,
    build_manifest,
    build_sample,
    guard_existing_manifest,
    mask_ego_vehicle,
    split_windows,
)
from visual_perception_experiments.review_package import render_review_page, write_review_package

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# Executa um CLI de experimento como documentado, sem depender do pythonpath
# injetado pelo pytest no processo pai.
def test_experiment_entrypoint_prepares_composition_import_roots() -> None:
    """Confirma que o pacote resolve suas dependências locais ao iniciar."""
    environment = {**os.environ, "PYTHONPATH": str(_REPOSITORY_ROOT / "experiments")}
    completed = subprocess.run(
        [sys.executable, "-m", "visual_perception_experiments.review_package", "--help"],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--output-dir" in completed.stdout


def test_split_windows_are_contiguous_and_separated_by_guard_bands() -> None:
    windows = split_windows(1000, guard_band_fraction=0.03)

    assert [window.split for window in windows] == [Split.DEVELOPMENT, Split.CALIBRATION, Split.TEST]
    assert windows[0].start == 0
    # Cada bloco começa 30 frames (3% de 1000) depois do fim do anterior.
    assert windows[1].start == windows[0].stop + 30
    assert windows[2].start == windows[1].stop + 30
    assert windows[2].stop == 1000


def test_sampling_inside_a_window_is_deterministic_and_spread_out() -> None:
    window = split_windows(1000)[0]

    first = window.sample(4)
    second = window.sample(4)

    assert first == second
    assert all(window.start <= index < window.stop for index in first)
    assert len(set(first)) == 4


def test_a_sequence_too_short_for_three_blocks_fails_instead_of_overlapping() -> None:
    with pytest.raises(ValueError, match="not enough for"):
        split_windows(4, guard_band_fraction=0.5)
    with pytest.raises(ValueError, match="only 400 frames available"):
        split_windows(1000)[0].sample(500)


def test_the_ego_vehicle_band_is_zeroed_before_the_frame_is_stored() -> None:
    pixels = np.full((480, 640, 3), 255, dtype=np.uint8)

    masked = mask_ego_vehicle(pixels)

    assert masked[EGO_VEHICLE_ROW_START:].sum() == 0
    assert masked[: EGO_VEHICLE_ROW_START].all()
    assert pixels.all()  # a entrada não é modificada


def test_capture_conditions_are_measured_and_never_guessed() -> None:
    dark = np.full((32, 32, 3), 10, dtype=np.uint8)
    bright_flat = np.full((32, 32, 3), 200, dtype=np.uint8)
    textured = np.zeros((32, 32, 3), dtype=np.uint8)
    textured[::2] = 255

    assert measure_conditions(dark) == ("low_light", "motion_blur")
    assert measure_conditions(bright_flat) == ("normal_light", "motion_blur")
    assert measure_conditions(textured)[1] == "sharp"
    assert laplacian_variance(np.zeros((8, 8))) == 0.0


def test_an_extracted_sample_carries_identity_and_no_invented_annotation(tmp_path: Path) -> None:
    dataset_root = tmp_path / "corridor-02"
    frames_dir = dataset_root / "visual-reference"
    frames_dir.mkdir(parents=True)
    frame_path = frames_dir / "sample.png"
    pixels = render_frame(fixture_shapes(Split.DEVELOPMENT))
    frame_path.write_bytes(b"deterministic-bytes")

    sample = build_sample(
        "corridor-02-development-000001",
        frame_path,
        pixels,
        Split.DEVELOPMENT,
        "corridor-02:12345",
        dataset_root=dataset_root,
    )

    assert sample.regions == ()
    assert sample.review_state is ReviewState.PENDING_REVIEW
    assert sample.provenance.annotator == "unassigned"
    assert sample.artifact.uri == "visual-reference/sample.png"
    assert "ego_vehicle_masked" in sample.capture_conditions
    assert sample.source_frame_id == "corridor-02:12345"


def test_rerunning_the_tool_refuses_to_discard_reviewed_work(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    save_reference_manifest(build_fixture_manifest(), path)

    guard_existing_manifest(tmp_path / "missing.json", force=False)  # nada a proteger
    guard_existing_manifest(path, force=True)  # explicitamente autorizado
    with pytest.raises(SystemExit, match="already contains reviewed or annotated samples"):
        guard_existing_manifest(path, force=False)


def test_the_generated_manifest_passes_reference_validation() -> None:
    manifest = build_manifest(build_fixture_manifest().samples, reference_id="generated")

    validate_reference(manifest)

    assert len(manifest.samples) == 3
    assert manifest.policy_uri.endswith("annotation-policy.md")


def test_the_synthetic_fixture_is_a_complete_reviewed_reference_set() -> None:
    manifest = build_fixture_manifest()

    validate_reference(manifest, require_reviewed=True)

    assert manifest.reference_id == FIXTURE_REFERENCE_ID
    test_sample = manifest.samples_in(Split.TEST)[0]
    assert [region.annotation_id for region in test_sample.scored_regions] == ["test-floor"]
    calibration_sample = manifest.samples_in(Split.CALIBRATION)[0]
    assert calibration_sample.regions[0].labels == ("porta", "armário")


def test_the_synthetic_frames_match_the_annotations_that_describe_them(tmp_path: Path) -> None:
    frames = write_fixture_frames(tmp_path)
    manifest = build_fixture_manifest(tmp_path)

    sample = manifest.samples_in(Split.DEVELOPMENT)[0]
    shape = fixture_shapes(Split.DEVELOPMENT)[0]
    annotation = sample.regions[0]

    assert frames[sample.sample_id].exists()
    assert annotation.mask.area == shape.size * shape.size
    assert sample.artifact.digest == hashlib.sha256(frames[sample.sample_id].read_bytes()).hexdigest()


def test_writing_the_fixture_frames_is_byte_for_byte_reproducible(tmp_path: Path) -> None:
    first = write_fixture_frames(tmp_path / "a")
    second = write_fixture_frames(tmp_path / "b")

    for sample_id, path in first.items():
        assert path.read_bytes() == second[sample_id].read_bytes()


def test_the_review_package_shows_what_was_measured_and_what_is_missing(tmp_path: Path) -> None:
    write_fixture_frames(tmp_path)
    manifest = build_fixture_manifest(tmp_path)

    page = render_review_page(manifest, frames_dir=tmp_path)

    assert "Checklist de revisão" in page
    assert "synthetic-development" in page
    assert "data:image/png;base64," in page
    assert "porta, armário" in page
    assert "ignorada" in page


def test_the_review_package_still_renders_without_the_raw_frames(tmp_path: Path) -> None:
    manifest = build_fixture_manifest()

    page = render_review_page(manifest, frames_dir=tmp_path / "absent")
    destination = write_review_package(manifest, frames_dir=tmp_path / "absent", output_dir=tmp_path)

    assert "Frame não disponível localmente" in page
    assert destination.exists()
    assert destination.read_text(encoding="utf-8") == page


def test_the_manifest_written_by_the_tool_round_trips_through_disk(tmp_path: Path) -> None:
    manifest = build_fixture_manifest()
    path = tmp_path / "reference.json"

    save_reference_manifest(manifest, path)

    assert load_reference_manifest(path) == manifest
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "visual-reference/1"


# Confirma que predições podem reduzir o trabalho mecânico sem se passarem
# por anotações ou revisão humana concluída.
def test_prediction_drafts_remain_pending_human_review() -> None:
    """Converte máscara/label previstos e mantém estado e proveniência honestos."""
    manifest = build_fixture_manifest()
    sample = manifest.samples[0]
    source_region = sample.regions[0]
    predictions = PredictionSet(
        run_id="draft-run",
        configuration="multi_context",
        config_fingerprint="draft-config",
        code_revision="deadbeef",
        samples=(
            SamplePrediction(
                sample.sample_id,
                sample.width,
                sample.height,
                regions=(
                    PredictedRegion(
                        "predicted-a",
                        sample.width,
                        sample.height,
                        source_region.mask.runs,
                        labels=(ScoredValue("porta", raw_confidence=0.8),),
                    ),
                ),
            ),
        ),
    )

    draft = build_draft_manifest(manifest, predictions)
    drafted = draft.samples[0]

    assert drafted.review_state is ReviewState.PENDING_REVIEW
    assert drafted.provenance.method == "model_assisted_draft"
    assert drafted.regions[0].labels == ("porta",)
    assert "revisão humana" in drafted.regions[0].notes


# Confirma que o pacote visual realmente incorpora overlays das masks, não
# apenas a imagem bruta e uma lista textual de IDs.
def test_review_package_renders_annotation_overlay(tmp_path: Path) -> None:
    """Gera uma imagem PNG embutida diferente do frame sem overlay."""
    frames = write_fixture_frames(tmp_path)
    manifest = build_fixture_manifest(tmp_path)

    page = render_review_page(manifest, frames_dir=tmp_path)
    raw_base64 = __import__("base64").b64encode(
        frames[manifest.samples[0].sample_id].read_bytes()
    ).decode("ascii")

    assert raw_base64 not in page
