"""Testes end-to-end do pipeline canônico (#169). Sem GPU, sem download de modelo."""

from __future__ import annotations

import dataclasses

from fixtures import blank_payload, default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.infrastructure.fakes.fake_feature_extractor import FakeDenseFeatureExtractor


# Verifica o caso degenerado: uma imagem sem nenhuma região descoberta ainda produz uma
# observação válida (regions vazio) que passa no quality audit, sem falhas de interpretação.
def test_canonical_pipeline_with_no_regions_passes_audit() -> None:
    result = run_canonical_pipeline(image_observation(), blank_payload(), default_config(), default_ports())

    assert result.observation.regions == ()
    assert result.audit.passed
    assert result.region_interpretation_failures == ()


# Confirma o caminho principal do pipeline: uma única região descoberta recebe claims
# semânticos e ambos os embeddings (visual e de linguagem), e a observação passa no audit.
def test_canonical_pipeline_with_one_region_produces_claims_and_embeddings() -> None:
    payload = payload_with_blobs(blobs=((4, 4, 12, 12, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    assert len(result.observation.regions) == 1
    region = result.observation.regions[0]
    assert region.claims
    assert region.visual_embedding_ref is not None
    assert region.language_embedding_ref is not None
    assert result.audit.passed


# Garante que, com múltiplas regiões na imagem, o estágio de geração de relações roda e o
# pipeline continua consistente (audit passa).
def test_canonical_pipeline_with_multiple_regions_generates_relations() -> None:
    payload = payload_with_blobs(
        blobs=(
            (2, 2, 8, 8, (200, 30, 30)),
            (20, 20, 28, 28, (30, 200, 30)),
        )
    )
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    assert len(result.observation.regions) == 2
    assert result.audit.passed


# Confirma que o pipeline sempre termina rodando o quality auditor (#168) sobre sua
# própria saída, e que essa saída não produz nenhum erro de auditoria.
def test_canonical_pipeline_output_passes_the_quality_auditor() -> None:
    payload = payload_with_blobs()
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    assert result.audit.errors == ()


# Um fallback de backend denso é uma execução legítima, mas não é a execução
# pedida. Sem o motivo subindo até PipelineResult, o manifest de validação
# registraria o backend configurado e o run pareceria tê-lo usado — que é
# exatamente o "fallback silencioso" que o checklist de #190 proíbe.
def test_a_dense_feature_fallback_is_reported_instead_of_passing_as_the_configured_backend() -> None:
    """O motivo do fallback do extractor denso chega ao PipelineResult."""

    class _FallenBackExtractor:
        """Envolve o fake e carimba o motivo de fallback que o adapter real gravaria."""

        def extract(self, image: ImagePayload, config: FeatureExtractionConfig) -> FeatureMap:
            feature_map = FakeDenseFeatureExtractor().extract(image, config)
            return dataclasses.replace(
                feature_map, fallback_reason="BackendUnavailableError: featup checkpoint missing"
            )

    payload = payload_with_blobs(blobs=((4, 4, 12, 12, (200, 30, 30)),))
    ports = dataclasses.replace(default_ports(), feature_extractor=_FallenBackExtractor())

    result = run_canonical_pipeline(image_observation(), payload, default_config(), ports)

    assert result.feature_fallback_reason == "BackendUnavailableError: featup checkpoint missing"
    assert result.audit.passed


# O outro lado da mesma garantia: uma execução sem fallback não inventa um
# motivo, para que a presença do campo signifique sempre um desvio real.
def test_a_run_without_fallback_reports_no_reason() -> None:
    """Sem fallback, ``feature_fallback_reason`` permanece ``None``."""
    payload = payload_with_blobs(blobs=((4, 4, 12, 12, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    assert result.feature_fallback_reason is None


# As proposals cruas de discovery precisam sobreviver ao pipeline: elas são o
# único registro do estágio pré-merge, e recomputá-las depois não serve porque
# discovery não é determinística — uma segunda passada produziria proposals que
# não são as que geraram estas regiões.
def test_pipeline_exposes_the_raw_discovery_proposals() -> None:
    """O resultado carrega as proposals anteriores ao merge geométrico."""
    result = run_canonical_pipeline(
        image_observation(), payload_with_blobs(), default_config(), default_ports()
    )
    assert result.proposals != ()
    assert all(proposal.proposal_id for proposal in result.proposals)


# Amarra os dois estágios: toda proposal descoberta precisa aparecer em
# exatamente uma região. Se um dia divergirem, alguma proposal terá sido
# descartada em silêncio entre discovery e merge — e é justamente essa
# divergência que o diagnóstico do frame existe para tornar visível.
def test_every_proposal_is_accounted_for_by_exactly_one_region() -> None:
    """A soma das proposals contribuintes das regiões bate com as descobertas."""
    result = run_canonical_pipeline(
        image_observation(),
        payload_with_blobs(blobs=((4, 4, 10, 10, (200, 30, 30)), (20, 20, 28, 28, (30, 30, 200)))),
        default_config(),
        default_ports(),
    )
    contributing = [
        proposal_id
        for region in result.observation.regions
        for proposal_id in region.contributing_proposal_ids
    ]
    assert sorted(contributing) == sorted(p.proposal_id for p in result.proposals)


# Um consumidor que não pede diagnóstico não deve ser obrigado a preencher o
# campo: ele é aditivo e tem default vazio.
def test_proposals_default_to_empty_when_not_provided() -> None:
    """PipelineResult pode ser construído sem proposals."""
    from visual_perception.domain.audit import AuditResult

    result = run_canonical_pipeline(
        image_observation(), blank_payload(), default_config(), default_ports()
    )
    replaced = dataclasses.replace(result, proposals=())
    assert replaced.proposals == ()
    assert isinstance(result.audit, AuditResult)
