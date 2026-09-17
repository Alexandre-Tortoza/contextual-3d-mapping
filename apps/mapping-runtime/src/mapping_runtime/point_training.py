"""Adaptação de associações visuais para supervisão de `point-representation`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from point_representation import PointCloud, PointTrainingSample
from sensor_association import PointVisualAssociation

from contextual_mapping_contracts import FrameId


# Expõe a proveniência que não cabe em PointTrainingSample sem fazer o módulo
# de representação conhecer calibração ou câmera. A aplicação a preserva ao
# lado do sample para treino, avaliação e auditoria reproduzíveis.
@dataclass(frozen=True)
class VisualTeacherProvenance:
    """Identidade verificável dos alvos visuais alinhados por ponto.

    Argumentos:
        embedding_space: espaço compartilhado por todos os vetores do sample.
        producer: estágio e configuração que geraram os vetores.
        artifact_references: artifacts de feature de origem, em ordem estável.
        observation_ids: observações RGB que contribuíram os alvos.
        calibration_ids: calibrações usadas nas projeções correspondentes.
    """

    embedding_space: str
    producer: str
    artifact_references: tuple[str, ...]
    observation_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]


# Constrói um sample de treino somente com associações que carregam feature
# densa válida. Pontos ocluídos, fora de suporte ou sem artifact não recebem
# vetor sintético: simplesmente não entram no conjunto supervisionado, e os
# correspondence_indices preservam a identidade na nuvem de origem.
def visual_teacher_sample(
    coordinates: np.ndarray,
    frame_id: FrameId,
    associations: Sequence[PointVisualAssociation],
) -> tuple[PointTrainingSample, VisualTeacherProvenance]:
    """Converte associações ponto-imagem em um sample supervisionado.

    Argumentos:
        coordinates: XYZ canônico da nuvem de origem, um por associação.
        frame_id: frame espacial em que os XYZ estão expressos.
        associations: resultados na mesma ordem de ``coordinates``.
    Retorna:
        sample de treino contendo somente pontos com alvo visual e a
        proveniência compartilhada desses alvos.
    Levanta:
        ValueError: se comprimentos, espaços ou proveniência forem incompatíveis,
            ou se não existir alvo visual válido.
    """
    source = np.asarray(coordinates, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) != len(associations):
        raise ValueError("coordinates must be shaped (N, 3) and align with associations.")
    selected = [
        (index, association)
        for index, association in enumerate(associations)
        if association.dense_feature is not None
    ]
    if not selected:
        raise ValueError("visual_teacher_sample requires at least one valid dense feature target.")
    spaces = {(item.embedding_space, item.feature_producer, item.feature_dimension) for _, item in selected}
    if len(spaces) != 1:
        raise ValueError("visual teacher targets must share embedding space, producer and dimension.")
    if any(item.dense_feature_reference is None for _, item in selected):
        raise ValueError("visual teacher targets must preserve their feature artifact reference.")
    indices = np.asarray([index for index, _ in selected], dtype=np.int64)
    vectors = np.asarray([item.dense_feature for _, item in selected], dtype=np.float64)
    space, producer, _ = next(iter(spaces))
    sample = PointTrainingSample(
        point_cloud=PointCloud(source[indices], frame_id),
        teacher_features=vectors,
        correspondence_indices=indices,
    )
    return sample, VisualTeacherProvenance(
        embedding_space=str(space),
        producer=str(producer),
        artifact_references=tuple(sorted({str(item.dense_feature_reference) for _, item in selected})),
        observation_ids=tuple(sorted({item.rgb_observation.observation_id for _, item in selected})),
        calibration_ids=tuple(sorted({item.calibration.calibration_id for _, item in selected})),
    )
