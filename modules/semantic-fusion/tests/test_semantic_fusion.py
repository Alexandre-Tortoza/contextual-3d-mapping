"""Testes da regra de fusão semântica multi-observação."""

from __future__ import annotations

import pytest
from semantic_fusion import SemanticContribution, fuse_point_contributions


# Constrói uma contribuição completa para variar um sinal por vez nos testes.
def _contribution(
    observation_id: str,
    label: str,
    *,
    timestamp_ns: int = 10,
    confidence: float | None = 0.9,
    calibrated_confidence: float | None = None,
    visual_support: float | None = None,
    region_quality: float | None = None,
) -> SemanticContribution:
    """Cria uma contribuição semântica de teste."""
    return SemanticContribution(
        observation_id=observation_id,
        timestamp_ns=timestamp_ns,
        region_id=f"region-{observation_id}",
        label=label,
        confidence=confidence,
        calibrated_confidence=calibrated_confidence,
        support_state="raw",
        visual_support=visual_support,
        region_quality=region_quality,
    )


# Um único keyframe continua sendo o caso mais comum de um trecho curto, e a
# fusão não pode alterar o resultado nesse caso.
def test_fonte_unica_preserva_o_claim_original() -> None:
    """Confere que uma contribuição isolada vence com concordância total."""
    fused = fuse_point_contributions([_contribution("frame-a", "porta")])
    assert fused.label == "porta"
    assert fused.agreement == 1.0
    assert fused.contribution_count == 1
    assert fused.confidence == 0.9


# Concordância é evidência: quando vários keyframes veem a mesma superfície e
# dizem a mesma coisa, isso precisa aparecer no artifact.
def test_fontes_concordantes_registram_concordancia_total() -> None:
    """Confere concordância quando todas as observações apontam o mesmo label."""
    fused = fuse_point_contributions(
        [_contribution("frame-a", "porta"), _contribution("frame-b", "porta", timestamp_ns=20)]
    )
    assert fused.agreement == 1.0
    assert fused.contribution_count == 2


# O conflito é o caso que justifica o módulo: a confiança decide, e a proposta
# perdedora continua registrada em vez de desaparecer.
def test_conflito_escolhe_a_maior_confianca_sem_descartar_a_perdedora() -> None:
    """Confere escolha por confiança e preservação dos contribuintes."""
    fused = fuse_point_contributions(
        [
            _contribution("frame-a", "parede", confidence=0.4),
            _contribution("frame-b", "porta", confidence=0.8, timestamp_ns=20),
        ]
    )
    assert fused.label == "porta"
    assert fused.observation_id == "frame-b"
    assert fused.agreement == 0.5
    assert [item.label for item in fused.contributions] == ["porta", "parede"]


# Confiança calibrada, quando existe, tem precedência sobre a confiança bruta
# declarada pelo modelo, que este pipeline ainda não calibra.
def test_confianca_calibrada_tem_precedencia_sobre_a_bruta() -> None:
    """Confere que a calibração vence a confiança declarada."""
    fused = fuse_point_contributions(
        [
            _contribution("frame-a", "parede", confidence=0.95),
            _contribution("frame-b", "porta", confidence=0.10, calibrated_confidence=0.99),
        ]
    )
    assert fused.label == "porta"
    assert fused.confidence == 0.99


# Sem determinismo, dois runs sobre a mesma entrada produziriam mapas
# diferentes, e nenhuma comparação de experimento seria válida.
def test_empate_e_resolvido_de_forma_deterministica() -> None:
    """Confere o desempate por suporte, instante e identificador de região."""
    primeiro = _contribution("frame-a", "parede", timestamp_ns=10)
    segundo = _contribution("frame-b", "porta", timestamp_ns=20)
    assert fuse_point_contributions([primeiro, segundo]).label == "parede"
    assert fuse_point_contributions([segundo, primeiro]).label == "parede"
    com_suporte = _contribution("frame-b", "porta", timestamp_ns=20, visual_support=0.9)
    assert fuse_point_contributions([primeiro, com_suporte]).label == "porta"


# Entradas impossíveis precisam falhar na fronteira, e não produzir um label
# vazio que o viewer apresentaria como classificação.
def test_entradas_invalidas_falham_na_fronteira() -> None:
    """Confere validação de lista vazia, label vazio e medida fora de faixa."""
    with pytest.raises(ValueError, match="at least one contribution"):
        fuse_point_contributions([])
    with pytest.raises(ValueError, match="label must not be empty"):
        _contribution("frame-a", "   ")
    with pytest.raises(ValueError, match="confidence"):
        _contribution("frame-a", "porta", confidence=1.5)
