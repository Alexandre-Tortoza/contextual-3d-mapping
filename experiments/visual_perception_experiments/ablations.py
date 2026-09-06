"""Matriz de configurações comparadas pela ablation de percepção visual.

Issue: #200 (ablation), sobre os mecanismos das #191-#196.

Cada configuração isola *um* mecanismo em relação ao baseline, para que a
diferença medida possa ser atribuída a ele. As combinações só aparecem
depois dos isolados, e apenas as que fazem sentido conjunto.

O baseline não é "os defaults do módulo": é a configuração que representa o
estado anterior a esta fase — pooling em grade de patches, sem evidência
contextual, sem calibração. Sem esse ponto fixo, "melhorou" não teria
referência.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from visual_perception.application.execution_profile import research_quality_config
from visual_perception.config import CalibrationConfig, ModuleConfig, MultiContextConfig


# Nomeia uma configuração da matriz junto do mecanismo que ela isola. Existe
# para que o relatório de ablation possa dizer *o que* cada linha testa, em
# vez de só mostrar um nome de configuração.
@dataclass(frozen=True)
class Ablation:
    """Uma configuração da matriz, com o mecanismo que ela isola."""

    name: str
    mechanism: str
    config: ModuleConfig

    # Valida que nome e mecanismo estão presentes: uma linha de ablation sem
    # mecanismo declarado não é interpretável em um relatório.
    def __post_init__(self) -> None:
        """Rejeita configurações sem nome ou sem mecanismo declarado."""
        if not self.name or not self.mechanism:
            raise ValueError("An ablation must declare both a name and the mechanism it isolates.")


# Constrói a configuração baseline da ablation: o estado anterior a esta
# fase de pesquisa. Existe como o ponto fixo contra o qual todo mecanismo é
# comparado.
def baseline_config(*, real_backends: bool, gpu_memory_budget_gb: float = 8.0) -> ModuleConfig:
    """Retorna a configuração baseline: grade de patches, sem contexto, sem calibração."""
    config = research_quality_config(
        multi_scale_justified=False,
        gpu_memory_budget_gb=gpu_memory_budget_gb,
        real_backends=real_backends,
    )
    return dataclasses.replace(
        config,
        feature_extraction=dataclasses.replace(config.feature_extraction, upsampling="patch_grid"),
        multi_context=MultiContextConfig(
            contextual_crop_enabled=False, scene_conditioned_enabled=False
        ),
        calibration=CalibrationConfig(enabled=False),
    )


# Enumera a matriz completa de ablation. Cada entrada muda exatamente um
# mecanismo em relação ao baseline, exceto as combinações finais, que são
# declaradas como tais. Chamada pelo runner de predições e pelo relatório.
def ablation_matrix(
    *,
    real_backends: bool,
    calibration_artifact: Path | None = None,
    calibration_domain: str = "indoor_corridor",
    gpu_memory_budget_gb: float = 8.0,
) -> tuple[Ablation, ...]:
    """Constrói a matriz de ablation completa.

    Argumentos:
        real_backends: usa os backends reais selecionados por benchmark (#174).
        calibration_artifact: tabela de calibração medida; sem ela as
            configurações calibradas são omitidas em vez de rodarem sem dados.
        calibration_domain: domínio declarado da execução calibrada.
        gpu_memory_budget_gb: budget de memória da GPU de referência.
    Retorna:
        as configurações a comparar, começando pelo baseline.
    """
    base = baseline_config(real_backends=real_backends, gpu_memory_budget_gb=gpu_memory_budget_gb)
    entries = [
        Ablation("baseline", "patch-grid pooling, no context evidence, no calibration", base),
        Ablation(
            "high_resolution_nearest",
            "pixel-aligned dense evidence sampled with nearest neighbour (#191/#192)",
            _with_upsampling(base, "nearest"),
        ),
        Ablation(
            "high_resolution_bilinear",
            "pixel-aligned dense evidence sampled bilinearly (#191/#192)",
            _with_upsampling(base, "bilinear"),
        ),
        Ablation(
            "multi_context",
            "contextual crop and global scene evidence slots (#193/#194)",
            _with_multi_context(base),
        ),
    ]
    if calibration_artifact is not None:
        calibrated = _with_calibration(base, calibration_artifact, calibration_domain)
        entries.append(
            Ablation(
                "calibrated_semantics",
                "calibrated claim scoring and abstention (#195/#196)",
                calibrated,
            )
        )
        entries.append(
            Ablation(
                "high_resolution_and_calibrated",
                "pixel-aligned evidence combined with calibrated scoring",
                _with_upsampling(calibrated, "bilinear"),
            )
        )
        entries.append(
            Ablation(
                "all_mechanisms",
                "pixel-aligned evidence, multi-context slots and calibrated scoring together",
                _with_multi_context(_with_upsampling(calibrated, "bilinear")),
            )
        )
    return tuple(entries)


# Itera a matriz produzindo apenas as configurações pedidas pelo chamador.
# Existe para que uma execução parcial (por exemplo, só os caminhos densos)
# use exatamente as mesmas configurações da execução completa.
def selected_ablations(matrix: tuple[Ablation, ...], names: tuple[str, ...]) -> Iterator[Ablation]:
    """Percorre ``matrix`` filtrando por ``names``, preservando a ordem da matriz."""
    if not names:
        yield from matrix
        return
    known = {ablation.name for ablation in matrix}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown ablation names {unknown}; the matrix declares {sorted(known)}.")
    for ablation in matrix:
        if ablation.name in names:
            yield ablation


# Troca a regra de amostragem densa mantendo todo o resto igual.
def _with_upsampling(config: ModuleConfig, upsampling: str) -> ModuleConfig:
    """Retorna ``config`` com outra regra de amostragem de evidência densa."""
    return dataclasses.replace(
        config,
        feature_extraction=dataclasses.replace(
            config.feature_extraction, upsampling=upsampling
        ),
    )


# Liga os slots de evidência contextual mantendo todo o resto igual.
def _with_multi_context(config: ModuleConfig) -> ModuleConfig:
    """Retorna ``config`` com os slots de crop contextual e de cena habilitados."""
    return dataclasses.replace(
        config,
        multi_context=MultiContextConfig(
            contextual_crop_enabled=True, scene_conditioned_enabled=True, context_expansion=0.25
        ),
    )


# Liga a calibração apontando para o artifact medido.
def _with_calibration(config: ModuleConfig, artifact: Path, domain: str) -> ModuleConfig:
    """Retorna ``config`` com a calibração habilitada sobre ``artifact``."""
    return dataclasses.replace(
        config,
        calibration=CalibrationConfig(enabled=True, artifact_path=str(artifact), domain=domain),
    )


__all__ = ["Ablation", "ablation_matrix", "baseline_config", "selected_ablations"]
