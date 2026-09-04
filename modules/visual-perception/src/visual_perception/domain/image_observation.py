"""Contract de entrada canônico de observação de imagem.

Issue: #153.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception.domain.references import ObservationReference, SourceArtifactReference

_SUPPORTED_ENCODINGS = frozenset({"rgb8", "bgr8", "rgba8"})


# Representa um frame RGB(-like) de entrada consumido pela percepção visual.
# Existe como o contract de entrada agnóstico de implementação do módulo,
# desacoplando o restante do pipeline de qualquer tipo concreto de
# imagem/tensor.
@dataclass(frozen=True)
class ImageObservation:
    """Um frame RGB(-like) consumido pela percepção visual.

    Agnóstico de implementação: carrega uma :class:`SourceArtifactReference`
    para o payload de pixels em vez de um tipo concreto de tensor/array,
    então este contract é serializável e não depende de nenhuma biblioteca
    de imaging. Identidade, timing e proveniência de frame vivem em
    ``source`` (a :class:`ObservationReference` compartilhada), sem
    duplicação aqui.
    """

    width: int
    height: int
    encoding: str
    image: SourceArtifactReference
    source: ObservationReference

    # Valida que width/height são positivos e que o encoding é um dos
    # formatos suportados, falhando cedo em uma observação malformada.
    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")
        if self.encoding not in _SUPPORTED_ENCODINGS:
            raise ValueError(
                f"encoding must be one of {sorted(_SUPPORTED_ENCODINGS)}, got {self.encoding!r}."
            )

    # Expõe o observation_id a partir de source, sem duplicar o campo no
    # próprio ImageObservation.
    @property
    def observation_id(self) -> str:
        return str(self.source.observation_id)
