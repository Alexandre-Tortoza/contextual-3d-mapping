"""Contracts públicos de geometric-map."""

from __future__ import annotations

from dataclasses import dataclass

from contextual_mapping_contracts import FrameId, MapId, ObservationReference, Provenance


# Identifica uma geometria persistente sem expor o formato interno do mapa.
# Sensor-association e aplicações usam esta referência para apontar pontos estáveis.
@dataclass(frozen=True, order=True)
class GeometryReference:
    """Referência estável a um ponto persistido em um mapa.

    Argumentos:
        map_id: mapa proprietário da geometria.
        geometry_id: identidade única dentro do mapa.
    """

    map_id: MapId
    geometry_id: str

    def __post_init__(self) -> None:
        if not self.geometry_id.strip():
            raise ValueError("geometry_id must not be empty.")


# Guarda um ponto no frame global junto à cadeia mínima de observações que o
# originou. O mapa é dono deste estado; evidência visual permanece referenciada.
@dataclass(frozen=True)
class GeometryPoint:
    """Ponto persistido no frame do mapa com proveniência.

    Argumentos:
        reference: identidade estável do ponto persistido.
        coordinates_m: coordenadas x, y, z no frame do mapa.
        source_observation: observação LiDAR de origem.
        provenance: pose e observações que produziram o ponto.
    """

    reference: GeometryReference
    coordinates_m: tuple[float, float, float]
    source_observation: ObservationReference
    provenance: Provenance

    def __post_init__(self) -> None:
        if len(self.coordinates_m) != 3:
            raise ValueError("coordinates_m must contain exactly three values.")


# Representa bounds explícitos de busca para não vazar estruturas de índice.
@dataclass(frozen=True)
class Bounds3D:
    """Bounds inclusivos em metros no frame informado.

    Argumentos:
        frame_id: frame das coordenadas.
        minimum_m: canto mínimo x, y, z.
        maximum_m: canto máximo x, y, z.
    """

    frame_id: FrameId
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.minimum_m) != 3 or len(self.maximum_m) != 3:
            raise ValueError("bounds must contain xyz triples.")
        if any(low > high for low, high in zip(self.minimum_m, self.maximum_m, strict=True)):
            raise ValueError("minimum_m must not exceed maximum_m.")

    # Decide se um ponto pertence ao volume solicitado; usada pelo estado
    # em memória e mantém a regra de borda inclusiva em um único local.
    def contains(self, coordinates_m: tuple[float, float, float]) -> bool:
        """Retorna se coordenadas pertencem aos bounds inclusivos."""
        return all(low <= value <= high for value, low, high in zip(coordinates_m, self.minimum_m, self.maximum_m, strict=True))
