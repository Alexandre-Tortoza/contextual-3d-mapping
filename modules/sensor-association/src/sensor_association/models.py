"""Contracts públicos de calibração e associação RGB–LiDAR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from geometric_map import GeometryReference

from contextual_mapping_contracts import ObservationReference, RigidTransform, SourceArtifactReference


# Enumera modelos de projeção suportados sem deixar cada caller inferir a
# semântica dos parâmetros. O primeiro slice usa pinhole ou equidistant fisheye.
class CameraModel(StrEnum):
    """Modelo geométrico usado para projetar pontos 3D na imagem."""

    PINHOLE = "pinhole"
    EQUIDISTANT_FISHEYE = "equidistant_fisheye"
    MEI = "mei"


# Agrupa intrínsecos e extrínseca de câmera–LiDAR com uma referência externa
# auditável. Sensor-association é dono deste contract, não visual-perception.
@dataclass(frozen=True)
class CameraLidarCalibration:
    """Calibração câmera–LiDAR usada por projeções rastreáveis.

    Argumentos:
        calibration_id: identidade estável da calibração.
        artifact: arquivo de origem versionado da calibração.
        model: modelo de projeção da câmera.
        fx, fy, cx, cy: intrínsecos em pixels.
        lidar_to_camera: extrínseca do frame LiDAR para o frame da câmera.
        mirror_xi: parâmetro do espelho no modelo unificado MEI.
        distortion_k1, distortion_k2: distorção radial do modelo MEI.
        distortion_p1, distortion_p2: distorção tangencial do modelo MEI.
        front_hemisphere_only: restringe a projeção a raios com profundidade
            positiva no eixo óptico da câmera.
    """

    calibration_id: str
    artifact: SourceArtifactReference
    model: CameraModel
    fx: float
    fy: float
    cx: float
    cy: float
    lidar_to_camera: RigidTransform
    mirror_xi: float | None = None
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    distortion_p1: float = 0.0
    distortion_p2: float = 0.0
    front_hemisphere_only: bool = True

    # Rejeita calibrações sem identidade ou intrínsecos não físicos antes da
    # projeção, onde o erro seria mais difícil de diagnosticar.
    def __post_init__(self) -> None:
        """Valida identidade e parâmetros intrínsecos da calibração."""
        if not self.calibration_id.strip():
            raise ValueError("calibration_id must not be empty.")
        intrinsics = (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.distortion_k1,
            self.distortion_k2,
            self.distortion_p1,
            self.distortion_p2,
        )
        if not all(isfinite(float(value)) for value in intrinsics):
            raise ValueError("camera intrinsics must be finite.")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive.")
        if not isinstance(self.front_hemisphere_only, bool):
            raise TypeError("front_hemisphere_only must be a boolean.")
        if self.model is CameraModel.MEI:
            if self.mirror_xi is None or not isfinite(float(self.mirror_xi)):
                raise ValueError("MEI calibration requires a finite mirror_xi.")
        elif self.mirror_xi is not None:
            raise ValueError("mirror_xi is only valid for MEI calibration.")


# Carrega os pixels RGB e suporte válido necessários para colorização sem
# acoplar association ao decoder de imagem ou ao backend visual.
@dataclass(frozen=True)
class RgbFrame:
    """Pixels RGB e suporte geométrico de uma observação de câmera.

    Argumentos:
        reference: observação RGB de origem.
        width, height: resolução dos pixels.
        pixels: pixels RGB em ordem de linhas.
        valid_pixels: pixels que pertencem ao suporte fisheye válido.
    """

    reference: ObservationReference
    width: int
    height: int
    pixels: tuple[tuple[int, int, int], ...]
    valid_pixels: frozenset[tuple[int, int]]

    # Confirma que resolução, canais e suporte válido pertencem ao mesmo frame.
    def __post_init__(self) -> None:
        """Valida resolução, pixels RGB e suporte geométrico."""
        if self.width <= 0 or self.height <= 0 or len(self.pixels) != self.width * self.height:
            raise ValueError("RGB dimensions and pixel count must agree.")
        if any(
            len(pixel) != 3
            or any(
                isinstance(channel, bool) or not isinstance(channel, int) or channel < 0 or channel > 255
                for channel in pixel
            )
            for pixel in self.pixels
        ):
            raise ValueError("RGB channels must be in [0, 255].")
        if any(not (0 <= x < self.width and 0 <= y < self.height) for x, y in self.valid_pixels):
            raise ValueError("valid_pixels must stay inside the RGB frame.")

    # Lê uma cor usando a convenção top-left de visual-perception. Existe
    # para centralizar a conversão de coordenada 2D para índice linear.
    def color_at(self, pixel: tuple[int, int]) -> tuple[int, int, int]:
        """Retorna a cor RGB no pixel validado."""
        x, y = pixel
        return self.pixels[y * self.width + x]


# Representa a evidência visual já canonizada, sem duplicar geometrias de
# máscara ou carregar objetos privados de visual-perception na associação.
@dataclass(frozen=True)
class VisualRegionEvidence:
    """Região visual canônica representada por pixels e referências.

    Argumentos:
        region_id: identidade estável da região visual.
        pixels: footprint após grounding, validade e ownership; nunca discovery implícita.
        label: claim visual bruto opcional.
        feature_reference: referência opcional a dense feature ou embedding.
        grounding_status: refined autoriza evidência espacial; outros estados
            continuam tentativos. legacy_discovery exige ablação explícita.
        grounding_reference: referência à geometria e proveniência do grounding.
    """

    region_id: str
    pixels: frozenset[tuple[int, int]]
    label: str | None = None
    feature_reference: str | None = None
    grounding_status: str = "grounding_unavailable"
    grounding_reference: str | None = None

    # Mantém a região identificável e exige coordenadas inteiras não negativas,
    # pois o footprint é indexado diretamente no raster RGB.
    def __post_init__(self) -> None:
        """Valida identidade e coordenadas básicas da região visual."""
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty.")
        if any(
            len(pixel) != 2
            or any(isinstance(coordinate, bool) or not isinstance(coordinate, int) for coordinate in pixel)
            for pixel in self.pixels
        ):
            raise ValueError("region pixels must use integer coordinates.")
        if any(x < 0 or y < 0 for x, y in self.pixels):
            raise ValueError("region pixels must be non-negative.")
        if self.grounding_status == "refined" and not self.grounding_reference:
            raise ValueError("Refined evidence requires an auditable grounding reference.")


# Separa visibilidade geométrica de força semântica: uma boundary ainda tem
# cor RGB válida, mas seu conceito não deve entrar como contribuição forte.
class SemanticAssociationStatus(StrEnum):
    """Relação do pixel projetado com o suporte semântico disponível."""

    INTERIOR = "interior"
    BOUNDARY = "boundary"
    OUTSIDE = "outside"
    UNGROUNDED = "ungrounded"
    LEGACY = "legacy_discovery"
    SURFACE_UNSUPPORTED = "surface_unsupported"


# Representa um ponto do mapa persistente como entrada de associação. Existe
# porque um ponto acumulado não pertence a um único scan: ele não tem
# coordenada no frame do LiDAR nem observação de origem única, que é o que
# ``GeometryPoint`` exige. É consumido por ``associate_map_points``.
@dataclass(frozen=True)
class MapAnchoredPoint:
    """Ponto de geometria persistente candidato à associação visual.

    Argumentos:
        geometry: identidade estável do ponto dentro do mapa.
        coordinates_m: coordenadas no frame do mapa, em metros.
    """

    geometry: GeometryReference
    coordinates_m: tuple[float, float, float]

    # Rejeita coordenadas não físicas antes da projeção, onde o erro apareceria
    # como um pixel arbitrário em vez de dado inválido.
    def __post_init__(self) -> None:
        """Valida dimensão e finitude das coordenadas do ponto."""
        if len(self.coordinates_m) != 3:
            raise ValueError("coordinates_m must contain exactly three values.")
        if not all(isfinite(float(value)) for value in self.coordinates_m):
            raise ValueError("coordinates_m must be finite.")


# Explicita o resultado de toda tentativa de associação, inclusive rejeições.
# Isso torna pixels inválidos e oclusões depuráveis no viewer e em artifacts.
class AssociationStatus(StrEnum):
    """Estado observável de uma tentativa de associação ponto–imagem."""

    ASSOCIATED = "associated"
    BEHIND_CAMERA = "behind_camera"
    OUTSIDE_IMAGE = "outside_image"
    OUTSIDE_VALID_SUPPORT = "outside_valid_support"
    OCCLUDED = "occluded"
    VISIBILITY_UNCONFIRMED = "visibility_unconfirmed"


# Explica decisões geométricas mesmo quando não há evidência visual aceita.
# Distâncias são radiais em metros, no frame óptico da observação RGB.
@dataclass(frozen=True)
class SurfaceAssociationEvidence:
    """Primeira interseção medida e vínculo semântico, sem criar geometria."""

    point_depth_m: float
    first_surface_depth_m: float | None
    tolerance_m: float
    source_index: int | None
    reason: str
    surface_support: str = "not_evaluated"
    component_id: int | None = None
    support_reason: str | None = None

    # Rejeita medidas não físicas antes da serialização dos diagnostics.
    def __post_init__(self) -> None:
        """Valida profundidades, tolerância e índice de origem."""
        for value in (self.point_depth_m, self.first_surface_depth_m, self.tolerance_m):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("Surface depths and tolerance must be finite and non-negative.")
        if self.source_index is not None and self.source_index < 0:
            raise ValueError("Surface source_index must be non-negative.")


# É a saída pública ancorada em geometria persistente para o primeiro slice.
# Semantic fusion poderá consumir múltiplos resultados sem mudar este contract.
@dataclass(frozen=True)
class PointVisualAssociation:
    """Correspondência rastreável entre ponto persistido e evidência RGB.

    Argumentos:
        geometry: ponto persistido associado.
        lidar_observation: observação LiDAR de origem.
        rgb_observation: frame RGB de origem.
        calibration: calibração usada.
        status: resultado da projeção e visibilidade.
        pixel: pixel associado quando visível.
        color_rgb: cor amostrada quando visível.
        region_id: região visual opcional.
        label: claim visual bruto opcional.
        feature_reference: referência visual opcional.
    """

    geometry: GeometryReference
    lidar_observation: ObservationReference
    rgb_observation: ObservationReference
    calibration: CameraLidarCalibration
    status: AssociationStatus
    pixel: tuple[int, int] | None = None
    color_rgb: tuple[int, int, int] | None = None
    region_id: str | None = None
    label: str | None = None
    feature_reference: str | None = None
    semantic_status: SemanticAssociationStatus = SemanticAssociationStatus.OUTSIDE
    tentative_label: str | None = None
    distance_to_boundary_px: float | None = None
    boundary_margin_px: float | None = None
    grounding_reference: str | None = None
    surface_evidence: SurfaceAssociationEvidence | None = None

    # Mantém o payload coerente com o status para que rejeições não carreguem
    # evidência visual residual e associações sempre tenham pixel e cor.
    def __post_init__(self) -> None:
        """Valida a coerência entre status e evidência associada."""
        if self.status is AssociationStatus.ASSOCIATED and (self.pixel is None or self.color_rgb is None):
            raise ValueError("associated points require pixel and color_rgb.")
        if self.status is not AssociationStatus.ASSOCIATED and any(
            value is not None
            for value in (
                self.pixel,
                self.color_rgb,
                self.region_id,
                self.label,
                self.feature_reference,
                self.tentative_label,
                self.distance_to_boundary_px,
                self.boundary_margin_px,
                self.grounding_reference,
            )
        ):
            raise ValueError("rejected points must not carry visual evidence.")
        if self.label is not None and self.semantic_status not in (
            SemanticAssociationStatus.INTERIOR, SemanticAssociationStatus.LEGACY,
        ):
            raise ValueError("Strong labels require semantic interior or explicit legacy ablation.")
        if self.label is not None and self.semantic_status is SemanticAssociationStatus.INTERIOR and not self.grounding_reference:
            raise ValueError("Semantic interior labels require a grounding reference.")
        if self.label is not None and self.surface_evidence is not None and self.surface_evidence.surface_support != "supported":
            raise ValueError("Measured surface labels require explicit surface support.")
        for value in (self.distance_to_boundary_px, self.boundary_margin_px):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("Boundary measurements must be finite and non-negative.")
