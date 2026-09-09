const UNOBSERVED_KEY = "__unobserved__";
const UNOBSERVED_COLOR = [28, 31, 38];

const SEMANTIC_COLORS = Object.freeze({
  door: [255, 82, 82],
  wall: [45, 181, 255],
  "plain wall": [92, 124, 250],
  floor: [255, 214, 51],
  "wooden floor": [255, 145, 48],
  ceiling: [190, 118, 255],
  "ceiling tiles": [255, 105, 180],
  "wooden panel": [61, 220, 132],
});

const FALLBACK_COLORS = Object.freeze([
  [0, 229, 255],
  [255, 112, 67],
  [174, 234, 0],
  [255, 64, 129],
  [124, 77, 255],
  [255, 171, 0],
  [29, 233, 182],
  [236, 64, 122],
]);

// Valida a fronteira mínima consumida pelo viewer antes de qualquer estado de
// câmera ou renderização ser criado.
export function validateSlice(value) {
  if (![1, 2].includes(value?.schema_version)) {
    throw new Error("O artifact precisa usar schema_version 1 ou 2.");
  }
  if (typeof value.map_id !== "string" || typeof value.map_frame !== "string") {
    throw new Error("O artifact precisa declarar map_id e map_frame.");
  }
  if (!Array.isArray(value.points)) {
    throw new Error("O artifact precisa conter uma lista points.");
  }
  value.points.forEach((point, index) => {
    if (!Array.isArray(point.coordinates_m) || point.coordinates_m.length !== 3) {
      throw new Error(`O ponto ${index} não possui coordinates_m tridimensional.`);
    }
    if (point.coordinates_m.some((coordinate) => !Number.isFinite(coordinate))) {
      throw new Error(`O ponto ${index} possui coordenadas não finitas.`);
    }
  });
  if (
    value.schema_version === 2
    && (!Array.isArray(value.observations) || !Array.isArray(value.regions))
  ) {
    throw new Error("O artifact contextual precisa declarar observations e regions.");
  }
  return value;
}

// Resume bounds e escala física uma única vez ao abrir o mapa. A câmera usa
// esta informação estável, nunca a seleção ou o subconjunto filtrado.
export function measureMap(points) {
  if (!points.length) {
    return {
      minimum: [-1, -1, -1],
      maximum: [1, 1, 1],
      center: [0, 0, 0],
      diagonal: 2 * Math.sqrt(3),
      minimumHeight: -1,
      maximumHeight: 1,
    };
  }
  const minimum = [...points[0].coordinates_m];
  const maximum = [...points[0].coordinates_m];
  points.forEach((point) => {
    point.coordinates_m.forEach((coordinate, axis) => {
      minimum[axis] = Math.min(minimum[axis], coordinate);
      maximum[axis] = Math.max(maximum[axis], coordinate);
    });
  });
  const center = minimum.map((value, axis) => (value + maximum[axis]) / 2);
  const diagonal = Math.max(
    Math.sqrt(minimum.reduce((sum, value, axis) => sum + (maximum[axis] - value) ** 2, 0)),
    0.001,
  );
  return {
    minimum,
    maximum,
    center,
    diagonal,
    minimumHeight: minimum[2],
    maximumHeight: maximum[2],
  };
}

// Classifica pontos sem label separadamente para que ausência de contexto não
// desapareça da legenda nem pareça uma categoria semântica.
export function contextKey(point) {
  if (point.association?.label) return point.association.label.trim().toLowerCase();
  return UNOBSERVED_KEY;
}

// Escolhe uma cor semântica saturada e estável. Labels conhecidas preservam
// convenções visuais e labels futuras recebem um fallback determinístico.
export function semanticColor(label) {
  const normalized = label?.trim().toLowerCase();
  if (!normalized) return UNOBSERVED_COLOR;
  if (SEMANTIC_COLORS[normalized]) return SEMANTIC_COLORS[normalized];
  let hash = 2166136261;
  for (const character of normalized) {
    hash ^= character.codePointAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return FALLBACK_COLORS[(hash >>> 0) % FALLBACK_COLORS.length];
}

// Gera uma legenda ordenada por cobertura e preserva as cores publicadas pelo
// viewer. A ausência de label é uma só categoria, independentemente de existir
// uma associação RGB que não produziu classificação.
export function buildContextLegend(points) {
  const byKey = new Map();
  points.forEach((point) => {
    const key = contextKey(point);
    const current = byKey.get(key) ?? {
      key,
      label: key,
      count: 0,
      color: semanticColor(key),
      semantic: true,
    };
    current.count += 1;
    byKey.set(key, current);
  });
  if (byKey.has(UNOBSERVED_KEY)) {
    Object.assign(byKey.get(UNOBSERVED_KEY), {
      label: "Sem contexto",
      color: UNOBSERVED_COLOR,
      semantic: false,
    });
  }
  return [...byKey.values()].sort((left, right) => {
    if (left.semantic !== right.semantic) return left.semantic ? -1 : 1;
    return right.count - left.count || left.label.localeCompare(right.label);
  });
}

// Aplica os filtros somente à camada de contexto. Geometria e RGB continuam
// completas para que uma legenda semântica não altere silenciosamente a base.
export function filterPoints(points, enabledContextKeys) {
  if (enabledContextKeys === null) return points;
  return points.filter((point) => enabledContextKeys.has(contextKey(point)));
}

// Retorna exclusivamente a cor da classe contextual usada pelo mapa.
export function pointColor(point) {
  return semanticColor(contextKey(point) === UNOBSERVED_KEY ? null : contextKey(point));
}

// Converte canais sRGB de interface para o espaço linear esperado pelos
// vertex colors do Three.js, evitando cores claras e lavadas no canvas.
export function srgbColorToLinear(color) {
  return color.map((channel) => {
    const normalized = channel / 255;
    return normalized <= 0.04045
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  });
}

// Encapsula o artifact atual como uma fonte de geometria estática. A mesma
// fronteira poderá receber bounds/LOD da API sem mudar os componentes visuais.
export class StaticArtifactGeometrySource {
  // Retém o artifact validado e declara que esta implementação ainda não
  // oferece demanda espacial nem níveis progressivos.
  constructor(slice) {
    this.slice = slice;
    this.capabilities = Object.freeze({ boundedGeometry: false, levelsOfDetail: false });
  }

  // Mantém assinatura assíncrona e cancelável compatível com uma futura fonte
  // de chunks, embora o artifact local devolva um único chunk completo.
  async getGeometry({ signal } = {}) {
    if (signal?.aborted) throw new DOMException("Operação cancelada.", "AbortError");
    return {
      chunks: [{ chunkId: `${this.slice.map_id}:static`, points: this.slice.points }],
      complete: true,
    };
  }
}

export { UNOBSERVED_KEY };
