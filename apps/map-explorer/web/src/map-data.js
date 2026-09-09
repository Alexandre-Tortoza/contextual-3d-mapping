const UNOBSERVED_KEY = "__unobserved__";
const RGB_ONLY_KEY = "__rgb_only__";

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
  if (point.association?.label) return point.association.label;
  if (point.association?.color_rgb) return RGB_ONLY_KEY;
  return UNOBSERVED_KEY;
}

// Gera uma legenda ordenada por cobertura e preserva as cores publicadas pelo
// artifact. Categorias especiais usam cores neutras e nomes explícitos.
export function buildContextLegend(points) {
  const byKey = new Map();
  points.forEach((point) => {
    const key = contextKey(point);
    const current = byKey.get(key) ?? {
      key,
      label: key,
      count: 0,
      color: point.association?.semantic_color_rgb ?? [48, 53, 64],
      semantic: true,
    };
    current.count += 1;
    byKey.set(key, current);
  });
  if (byKey.has(RGB_ONLY_KEY)) {
    Object.assign(byKey.get(RGB_ONLY_KEY), {
      label: "RGB sem label",
      color: [91, 107, 128],
      semantic: false,
    });
  }
  if (byKey.has(UNOBSERVED_KEY)) {
    Object.assign(byKey.get(UNOBSERVED_KEY), {
      label: "Sem observação",
      color: [48, 53, 64],
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
export function filterPoints(points, colorMode, enabledContextKeys) {
  if (colorMode !== "context" || enabledContextKeys === null) return points;
  return points.filter((point) => enabledContextKeys.has(contextKey(point)));
}

// Escolhe a cor sem misturar altura, cor física e claim semântico.
export function pointColor(point, colorMode) {
  if (colorMode === "context") {
    return point.association?.semantic_color_rgb ?? [48, 53, 64];
  }
  if (colorMode === "rgb") {
    return point.association?.color_rgb ?? [48, 53, 64];
  }
  return point.display_color_rgb ?? [120, 130, 145];
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

export { RGB_ONLY_KEY, UNOBSERVED_KEY };
