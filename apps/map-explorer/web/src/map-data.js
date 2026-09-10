const UNOBSERVED_KEY = "__unobserved__";
const UNOBSERVED_COLOR = [28, 31, 38];
const OBSERVED_UNLABELED_KEY = "__observed_unlabeled__";
const OBSERVED_UNLABELED_COLOR = [104, 111, 124];
const DIMMED_COLOR = [70, 74, 84];

const NEUTRAL_LABELS = Object.freeze({
  [UNOBSERVED_KEY]: "Sem contexto",
  [OBSERVED_UNLABELED_KEY]: "Observado, sem label",
});

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

// Resolve a evidência visual de um ponto tolerando as duas composições de
// artifact: a associação anexada por scan e o contexto gravado no próprio ponto
// do mapa. Existe para que legenda, cor e inspeção não precisem saber qual
// composição gerou o arquivo aberto.
export function visualEvidence(point) {
  return point.association ?? point.context ?? null;
}

// Resolve a observação de origem entre os dois nomes que os artifacts usaram:
// a composição por scan publicava ``rgb_observation_id``, e a ancorada no mapa
// publica ``observation_id``. Existe para que o inspector abra a evidência dos
// dois formatos sem ramificar em cada uso.
export function observationIdOf(evidence) {
  return evidence?.observation_id ?? evidence?.rgb_observation_id ?? null;
}

// Decide se a câmera realmente enxergou o ponto. Uma evidência rejeitada
// (ocluída, fora da imagem, fora do suporte válido) descreve por que o ponto
// não foi observado, e por isso não conta como observação.
function isVisuallyObserved(evidence) {
  if (!evidence) return false;
  if (evidence.status) return evidence.status === "associated";
  return Boolean(evidence.color_rgb || evidence.pixel);
}

// Classifica cada ponto em uma de três categorias, porque "a câmera viu e não
// classificou" e "a câmera nunca viu" são diagnósticos diferentes: o primeiro
// aponta para a máscara ou para o reasoner, o segundo para cobertura de frames.
export function contextKey(point) {
  const evidence = visualEvidence(point);
  if (evidence?.label) return evidence.label.trim().toLowerCase();
  if (isVisuallyObserved(evidence)) return OBSERVED_UNLABELED_KEY;
  return UNOBSERVED_KEY;
}

// Centraliza a cor de uma categoria para que legenda e nuvem nunca divirjam.
function colorForKey(key) {
  if (key === UNOBSERVED_KEY) return UNOBSERVED_COLOR;
  if (key === OBSERVED_UNLABELED_KEY) return OBSERVED_UNLABELED_COLOR;
  return semanticColor(key);
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
      label: NEUTRAL_LABELS[key] ?? key,
      count: 0,
      color: colorForKey(key),
      semantic: !(key in NEUTRAL_LABELS),
    };
    current.count += 1;
    byKey.set(key, current);
  });
  return [...byKey.values()].sort((left, right) => {
    if (left.semantic !== right.semantic) return left.semantic ? -1 : 1;
    return right.count - left.count || left.label.localeCompare(right.label);
  });
}

// Separa o mapa entre a classe em foco e o restante, em vez de descartar
// pontos. A legenda responde "onde está esta classe dentro do mapa", e isso
// exige que a geometria vizinha continue desenhada como referência espacial.
export function partitionByFocus(points, enabledContextKeys) {
  if (enabledContextKeys === null) return { focused: points, dimmed: [] };
  const focused = [];
  const dimmed = [];
  points.forEach((point) => {
    if (enabledContextKeys.has(contextKey(point))) focused.push(point);
    else dimmed.push(point);
  });
  return { focused, dimmed };
}

// Retorna exclusivamente a cor da classe contextual usada pelo mapa.
export function pointColor(point) {
  return colorForKey(contextKey(point));
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

// Valida o índice de mapas publicado pelo servidor local. O seletor precisa
// refletir o que existe no diretório servido, e não uma lista fixa no código,
// mas um índice malformado nunca pode impedir a abertura do mapa atual.
export function mapEntriesFromIndex(payload) {
  if (!Array.isArray(payload)) return [];
  return payload.filter(
    (entry) => typeof entry?.url === "string" && typeof entry?.label === "string",
  ).map((entry) => ({ url: entry.url, label: entry.label }));
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

export { DIMMED_COLOR, OBSERVED_UNLABELED_KEY, UNOBSERVED_KEY };
