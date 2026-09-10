const UNOBSERVED_KEY = "__unobserved__";
const UNOBSERVED_COLOR = [28, 31, 38];
const OBSERVED_UNLABELED_KEY = "__observed_unlabeled__";
const OBSERVED_UNLABELED_COLOR = [104, 111, 124];
const OTHER_FAMILY_KEY = "__other__";
const OTHER_FAMILY_COLOR = [185, 189, 199];
const DIMMED_COLOR = [70, 74, 84];

// Quatro matizes, nesta ordem, e nenhum a mais.
//
// Uma nuvem de pontos é um caso "all-pairs": qualquer classe pode encostar em
// qualquer outra no espaço, então toda combinação precisa ser distinguível — e
// não apenas as vizinhas de uma legenda ordenada. Sob essa restrição, contra o
// fundo #0d0d0d deste viewer, apenas dois conjuntos de quatro matizes passam nos
// limiares de separação para visão normal e para daltonismo; nenhum conjunto de
// cinco passa. Famílias além da quarta recebem um neutro claro em vez de um
// matiz gerado que o leitor não conseguiria separar dos demais.
const FAMILY_COLORS = Object.freeze([
  [57, 135, 229],
  [201, 133, 0],
  [213, 81, 129],
  [0, 131, 0],
]);

const NEUTRAL_LABELS = Object.freeze({
  [UNOBSERVED_KEY]: "Sem contexto",
  [OBSERVED_UNLABELED_KEY]: "Observado, sem label",
  [OTHER_FAMILY_KEY]: "Outros labels",
});

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

// Verifica se uma sequência de tokens aparece inteira e contígua dentro de
// outra. É a relação que define parentesco entre labels: "wall" aparece dentro
// de "plain wall" e de "wall tiles".
function containsTokens(tokens, needle) {
  if (needle.length >= tokens.length) return false;
  for (let start = 0; start + needle.length <= tokens.length; start += 1) {
    if (needle.every((token, offset) => tokens[start + offset] === token)) return true;
  }
  return false;
}

// Agrupa labels de vocabulário aberto em famílias derivadas dos próprios dados.
//
// Existe porque o reasoner produz quase-sinônimos — "wall", "plain wall" e
// "wall tiles" descrevem a mesma superfície — e tratá-los como categorias
// independentes gasta matizes que o leitor não consegue separar. Um label entra
// na família do label mais frequente que aparece inteiro dentro dele, o que
// dispensa uma tabela escrita à mão e funciona em um dataset novo.
export function buildLabelFamilies(labelCounts) {
  const ordered = [...labelCounts.entries()].sort(
    (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
  );
  const host = new Map();
  ordered.forEach(([label]) => {
    const tokens = label.split(/\s+/);
    const parent = ordered.find(
      ([candidate]) => candidate !== label && containsTokens(tokens, candidate.split(/\s+/)),
    );
    host.set(label, parent ? parent[0] : label);
  });
  const families = new Map();
  host.forEach((_, label) => {
    const seen = new Set();
    let root = label;
    while (host.get(root) !== root && !seen.has(root)) {
      seen.add(root);
      root = host.get(root);
    }
    families.set(label, root);
  });
  return families;
}

// Constrói a paleta e a legenda a partir de um artifact concreto.
//
// A cor segue a família, e a família é decidida uma vez para o mapa aberto:
// filtrar ou isolar na legenda nunca repinta o que sobrou. Famílias além da
// quarta compartilham um neutro claro, porque só quatro matizes se separam com
// segurança sobre este fundo.
export function buildContextPalette(points) {
  const labelCounts = new Map();
  const neutralCounts = new Map();
  points.forEach((point) => {
    const key = contextKey(point);
    const target = key in NEUTRAL_LABELS ? neutralCounts : labelCounts;
    target.set(key, (target.get(key) ?? 0) + 1);
  });
  const families = buildLabelFamilies(labelCounts);
  const grouped = new Map();
  labelCounts.forEach((count, label) => {
    const family = families.get(label) ?? label;
    const entry = grouped.get(family) ?? { key: family, count: 0, members: [] };
    entry.count += count;
    entry.members.push({ key: label, label, count });
    grouped.set(family, entry);
  });
  const ranked = [...grouped.values()].sort(
    (left, right) => right.count - left.count || left.key.localeCompare(right.key),
  );
  const colorByFamily = new Map();
  ranked.forEach((entry, rank) => {
    colorByFamily.set(entry.key, FAMILY_COLORS[rank] ?? OTHER_FAMILY_COLOR);
  });
  const byCoverage = (left, right) => right.count - left.count;
  const legend = ranked.slice(0, FAMILY_COLORS.length).map((entry) => ({
    key: entry.key,
    label: entry.key,
    count: entry.count,
    color: colorByFamily.get(entry.key),
    semantic: true,
    members: [...entry.members].sort(byCoverage),
  }));
  // A cauda vira uma linha só. Espalhá-la em várias linhas do mesmo neutro
  // sugeriria categorias distinguíveis no mapa, quando elas compartilham cor.
  const tail = ranked.slice(FAMILY_COLORS.length);
  if (tail.length) {
    legend.push({
      key: OTHER_FAMILY_KEY,
      label: NEUTRAL_LABELS[OTHER_FAMILY_KEY],
      count: tail.reduce((total, entry) => total + entry.count, 0),
      color: OTHER_FAMILY_COLOR,
      semantic: true,
      members: tail.flatMap((entry) => entry.members).sort(byCoverage),
    });
  }
  [OBSERVED_UNLABELED_KEY, UNOBSERVED_KEY].forEach((key) => {
    const count = neutralCounts.get(key);
    if (!count) return;
    legend.push({
      key,
      label: NEUTRAL_LABELS[key],
      count,
      color: key === UNOBSERVED_KEY ? UNOBSERVED_COLOR : OBSERVED_UNLABELED_COLOR,
      semantic: false,
      members: [],
    });
  });
  const colorOf = (point) => {
    const key = contextKey(point);
    if (key === UNOBSERVED_KEY) return UNOBSERVED_COLOR;
    if (key === OBSERVED_UNLABELED_KEY) return OBSERVED_UNLABELED_COLOR;
    return colorByFamily.get(families.get(key) ?? key) ?? OTHER_FAMILY_COLOR;
  };
  return { legend, colorOf, familyOf: (label) => families.get(label) ?? label };
}

// Resolve as chaves de contexto que uma linha da legenda representa. Uma
// família cobre todos os seus labels brutos; uma categoria neutra cobre a si
// mesma. Existe para que alternar e isolar funcionem igual nos dois níveis.
export function legendKeys(entry) {
  return entry.members?.length ? entry.members.map((member) => member.key) : [entry.key];
}

// Reúne todas as chaves presentes no mapa aberto, que é o estado inicial de
// "tudo em foco" materializado no primeiro filtro.
export function allLegendKeys(legend) {
  return legend.flatMap(legendKeys);
}

// Conta os pontos por estado de corroboração para que a legenda diga quanto do
// mapa cada opção de atenuação afeta, em vez de oferecer um interruptor cego.
export function countSupportStates(points) {
  const counts = { corroborated: 0, uncorroborated: 0, weak: 0 };
  points.forEach((point) => {
    const state = visualEvidence(point)?.support_state;
    if (state in counts) counts[state] += 1;
  });
  return counts;
}

// Decide se um ponto tem sustentação suficiente para ser desenhado com cor
// plena. Um label pode estar geometricamente deslocado sem que nada na imagem
// o denuncie — é o caso do ponto atrás da parede que herda a cor da superfície
// da frente. O artifact publica os dois sinais que detectam isso; aqui eles
// viram uma decisão de desenho, nunca uma alteração do claim.
function isWeaklySupported(point, { dimWeak, dimUncorroborated }) {
  const state = visualEvidence(point)?.support_state;
  if (state === "weak") return dimWeak;
  if (state === "uncorroborated") return dimUncorroborated;
  return false;
}

// Separa o mapa entre o que é desenhado com cor plena e o que fica atenuado,
// em vez de descartar pontos. A legenda responde "onde está esta classe dentro
// do mapa", e isso exige que a geometria vizinha continue desenhada como
// referência espacial.
export function partitionByFocus(points, enabledContextKeys, support = {}) {
  const rules = { dimWeak: true, dimUncorroborated: false, ...support };
  const focused = [];
  const dimmed = [];
  points.forEach((point) => {
    const inFocus = enabledContextKeys === null || enabledContextKeys.has(contextKey(point));
    if (inFocus && !isWeaklySupported(point, rules)) focused.push(point);
    else dimmed.push(point);
  });
  return { focused, dimmed };
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
