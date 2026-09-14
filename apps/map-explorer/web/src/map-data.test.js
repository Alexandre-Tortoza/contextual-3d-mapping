import assert from "node:assert/strict";
import test from "node:test";

import {
  StaticArtifactGeometrySource,
  buildContextPalette,
  buildLabelFamilies,
  contextKey,
  countSupportStates,
  geometryIdentity,
  legendKeys,
  mapEntriesFromIndex,
  measureMap,
  partitionByFocus,
  srgbColorToLinear,
  validateSlice,
} from "./map-data.js";

// Reproduz o vocabulário aberto real do pipeline: quase-sinônimos em volta de
// poucas superfícies, mais uma cauda de labels raros.
const VOCABULARIO = [
  ["wall", 320], ["floor", 140], ["ceiling", 120], ["door", 70],
  ["wall tiles", 22], ["plain wall", 19], ["floor tiles", 25],
  ["ceiling tiles", 38], ["red door", 34], ["wooden pallet", 10],
  ["window", 8], ["rock", 3],
];
const CENA = VOCABULARIO.flatMap(([label, count], familia) =>
  Array.from({ length: count }, (_, index) => ({
    geometry_id: `${label}-${index}`,
    coordinates_m: [familia, index, 0],
    context: { status: "associated", color_rgb: [1, 2, 3], label },
  })),
);

const POINTS = [
  { geometry_id: "a", coordinates_m: [0, 1, -2], display_color_rgb: [1, 2, 3] },
  {
    geometry_id: "b",
    coordinates_m: [4, 5, 6],
    association: { label: "door", color_rgb: [20, 30, 40], semantic_color_rgb: [200, 10, 20] },
  },
  {
    geometry_id: "c",
    coordinates_m: [2, 3, 1],
    association: { label: "door", color_rgb: [30, 40, 50], semantic_color_rgb: [200, 10, 20] },
  },
  {
    geometry_id: "d",
    coordinates_m: [1, 2, 0],
    association: { status: "associated", color_rgb: [90, 80, 70] },
  },
  {
    geometry_id: "e",
    coordinates_m: [1, 2, 0],
    context: { status: "occluded" },
  },
];

// Protege a compatibilidade dos dois schemas e os diagnósticos de coordenadas
// inválidas antes da renderização WebGL.
test("validateSlice aceita v1/v2 e rejeita coordenadas inválidas", () => {
  assert.equal(validateSlice({ schema_version: 1, map_id: "m", map_frame: "map", points: POINTS }).map_id, "m");
  assert.equal(validateSlice({ schema_version: 2, map_id: "m", map_frame: "map", points: POINTS, observations: [], regions: [] }).schema_version, 2);
  assert.throws(
    () => validateSlice({ schema_version: 1, map_id: "m", map_frame: "map", points: [{ coordinates_m: [0, Number.NaN, 0] }] }),
    /não finitas/,
  );
  assert.equal(
    validateSlice({
      schema_version: 1, artifact_type: "consolidated_contextual_map", map_id: "m", map_frame: "map",
      points: POINTS, observations: [], regions: [],
    }).artifact_type,
    "consolidated_contextual_map",
  );
});

// Garante que a câmera enquadre os bounds do mapa completo, não o subconjunto
// atualmente visível por filtros.
test("measureMap calcula centro, alturas e diagonal", () => {
  const metrics = measureMap(POINTS);
  assert.deepEqual(metrics.minimum, [0, 1, -2]);
  assert.deepEqual(metrics.maximum, [4, 5, 6]);
  assert.deepEqual(metrics.center, [2, 3, 2]);
  assert.equal(metrics.diagonal, Math.sqrt(96));
});

// Verifica contagem e ordenação semântica, e protege a distinção entre um
// ponto visto sem evidência publicada e um que os keyframes nunca observaram.
test("legenda separa ausência de evidência publicada de falta de cobertura", () => {
  const legend = buildContextPalette(POINTS).legend;
  // As duas categorias neutras têm ordem fixa, e não por cobertura: elas não
  // competem entre si, e a mais informativa vem primeiro.
  assert.deepEqual(legend.map((entry) => [entry.label, entry.count]), [
    ["door", 2],
    ["Observado, sem evidência publicada", 1],
    ["Não observado pelos keyframes", 2],
  ]);
  assert.equal(contextKey(POINTS[0]), "__unobserved__");
  assert.equal(contextKey(POINTS[3]), "__observed_unlabeled__");
  assert.equal(contextKey(POINTS[4]), "__unobserved__");
});

// Garante que a legenda mude o foco sem descartar geometria: o restante do
// mapa continua entregue à cena como referência espacial e continua clicável.
test("partitionByFocus preserva todos os pontos em duas camadas", () => {
  const { focused, dimmed } = partitionByFocus(POINTS, new Set(["door"]));
  assert.deepEqual(focused.map((point) => point.geometry_id), ["b", "c"]);
  assert.deepEqual(dimmed.map((point) => point.geometry_id), ["a", "d", "e"]);
  assert.equal(focused.length + dimmed.length, POINTS.length);
  const everything = partitionByFocus(POINTS, null);
  assert.equal(everything.focused.length, POINTS.length);
  assert.deepEqual(everything.dimmed, []);
});

// O vocabulário do reasoner é aberto e cheio de quase-sinônimos; agrupá-los é
// o que impede que uma superfície só consuma quatro matizes.
test("labels quase-sinônimos caem na família do label mais frequente", () => {
  const familias = buildLabelFamilies(new Map(VOCABULARIO));
  assert.equal(familias.get("plain wall"), "wall");
  assert.equal(familias.get("wall tiles"), "wall");
  assert.equal(familias.get("ceiling tiles"), "ceiling");
  assert.equal(familias.get("red door"), "door");
  assert.equal(familias.get("wall"), "wall");
  assert.equal(familias.get("rock"), "rock");
});

// É a regressão do defeito relatado: 32 labels caíam em 15 cores, seis delas
// compartilhadas, e `floor tiles` tinha exatamente a cor de `wall tiles`. Duas
// categorias visíveis nunca podem dividir a mesma cor.
test("nenhuma categoria visível compartilha cor com outra", () => {
  const { legend } = buildContextPalette(CENA);
  const porCor = new Map();
  legend.forEach((entry) => {
    const cor = entry.color.join(",");
    porCor.set(cor, [...(porCor.get(cor) ?? []), entry.label]);
  });
  const colisoes = [...porCor.values()].filter((labels) => labels.length > 1);
  assert.deepEqual(colisoes, [], `cores compartilhadas: ${JSON.stringify(colisoes)}`);
});

// Só quatro matizes se separam com segurança sobre o fundo escuro do viewer, e
// a quinta família precisa cair no neutro em vez de ganhar uma cor gerada.
test("apenas as quatro maiores famílias recebem matiz", () => {
  const { legend } = buildContextPalette(CENA);
  const familias = legend.filter((entry) => entry.semantic && !entry.structural);
  assert.deepEqual(familias.slice(0, 4).map((entry) => entry.label), ["door", "ceiling tiles", "floor tiles", "wall tiles"]);
  const neutro = [185, 189, 199];
  assert.ok(familias.slice(4).every((entry) => entry.color.join() === neutro.join()));
  assert.equal(new Set(familias.slice(0, 4).map((entry) => entry.color.join())).size, 4);
});

// A cor segue a entidade: filtrar não pode repintar o que sobrou no mapa.
test("cor de uma família não depende do que está filtrado", () => {
  const paleta = buildContextPalette(CENA);
  const porta = CENA.find((point) => point.context.label === "red door");
  const antes = paleta.colorOf(porta);
  const soPorta = buildContextPalette(CENA.filter((p) => p.context.label.includes("door")));
  assert.deepEqual(paleta.colorOf(porta), antes);
  assert.deepEqual(soPorta.colorOf(porta), [57, 135, 229]);
});

// Regressão: a família estrutural agrupava os labels sob uma chave sintética
// que nenhum ponto carregava, e por isso não dava para escondê-la nem isolá-la.
test("as chaves da família estrutural classificam os pontos estruturais", () => {
  const { legend } = buildContextPalette(CENA);
  const estruturas = legend.find((entry) => entry.structural);
  const chaves = new Set(legendKeys(estruturas));
  const alcancados = CENA.filter((point) => chaves.has(contextKey(point)));
  assert.equal(alcancados.length, estruturas.count);

  const restante = new Set(CENA.map(contextKey).filter((key) => !chaves.has(key)));
  const { focused } = partitionByFocus(CENA, restante, { dimWeak: false });
  assert.equal(focused.length, CENA.length - estruturas.count);
});

// Alternar uma família precisa alcançar todos os labels que ela agrupa.
test("uma linha de família cobre as chaves de todos os seus labels", () => {
  const { legend } = buildContextPalette(CENA);
  const porta = legend.find((entry) => entry.label === "door");
  assert.deepEqual(legendKeys(porta).sort(), ["door", "red door"]);
  const semContexto = { key: "__unobserved__", members: [] };
  assert.deepEqual(legendKeys(semContexto), ["__unobserved__"]);
});

// O viewer precisa dizer quanto do mapa cada opção de atenuação afeta.
test("countSupportStates conta os estados de corroboração", () => {
  const pontos = [
    { context: { label: "a", support_state: "weak" } },
    { context: { label: "b", support_state: "corroborated" } },
    { context: { label: "c", support_state: "uncorroborated" } },
    { context: { label: "d" } },
    { display_color_rgb: [0, 0, 0] },
  ];
  assert.deepEqual(countSupportStates(pontos), { corroborated: 1, uncorroborated: 1, weak: 1 });
});

// Confirma a conversão necessária antes de gravar cores sRGB nos buffers
// lineares do Three.js.
test("conversão sRGB preserva extremos e lineariza meios-tons", () => {
  const converted = srgbColorToLinear([0, 128, 255]);
  assert.equal(converted[0], 0);
  assert.ok(Math.abs(converted[1] - 0.21586) < 0.00001);
  assert.equal(converted[2], 1);
});

// O seletor não pode quebrar quando o índice publicado estiver ausente ou
// malformado: abrir o mapa é mais importante que listar alternativas.
test("mapEntriesFromIndex aceita entradas válidas e descarta o resto", () => {
  const entries = mapEntriesFromIndex([
    { url: "/runs/a/context.json", label: "Run A · 16 frames", artifact_type: "contextual_rgb_lidar_slice" },
    { url: "/maps/consolidated/a/context.json", label: "Mapa consolidado · m", artifact_type: "consolidated_contextual_map" },
    { url: "/maps/geometry.json", label: "Geometria", artifact_type: "geometric_slice" },
    { url: "/maps/x.json" },
    { label: "sem url" },
    "texto",
  ]);
  assert.deepEqual(entries, [
    { url: "/runs/a/context.json", label: "Run A · 16 frames", artifactType: "contextual_rgb_lidar_slice" },
    { url: "/maps/consolidated/a/context.json", label: "Mapa consolidado · m", artifactType: "consolidated_contextual_map" },
  ]);
  assert.deepEqual(mapEntriesFromIndex(null), []);
  assert.deepEqual(mapEntriesFromIndex({}), []);
});

// Alternar evidência sobre a mesma nuvem deve preservar a câmera; mudar a
// geometria, seu frame ou a amostragem precisa produzir outra identidade.
test("geometryIdentity preserva a câmera entre runs da mesma geometria", () => {
  const primeira = { map_id: "m", map_frame: "map", source: { sha256: "cloud-a" }, points: POINTS };
  const segunda = { ...primeira, points: POINTS.map((point) => ({ ...point, context: { label: "pallet" } })) };
  assert.equal(geometryIdentity(primeira), geometryIdentity(segunda));
  assert.notEqual(geometryIdentity(primeira), geometryIdentity({ ...segunda, source: { sha256: "cloud-b" } }));
  assert.notEqual(geometryIdentity(primeira), geometryIdentity({ ...segunda, points: POINTS.slice(1) }));
  assert.notEqual(geometryIdentity(primeira), geometryIdentity({ ...segunda, map_frame: "other" }));
});

// Mantém a fonte estática compatível com a futura fronteira de chunks e com
// cancelamento de requisição.
test("fonte estática devolve um chunk e respeita cancelamento", async () => {
  const slice = { map_id: "m", points: POINTS };
  const source = new StaticArtifactGeometrySource(slice);
  const geometry = await source.getGeometry();
  assert.equal(geometry.chunks[0].points, POINTS);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(source.getGeometry({ signal: controller.signal }), { name: "AbortError" });
});

// Uma paisagem visível com hipótese de janela não pode ganhar a cor do label.
test("hipótese 2D e visibilidade incerta permanecem fora dos labels da legenda", () => {
  const landscape = { geometry_id: "landscape", context: {
    status: "associated", tentative_label: "window", semantic_status: "surface_unsupported", color_rgb: [1, 2, 3],
  } };
  const unsupported = { geometry_id: "unsupported", context: { status: "visibility_unconfirmed" } };
  assert.equal(contextKey(landscape), contextKey(POINTS[3]));
  assert.equal(contextKey(unsupported), contextKey(POINTS[4]));
  assert.notEqual(contextKey(landscape), "window");
});
