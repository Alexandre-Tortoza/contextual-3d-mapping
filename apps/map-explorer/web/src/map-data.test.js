import assert from "node:assert/strict";
import test from "node:test";

import {
  StaticArtifactGeometrySource,
  buildContextLegend,
  filterPoints,
  measureMap,
  pointColor,
  semanticColor,
  srgbColorToLinear,
  validateSlice,
} from "./map-data.js";

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
    association: { color_rgb: [90, 80, 70] },
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

// Verifica contagem, ordenação semântica e isolamento sem perder o vínculo
// entre índice renderizado e identidade original.
test("legenda contextual conta e filtra categorias", () => {
  const legend = buildContextLegend(POINTS);
  assert.deepEqual(legend.map((entry) => [entry.label, entry.count]), [
    ["door", 2],
    ["Sem contexto", 2],
  ]);
  assert.deepEqual(filterPoints(POINTS, new Set(["door"])).map((point) => point.geometry_id), ["b", "c"]);
  assert.equal(filterPoints(POINTS, null).length, POINTS.length);
});

// Protege a paleta apresentada ao usuário contra regressões para cores opacas,
// inconsistentes ou dependentes da cor publicada pelo artifact.
test("cores contextuais são estáveis e pontos sem contexto ficam discretos", () => {
  assert.deepEqual(semanticColor("door"), [255, 82, 82]);
  assert.deepEqual(semanticColor(" Door "), semanticColor("door"));
  assert.deepEqual(semanticColor("label futura"), semanticColor("label futura"));
  assert.notDeepEqual(semanticColor("door"), semanticColor("wall"));
  assert.deepEqual(pointColor(POINTS[0]), [28, 31, 38]);
  assert.deepEqual(pointColor(POINTS[1]), [255, 82, 82]);
});

// Confirma a conversão necessária antes de gravar cores sRGB nos buffers
// lineares do Three.js.
test("conversão sRGB preserva extremos e lineariza meios-tons", () => {
  const converted = srgbColorToLinear([0, 128, 255]);
  assert.equal(converted[0], 0);
  assert.ok(Math.abs(converted[1] - 0.21586) < 0.00001);
  assert.equal(converted[2], 1);
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
