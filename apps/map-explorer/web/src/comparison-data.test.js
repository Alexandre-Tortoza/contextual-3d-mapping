import assert from "node:assert/strict";
import test from "node:test";

import { comparisonGroups, comparisonMatrix, sharedObservationIds } from "./comparison-data.js";

const run = (label, discovery, reasoner) => ({
  label,
  artifactType: "contextual_rgb_lidar_slice",
  comparison: { groupId: "frame-184", axes: { region_discovery: discovery, multimodal_reasoner: reasoner } },
});

// A matriz precisa preservar combinações ausentes para que a ausência de uma
// execução seja observável e não seja confundida com uma saída vazia do modelo.
test("comparisonMatrix organiza dois backends por eixo e preserva célula ausente", () => {
  const matrix = comparisonMatrix([
    run("SAM Qwen", "sam", "qwen"), run("SAM Gemini", "sam", "gemini"), run("Florence Qwen", "florence", "qwen"),
  ]);
  assert.deepEqual(matrix.axes.map((axis) => axis.name), ["region_discovery", "multimodal_reasoner"]);
  assert.equal(matrix.rows.length, 2);
  assert.equal(matrix.rows.flatMap((row) => row.cells).filter(Boolean).length, 3);
  assert.equal(matrix.rows.flatMap((row) => row.cells).filter((cell) => cell === null).length, 1);
});

// Com 3+ eixos variáveis, a matriz não pode mais dividir a comparação em
// telas por eixo fixado: o primeiro eixo vira linha e o resto é achatado
// numa única dimensão de coluna, para que toda combinação apareça na mesma
// grade em vez de exigir uma tela por eixo mantido constante.
test("comparisonMatrix achata 3+ eixos numa única grade", () => {
  const runWithFeatures = (label, discovery, features, reasoner) => ({
    label,
    artifactType: "contextual_rgb_lidar_slice",
    comparison: {
      groupId: "frame-184",
      axes: { region_discovery: discovery, feature_extraction: features, multimodal_reasoner: reasoner },
    },
  });
  const matrix = comparisonMatrix([
    runWithFeatures("A", "sam", "dino", "qwen"),
    runWithFeatures("B", "sam", "dino", "gemini"),
    runWithFeatures("C", "sam", "featup", "qwen"),
    runWithFeatures("D", "sam", "featup", "gemini"),
    runWithFeatures("E", "florence", "dino", "qwen"),
    runWithFeatures("F", "florence", "dino", "gemini"),
    runWithFeatures("G", "florence", "featup", "qwen"),
    runWithFeatures("H", "florence", "featup", "gemini"),
  ]);
  assert.deepEqual(matrix.axes.map((axis) => axis.name), ["region_discovery", "multimodal_reasoner", "feature_extraction"]);
  assert.equal(matrix.rows.length, 2);
  assert.equal(matrix.columns.length, 4);
  assert.equal(matrix.rows.flatMap((row) => row.cells).filter(Boolean).length, 8);
});

// O frame só pode ser exibido lado a lado se estiver presente em todas as
// execuções: a identidade publicada, não a posição na lista, é o contract.
test("sharedObservationIds usa a interseção por observation_id", () => {
  assert.deepEqual(sharedObservationIds([
    { observations: [{ observation_id: "b" }, { observation_id: "a" }] },
    { observations: [{ observation_id: "c" }, { observation_id: "a" }] },
  ]), ["a"]);
  assert.deepEqual(comparisonGroups([run("a", "sam", "qwen")]).map((group) => group.id), ["frame-184"]);
});
