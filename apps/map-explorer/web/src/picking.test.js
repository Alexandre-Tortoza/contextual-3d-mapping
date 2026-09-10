import assert from "node:assert/strict";
import test from "node:test";

import { nearestIntersection, pickThreshold, pointFromIntersection } from "./picking.js";

const LAYER = { userData: { points: [{ geometry_id: "a" }, { geometry_id: "b" }] } };

// Constrói uma interseção no formato publicado pelo raycaster de nuvem de
// pontos do Three.js, incluindo a distância ao raio que a regra de seleção usa.
function intersection({ index, distance, distanceToRay, object = LAYER }) {
  return { index, distance, distanceToRay, object };
}

// Protege o raio de captura contra o default de 1 metro do Three.js, que em um
// mapa denso aceita vizinhos em vez do ponto apontado.
test("pickThreshold acompanha a escala do mapa dentro de limites úteis", () => {
  assert.equal(pickThreshold(40), 0.06);
  assert.equal(pickThreshold(1), 0.01);
  assert.equal(pickThreshold(10000), 0.15);
  assert.equal(pickThreshold(Number.NaN), 0.01);
});

// É a regressão do bug observado no viewer: o ponto mais próximo da câmera
// vencia o ponto sob o cursor, e a seleção parecia pular para um vizinho.
test("nearestIntersection prefere o ponto sob o cursor, não o mais próximo da câmera", () => {
  const vizinhoNaFrente = intersection({ index: 0, distance: 2, distanceToRay: 0.09 });
  const pontoApontado = intersection({ index: 1, distance: 5, distanceToRay: 0.004 });
  assert.equal(nearestIntersection([vizinhoNaFrente, pontoApontado]).index, 1);
});

// Empates precisam ser determinísticos para que o mesmo clique selecione
// sempre o mesmo ponto.
test("nearestIntersection desempata por distância e depois por índice", () => {
  const longe = intersection({ index: 0, distance: 9, distanceToRay: 0.01 });
  const perto = intersection({ index: 1, distance: 3, distanceToRay: 0.01 });
  assert.equal(nearestIntersection([longe, perto]).index, 1);
  const primeiro = intersection({ index: 0, distance: 3, distanceToRay: 0.01 });
  assert.equal(nearestIntersection([primeiro, perto]).index, 0);
});

// A cena também contém grid e eixos; eles não têm distância ao raio nem array
// de pontos, e nunca podem virar uma seleção.
test("nearestIntersection ignora objetos que não são camadas de pontos", () => {
  const grid = { index: 0, distance: 1, object: { userData: {} } };
  const ponto = intersection({ index: 1, distance: 8, distanceToRay: 0.02 });
  assert.equal(nearestIntersection([grid, ponto]).index, 1);
  assert.equal(nearestIntersection([grid]), null);
  assert.equal(nearestIntersection([]), null);
  assert.equal(nearestIntersection(undefined), null);
});

// A identidade do ponto vem da camada que preencheu o buffer, o que mantém as
// duas camadas independentes uma da outra.
test("pointFromIntersection resolve o ponto pela camada de origem", () => {
  assert.equal(pointFromIntersection(intersection({ index: 1, distance: 1, distanceToRay: 0 })).geometry_id, "b");
  assert.equal(pointFromIntersection(null), null);
  assert.equal(pointFromIntersection({ index: 7, object: LAYER }), null);
});
