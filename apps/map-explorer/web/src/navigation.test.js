import assert from "node:assert/strict";
import test from "node:test";

import { cameraPreset, movementFromKeys, navigationScale } from "./navigation.js";

// Valida deslocamento tridimensional, oposição de teclas e aceleração do modo
// voo independentemente do frame loop do Three.js.
test("movementFromKeys produz movimento local previsível", () => {
  assert.deepEqual(movementFromKeys(new Set(["KeyW", "KeyD", "KeyE"]), 0.5, 2), {
    horizontal: 1,
    vertical: 1,
    forward: 1,
  });
  assert.deepEqual(movementFromKeys(new Set(["KeyW", "KeyS", "KeyA"]), 0.5, 2, true), {
    horizontal: -4,
    vertical: 0,
    forward: 0,
  });
});

// Protege os limites de velocidade, foco e marcador para mapas extremamente
// pequenos ou grandes.
test("navigationScale limita parâmetros dependentes do mapa", () => {
  assert.deepEqual(navigationScale(0.1), { flySpeed: 0.5, focusDistance: 1.5, markerRadius: 0.04 });
  assert.deepEqual(navigationScale(10_000), { flySpeed: 1200, focusDistance: 30, markerRadius: 0.35 });
});

// Garante que presets usem o centro/bounds originais, sem herdar alvo deslocado
// ou seleção filtrada.
test("cameraPreset cria vistas superior e isométrica", () => {
  const metrics = { center: [10, 20, 30], diagonal: 100, minimum: [0, 15, 20], maximum: [100, 25, 40] };
  const top = cameraPreset(metrics, "top");
  const isometric = cameraPreset(metrics, "isometric");
  assert.deepEqual(top.target, metrics.center);
  assert.deepEqual(top.position, [10, 19.92, 110]);
  assert.deepEqual(isometric.target, metrics.center);
  assert.ok(isometric.position[0] > 10);
  assert.ok(isometric.position[1] < 20);
  assert.ok(isometric.position[2] > 30);
});
