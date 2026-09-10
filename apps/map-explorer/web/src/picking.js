// Raio de captura do clique, em unidades de mapa. O default do Three.js para
// nuvens de pontos é 1 metro, o que aceita qualquer ponto a até um metro do
// raio do cursor; em um mapa denso isso seleciona um vizinho em vez do ponto
// apontado. Deriva da diagonal para acompanhar mapas de escalas diferentes.
export function pickThreshold(diagonal) {
  const scaled = (Number.isFinite(diagonal) ? diagonal : 0) * 0.0015;
  return Math.min(Math.max(scaled, 0.01), 0.15);
}

// Escolhe a interseção realmente apontada pelo cursor. Existe porque o
// raycaster ordena os hits por distância à câmera, e não por distância ao raio:
// sem esta regra um ponto lateral mais próximo da câmera vence o ponto sob o
// cursor. Chamada pelos handlers de clique das camadas da nuvem.
export function nearestIntersection(intersections) {
  const candidates = (intersections ?? []).filter(
    (item) =>
      Number.isFinite(item?.distanceToRay)
      && Number.isInteger(item?.index)
      && Array.isArray(item?.object?.userData?.points),
  );
  if (!candidates.length) return null;
  return candidates.reduce((best, item) => {
    if (item.distanceToRay !== best.distanceToRay) {
      return item.distanceToRay < best.distanceToRay ? item : best;
    }
    if (item.distance !== best.distance) return item.distance < best.distance ? item : best;
    return item.index < best.index ? item : best;
  });
}

// Resolve o ponto de domínio por trás de uma interseção. A camada publica seu
// próprio array em ``userData`` para que o índice do buffer nunca precise ser
// reconciliado fora do componente que o construiu.
export function pointFromIntersection(intersection) {
  const points = intersection?.object?.userData?.points;
  if (!Array.isArray(points)) return null;
  return points[intersection.index] ?? null;
}
