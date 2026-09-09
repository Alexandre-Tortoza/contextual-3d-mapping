// Impede que atalhos de navegação capturem digitação e interação com controles
// HTML do inspector ou da barra superior.
export function isEditableTarget(target) {
  if (typeof Element === "undefined" || !(target instanceof Element)) return false;
  return Boolean(target.closest("input, textarea, select, button, [contenteditable='true']"));
}

// Converte teclas pressionadas em deslocamento local da câmera. Separada do
// render loop para que direção, velocidade e aceleração sejam testáveis.
export function movementFromKeys(keys, deltaSeconds, baseSpeed, accelerated = false) {
  const speed = baseSpeed * (accelerated ? 4 : 1) * deltaSeconds;
  const horizontal = (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0);
  const vertical = (keys.has("KeyE") ? 1 : 0) - (keys.has("KeyQ") ? 1 : 0);
  const forward = (keys.has("KeyW") ? 1 : 0) - (keys.has("KeyS") ? 1 : 0);
  return { horizontal: horizontal * speed, vertical: vertical * speed, forward: forward * speed };
}

// Dimensiona movimento e foco pela extensão do mapa, evitando velocidade fixa
// muito lenta em mapas grandes ou agressiva em fixtures pequenas.
export function navigationScale(diagonal) {
  return {
    flySpeed: Math.max(diagonal * 0.12, 0.5),
    focusDistance: Math.min(Math.max(diagonal * 0.04, 1.5), 30),
    markerRadius: Math.min(Math.max(diagonal * 0.0015, 0.04), 0.35),
  };
}

// Define poses previsíveis para os botões de vista sem depender da orientação
// atual da câmera ou de um alvo que o usuário deslocou.
export function cameraPreset(metrics, preset) {
  const [x, y, z] = metrics.center;
  const distance = Math.max(metrics.diagonal * 0.8, 2);
  if (preset === "top") {
    return { position: [x, y - distance * 0.001, z + distance], target: [x, y, z] };
  }
  const spanX = metrics.maximum[0] - metrics.minimum[0];
  const spanY = metrics.maximum[1] - metrics.minimum[1];
  const position = spanX >= spanY
    ? [x + distance * 0.08, y - distance * 0.78, z + distance * 0.42]
    : [x + distance * 0.78, y - distance * 0.08, z + distance * 0.42];
  return {
    position,
    target: [x, y, z],
  };
}
