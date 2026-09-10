import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Canvas } from "@react-three/fiber";
import { PointMaterial } from "@react-three/drei";
import * as THREE from "three";

import { CameraRig } from "./camera-rig.jsx";
import { Inspector } from "./inspector.jsx";
import {
  DIMMED_COLOR,
  StaticArtifactGeometrySource,
  allLegendKeys,
  buildContextPalette,
  contextKey,
  countSupportStates,
  legendKeys,
  mapEntriesFromIndex,
  measureMap,
  partitionByFocus,
  srgbColorToLinear,
  validateSlice,
} from "./map-data.js";
import { isEditableTarget } from "./navigation.js";
import { nearestIntersection, pickThreshold, pointFromIntersection } from "./picking.js";
import "./styles.css";

const DEFAULT_MAP_URL = "/current-map.json";
const MAP_INDEX_URL = "/maps/index.json";
const FALLBACK_MAPS = Object.freeze([{ url: DEFAULT_MAP_URL, label: "Mapa atual" }]);

// Cor única da camada atenuada. Fica fora do componente para que o builder de
// geometria continue memoizável por identidade da função.
const dimmedColor = () => DIMMED_COLOR;

// Converte um subconjunto de pontos em buffers e publica o array de origem no
// objeto renderizado. Existe para que a seleção resolva a identidade do ponto
// pelo mesmo índice de buffer que esta camada preencheu, sem reconciliar
// índices entre camadas.
function PointLayer({ points, colorOf, size, opacity, onPointerDown, onClick, onDoubleClick }) {
  const geometry = useMemo(() => {
    const positions = [];
    const colors = [];
    points.forEach((point) => {
      positions.push(...point.coordinates_m);
      colors.push(...srgbColorToLinear(colorOf(point)));
    });
    const value = new THREE.BufferGeometry();
    value.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    value.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return value;
  }, [points, colorOf]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  if (!points.length) return null;
  return (
    <points
      geometry={geometry}
      userData={{ points }}
      onPointerDown={onPointerDown}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      <PointMaterial
        vertexColors
        size={size}
        sizeAttenuation={false}
        toneMapped={false}
        fog={false}
        transparent={opacity < 1}
        opacity={opacity}
        depthWrite={opacity >= 1}
      />
    </points>
  );
}

// Desenha o mapa em duas camadas: a classe em foco com cor plena e o restante
// atenuado como referência espacial. Remover a geometria vizinha esconderia
// justamente o contexto que permite localizar a classe dentro do mapa, então
// pontos atenuados continuam desenhados e selecionáveis.
function Cloud({ focused, dimmed, colorOf, onSelect, onFocus }) {
  const pointerOrigin = useRef(null);

  // Distingue clique de arraste para que orbitar ou fazer pan não selecione um
  // ponto acidentalmente ao soltar o mouse. A escolha do ponto considera todas
  // as interseções do evento, e não a camada que recebeu o clique.
  const selectIfStationary = (event, focus = false) => {
    const origin = pointerOrigin.current;
    const distance = origin ? Math.hypot(event.clientX - origin.x, event.clientY - origin.y) : 0;
    if (distance > 4) return;
    event.stopPropagation();
    const point = pointFromIntersection(nearestIntersection(event.intersections));
    onSelect(point);
    if (focus && point) onFocus(point);
  };

  const handlers = {
    onPointerDown: (event) => { pointerOrigin.current = { x: event.clientX, y: event.clientY }; },
    onClick: (event) => selectIfStationary(event),
    onDoubleClick: (event) => selectIfStationary(event, true),
  };

  return (
    <>
      <PointLayer points={dimmed} colorOf={dimmedColor} size={1.6} opacity={0.35} {...handlers} />
      <PointLayer points={focused} colorOf={colorOf} size={2.6} opacity={1} {...handlers} />
    </>
  );
}

// Marca o ponto inspecionado sem alterar o centro da câmera.
function SelectionMarker({ point, radius }) {
  if (!point) return null;
  return (
    <mesh position={point.coordinates_m} renderOrder={2}>
      <sphereGeometry args={[radius, 16, 16]} />
      <meshBasicMaterial color="#ff3535" depthTest={false} />
    </mesh>
  );
}

// Mantém uma referência espacial discreta no plano XY do menor Z do mapa.
function SpatialReference({ metrics }) {
  const size = Math.max(metrics.diagonal, 2);
  return (
    <group position={[metrics.center[0], metrics.center[1], metrics.minimumHeight]}>
      <gridHelper args={[size, 20, "#173dff", "#252525"]} rotation={[Math.PI / 2, 0, 0]} />
      <axesHelper args={[Math.max(size * 0.08, 0.5)]} />
    </group>
  );
}

// Oferece a leitura e os filtros das cores em um painel pequeno e recolhível,
// mantendo a sidebar reservada à evidência do ponto selecionado.
function ContextLegend({
  entries,
  enabledKeys,
  focusedPointCount,
  totalPointCount,
  supportRules,
  supportCounts,
  onToggle,
  onIsolate,
  onReset,
  onToggleSupport,
}) {
  const [expanded, setExpanded] = useState(new Set());
  const toggleExpanded = (key) => setExpanded((current) => {
    const next = new Set(current);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  });
  return (
    <details className="map-overlay context-legend" open>
      <summary>
        <span>Legenda</span>
        <small>{focusedPointCount.toLocaleString("pt-BR")} em foco / {totalPointCount.toLocaleString("pt-BR")}</small>
      </summary>
      <div className="legend-actions">
        <span>Contexto</span>
        <button type="button" onClick={onReset}>Focar tudo</button>
      </div>
      <div className="legend-support">
        <label>
          <input
            type="checkbox"
            checked={supportRules.dimWeak}
            onChange={() => onToggleSupport("dimWeak")}
          />
          <span>Atenuar contexto sem suporte</span>
          <b>{supportCounts.weak.toLocaleString("pt-BR")}</b>
        </label>
        <label>
          <input
            type="checkbox"
            checked={supportRules.dimUncorroborated}
            onChange={() => onToggleSupport("dimUncorroborated")}
          />
          <span>Atenuar visto por um frame só</span>
          <b>{supportCounts.uncorroborated.toLocaleString("pt-BR")}</b>
        </label>
      </div>
      <div className="legend-list">
        {entries.map((entry) => {
          const keys = legendKeys(entry);
          const enabled = enabledKeys === null || keys.every((key) => enabledKeys.has(key));
          const detailed = entry.members.length > 1;
          const open = expanded.has(entry.key);
          return (
            <div key={entry.key}>
              <div className={`legend-row ${enabled ? "" : "disabled"}`}>
                <button type="button" className="legend-toggle" aria-pressed={enabled} onClick={() => onToggle(keys)}>
                  <i style={{ background: `rgb(${entry.color.join(" ")})` }} />
                  <span>{entry.label}</span>
                  <b>{entry.count.toLocaleString("pt-BR")}</b>
                </button>
                {detailed && (
                  <button type="button" className="isolate-action" aria-expanded={open}
                    aria-label={`Labels de ${entry.label}`} onClick={() => toggleExpanded(entry.key)}>
                    {open ? "▾" : `${entry.members.length}`}
                  </button>
                )}
                <button type="button" className="isolate-action" onClick={() => onIsolate(keys)}>Isolar</button>
              </div>
              {detailed && open && entry.members.map((member) => {
                const memberEnabled = enabledKeys === null || enabledKeys.has(member.key);
                return (
                  <div className={`legend-row legend-member ${memberEnabled ? "" : "disabled"}`} key={member.key}>
                    <button type="button" className="legend-toggle" aria-pressed={memberEnabled}
                      onClick={() => onToggle([member.key])}>
                      <i />
                      <span>{member.label}</span>
                      <b>{member.count.toLocaleString("pt-BR")}</b>
                    </button>
                    <button type="button" className="isolate-action" onClick={() => onIsolate([member.key])}>Isolar</button>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </details>
  );
}

// Expõe os artifacts realmente publicados pelo servidor local. A lista vem do
// índice gravado ao publicar os mapas, e não de constantes no frontend, para
// que um trecho novo apareça no seletor sem alterar o viewer.
function MapSelector({ entries, value, onChange }) {
  const known = entries.some((entry) => entry.url === value);
  return (
    <label className="map-overlay map-selector">
      <span>Mapa</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {!known && <option value={value}>Mapa solicitado pela URL</option>}
        {entries.map((entry) => <option value={entry.url} key={entry.url}>{entry.label}</option>)}
      </select>
    </label>
  );
}

// Expõe os comandos de câmera no próprio mapa para que navegação não dependa
// de conhecer atalhos de teclado.
function CameraToolbar({ flyMode, hasSelection, onReset, onTop, onIsometric, onFocus, onToggleFly, onHelp }) {
  return (
    <div className="map-overlay camera-toolbar" aria-label="Controles da câmera">
      <button type="button" onClick={onReset}><kbd>Home</kbd><span>Mapa inteiro</span></button>
      <button type="button" onClick={onTop}><kbd>1</kbd><span>Topo</span></button>
      <button type="button" onClick={onIsometric}><kbd>2</kbd><span>Perspectiva</span></button>
      <button type="button" onClick={onFocus} disabled={!hasSelection}><kbd>F</kbd><span>Focar</span></button>
      <button type="button" className={flyMode ? "active" : ""} aria-pressed={flyMode} onClick={onToggleFly}><kbd>V</kbd><span>{flyMode ? "Voando" : "Modo voo"}</span></button>
      <button type="button" className="help-action" onClick={onHelp} aria-label="Ajuda de navegação">?</button>
    </div>
  );
}

// Mostra os gestos e atalhos somente sob demanda para manter o mapa dominante.
function NavigationHelp({ onClose }) {
  return (
    <div className="help-popover" role="dialog" aria-label="Ajuda de navegação">
      <div className="overlay-title"><strong>Navegação</strong><button type="button" onClick={onClose}>×</button></div>
      <dl>
        <dt>Arrastar esquerdo</dt><dd>Orbitar</dd>
        <dt>Arrastar direito/meio</dt><dd>Mover alvo e câmera</dd>
        <dt>Scroll</dt><dd>Zoom no cursor</dd>
        <dt>Duplo clique</dt><dd>Selecionar e focar ponto</dd>
        <dt>W A S D</dt><dd>Mover no modo voo</dd>
        <dt>Q / E</dt><dd>Descer / subir</dd>
        <dt>Shift</dt><dd>Velocidade 4×</dd>
      </dl>
    </div>
  );
}

// Compõe carregamento, navegação, filtros e inspeção sem deixar um desses
// estados reposicionar ou reconstruir os demais implicitamente.
function Explorer() {
  const [artifactPath, setArtifactPath] = useState(
    () => new URLSearchParams(window.location.search).get("artifact") ?? DEFAULT_MAP_URL,
  );
  const [availableMaps, setAvailableMaps] = useState(FALLBACK_MAPS);
  const [supportRules, setSupportRules] = useState({ dimWeak: true, dimUncorroborated: false });
  const [slice, setSlice] = useState(null);
  const [points, setPoints] = useState([]);
  const [selected, setSelected] = useState(null);
  const [enabledContextKeys, setEnabledContextKeys] = useState(null);
  const [artifactUrl, setArtifactUrl] = useState(null);
  const [dockOpen, setDockOpen] = useState(true);
  const [flyMode, setFlyMode] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [error, setError] = useState(null);
  const cameraActions = useRef(null);
  const reducedMotion = useMemo(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    [],
  );
  const metrics = useMemo(() => measureMap(points), [points]);
  const palette = useMemo(() => buildContextPalette(points), [points]);
  const supportCounts = useMemo(() => countSupportStates(points), [points]);
  const { focused, dimmed } = useMemo(
    () => partitionByFocus(points, enabledContextKeys, supportRules),
    [points, enabledContextKeys, supportRules],
  );
  const selectedOutOfFocus = Boolean(
    selected
    && enabledContextKeys !== null
    && !enabledContextKeys.has(contextKey(selected)),
  );

  // Abre um artifact validado através da fronteira de geometria, permitindo
  // substituir a fonte estática por chunks sem reescrever a interface.
  const openSlice = async (payload, resolvedUrl = null) => {
    const validated = validateSlice(payload);
    const source = new StaticArtifactGeometrySource(validated);
    const geometry = await source.getGeometry();
    const loadedPoints = geometry.chunks.flatMap((chunk) => chunk.points);
    setSlice(validated);
    setPoints(loadedPoints);
    setSelected(null);
    setDockOpen(true);
    setFlyMode(false);
    setHelpOpen(false);
    setArtifactUrl(resolvedUrl);
    setEnabledContextKeys(null);
    setError(null);
  };

  // Carrega a opção inicial ou escolhida e cancela a leitura anterior quando o
  // usuário troca de mapa antes do término do download.
  useEffect(() => {
    const resolvedUrl = new URL(artifactPath, window.location.href).href;
    const controller = new AbortController();
    setError(null);
    fetch(resolvedUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`O servidor respondeu HTTP ${response.status}.`);
        return response.json();
      })
      .then((payload) => openSlice(payload, resolvedUrl))
      .catch((failure) => {
        if (failure.name !== "AbortError") {
          setError(failure instanceof Error ? failure.message : "Não foi possível abrir o artifact.");
        }
    });
    return () => controller.abort();
  }, [artifactPath]);

  // Lê o índice publicado uma única vez. A ausência do índice não é erro: um
  // artifact aberto por URL continua abrindo com o seletor mínimo.
  useEffect(() => {
    const controller = new AbortController();
    fetch(MAP_INDEX_URL, { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : []))
      .then((payload) => {
        const entries = mapEntriesFromIndex(payload);
        if (entries.length) setAvailableMaps(entries);
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  // Mantém a URL compartilhável sincronizada sem recarregar a aplicação nem
  // perder o estado de conexão do viewer.
  const selectMap = (path) => {
    const url = new URL(window.location.href);
    url.searchParams.set("artifact", path);
    window.history.replaceState(null, "", url);
    setArtifactPath(path);
  };

  // Centraliza atalhos globais e os desativa quando o usuário interage com
  // elementos HTML, preservando acessibilidade e edição de formulários.
  useEffect(() => {
    const handleShortcut = (event) => {
      if (isEditableTarget(event.target)) return;
      if (event.code === "Home") {
        event.preventDefault();
        cameraActions.current?.reset();
      } else if (event.code === "Digit1") {
        cameraActions.current?.preset("top");
      } else if (event.code === "Digit2") {
        cameraActions.current?.preset("isometric");
      } else if (event.code === "KeyF" && selected) {
        cameraActions.current?.focus(selected);
      } else if (event.code === "KeyV") {
        setFlyMode((value) => !value);
      } else if (event.code === "Escape") {
        if (flyMode) setFlyMode(false);
        else if (helpOpen) setHelpOpen(false);
        else {
          setSelected(null);
          setDockOpen(false);
        }
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [flyMode, helpOpen, selected]);

  // Mantém seleção e abertura do dock como uma única intenção de usuário.
  const selectPoint = (point) => {
    setSelected(point);
    setDockOpen(Boolean(point));
  };
  // Materializa o conjunto completo somente no primeiro filtro, permitindo que
  // ``null`` continue representando o estado barato “todos visíveis”. Alternar
  // uma família alterna todos os labels que ela agrupa, de uma vez.
  const toggleContextKeys = (keys) => {
    setEnabledContextKeys((current) => {
      const next = new Set(current ?? allLegendKeys(palette.legend));
      if (keys.every((key) => next.has(key))) keys.forEach((key) => next.delete(key));
      else keys.forEach((key) => next.add(key));
      return next;
    });
  };

  return (
    <main className="app-shell">
      {error && <p role="alert" className="error-banner">{error}</p>}
      {!slice && !error && <div className="loading-stage" aria-label="Carregando mapa" />}
      {slice && (
        <div className={`workspace ${dockOpen ? "with-dock" : ""}`}>
          <div className="map-column">
            <div className={`viewport ${flyMode ? "fly-active" : ""}`}>
              <Canvas
                camera={{ position: [0, -4, 2], near: Math.max(metrics.diagonal * 0.00001, 0.001), far: metrics.diagonal * 100, up: [0, 0, 1] }}
                raycaster={{ params: { Points: { threshold: pickThreshold(metrics.diagonal) } } }}
              >
                <color attach="background" args={["#0d0d0d"]} />
                <SpatialReference metrics={metrics} />
                <Cloud focused={focused} dimmed={dimmed} colorOf={palette.colorOf} onSelect={selectPoint}
                  onFocus={(point) => cameraActions.current?.focus(point)} />
                <SelectionMarker point={selected} radius={Math.min(Math.max(metrics.diagonal * 0.0015, 0.04), 0.35)} />
                <CameraRig ref={cameraActions} metrics={metrics} flyMode={flyMode}
                  mapKey={slice.map_id} reducedMotion={reducedMotion} />
              </Canvas>
              <CameraToolbar flyMode={flyMode} hasSelection={Boolean(selected)}
                onReset={() => cameraActions.current?.reset()}
                onTop={() => cameraActions.current?.preset("top")}
                onIsometric={() => cameraActions.current?.preset("isometric")}
                onFocus={() => cameraActions.current?.focus(selected)}
                onToggleFly={() => setFlyMode((value) => !value)}
                onHelp={() => setHelpOpen((value) => !value)} />
              <MapSelector entries={availableMaps} value={artifactPath} onChange={selectMap} />
              <ContextLegend entries={palette.legend} enabledKeys={enabledContextKeys}
                focusedPointCount={focused.length} totalPointCount={points.length}
                supportRules={supportRules} supportCounts={supportCounts}
                onToggleSupport={(key) => setSupportRules((current) => ({ ...current, [key]: !current[key] }))}
                onToggle={toggleContextKeys}
                onIsolate={(keys) => setEnabledContextKeys(new Set(keys))}
                onReset={() => setEnabledContextKeys(null)} />
              {helpOpen && <NavigationHelp onClose={() => setHelpOpen(false)} />}
              {!dockOpen && (
                <button type="button" className="dock-trigger" onClick={() => setDockOpen(true)}>
                  <span>{selected ? "Abrir inspeção" : "Inspeção"}</span><b>›</b>
                </button>
              )}
            </div>
          </div>
          {dockOpen && (
            <Inspector point={selected} slice={slice} artifactUrl={artifactUrl}
              onClose={() => setDockOpen(false)} onFocus={() => cameraActions.current?.focus(selected)}
              outOfFocus={selectedOutOfFocus} />
          )}
        </div>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<Explorer />);
