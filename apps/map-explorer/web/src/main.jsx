import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Canvas } from "@react-three/fiber";
import { PointMaterial } from "@react-three/drei";
import * as THREE from "three";

import { CameraRig } from "./camera-rig.jsx";
import { Inspector } from "./inspector.jsx";
import {
  StaticArtifactGeometrySource,
  buildContextLegend,
  contextKey,
  filterPoints,
  measureMap,
  pointColor,
  srgbColorToLinear,
  validateSlice,
} from "./map-data.js";
import { isEditableTarget } from "./navigation.js";
import "./styles.css";

// Converte o subconjunto visível em buffers e mantém o índice de picking
// alinhado aos pontos depois de filtros de legenda.
function Cloud({ points, onSelect, onFocus }) {
  const pointerOrigin = useRef(null);
  const geometry = useMemo(() => {
    const positions = [];
    const colors = [];
    points.forEach((point) => {
      positions.push(...point.coordinates_m);
      colors.push(...srgbColorToLinear(pointColor(point)));
    });
    const value = new THREE.BufferGeometry();
    value.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    value.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return value;
  }, [points]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  // Distingue clique de arraste para que orbitar ou fazer pan não selecione um
  // ponto acidentalmente ao soltar o mouse.
  const selectIfStationary = (event, focus = false) => {
    const origin = pointerOrigin.current;
    const distance = origin ? Math.hypot(event.clientX - origin.x, event.clientY - origin.y) : 0;
    if (distance > 4) return;
    event.stopPropagation();
    const point = points[event.index] ?? null;
    onSelect(point);
    if (focus && point) onFocus(point);
  };

  return (
    <points
      geometry={geometry}
      onPointerDown={(event) => { pointerOrigin.current = { x: event.clientX, y: event.clientY }; }}
      onClick={(event) => selectIfStationary(event)}
      onDoubleClick={(event) => selectIfStationary(event, true)}
    >
      <PointMaterial vertexColors size={2.2} sizeAttenuation={false} toneMapped={false} fog={false} />
    </points>
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
function ContextLegend({ entries, enabledKeys, visiblePointCount, totalPointCount, onToggle, onIsolate, onReset }) {
  return (
    <details className="map-overlay context-legend" open>
      <summary>
        <span>Legenda</span>
        <small>{visiblePointCount.toLocaleString("pt-BR")} / {totalPointCount.toLocaleString("pt-BR")}</small>
      </summary>
      <div className="legend-actions">
        <span>Contexto</span>
        <button type="button" onClick={onReset}>Mostrar tudo</button>
      </div>
      <div className="legend-list">
        {entries.map((entry) => {
          const enabled = enabledKeys === null || enabledKeys.has(entry.key);
          return (
            <div className={`legend-row ${enabled ? "" : "disabled"}`} key={entry.key}>
              <button type="button" className="legend-toggle" aria-pressed={enabled} onClick={() => onToggle(entry.key)}>
                <i style={{ background: `rgb(${entry.color.join(" ")})` }} />
                <span>{entry.label}</span>
                <b>{entry.count.toLocaleString("pt-BR")}</b>
              </button>
              <button type="button" className="isolate-action" onClick={() => onIsolate(entry.key)}>Isolar</button>
            </div>
          );
        })}
      </div>
    </details>
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
  const legendEntries = useMemo(() => buildContextLegend(points), [points]);
  const visiblePoints = useMemo(
    () => filterPoints(points, enabledContextKeys),
    [points, enabledContextKeys],
  );
  const selectedHidden = Boolean(
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

  // Carrega automaticamente a URL pedida e cancela a leitura se a aplicação
  // desmontar antes da resposta.
  useEffect(() => {
    const requestedUrl = new URLSearchParams(window.location.search).get("artifact") ?? "/current-map.json";
    const resolvedUrl = new URL(requestedUrl, window.location.href).href;
    const controller = new AbortController();
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
  }, []);

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
  // ``null`` continue representando o estado barato “todos visíveis”.
  const toggleContextKey = (key) => {
    setEnabledContextKeys((current) => {
      const next = new Set(current ?? legendEntries.map((entry) => entry.key));
      if (next.has(key)) next.delete(key);
      else next.add(key);
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
              <Canvas camera={{ position: [0, -4, 2], near: Math.max(metrics.diagonal * 0.00001, 0.001), far: metrics.diagonal * 100, up: [0, 0, 1] }}>
                <color attach="background" args={["#0d0d0d"]} />
                <SpatialReference metrics={metrics} />
                <Cloud points={visiblePoints} onSelect={selectPoint}
                  onFocus={(point) => cameraActions.current?.focus(point)} />
                <SelectionMarker point={selectedHidden ? null : selected} radius={Math.min(Math.max(metrics.diagonal * 0.0015, 0.04), 0.35)} />
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
              <ContextLegend entries={legendEntries} enabledKeys={enabledContextKeys}
                visiblePointCount={visiblePoints.length} totalPointCount={points.length}
                onToggle={toggleContextKey}
                onIsolate={(key) => setEnabledContextKeys(new Set([key]))}
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
              hiddenByFilter={selectedHidden} />
          )}
        </div>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<Explorer />);
