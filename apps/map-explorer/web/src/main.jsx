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
  validateSlice,
} from "./map-data.js";
import { isEditableTarget } from "./navigation.js";
import "./styles.css";

const COLOR_MODES = { geometry: "Geometria", rgb: "RGB", context: "Contexto" };

// Converte o subconjunto visível em buffers e mantém o índice de picking
// alinhado aos pontos depois de filtros de legenda.
function Cloud({ points, colorMode, onSelect, onFocus }) {
  const pointerOrigin = useRef(null);
  const geometry = useMemo(() => {
    const positions = [];
    const colors = [];
    points.forEach((point) => {
      positions.push(...point.coordinates_m);
      colors.push(...pointColor(point, colorMode).map((channel) => channel / 255));
    });
    const value = new THREE.BufferGeometry();
    value.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    value.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return value;
  }, [points, colorMode]);
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
      <PointMaterial vertexColors size={2.2} sizeAttenuation={false} />
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

// Resume cobertura em blocos compactos e mantém o status de predição visível
// sem ocupar uma faixa textual extensa acima do mapa.
function MapStats({ slice }) {
  const summary = slice.context_summary;
  return (
    <div className="map-stats" aria-label="Resumo do mapa">
      <div><span>Pontos</span><strong>{slice.points.length.toLocaleString("pt-BR")}</strong></div>
      <div><span>RGB</span><strong>{(summary?.rgb_point_count ?? 0).toLocaleString("pt-BR")}</strong></div>
      <div><span>Contexto</span><strong>{(summary?.contextual_point_count ?? 0).toLocaleString("pt-BR")}</strong></div>
      <div className="status-stat"><span>Status</span><strong>{summary ? "Parcial · VLM" : "Geométrico"}</strong></div>
    </div>
  );
}

// Troca a explicação da cor junto com a camada ativa e fornece filtros somente
// onde existem categorias contextuais.
function Legend({ mode, entries, enabledKeys, metrics, points, onToggle, onIsolate, onReset }) {
  if (mode === "geometry") {
    return (
      <div className="map-overlay legend-panel">
        <div className="overlay-title"><span>Legenda</span><strong>Altura</strong></div>
        <div className="height-ramp" />
        <div className="ramp-labels"><span>{metrics.minimumHeight.toFixed(1)} m</span><span>{metrics.maximumHeight.toFixed(1)} m</span></div>
        <p>Cor técnica calculada pelo eixo Z.</p>
      </div>
    );
  }
  if (mode === "rgb") {
    const observed = points.filter((point) => point.association?.color_rgb).length;
    return (
      <div className="map-overlay legend-panel">
        <div className="overlay-title"><span>Legenda</span><strong>RGB</strong></div>
        <div className="legend-static"><i className="rainbow-swatch" /><span>Cor física da câmera</span><b>{observed.toLocaleString("pt-BR")}</b></div>
        <div className="legend-static"><i style={{ background: "rgb(48 53 64)" }} /><span>Sem RGB</span><b>{(points.length - observed).toLocaleString("pt-BR")}</b></div>
      </div>
    );
  }
  return (
    <div className="map-overlay legend-panel context-legend">
      <div className="overlay-title">
        <span>Legenda</span><button type="button" onClick={onReset}>Mostrar tudo</button>
      </div>
      <div className="legend-list">
        {entries.map((entry) => {
          const enabled = enabledKeys === null || enabledKeys.has(entry.key);
          return (
            <div className={`legend-row ${enabled ? "" : "disabled"}`} key={entry.key}>
              <button type="button" className="legend-toggle" aria-pressed={enabled} onClick={() => onToggle(entry.key)}>
                <i style={{ background: `rgb(${entry.color.join(" ")})` }} />
                <span>{entry.label}</span><b>{entry.count.toLocaleString("pt-BR")}</b>
              </button>
              <button type="button" className="isolate-action" onClick={() => onIsolate(entry.key)}>Isolar</button>
            </div>
          );
        })}
      </div>
    </div>
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
  const [colorMode, setColorMode] = useState("geometry");
  const [enabledContextKeys, setEnabledContextKeys] = useState(null);
  const [artifactUrl, setArtifactUrl] = useState(null);
  const [dockOpen, setDockOpen] = useState(false);
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
    () => filterPoints(points, colorMode, enabledContextKeys),
    [points, colorMode, enabledContextKeys],
  );
  const selectedHidden = Boolean(
    selected
    && colorMode === "context"
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
    setDockOpen(false);
    setFlyMode(false);
    setHelpOpen(false);
    setArtifactUrl(resolvedUrl);
    setEnabledContextKeys(null);
    setColorMode(validated.schema_version === 2 ? "context" : "geometry");
    setError(null);
  };

  // Carrega automaticamente a URL pedida e cancela a leitura se a aplicação
  // desmontar antes da resposta.
  useEffect(() => {
    const requestedUrl = new URLSearchParams(window.location.search).get("artifact");
    if (!requestedUrl) return undefined;
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

  // Lê um arquivo local mantendo previews relativos explicitamente
  // indisponíveis, já que o browser não expõe sua pasta de origem.
  const loadFile = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      await openSlice(JSON.parse(await file.text()));
    } catch (failure) {
      setSlice(null);
      setPoints([]);
      setSelected(null);
      setError(failure instanceof Error ? failure.message : "Não foi possível abrir o artifact.");
    }
  };

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
      <header className="topbar">
        <div className="brand-block"><span>CM/3D</span><div><h1>Contextual Map Explorer</h1><p>Geometria + evidência + contexto</p></div></div>
        {slice && <div className="map-identity"><span>Mapa</span><strong>{slice.map_id}</strong><code>{slice.map_frame}</code></div>}
        <label className="file-action"><span>Abrir artifact</span><input aria-label="Artifact do mapa" type="file" accept="application/json,.json" onChange={loadFile} /></label>
      </header>
      {error && <p role="alert" className="error-banner">{error}</p>}
      {!slice && !error && <div className="empty-stage"><strong>ABRA UM MAPA</strong><p>Selecione um artifact JSON para iniciar a exploração.</p></div>}
      {slice && (
        <div className={`workspace ${dockOpen ? "with-dock" : ""}`}>
          <div className="map-column">
            <div className="map-header">
              <MapStats slice={slice} />
              <nav className="layer-switch" aria-label="Camada de coloração">
                {Object.entries(COLOR_MODES).map(([value, label]) => (
                  <button type="button" key={value} className={colorMode === value ? "active" : ""}
                    aria-pressed={colorMode === value} onClick={() => setColorMode(value)}
                    disabled={value !== "geometry" && slice.schema_version < 2}>
                    {label}
                  </button>
                ))}
              </nav>
            </div>
            <div className={`viewport ${flyMode ? "fly-active" : ""}`}>
              <Canvas camera={{ position: [0, -4, 2], near: Math.max(metrics.diagonal * 0.00001, 0.001), far: metrics.diagonal * 100, up: [0, 0, 1] }}>
                <color attach="background" args={["#0d0d0d"]} />
                <fog attach="fog" args={["#0d0d0d", metrics.diagonal * 2, metrics.diagonal * 12]} />
                <SpatialReference metrics={metrics} />
                <Cloud points={visiblePoints} colorMode={colorMode} onSelect={selectPoint}
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
              <Legend mode={colorMode} entries={legendEntries} enabledKeys={enabledContextKeys}
                metrics={metrics} points={points} onToggle={toggleContextKey}
                onIsolate={(key) => setEnabledContextKeys(new Set([key]))}
                onReset={() => setEnabledContextKeys(null)} />
              <div className="viewport-status">
                <span className={flyMode ? "active-dot" : ""}>{flyMode ? "MODO VOO" : "EXPLORAR"}</span>
                <span>{visiblePoints.length.toLocaleString("pt-BR")} / {points.length.toLocaleString("pt-BR")} pontos</span>
              </div>
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
