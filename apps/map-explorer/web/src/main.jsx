import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { createPortal } from "react-dom";
import { Canvas } from "@react-three/fiber";
import { PointMaterial } from "@react-three/drei";
import * as THREE from "three";

import { CameraRig } from "./camera-rig.jsx";
import { DebugExplorer } from "./debug-explorer.jsx";
import { ComparisonExplorer } from "./comparison-explorer.jsx";
import { Inspector } from "./inspector.jsx";
import {
  DIMMED_COLOR,
  CONSOLIDATED_ARTIFACT_TYPE,
  CONTEXT_ARTIFACT_TYPE,
  allLegendKeys,
  buildContextPalette,
  contextKey,
  countSupportStates,
  geometryIdentity,
  legendKeys,
  measureMap,
  partitionByFocus,
  srgbColorToLinear,
} from "./map-data.js";
import { isEditableTarget } from "./navigation.js";
import { nearestIntersection, pickThreshold, pointFromIntersection } from "./picking.js";
import { useMapCatalog } from "./use-map-catalog.js";
import "./styles.css";

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

// Desenha a dica flutuante fora do painel, que recorta o próprio conteúdo.
// Existe porque a legenda é estreita por design: nomes cortados e ações com
// ícone precisam de um lugar para dizer o que são sem alargar a coluna.
function HoverTip({ text, anchor }) {
  const style = {
    left: Math.max(8, Math.min(anchor.left, window.innerWidth - 268)),
    top: anchor.placement === "above" ? anchor.top : anchor.bottom,
    transform: anchor.placement === "above" ? "translateY(-100%)" : "none",
  };
  return createPortal(
    <span className="hover-tip" role="tooltip" style={style}>{text}</span>,
    document.body,
  );
}

// Só uma dica fica aberta por vez. O registro é de módulo porque cada linha da
// legenda tem a sua, e o ponteiro pode sair de uma linha sem passar por outra
// quando a lista rola sob o cursor.
let openTip = null;

// Controla uma dica ancorada em um elemento da legenda. Com `whenTruncated`,
// mede o elemento e só arma a dica quando o texto não coube — é o caso dos
// rótulos, que variam muito de comprimento e não devem ganhar tooltip redundante.
// Sem ele, a dica vale sempre: é o caso dos botões de ação, que são só ícone.
function useHoverTip(text, { whenTruncated = false } = {}) {
  const ref = useRef(null);
  const [eligible, setEligible] = useState(!whenTruncated);
  const [anchor, setAnchor] = useState(null);
  const dismiss = useRef(null);
  if (!dismiss.current) dismiss.current = () => setAnchor(null);

  useEffect(() => {
    if (!whenTruncated) return undefined;
    const node = ref.current;
    if (!node) return undefined;
    const measure = () => setEligible(node.scrollWidth > node.clientWidth + 1);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [text, whenTruncated]);

  const place = () => {
    const node = ref.current;
    if (!node) return;
    if (openTip && openTip !== dismiss.current) openTip();
    openTip = dismiss.current;
    const box = node.getBoundingClientRect();
    setAnchor({
      left: box.left - 9,
      top: box.top - 8,
      bottom: box.bottom + 8,
      placement: box.top > 96 ? "above" : "below",
    });
  };
  const hide = () => {
    if (openTip === dismiss.current) openTip = null;
    dismiss.current();
  };

  // Libera o registro se o elemento sair da tela com a dica aberta: a lista
  // recolhe famílias e troca de mapa, e a próxima dica não deve ficar presa
  // atrás de um dono que não existe mais.
  useEffect(() => () => {
    if (openTip === dismiss.current) openTip = null;
  }, []);

  // Reancora enquanto a dica está aberta: a lista da legenda rola, e uma dica
  // de posição fixa ficaria apontando para a linha errada.
  useEffect(() => {
    if (!anchor) return undefined;
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [anchor !== null]);

  const handlers = eligible
    ? { onMouseEnter: place, onMouseLeave: hide, onFocus: place, onBlur: hide }
    : {};
  return { ref, handlers, tip: eligible && anchor ? <HoverTip text={text} anchor={anchor} /> : null };
}

// Desenha o botão que liga e desliga uma linha da legenda, seja uma família ou
// um label bruto. Concentra swatch, rótulo, contagem e dica para que os dois
// níveis da lista tenham exatamente o mesmo comportamento de leitura.
function LegendToggle({ label, count, state, member = false, onClick }) {
  const { ref, handlers, tip } = useHoverTip(label, { whenTruncated: true });
  const pressed = state === "partial" ? "mixed" : state === "on";
  return (
    <button
      type="button"
      className="legend-toggle"
      aria-pressed={pressed}
      aria-label={`${label}, ${count.toLocaleString("pt-BR")} pontos`}
      onClick={onClick}
      onFocus={handlers.onFocus}
      onBlur={handlers.onBlur}
    >
      <i className={member ? "legend-dot" : "legend-swatch"} />
      <span
        className="legend-label"
        ref={ref}
        onMouseEnter={handlers.onMouseEnter}
        onMouseLeave={handlers.onMouseLeave}
      >
        {label}
      </span>
      <b>{count.toLocaleString("pt-BR")}</b>
      {tip}
    </button>
  );
}

// Botão de ação de uma linha: só ícone, com a dica explicando o que ele faz
// sobre qual classe. Mantém a lista legível quando há muitas linhas, já que a
// mesma palavra repetida em cada uma competia com os nomes das classes.
function LegendAction({ hint, className = "", children, ...rest }) {
  const { ref, handlers, tip } = useHoverTip(hint);
  return (
    <button type="button" className={`legend-action ${className}`.trim()} aria-label={hint}
      ref={ref} {...handlers} {...rest}>
      {children}
      {tip}
    </button>
  );
}

// Compara dois recortes de foco. Existe para que os botões do painel saibam
// quando não têm mais nada a fazer: `null` significa "todas as chaves", e dois
// conjuntos com as mesmas chaves são o mesmo recorte, ainda que sejam objetos
// diferentes a cada alternância.
function sameKeySet(left, right) {
  if (left === right) return true;
  if (left === null || right === null) return false;
  return left.size === right.size && [...left].every((key) => right.has(key));
}

// Oferece a leitura e os filtros das cores em um painel pequeno e recolhível,
// mantendo a sidebar reservada à evidência do ponto selecionado.
function ContextLegend({
  entries,
  enabledKeys,
  defaultKeys,
  focusedPointCount,
  totalPointCount,
  supportRules,
  supportCounts,
  onToggle,
  onIsolate,
  onReset,
  onFocusAll,
  onToggleSupport,
}) {
  const [expanded, setExpanded] = useState(new Set());
  const toggleExpanded = (key) => setExpanded((current) => {
    const next = new Set(current);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  });

  // A barra de cada linha compara classes dentro da própria seção. Contra o
  // total do mapa, ou contra a cobertura visual, toda classe publicada viraria
  // um traço invisível — e é justamente a classe rara que interessa ler.
  const peakOf = (semantic) => entries.reduce(
    (peak, entry) => (entry.semantic === semantic ? Math.max(peak, entry.count) : peak),
    0,
  ) || 1;
  const peaks = { published: peakOf(true), coverage: peakOf(false) };
  const share = (count, peak) => `${Math.max((count / peak) * 100, 2)}%`;
  const isOn = (key) => enabledKeys === null || enabledKeys.has(key);
  const hiddenCount = entries.filter((entry) => !legendKeys(entry).some(isOn)).length;
  const focusShare = totalPointCount > 0 ? (focusedPointCount / totalPointCount) * 100 : 0;
  const filtered = focusedPointCount < totalPointCount;
  const everythingOn = entries.every((entry) => legendKeys(entry).every(isOn));
  const atDefault = sameKeySet(enabledKeys, defaultKeys);

  return (
    <details className="map-overlay context-legend" open>
      <summary>
        <span>Legenda</span>
        <small>
          <b>{focusedPointCount.toLocaleString("pt-BR")}</b>
          <span className="of-total"> / {totalPointCount.toLocaleString("pt-BR")}</span> pts
        </small>
        <b className="legend-caret" aria-hidden="true">›</b>
        {filtered && <div className="legend-focus-bar" style={{ "--focus-share": `${focusShare}%` }} />}
      </summary>
      <div className="legend-actions">
        <span>
          {hiddenCount > 0
            ? `${Math.round(focusShare)}% em foco · ${hiddenCount} oculta${hiddenCount > 1 ? "s" : ""}`
            : filtered ? `${Math.round(focusShare)}% em foco` : "Mapa inteiro em foco"}
        </span>
        {defaultKeys !== null && (
          <LegendAction hint="Padrão: esconde as estruturas genéricas"
            onClick={onReset} disabled={atDefault}>Padrão</LegendAction>
        )}
        <LegendAction hint="Tudo: traz todas as classes de volta ao foco"
          onClick={onFocusAll} disabled={everythingOn}>Tudo</LegendAction>
      </div>
      <div className="legend-support">
        <strong>Sustentação da evidência</strong>
        <label className={supportCounts.weak === 0 ? "empty" : ""}>
          <input
            type="checkbox"
            checked={supportRules.dimWeak}
            onChange={() => onToggleSupport("dimWeak")}
          />
          <span>Atenuar evidência com suporte fraco</span>
          <b>{supportCounts.weak.toLocaleString("pt-BR")}</b>
        </label>
        <label className={supportCounts.uncorroborated === 0 ? "empty" : ""}>
          <input
            type="checkbox"
            checked={supportRules.dimUncorroborated}
            onChange={() => onToggleSupport("dimUncorroborated")}
          />
          <span>Atenuar evidência de um único frame</span>
          <b>{supportCounts.uncorroborated.toLocaleString("pt-BR")}</b>
        </label>
      </div>
      <div className="legend-list">
        {entries.map((entry, index) => {
          const keys = legendKeys(entry);
          const activeKeys = keys.filter((key) => enabledKeys === null || enabledKeys.has(key)).length;
          const state = activeKeys === keys.length ? "on" : activeKeys === 0 ? "off" : "partial";
          const detailed = entry.members.length > 1;
          const open = expanded.has(entry.key);
          return (
            <React.Fragment key={entry.key}>
              {(index === 0 || entry.semantic !== entries[index - 1].semantic) && (
                <div className="legend-section">
                  {entry.semantic ? "Evidência publicada" : "Cobertura visual"}
                </div>
              )}
              <div className="legend-group">
                <div
                  className={`legend-row state-${state}`}
                  style={{
                    "--tint": entry.color.join(" "),
                    "--share": share(entry.count, entry.semantic ? peaks.published : peaks.coverage),
                  }}
                >
                  <LegendToggle label={entry.label} count={entry.count} state={state}
                    onClick={() => onToggle(keys)} />
                  {detailed && (
                    <LegendAction className={`expand-action ${open ? "" : "closed"}`} aria-expanded={open}
                      hint={`${open ? "Recolher" : "Ver"} os ${entry.members.length} labels de ${entry.label}`}
                      onClick={() => toggleExpanded(entry.key)}>
                      <span>{entry.members.length}</span><b>›</b>
                    </LegendAction>
                  )}
                  <LegendAction className="isolate-action" hint={`Focar somente ${entry.label}`}
                    onClick={() => onIsolate(keys)}>◎</LegendAction>
                </div>
                {detailed && open && entry.members.map((member) => {
                  const memberEnabled = enabledKeys === null || enabledKeys.has(member.key);
                  return (
                    <div
                      className={`legend-row legend-member state-${memberEnabled ? "on" : "off"}`}
                      key={member.key}
                      style={{ "--share": share(member.count, entry.members[0].count) }}
                    >
                      <LegendToggle label={member.label} count={member.count} member
                        state={memberEnabled ? "on" : "off"} onClick={() => onToggle([member.key])} />
                      <LegendAction className="isolate-action" hint={`Focar somente ${member.label}`}
                        onClick={() => onIsolate([member.key])}>◎</LegendAction>
                    </div>
                  );
                })}
              </div>
            </React.Fragment>
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
  const runs = entries.filter((entry) => entry.artifactType === CONTEXT_ARTIFACT_TYPE);
  const consolidated = entries.filter((entry) => entry.artifactType === CONSOLIDATED_ARTIFACT_TYPE);
  return (
    <label className="map-overlay map-selector">
      <span>Run</span>
      <select aria-label="Run com contexto" value={value} onChange={(event) => onChange(event.target.value)}>
        {!known && <option value={value}>Run aberta pela URL</option>}
        {runs.length > 0 && <optgroup label="Runs">{runs.map((entry) => <option value={entry.url} key={entry.url}>{entry.label}</option>)}</optgroup>}
        {consolidated.length > 0 && <optgroup label="Mapas consolidados">{consolidated.map((entry) => <option value={entry.url} key={entry.url}>{entry.label}</option>)}</optgroup>}
      </select>
    </label>
  );
}

// Substitui um erro solto por uma tela explicativa quando nenhuma run está
// aberta. Existe porque um erro de fetch/parse (ex.: JSON inválido) não diz
// por si só o que aconteceu nem o que fazer — esta tela explica a causa
// provável e, quando há outras runs publicadas, oferece a troca direta.
function ArtifactErrorScreen({ error, entries, artifactPath, onChange }) {
  return (
    <div className="artifact-error-screen" role="alert">
      <strong>Não foi possível abrir esta run</strong>
      <p className="artifact-error-message">{error.message}</p>
      {error.explanation && <p className="artifact-error-explanation">{error.explanation}</p>}
      {entries.length > 0 && (
        <MapSelector entries={entries} value={artifactPath} onChange={onChange} />
      )}
    </div>
  );
}

// Expõe os comandos de câmera no próprio mapa para que navegação não dependa
// de conhecer atalhos de teclado.
function CameraToolbar({ flyMode, hasSelection, onReset, onTop, onIsometric, onFocus, onToggleFly, onHelp, onOpenDebug, onOpenCompare }) {
  return (
    <div className="map-overlay camera-toolbar" aria-label="Controles da câmera">
      <button type="button" onClick={onReset}><kbd>Home</kbd><span>Mapa inteiro</span></button>
      <button type="button" onClick={onTop}><kbd>1</kbd><span>Topo</span></button>
      <button type="button" onClick={onIsometric}><kbd>2</kbd><span>Perspectiva</span></button>
      <button type="button" onClick={onFocus} disabled={!hasSelection}><kbd>F</kbd><span>Focar</span></button>
      <button type="button" className={flyMode ? "active" : ""} aria-pressed={flyMode} onClick={onToggleFly}><kbd>V</kbd><span>{flyMode ? "Voando" : "Modo voo"}</span></button>
      <button type="button" onClick={onOpenCompare}><span>Comparar</span></button>
      <button type="button" onClick={onOpenDebug}><span>Explicabilidade</span></button>
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
// estados reposicionar ou reconstruir os demais implicitamente. O catálogo de
// runs (qual artifact está aberto, o índice publicado, os pontos carregados)
// vem de useMapCatalog — compartilhado com DebugExplorer — e este componente
// só possui o estado de navegação e filtro específico do viewer 3D.
function Explorer({ catalog, onOpenDebug, onOpenCompare }) {
  const { artifactPath, availableMaps, catalogLoaded, slice, points, artifactUrl, error, selectMap } = catalog;
  const catalogEmpty = catalogLoaded && availableMaps.length === 0 && !artifactPath;
  const [supportRules, setSupportRules] = useState({ dimWeak: true, dimUncorroborated: false });
  const [selected, setSelected] = useState(null);
  const [enabledContextKeys, setEnabledContextKeys] = useState(() => {
    // Inicializa com estrutural oculto quando o primeiro artifact carrega.
    // Será sobrescrito abaixo quando temos os pontos/paleta.
    return null;
  });
  const [dockOpen, setDockOpen] = useState(true);
  const [flyMode, setFlyMode] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const cameraActions = useRef(null);
  const reducedMotion = useMemo(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    [],
  );
  const metrics = useMemo(() => measureMap(points), [points]);
  const palette = useMemo(() => buildContextPalette(points), [points]);
  const supportCounts = useMemo(() => countSupportStates(points), [points]);

  // Calcula o default de enabledContextKeys: todas as chaves exceto as da
  // família estrutural, que são os labels genéricos de superfície. Usado em
  // dois contextos: ao abrir o artifact e ao resetar o filtro.
  const defaultEnabledKeys = useMemo(() => {
    if (!palette.legend || palette.legend.length === 0) return null;
    const structural = palette.legend.find((entry) => entry.structural);
    if (!structural) return null; // null quando não há o que esconder
    const next = new Set(allLegendKeys(palette.legend));
    legendKeys(structural).forEach((key) => next.delete(key));
    return next;
  }, [palette.legend]);

  const { focused, dimmed } = useMemo(
    () => partitionByFocus(points, enabledContextKeys, supportRules),
    [points, enabledContextKeys, supportRules],
  );
  const selectedOutOfFocus = Boolean(
    selected
    && enabledContextKeys !== null
    && !enabledContextKeys.has(contextKey(selected)),
  );

  // Reseta a navegação e o dock sempre que uma run diferente termina de abrir
  // (useMapCatalog troca a identidade de `slice`). Filtro de contexto é
  // tratado à parte, no efeito de defaultEnabledKeys abaixo.
  useEffect(() => {
    setSelected(null);
    setDockOpen(true);
    setFlyMode(false);
    setHelpOpen(false);
  }, [slice]);

  // Após paleta ser construída (carregamento bem-sucedido de novo artifact),
  // aplica o default de estrutural oculto.
  useEffect(() => {
    if (defaultEnabledKeys !== undefined) {
      setEnabledContextKeys(defaultEnabledKeys);
    }
  }, [defaultEnabledKeys]);

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
  // Materializa o conjunto completo somente no primeiro filtro. Alternar
  // uma família alterna todos os labels que ela agrupa, de uma vez.
  const toggleContextKeys = (keys) => {
    setEnabledContextKeys((current) => {
      // Se está usando o default (estrutural oculto), expande primeiro.
      const base = current ?? defaultEnabledKeys ?? allLegendKeys(palette.legend);
      const next = new Set(base);
      if (keys.every((key) => next.has(key))) keys.forEach((key) => next.delete(key));
      else keys.forEach((key) => next.add(key));
      return next;
    });
  };

  return (
    <main className="app-shell">
      {catalogEmpty ? (
        <div className="empty-catalog" role="status">
          <strong>Nenhuma run publicada ainda</strong>
          <p>
            Publique uma run contextual para começar a explorar o mapa e a
            página de explicabilidade.
          </p>
          <pre>python apps/map-explorer/scripts/publish_map_index.py \{"\n"}  apps/map-explorer/web/public --artifact &lt;context.json&gt;</pre>
        </div>
      ) : (
        <>
          {!slice && error && (
            <ArtifactErrorScreen
              error={error}
              entries={availableMaps}
              artifactPath={artifactPath}
              onChange={selectMap}
            />
          )}
          {!slice && !error && (
            <div className="loading-stage" role="status">
              <span className="loading-spinner" aria-hidden="true" />
              <span>Carregando mapa…</span>
            </div>
          )}
        </>
      )}
      {slice && (
        <div className={`workspace ${dockOpen ? "with-dock" : ""}`}>
          {error && <p role="alert" className="error-banner">{error.message}</p>}
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
                  mapKey={geometryIdentity(slice)} reducedMotion={reducedMotion} />
              </Canvas>
              <CameraToolbar flyMode={flyMode} hasSelection={Boolean(selected)}
                onReset={() => cameraActions.current?.reset()}
                onTop={() => cameraActions.current?.preset("top")}
                onIsometric={() => cameraActions.current?.preset("isometric")}
                onFocus={() => cameraActions.current?.focus(selected)}
                onToggleFly={() => setFlyMode((value) => !value)}
                onHelp={() => setHelpOpen((value) => !value)}
                onOpenCompare={onOpenCompare}
                onOpenDebug={onOpenDebug} />
              <MapSelector entries={availableMaps} value={artifactPath} onChange={selectMap} />
              <ContextLegend entries={palette.legend} enabledKeys={enabledContextKeys}
                defaultKeys={defaultEnabledKeys}
                focusedPointCount={focused.length} totalPointCount={points.length}
                supportRules={supportRules} supportCounts={supportCounts}
                onToggleSupport={(key) => setSupportRules((current) => ({ ...current, [key]: !current[key] }))}
                onToggle={toggleContextKeys}
                onIsolate={(keys) => setEnabledContextKeys(new Set(keys))}
                onFocusAll={() => setEnabledContextKeys(new Set(allLegendKeys(palette.legend)))}
                onReset={() => setEnabledContextKeys(defaultEnabledKeys)} />
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
              onOpenDebug={onOpenDebug} onOpenCompare={onOpenCompare}
              outOfFocus={selectedOutOfFocus} />
          )}
        </div>
      )}
    </main>
  );
}

// Decide entre a vista de mapa (padrão) e a vista de explicabilidade a partir
// de ?view=debug, compartilhando um único useMapCatalog() entre as duas —
// trocar de vista não deve refazer fetch nem perder a run já carregada. Não
// há router: a troca de vista usa history.pushState e ?view na query string,
// o mesmo padrão já usado por ?artifact para selecionar a run.
function App() {
  const [view, setView] = useState(
    () => new URLSearchParams(window.location.search).get("view"),
  );
  useEffect(() => {
    const onPopState = () => setView(new URLSearchParams(window.location.search).get("view"));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigateTo = (nextView, observationId = null) => {
    const url = new URL(window.location.href);
    if (nextView) url.searchParams.set("view", nextView);
    else url.searchParams.delete("view");
    if (observationId) url.searchParams.set("observation", observationId);
    else url.searchParams.delete("observation");
    window.history.pushState(null, "", url);
    setView(nextView);
  };

  const catalog = useMapCatalog();

  if (view === "debug") {
    return (
      <DebugExplorer
        availableMaps={catalog.availableMaps}
        slice={catalog.slice}
        artifactUrl={catalog.artifactUrl}
        artifactPath={catalog.artifactPath}
        onSelectArtifact={catalog.selectMap}
        requestedObservationId={new URLSearchParams(window.location.search).get("observation")}
        onBack={() => navigateTo(null)}
      />
    );
  }
  if (view === "compare") {
    return (
      <ComparisonExplorer
        availableMaps={catalog.availableMaps}
        artifactPath={catalog.artifactPath}
        onSelectArtifact={catalog.selectMap}
        requestedObservationId={new URLSearchParams(window.location.search).get("observation")}
        onBack={() => navigateTo(null)}
      />
    );
  }
  return <Explorer catalog={catalog} onOpenDebug={(observationId) => navigateTo("debug", observationId)} onOpenCompare={(observationId) => navigateTo("compare", observationId)} />;
}

createRoot(document.getElementById("root")).render(<App />);
