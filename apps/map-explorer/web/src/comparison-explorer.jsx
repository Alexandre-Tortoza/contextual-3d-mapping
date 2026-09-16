import React, { useEffect, useMemo, useState } from "react";

import { comparisonGroups, comparisonMatrix, sharedObservationIds } from "./comparison-data.js";
import { resolveAssetUrl, validateSlice } from "./map-data.js";

// Busca os artifacts de todas as células da matriz em paralelo. O artifact
// completo é necessário aqui porque previews e observation_id pertencem à run,
// enquanto o catálogo publica deliberadamente só metadados compactos.
function useComparisonSlices(entries) {
  const [state, setState] = useState({ loading: false, slices: new Map(), error: null });
  const key = entries.map((entry) => entry.url).sort().join("|");

  useEffect(() => {
    if (!entries.length) {
      setState({ loading: false, slices: new Map(), error: null });
      return undefined;
    }
    const controller = new AbortController();
    setState({ loading: true, slices: new Map(), error: null });
    Promise.all(entries.map(async (entry) => {
      const url = new URL(entry.url, window.location.href).href;
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) throw new Error(`${entry.label}: HTTP ${response.status}.`);
      return [entry.url, { slice: validateSlice(await response.json()), artifactUrl: url }];
    }))
      .then((loaded) => {
        if (!controller.signal.aborted) setState({ loading: false, slices: new Map(loaded), error: null });
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setState({ loading: false, slices: new Map(), error: failure.message });
      });
    return () => controller.abort();
  }, [key]);

  return state;
}

// Resolve a observação de uma célula pelo contract estável observation_id, não
// pelo índice da lista. Isso impede que um backend com ordenação diferente
// pareça estar comparando o mesmo frame quando não está.
function observationFor(slice, observationId) {
  return slice?.observations?.find((item) => item.observation_id === observationId) ?? null;
}

// Desenha uma célula com o overlay da run e métricas mínimas para decisão
// rápida. O clique só altera a run ativa; a navegação ao viewport 3D continua
// uma ação explícita no cabeçalho, evitando perder a matriz por acidente.
function ComparisonCell({ entry, loaded, observationId, active, onActivate }) {
  if (!entry) return <div className="comparison-cell comparison-cell-empty">Não executada</div>;
  const observation = observationFor(loaded?.slice, observationId);
  const imageUri = observation?.overlay_image_uri ?? observation?.raw_image_uri;
  const imageUrl = imageUri ? resolveAssetUrl(imageUri, loaded?.artifactUrl) : null;
  const contextualPoints = loaded?.slice?.context_summary?.contextual_point_count;
  return (
    <button type="button" className={`comparison-cell ${active ? "active" : ""}`} onClick={() => onActivate(entry.url)}>
      <span className="comparison-cell-title">{entry.label}</span>
      {imageUrl ? <img src={imageUrl} alt={`Overlay da run ${entry.label}`} /> : <span className="comparison-no-preview">Preview indisponível</span>}
      <span className="comparison-cell-metrics">
        <b>{loaded?.slice?.regions?.length ?? "—"}</b> regiões · <b>{contextualPoints ?? "—"}</b> pontos
      </span>
    </button>
  );
}

// Página de comparação de experimentos sobre o mesmo frame. A matriz usa dois
// eixos publicados pelo pipeline e mantém uma única run ativa para integrar a
// leitura visual com o viewer 3D, sem criar múltiplos canvases WebGL caros.
export function ComparisonExplorer({ availableMaps, artifactPath, onSelectArtifact, requestedObservationId, onBack }) {
  const groups = useMemo(() => comparisonGroups(availableMaps), [availableMaps]);
  const activeGroupId = availableMaps.find((entry) => entry.url === artifactPath)?.comparison?.groupId ?? null;
  const [groupId, setGroupId] = useState(activeGroupId);

  useEffect(() => {
    setGroupId((current) => (
      groups.some((group) => group.id === current) ? current : activeGroupId ?? groups[0]?.id ?? null
    ));
  }, [activeGroupId, groups]);

  const group = groups.find((item) => item.id === groupId) ?? null;
  const entries = group?.runs ?? [];
  const matrix = useMemo(() => comparisonMatrix(entries), [entries]);
  const { loading, slices, error } = useComparisonSlices(entries);
  const sharedFrames = useMemo(
    () => sharedObservationIds([...slices.values()].map((item) => item.slice)),
    [slices],
  );
  const [observationId, setObservationId] = useState(null);

  useEffect(() => {
    setObservationId((current) => (
      sharedFrames.includes(requestedObservationId) ? requestedObservationId
        : sharedFrames.includes(current) ? current : sharedFrames[0] ?? null
    ));
  }, [groupId, requestedObservationId, sharedFrames]);

  return (
    <main className="app-shell comparison-explorer">
      <header className="comparison-header">
        <div className="overlay-title">
          <strong>Comparar backends</strong>
          <button type="button" className="text-action" onClick={onBack}>Ver mapa 3D</button>
        </div>
        {groups.length > 0 && (
          <label className="comparison-select">
            <span>Grupo / frame</span>
            <select value={groupId ?? ""} onChange={(event) => setGroupId(event.target.value)}>
              {groups.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.runs.length} runs</option>)}
            </select>
          </label>
        )}
        {sharedFrames.length > 1 && (
          <label className="comparison-select">
            <span>Frame</span>
            <select value={observationId ?? ""} onChange={(event) => setObservationId(event.target.value)}>
              {sharedFrames.map((id) => <option key={id} value={id}>{id}</option>)}
            </select>
          </label>
        )}
      </header>

      {!groups.length && (
        <section className="comparison-message">
          <strong>Não há grupos de comparação publicados.</strong>
          <p>Publique cada run com <code>comparison.group_id</code> comum ao frame e <code>comparison.axes</code> com os backends testados.</p>
        </section>
      )}
      {loading && <p className="comparison-message">Carregando runs da matriz…</p>}
      {error && <p role="alert" className="comparison-message">Não foi possível abrir a comparação: {error}</p>}
      {!loading && !error && group && matrix.axes.length < 2 && (
        <section className="comparison-message">
          <strong>Esta comparação não define dois ou mais eixos variáveis.</strong>
          <p>O viewer encontrou {matrix.axes.length} eixo(s) com mais de uma opção; a matriz exige ao menos dois.</p>
        </section>
      )}
      {!loading && !error && matrix.axes.length >= 2 && !observationId && (
        <section className="comparison-message">
          <strong>As runs não compartilham um frame publicado.</strong>
          <p>Para comparar, todas precisam declarar o mesmo <code>observation_id</code>.</p>
        </section>
      )}
      {!loading && !error && matrix.axes.length >= 2 && observationId && (
        <section className="comparison-workspace" aria-label={`Matriz de comparação do frame ${observationId}`}>
          <p className="comparison-caption">Frame <b>{observationId}</b> · clique em uma célula para torná-la a run ativa; depois abra o mapa 3D.</p>
          <div className="comparison-grid" style={{ "--comparison-columns": matrix.columns.length }}>
            <div className="comparison-axis-corner">
              {matrix.axes[0].name} ↓<br />
              {matrix.columnAxes.map((axis) => axis.name).join(" · ")} →
            </div>
            {matrix.columns.map((option) => <strong className="comparison-column" key={option}>{option}</strong>)}
            {matrix.rows.map((row) => (
              <React.Fragment key={row.option}>
                <strong className="comparison-row">{row.option}</strong>
                {row.cells.map((entry, index) => (
                  <ComparisonCell key={`${row.option}-${matrix.columns[index]}`} entry={entry}
                    loaded={entry ? slices.get(entry.url) : null} observationId={observationId}
                    active={entry?.url === artifactPath} onActivate={onSelectArtifact} />
                ))}
              </React.Fragment>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
