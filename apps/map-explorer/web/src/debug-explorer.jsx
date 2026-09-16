import React, { useEffect, useMemo, useState } from "react";

import { compositionEntriesOf, debugGalleryFor, pipelineBackendsOf } from "./debug-explorer-data.js";
import { CONSOLIDATED_ARTIFACT_TYPE, CONTEXT_ARTIFACT_TYPE, debugStagesOf, resolveAssetUrl } from "./map-data.js";

// Busca e formata um JSON de debug sob demanda, só quando o <details> é
// aberto. Uma run pode publicar dezenas de documentos (diagnostics, audits,
// composição), e baixar todos de uma vez tornaria a página inútil numa
// conexão lenta — o custo de rede só existe se alguém realmente quiser ler.
function JsonDisclosure({ name, uri, artifactUrl }) {
  const [state, setState] = useState({ status: "idle", text: "" });
  const href = resolveAssetUrl(uri, artifactUrl);

  const load = () => {
    if (state.status !== "idle" || !href) return;
    setState({ status: "loading", text: "" });
    fetch(href)
      .then((response) => {
        if (!response.ok) throw new Error(`O servidor respondeu HTTP ${response.status}.`);
        return response.json();
      })
      .then((payload) => setState({ status: "done", text: JSON.stringify(payload, null, 2) }))
      .catch((failure) => setState({ status: "error", text: failure.message }));
  };

  return (
    <details className="debug-json-disclosure" onToggle={(event) => event.target.open && load()}>
      <summary>{name}</summary>
      {!href && <p className="empty-copy">Indisponível sem o artifact aberto pelo servidor.</p>}
      {state.status === "loading" && <p className="empty-copy">Carregando…</p>}
      {state.status === "error" && <p className="empty-copy">Falha ao carregar: {state.text}</p>}
      {state.status === "done" && <pre>{state.text}</pre>}
    </details>
  );
}

// Mostra as etapas presentes numa observação como badges compactos, para que
// a lista lateral revele a cobertura de debug de cada frame sem precisar
// abri-lo primeiro.
function StageBadges({ observation }) {
  const stages = debugStagesOf(observation);
  if (!stages.length) return <span className="debug-badge debug-badge-empty">sem debug</span>;
  return stages.map(([stage]) => <span className="debug-badge" key={stage}>{stage}</span>);
}

// Reaproveita a mesma lista de runs do MapSelector (main.jsx), a partir do
// catálogo compartilhado por useMapCatalog — nenhuma lógica de índice é
// duplicada aqui, só a apresentação muda para caber no cabeçalho desta página.
function RunSelector({ availableMaps, artifactPath, onSelectArtifact }) {
  const runs = availableMaps.filter((entry) => entry.artifactType === CONTEXT_ARTIFACT_TYPE);
  const consolidated = availableMaps.filter((entry) => entry.artifactType === CONSOLIDATED_ARTIFACT_TYPE);
  return (
    <label className="map-overlay map-selector debug-run-selector">
      <span>Run</span>
      <select aria-label="Run" value={artifactPath ?? ""} onChange={(event) => onSelectArtifact(event.target.value)}>
        {runs.length > 0 && (
          <optgroup label="Runs">{runs.map((entry) => <option value={entry.url} key={entry.url}>{entry.label}</option>)}</optgroup>
        )}
        {consolidated.length > 0 && (
          <optgroup label="Mapas consolidados">{consolidated.map((entry) => <option value={entry.url} key={entry.url}>{entry.label}</option>)}</optgroup>
        )}
      </select>
    </label>
  );
}

// Mostra o backend e checkpoint de um estágio, e a alternativa configurada
// só quando ela existe — sem fallback, a chave nem aparece no manifest, então
// não há nada a listar como "opção não escolhida".
function BackendRow({ stage, backend }) {
  return (
    <div className="debug-backend-row">
      <span className="debug-backend-stage">{stage}</span>
      <strong>{backend.backend}</strong>
      {backend.checkpoint && <code className="debug-backend-checkpoint">{backend.checkpoint}</code>}
      {backend.fallback_backend && (
        <span className="debug-backend-fallback">
          alternativa: <strong>{backend.fallback_backend}</strong>
          {backend.fallback_checkpoint && <code className="debug-backend-checkpoint">{backend.fallback_checkpoint}</code>}
        </span>
      )}
    </div>
  );
}

// Responde "qual modelo gerou isso" sem exigir abrir o manifest.json da run
// de percepção manualmente. Uma run isolada mostra os backends direto; um
// mapa consolidado agrupa por run de origem, porque runs diferentes podem
// ter comparado backends diferentes para o mesmo trecho.
function PipelineBackends({ slice }) {
  const entries = pipelineBackendsOf(slice);
  if (!entries.length) return null;
  return (
    <section className="inspector-block debug-backends">
      <h3>Backends do pipeline</h3>
      {entries.map(({ runId, backends }) => (
        <div className="debug-backend-group" key={runId ?? "single-run"}>
          {runId && <p className="debug-backend-run-label">{runId}</p>}
          {Object.entries(backends).map(([stage, backend]) => (
            <BackendRow key={stage} stage={stage} backend={backend} />
          ))}
        </div>
      ))}
    </section>
  );
}

// Página inteira de explicabilidade: para a run selecionada, mostra todos os
// artefatos de debug publicados pelo pipeline — organizados por observação e
// por etapa — sem competir por espaço com o viewer 3D. Existe porque o
// Inspector (inspector.jsx) é só uma prévia rápida sobre o ponto selecionado;
// auditar um pipeline inteiro (SAM3, grounding, tiles de discovery, composição
// da run) precisa de uma tela própria com galeria e JSON completos.
export function DebugExplorer({ availableMaps, slice, artifactUrl, artifactPath, onSelectArtifact, requestedObservationId, onBack }) {
  const observations = slice?.observations ?? [];
  const [selectedId, setSelectedId] = useState(observations[0]?.observation_id ?? null);

  // Troca de run: a observação selecionada da run anterior não existe mais,
  // então volta para a primeira observação da run recém-aberta.
  useEffect(() => {
    setSelectedId(observations.some((item) => item.observation_id === requestedObservationId)
      ? requestedObservationId
      : observations[0]?.observation_id ?? null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slice, requestedObservationId]);

  const selected = observations.find((item) => item.observation_id === selectedId) ?? null;
  const gallery = useMemo(() => (selected ? debugGalleryFor(selected) : []), [selected]);
  const compositionEntries = useMemo(() => compositionEntriesOf(slice), [slice]);
  const hasAnyDebug = observations.some((item) => debugStagesOf(item).length > 0);

  return (
    <main className="app-shell debug-explorer">
      <header className="debug-header">
        <div className="overlay-title">
          <strong>Explicabilidade</strong>
          <button type="button" className="text-action" onClick={onBack}>Voltar ao mapa</button>
        </div>
        <RunSelector availableMaps={availableMaps} artifactPath={artifactPath} onSelectArtifact={onSelectArtifact} />
      </header>

      <PipelineBackends slice={slice} />

      {compositionEntries.length > 0 && (
        <section className="inspector-block debug-composition">
          <h3>Composição da run</h3>
          {compositionEntries.map(([name, uri]) => (
            <JsonDisclosure key={name} name={name} uri={uri} artifactUrl={artifactUrl} />
          ))}
        </section>
      )}

      {!slice ? (
        <p className="empty-copy debug-empty">Carregando run…</p>
      ) : !hasAnyDebug ? (
        <p className="empty-copy debug-empty">Esta run não tem artefatos de debug publicados.</p>
      ) : (
        <div className="debug-body">
          <ul className="debug-frame-list">
            {observations.map((observation) => (
              <li key={observation.observation_id}>
                <button
                  type="button"
                  className={`debug-frame-item ${observation.observation_id === selectedId ? "active" : ""}`}
                  onClick={() => setSelectedId(observation.observation_id)}
                >
                  <strong>{observation.frame_id ?? observation.observation_id}</strong>
                  <div className="debug-frame-badges"><StageBadges observation={observation} /></div>
                </button>
              </li>
            ))}
          </ul>
          <div className="debug-gallery">
            {gallery.length === 0 && <p className="empty-copy">Este frame não tem artefatos de debug publicados.</p>}
            {gallery.map(({ stage, images, documents }) => (
              <section className="debug-stage-group inspector-block" key={stage}>
                <h3>{stage}</h3>
                {images.length > 0 && (
                  <div className="debug-thumbs">
                    {images.map(([path, uri]) => (
                      <a
                        key={path}
                        className="debug-thumb"
                        href={resolveAssetUrl(uri, artifactUrl)}
                        target="_blank"
                        rel="noreferrer"
                        title={path}
                      >
                        <img src={resolveAssetUrl(uri, artifactUrl)} alt={path} />
                        <span>{path}</span>
                      </a>
                    ))}
                  </div>
                )}
                {documents.length > 0 && (
                  <div className="debug-documents">
                    {documents.map(([path, uri]) => (
                      <JsonDisclosure key={path} name={path} uri={uri} artifactUrl={artifactUrl} />
                    ))}
                  </div>
                )}
              </section>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}
