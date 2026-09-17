import { useEffect, useRef, useState } from "react";

import { loadApiBaseUrl, queryMap, saveApiBaseUrl } from "./semantic-search.js";

// Painel de busca textual sobre embeddings semânticos publicados por um mapa
// (#224). O serviço local decide se a capability existe; este componente só
// distingue os estados que o usuário precisa entender: carregando, sem
// resultado, capability indisponível, serviço fora do ar, ou uma lista
// ordenada de matches que reaproveita a seleção/foco já existentes do viewer.
export function SemanticSearchPanel({ mapId, onSelectGeometry }) {
  const [apiBaseUrl, setApiBaseUrl] = useState(loadApiBaseUrl);
  const [configOpen, setConfigOpen] = useState(false);
  const [text, setText] = useState("");
  const [status, setStatus] = useState("idle");
  const [results, setResults] = useState([]);
  const [reason, setReason] = useState(null);
  const controllerRef = useRef(null);

  // Cancela uma busca em andamento se o painel desmontar (troca de run) antes
  // do serviço local responder.
  useEffect(() => () => controllerRef.current?.abort(), []);
  // Uma run diferente invalida os resultados da anterior: nenhum geometry_id
  // buscado antes ainda corresponde à mesma geometria.
  useEffect(() => {
    setStatus("idle");
    setResults([]);
    setReason(null);
  }, [mapId]);

  const runSearch = async (event) => {
    event.preventDefault();
    const query = text.trim();
    if (!query || !mapId) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setStatus("loading");
    setReason(null);
    let outcome;
    try {
      outcome = await queryMap({ apiBaseUrl, mapId, text: query, limit: 10, signal: controller.signal });
    } catch (error) {
      if (error.name === "AbortError") return;
      outcome = { status: "service_unavailable", reason: error.message, results: [] };
    }
    if (outcome.status === "ok") {
      setResults(outcome.results);
      setStatus(outcome.results.length > 0 ? "ok" : "empty");
    } else {
      setResults([]);
      setStatus(outcome.status === "capability_unavailable" ? "unavailable" : "error");
      setReason(outcome.reason);
    }
  };

  const changeApiBaseUrl = (value) => {
    setApiBaseUrl(value);
    saveApiBaseUrl(value);
  };

  return (
    <div className="map-overlay semantic-search">
      <form className="semantic-search-form" onSubmit={runSearch}>
        <span>Busca</span>
        <input
          type="search"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="ex.: caixa de disjuntores"
          aria-label="Buscar por texto no mapa semântico"
        />
        <button type="submit" disabled={status === "loading" || !text.trim()}>
          {status === "loading" ? "…" : "Buscar"}
        </button>
        <button
          type="button"
          className="semantic-search-config-toggle"
          aria-label="Configurar serviço de busca local"
          aria-expanded={configOpen}
          onClick={() => setConfigOpen((value) => !value)}
        >
          ⚙
        </button>
      </form>
      {configOpen && (
        <label className="semantic-search-endpoint">
          <span>Serviço local</span>
          <input
            type="text"
            value={apiBaseUrl}
            onChange={(event) => changeApiBaseUrl(event.target.value)}
            placeholder="http://127.0.0.1:8765"
          />
        </label>
      )}
      {status === "unavailable" && (
        <p className="semantic-search-status" role="status">
          Este mapa não publicou busca semântica.
        </p>
      )}
      {status === "service_unavailable" || status === "error" ? (
        <p className="semantic-search-status semantic-search-status-error" role="alert">
          Não foi possível consultar o serviço local{reason ? `: ${reason}` : "."}
        </p>
      ) : null}
      {status === "map_not_found" && (
        <p className="semantic-search-status semantic-search-status-error" role="alert">
          O serviço local não conhece este mapa.
        </p>
      )}
      {status === "empty" && (
        <p className="semantic-search-status" role="status">
          Nenhum resultado para essa busca.
        </p>
      )}
      {status === "ok" && (
        <ol className="semantic-search-results">
          {results.map((result) => (
            <li key={result.semantic_id}>
              <button type="button" onClick={() => onSelectGeometry(result.geometry_id)}>
                <b>{result.label ?? result.geometry_id}</b>
                <span>{result.score.toFixed(3)}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
