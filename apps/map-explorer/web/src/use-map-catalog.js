import { useEffect, useState } from "react";

import {
  ChunkedArtifactGeometrySource,
  CONSOLIDATED_ARTIFACT_TYPE,
  CONTEXT_ARTIFACT_TYPE,
  StaticArtifactGeometrySource,
  mapEntriesFromIndex,
  validateSlice,
} from "./map-data.js";

const MAP_INDEX_URL = "/maps/index.json";
const FALLBACK_MAPS = Object.freeze([]);

// Resolve qual artifact está aberto a partir de ?artifact=... na query string,
// mantém o catálogo de runs sincronizado com o índice publicado pelo servidor
// local, e busca/valida o artifact selecionado. Existe para que a vista de
// mapa (Explorer) e a vista de explicabilidade (DebugExplorer) compartilhem
// exatamente a mesma run carregada e a mesma lista de opções, sem duplicar
// fetch nem validação de schema em dois lugares.
export function useMapCatalog() {
  const [artifactPath, setArtifactPath] = useState(
    () => new URLSearchParams(window.location.search).get("artifact"),
  );
  const [availableMaps, setAvailableMaps] = useState(FALLBACK_MAPS);
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const [slice, setSlice] = useState(null);
  const [points, setPoints] = useState([]);
  const [artifactUrl, setArtifactUrl] = useState(null);
  const [error, setError] = useState(null);

  // Abre um artifact validado através da fronteira de geometria. Um mapa
  // consolidado com geometry_manifest é buscado em partições, pintando a cena
  // progressivamente; uma run isolada ou um mapa pequeno continuam vindo como
  // um único chunk completo.
  const openSlice = async (payload, resolvedUrl = null, signal = null) => {
    const validated = validateSlice(payload);
    if (![CONTEXT_ARTIFACT_TYPE, CONSOLIDATED_ARTIFACT_TYPE].includes(validated.artifact_type)) {
      throw new Error("Selecione uma run com contexto ou um mapa consolidado.");
    }
    const hasGeometryManifest = Array.isArray(validated.geometry_manifest) && validated.geometry_manifest.length > 0;
    setSlice(validated);
    setPoints([]);
    setArtifactUrl(resolvedUrl);
    setError(null);
    const source = hasGeometryManifest
      ? new ChunkedArtifactGeometrySource(validated, resolvedUrl ?? window.location.href)
      : new StaticArtifactGeometrySource(validated);
    const geometry = await source.getGeometry({
      signal,
      onChunk: (chunk) => setPoints((previous) => [...previous, ...chunk.points]),
    });
    if (!hasGeometryManifest) {
      setPoints(geometry.chunks.flatMap((chunk) => chunk.points));
    }
  };

  // Carrega a opção inicial ou escolhida e cancela a leitura anterior quando o
  // usuário troca de mapa antes do término do download.
  useEffect(() => {
    if (!artifactPath) return undefined;
    const resolvedUrl = new URL(artifactPath, window.location.href).href;
    const controller = new AbortController();
    setError(null);
    fetch(resolvedUrl, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(
            response.status === 404
              ? "Esta run não está mais publicada. Escolha outra ou publique novamente."
              : `O servidor respondeu HTTP ${response.status}.`,
          );
        }
        // Um corpo que não é JSON (ex.: a página HTML de fallback do
        // servidor de desenvolvimento, quando o caminho não existe) precisa
        // de uma mensagem própria — o erro de parsing do navegador não diz
        // ao usuário o que fazer a respeito.
        try {
          return await response.json();
        } catch {
          const notJsonError = new Error("A run selecionada não é um artifact JSON válido.");
          notJsonError.explanation =
            "O servidor respondeu com um corpo que não é JSON — geralmente a página HTML de " +
            "fallback do servidor de desenvolvimento, sinal de que o arquivo apontado por esta " +
            "run não existe mais nesse caminho. Publique a run novamente com " +
            "publish_map_index.py ou escolha outra run na lista abaixo.";
          throw notJsonError;
        }
      })
      .then((payload) => openSlice(payload, resolvedUrl, controller.signal))
      .catch((failure) => {
        if (failure.name !== "AbortError") {
          setError(
            failure instanceof Error
              ? { message: failure.message, explanation: failure.explanation ?? null }
              : { message: "Não foi possível abrir o artifact.", explanation: null },
          );
        }
      });
    return () => controller.abort();
  }, [artifactPath]);

  // Atualiza o catálogo sem tirar o usuário da run escolhida. Na primeira
  // abertura, seleciona a mais recente; URLs explícitas continuam estáveis.
  useEffect(() => {
    const controller = new AbortController();
    // Um índice ausente ou vazio é o estado normal de um viewer recém-
    // instalado, não uma falha: só cai no artifact de demonstração fixo
    // (``current-map.json``) quando a query string já pedia explicitamente
    // por ele, para não trocar um catálogo vazio por um erro 404 confuso.
    const refresh = () => fetch(MAP_INDEX_URL, { signal: controller.signal, cache: "no-store" })
      .then((response) => (response.ok ? response.json().catch(() => []) : []))
      .then((payload) => {
        const entries = mapEntriesFromIndex(payload);
        setAvailableMaps(entries);
        setCatalogLoaded(true);
        setArtifactPath((current) => current ?? entries[0]?.url ?? null);
      })
      .catch((failure) => {
        if (failure.name !== "AbortError") setCatalogLoaded(true);
      });
    refresh();
    const interval = window.setInterval(refresh, 30000);
    window.addEventListener("focus", refresh);
    return () => {
      controller.abort();
      window.clearInterval(interval);
      window.removeEventListener("focus", refresh);
    };
  }, []);

  // Mantém a URL compartilhável sincronizada sem recarregar a aplicação nem
  // perder o estado de conexão do viewer.
  const selectMap = (path) => {
    const url = new URL(window.location.href);
    url.searchParams.set("artifact", path);
    window.history.replaceState(null, "", url);
    setArtifactPath(path);
  };

  return { artifactPath, availableMaps, catalogLoaded, slice, points, artifactUrl, error, selectMap };
}
