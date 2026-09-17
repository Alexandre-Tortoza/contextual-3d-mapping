// Cliente do serviço HTTP local de busca textual sobre mapas (#224). O
// serviço nunca devolve vetores — só QueryOutcome serializável — então o
// browser não precisa (e não deve) carregar nenhum archive de embeddings.

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8765";
const API_BASE_URL_STORAGE_KEY = "map-explorer.semantic-search.api-base-url";

// A URL do serviço local é uma preferência de máquina, não de mapa: persiste
// entre recarregamentos, mas nunca em um lugar que outro usuário do mesmo
// link compartilhado herdaria.
export function loadApiBaseUrl() {
  try {
    return window.localStorage.getItem(API_BASE_URL_STORAGE_KEY) ?? DEFAULT_API_BASE_URL;
  } catch {
    return DEFAULT_API_BASE_URL;
  }
}

export function saveApiBaseUrl(url) {
  try {
    window.localStorage.setItem(API_BASE_URL_STORAGE_KEY, url);
  } catch {
    // Armazenamento indisponível (modo privado, por exemplo): a sessão atual
    // segue funcionando, só não persiste entre recarregamentos.
  }
}

// Consulta o serviço local por texto e devolve sempre um QueryOutcome, nunca
// uma exceção de rede: uma falha de fetch vira "service_unavailable"
// explícito, para que o painel só precise tratar um formato de resultado.
export async function queryMap({ apiBaseUrl, mapId, text, limit = 10, signal }) {
  let response;
  try {
    response = await fetch(
      `${apiBaseUrl.replace(/\/+$/, "")}/v1/maps/${encodeURIComponent(mapId)}/query`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, limit }),
        signal,
      },
    );
  } catch (error) {
    if (error.name === "AbortError") throw error;
    return { status: "service_unavailable", reason: error.message, results: [] };
  }
  if (response.status === 404) {
    return { status: "map_not_found", reason: null, results: [] };
  }
  if (!response.ok) {
    return { status: "service_unavailable", reason: `HTTP ${response.status}`, results: [] };
  }
  try {
    return await response.json();
  } catch {
    return { status: "service_unavailable", reason: "resposta não é JSON válido", results: [] };
  }
}
