import { debugStagesOf } from "./map-data.js";

const IMAGE_EXTENSION = /\.(png|jpe?g|webp)$/i;
const JSON_EXTENSION = /\.json$/i;

// Percorre um valor do debug_manifest recursivamente e devolve todos os
// caminhos folha (string), com o prefixo de etapa/chave acumulado. Existe
// porque o manifesto mistura URIs soltas ("diagnostics"), sub-objetos
// ("images") e listas ("discovery_tiles", "region_views") sob a mesma etapa,
// e a galeria só precisa saber onde cada string apareceu, não a forma exata
// do JSON publicado.
export function flattenManifestValue(node, prefix) {
  if (typeof node === "string") return [[prefix, node]];
  if (Array.isArray(node)) {
    return node.flatMap((item, index) => flattenManifestValue(item, `${prefix}.${index}`));
  }
  if (node && typeof node === "object") {
    return Object.entries(node).flatMap(([key, value]) => flattenManifestValue(value, `${prefix}.${key}`));
  }
  return [];
}

// Agrupa a galeria de uma observação por etapa, separando imagens
// (thumbnails clicáveis) de documentos JSON (abertos sob demanda). Etapas
// desconhecidas aparecem do mesmo jeito que as conhecidas, pois esta função
// nunca lê nomes de etapa fixos — é o que permite ao DebugExplorer exibir uma
// etapa nova do pipeline sem precisar de uma mudança de código antes.
export function debugGalleryFor(observation) {
  return debugStagesOf(observation).map(([stage, value]) => {
    const leaves = flattenManifestValue(value, stage);
    return {
      stage,
      images: leaves.filter(([, uri]) => IMAGE_EXTENSION.test(uri)),
      documents: leaves.filter(([, uri]) => JSON_EXTENSION.test(uri)),
    };
  });
}

// Normaliza o debug_manifest.composition de nível de run em uma lista de
// entradas nome→uri. Uma run isolada publica uma única string; um mapa
// consolidado publica uma lista com uma entrada por run de origem
// ({run_id, uri}). Existe para que o painel "Composição da run" reaproveite o
// mesmo componente de <details>+fetch nos dois formatos.
export function compositionEntriesOf(slice) {
  const composition = slice?.debug_manifest?.composition;
  if (!composition) return [];
  if (typeof composition === "string") return [["composition", composition]];
  if (Array.isArray(composition)) {
    return composition
      .filter((entry) => entry && typeof entry.uri === "string")
      .map((entry) => [entry.run_id ?? entry.uri, entry.uri]);
  }
  return [];
}

// Normaliza o debug_manifest.pipeline_backends de nível de run em uma lista
// de {runId, backends}. Uma run isolada publica um único objeto por estágio
// (sem run_id, já que só há uma run); um mapa consolidado publica uma lista
// com uma entrada por run de origem, porque runs diferentes podem ter usado
// backends diferentes (ex.: uma comparação SAM3 vs. SAM2) e a explicabilidade
// precisa dizer qual run usou qual, não apenas listar tudo junto.
export function pipelineBackendsOf(slice) {
  const backends = slice?.debug_manifest?.pipeline_backends;
  if (!backends) return [];
  if (Array.isArray(backends)) {
    return backends
      .filter((entry) => entry && typeof entry.backends === "object")
      .map((entry) => ({ runId: entry.run_id ?? null, backends: entry.backends }));
  }
  if (typeof backends === "object") return [{ runId: null, backends }];
  return [];
}
