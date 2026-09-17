import assert from "node:assert/strict";
import test from "node:test";

import { queryMap } from "./semantic-search.js";

// Instala um fetch de teste e o restaura ao final, para não vazar mock entre
// casos que exercitam respostas diferentes do mesmo endpoint.
function withFetch(implementation, run) {
  const original = globalThis.fetch;
  globalThis.fetch = implementation;
  return run().finally(() => {
    globalThis.fetch = original;
  });
}

// O caminho comum: o serviço responde com um QueryOutcome de sucesso, e o
// cliente devolve exatamente esse corpo desserializado.
test("queryMap devolve o outcome de sucesso do serviço", async () => {
  await withFetch(
    async (url, init) => {
      assert.equal(url, "http://127.0.0.1:8765/v1/maps/corridor-02/query");
      assert.equal(JSON.parse(init.body).text, "porta");
      return {
        ok: true,
        status: 200,
        json: async () => ({ status: "ok", results: [{ geometry_id: "geom-1" }] }),
      };
    },
    async () => {
      const outcome = await queryMap({
        apiBaseUrl: "http://127.0.0.1:8765", mapId: "corridor-02", text: "porta",
      });
      assert.deepEqual(outcome, { status: "ok", results: [{ geometry_id: "geom-1" }] });
    },
  );
});

// Uma falha de rede (serviço local fora do ar) precisa virar um outcome
// explícito, não uma exceção que o painel teria que capturar separadamente.
test("queryMap converte falha de rede em service_unavailable", async () => {
  await withFetch(
    async () => {
      throw new Error("Failed to fetch");
    },
    async () => {
      const outcome = await queryMap({
        apiBaseUrl: "http://127.0.0.1:8765", mapId: "corridor-02", text: "porta",
      });
      assert.equal(outcome.status, "service_unavailable");
      assert.equal(outcome.reason, "Failed to fetch");
      assert.deepEqual(outcome.results, []);
    },
  );
});

// Um map_id que o serviço não conhece é distinto de uma falha de rede: o
// painel precisa poder mostrar mensagens diferentes para cada caso.
test("queryMap mapeia HTTP 404 para map_not_found", async () => {
  await withFetch(
    async () => ({ ok: false, status: 404, json: async () => ({}) }),
    async () => {
      const outcome = await queryMap({
        apiBaseUrl: "http://127.0.0.1:8765/", mapId: "corridor-02", text: "porta",
      });
      assert.equal(outcome.status, "map_not_found");
    },
  );
});
