"""Serviço HTTP local de consulta textual sobre um mapa publicado (#224).

Usa somente ``http.server`` da stdlib: o map-explorer não tinha nenhuma
dependência HTTP instalada, e a superfície deste serviço (duas rotas, sem
autenticação, uso local) não justifica adicionar um framework. A factory
recebe as entradas e o encoder já resolvidos pela aplicação — este módulo não
decide onde um mapa mora nem qual backend de texto roda.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from query_engine import MapQueryRequest, QueryOutcome, query_semantic_map
from semantic_memory import SemanticRetrievalEntry

#: Resolve um ``map_id`` para as entradas indexadas desse mapa, ou ``None``
#: se o mapa não existir no serviço.
MapEntriesLookup = Callable[[str], tuple[SemanticRetrievalEntry, ...] | None]
#: Resolve um ``map_id`` para o ``encode_text`` compatível com o espaço
#: publicado por esse mapa, ou ``None`` se a capability estiver indisponível.
TextEncoderLookup = Callable[[str], Callable[[str], tuple[float, ...]] | None]
#: Resolve um ``map_id`` para a identidade de espaço/produtor usada para
#: filtrar candidatos compatíveis na busca.
SpaceIdentityLookup = Callable[[str], str]


# Serializa um QueryOutcome sem publicar nenhum campo interno do módulo. É o
# único formato de resposta que a rota de consulta devolve, com sucesso ou não.
def _outcome_payload(outcome: QueryOutcome) -> dict[str, Any]:
    """Converte um ``QueryOutcome`` no payload JSON público da API."""
    return {
        "status": outcome.status,
        "reason": outcome.reason,
        "results": [asdict(result) for result in outcome.results],
    }


# Compõe o handler HTTP a partir das fronteiras já resolvidas pela aplicação.
# Existe como factory, e não uma classe fixa, porque ``BaseHTTPRequestHandler``
# não aceita injeção de dependência por construtor.
def create_app(
    *,
    load_entries: MapEntriesLookup,
    load_encoder: TextEncoderLookup,
    embedding_space: SpaceIdentityLookup,
    producer: SpaceIdentityLookup,
) -> type[BaseHTTPRequestHandler]:
    """Constrói o handler HTTP do serviço de consulta textual sobre mapas.

    Argumentos:
        load_entries: resolve ``map_id`` para as entradas indexadas do mapa.
        load_encoder: resolve ``map_id`` para o encoder de texto compatível.
        embedding_space: identidade de espaço declarada pelo mapa.
        producer: identidade de produtor declarada pelo mapa.
    Retorna:
        classe de handler pronta para um ``http.server.HTTPServer``.
    """

    class MapQueryHandler(BaseHTTPRequestHandler):
        # Escreve uma resposta JSON única, sempre com Content-Length explícito
        # para que o cliente não precise de chunked encoding. O viewer roda em
        # uma origem diferente (Vite em outra porta) do serviço local, então
        # toda resposta precisa do cabeçalho CORS — sem credenciais, sem
        # cookies, um serviço local só de leitura não tem por que restringir
        # a origem.
        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            """Serializa ``payload`` como corpo JSON da resposta HTTP."""
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        # O navegador envia um preflight OPTIONS antes de um POST com corpo
        # ``application/json`` (fora da lista CORS-safelisted). Sem responder
        # aqui, o fetch do viewer falha como erro de rede antes mesmo de
        # chegar em ``do_POST``.
        def do_OPTIONS(self) -> None:  # noqa: N802 - nome exigido pela stdlib
            """Responde ao preflight CORS do navegador para ``POST``."""
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 - nome exigido pela stdlib
            """Responde ``GET /health`` com o único diagnóstico público do serviço."""
            if self.path == "/health":
                self._write_json(200, {"status": "ok"})
                return
            self._write_json(404, {"status": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - nome exigido pela stdlib
            """Responde ``POST /v1/maps/{map_id}/query`` com um ``QueryOutcome``."""
            parts = self.path.strip("/").split("/")
            if len(parts) != 4 or (parts[0], parts[1], parts[3]) != ("v1", "maps", "query"):
                self._write_json(404, {"status": "not_found"})
                return
            map_id = parts[2]
            length = int(self.headers.get("Content-Length") or "0")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                request = MapQueryRequest(
                    map_id=map_id, text=str(body["text"]), limit=int(body.get("limit", 10)),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._write_json(400, {"status": "invalid_request", "reason": str(error)})
                return

            entries = load_entries(map_id)
            if entries is None:
                self._write_json(404, {"status": "map_not_found"})
                return
            encode_text = load_encoder(map_id)
            if encode_text is None or not entries:
                self._write_json(200, _outcome_payload(
                    QueryOutcome("capability_unavailable", reason="semantic_embedding_capability_unavailable")
                ))
                return
            try:
                outcome = query_semantic_map(
                    request, entries, encode_text,
                    embedding_space=embedding_space(map_id), producer=producer(map_id),
                )
            except ValueError as error:
                self._write_json(200, _outcome_payload(QueryOutcome("query_failed", reason=str(error))))
                return
            self._write_json(200, _outcome_payload(outcome))

        # A stdlib registra cada requisição em stderr por padrão; o serviço
        # local não precisa desse access log competindo com o run do runtime.
        def log_message(self, format: str, *args: object) -> None:
            """Silencia o access log padrão de ``BaseHTTPRequestHandler``."""

    return MapQueryHandler


# Sobe o servidor e devolve a instância para o chamador controlar o ciclo de
# vida (serve_forever/shutdown), em vez de bloquear dentro deste módulo.
def serve(handler_cls: type[BaseHTTPRequestHandler], *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Cria o servidor HTTP local para ``handler_cls`` sem iniciar o loop de serviço."""
    return ThreadingHTTPServer((host, port), handler_cls)
