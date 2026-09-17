"""Testes do serviço HTTP: health, consulta bem-sucedida e capability ausente."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

import pytest
from map_explorer_api.service import create_app, serve
from semantic_memory import SemanticRetrievalEntry

# Um único entry compatível com um encoder fake determinístico, suficiente
# para exercitar a rota de consulta de ponta a ponta sobre HTTP real.
_ENTRY = SemanticRetrievalEntry(
    semantic_id="semantic:geom-1", geometry_id="geom-1", coordinates_m=(1.0, 2.0, 3.0),
    values=(1.0, 0.0), embedding_space="fake:none:language_aligned",
    producer="language_embedding:fake:none", label="porta",
)


def _encode_text(_text: str) -> tuple[float, ...]:
    """Encoder de teste: sempre devolve o mesmo vetor do entry indexado."""
    return (1.0, 0.0)


# Sobe o serviço em uma porta efêmera e devolve a conexão já pronta, para que
# cada teste só precise fazer a chamada e checar a resposta.
@pytest.fixture()
def running_service():
    """Sobe o serviço com um mapa conhecido e devolve host/porta reais."""
    handler = create_app(
        load_entries=lambda map_id: (_ENTRY,) if map_id == "map-1" else (() if map_id == "map-empty" else None),
        load_encoder=lambda map_id: _encode_text if map_id in {"map-1", "map-empty"} else None,
        embedding_space=lambda _map_id: "fake:none:language_aligned",
        producer=lambda _map_id: "language_embedding:fake:none",
    )
    server = serve(handler, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(address: tuple[str, int], path: str, payload: dict) -> tuple[int, dict]:
    """Faz um POST JSON e devolve status e corpo desserializado."""
    connection = HTTPConnection(address[0], address[1], timeout=5)
    body = json.dumps(payload).encode("utf-8")
    connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    result = (response.status, json.loads(response.read()))
    connection.close()
    return result


# O health check é o único endpoint sem mapa, e precisa responder mesmo
# quando nenhum mapa foi carregado com sucesso.
def test_health_endpoint_responde_ok(running_service: tuple[str, int]) -> None:
    """Confere que ``GET /health`` responde 200 com status ok."""
    connection = HTTPConnection(*running_service, timeout=5)
    connection.request("GET", "/health")
    response = connection.getresponse()
    assert response.status == 200
    assert json.loads(response.read()) == {"status": "ok"}
    connection.close()


# O viewer roda em outra origem (Vite em outra porta) do serviço local: sem
# o cabeçalho CORS na resposta real e sem responder ao preflight, o fetch do
# browser falha como erro de rede antes mesmo de chegar à rota de consulta.
def test_respostas_incluem_cabecalho_cors(running_service: tuple[str, int]) -> None:
    """Confere ``Access-Control-Allow-Origin`` em health e no preflight OPTIONS."""
    connection = HTTPConnection(*running_service, timeout=5)
    connection.request("GET", "/health")
    response = connection.getresponse()
    assert response.getheader("Access-Control-Allow-Origin") == "*"
    response.read()
    connection.close()

    connection = HTTPConnection(*running_service, timeout=5)
    connection.request("OPTIONS", "/v1/maps/map-1/query")
    response = connection.getresponse()
    assert response.status == 204
    assert response.getheader("Access-Control-Allow-Origin") == "*"
    assert "POST" in response.getheader("Access-Control-Allow-Methods")
    connection.close()


# O caminho comum: um mapa com entradas e encoder devolve um QueryOutcome
# de sucesso com o resultado esperado.
def test_query_bem_sucedida_devolve_outcome_ok(running_service: tuple[str, int]) -> None:
    """Confere resposta 200 com status ok e o resultado indexado."""
    status, body = _post(running_service, "/v1/maps/map-1/query", {"text": "porta", "limit": 5})
    assert status == 200
    assert body["status"] == "ok"
    assert body["results"][0]["geometry_id"] == "geom-1"
    assert body["results"][0]["label"] == "porta"


# Um mapa sem nenhum embedding indexado precisa devolver capability
# indisponível, não uma lista vazia disfarçada de sucesso.
def test_mapa_sem_entradas_devolve_capability_indisponivel(running_service: tuple[str, int]) -> None:
    """Confere ``capability_unavailable`` quando o mapa não tem embeddings."""
    status, body = _post(running_service, "/v1/maps/map-empty/query", {"text": "porta"})
    assert status == 200
    assert body["status"] == "capability_unavailable"
    assert body["results"] == []


# Um map_id desconhecido é um erro do cliente, não uma capability ausente.
def test_mapa_desconhecido_devolve_404(running_service: tuple[str, int]) -> None:
    """Confere 404 quando o mapa não existe no serviço."""
    status, body = _post(running_service, "/v1/maps/map-nao-existe/query", {"text": "porta"})
    assert status == 404
    assert body["status"] == "map_not_found"


# Um corpo sem o campo obrigatório ``text`` é uma requisição inválida, e
# precisa falhar de forma explícita em vez de estourar um traceback interno.
def test_corpo_invalido_devolve_400(running_service: tuple[str, int]) -> None:
    """Confere 400 quando o corpo da requisição não declara ``text``."""
    status, body = _post(running_service, "/v1/maps/map-1/query", {"limit": 5})
    assert status == 400
    assert body["status"] == "invalid_request"


# Uma rota fora do contract público não deve ser confundida com um mapa
# inexistente: ambas devolvem 404, mas por razões diferentes e sem vazar
# nenhum detalhe de implementação do roteamento manual.
def test_rota_desconhecida_devolve_404(running_service: tuple[str, int]) -> None:
    """Confere 404 para um caminho fora do contract HTTP publicado."""
    connection = HTTPConnection(*running_service, timeout=5)
    connection.request("GET", "/v1/unknown")
    response = connection.getresponse()
    assert response.status == 404
    connection.close()
