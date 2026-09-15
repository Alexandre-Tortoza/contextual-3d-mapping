"""Testes do adapter remoto Gemini Robotics ER sem rede nem credencial real (#276)."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("google.genai")
from google.genai import errors  # noqa: E402

from fixtures import payload_with_blobs  # noqa: E402
from visual_perception.config import ModuleConfig, MultimodalReasoningConfig  # noqa: E402
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError  # noqa: E402
from visual_perception.domain.geometry import BoundingBox, CoordinateTransform  # noqa: E402
from visual_perception.domain.region_evidence import EvidenceSlot, SubjectEmphasis  # noqa: E402
from visual_perception.domain.region_reasoning import RegionReasoningRequest, RegionView  # noqa: E402
from visual_perception.infrastructure.adapters.factory import create_perception_ports  # noqa: E402
from visual_perception.infrastructure.adapters.gemini_reasoning_backend import (  # noqa: E402
    DEFAULT_GEMINI_ROBOTICS_ER_MODEL,
    GEMINI_API_KEY_ENV,
    OUTPUT_FORMAT_INSTRUCTION,
    GeminiRoboticsReasoningAdapter,
)
from visual_perception.infrastructure.adapters.reasoning_prompts import region_prompt, scene_prompt  # noqa: E402

SECRET = "chave-de-teste-que-nao-pode-vazar"


# Simula a superfície mínima da Gen AI SDK usada pelo adapter: registra cada
# requisição e devolve respostas ou exceções na ordem configurada.
class _FakeModels:
    """Substitui ``client.models`` com um roteiro de respostas."""

    def __init__(self, outcomes: list[Any]) -> None:
        """Guarda o roteiro de respostas e exceções."""
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
        """Registra a requisição e devolve o próximo resultado do roteiro."""
        self.requests.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# Monta uma resposta com texto e metadados de uso no formato da SDK.
def _response(text: str) -> SimpleNamespace:
    """Retorna uma resposta fake com contagem de tokens."""
    usage = SimpleNamespace(prompt_token_count=1230, candidates_token_count=86, thoughts_token_count=None)
    return SimpleNamespace(text=text, usage_metadata=usage)


# Cria adapter, cliente fake e registro de esperas para um roteiro de respostas.
def _adapter(outcomes: list[Any]) -> tuple[GeminiRoboticsReasoningAdapter, _FakeModels, list[float], list[str]]:
    """Retorna o adapter com cliente injetado e as listas observáveis do teste."""
    models = _FakeModels(outcomes)
    keys: list[str] = []
    sleeps: list[float] = []

    def factory(api_key: str) -> SimpleNamespace:
        keys.append(api_key)
        return SimpleNamespace(models=models)

    return GeminiRoboticsReasoningAdapter(client_factory=factory, sleep=sleeps.append), models, sleeps, keys


# Config remota mínima, com poucas tentativas para manter os testes curtos.
def _config(**overrides: Any) -> MultimodalReasoningConfig:
    """Retorna a config do backend Gemini com o modelo de referência."""
    values = {"backend": "gemini_robotics_er", "checkpoint": DEFAULT_GEMINI_ROBOTICS_ER_MODEL, "max_retries": 2}
    return MultimodalReasoningConfig(**{**values, **overrides})


# Garante que a cena usa o mesmo prompt do Qwen, uma imagem PNG, JSON estruturado,
# thinking desligado e o timeout configurado.
def test_scene_request_uses_shared_prompt_and_structured_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serializa a consulta de cena e devolve o objeto JSON bruto."""
    monkeypatch.setenv(GEMINI_API_KEY_ENV, SECRET)
    adapter, models, _, keys = _adapter([_response('{"scene_type": "hallway", "confidence": 0.63}')])

    response = adapter.analyze_scene(payload_with_blobs(width=8, height=6), _config(timeout_s=12.5))

    assert response == {"scene_type": "hallway", "confidence": 0.63}
    assert keys == [SECRET]
    request = models.requests[0]
    assert request["model"] == DEFAULT_GEMINI_ROBOTICS_ER_MODEL
    image, prompt = request["contents"]
    assert image.inline_data.mime_type == "image/png"
    assert image.inline_data.data.startswith(b"\x89PNG")
    assert prompt == scene_prompt()
    assert request["config"].response_mime_type == "application/json"
    assert request["config"].system_instruction == OUTPUT_FORMAT_INSTRUCTION
    assert request["config"].thinking_config.thinking_budget == 0
    assert request["config"].http_options.timeout == 12_500


# A região precisa chegar com as mesmas views, na mesma ordem, que o Qwen recebe;
# senão a comparação entre backends mede outra entrada.
def test_region_request_sends_every_view_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Envia uma imagem por view, seguida do prompt de região compartilhado."""
    monkeypatch.setenv(GEMINI_API_KEY_ENV, SECRET)
    adapter, models, _, _ = _adapter([_response('{"label": "door"}')])
    box = BoundingBox(0.0, 0.0, 4.0, 4.0)
    views = tuple(
        RegionView(
            slot=slot,
            payload=payload_with_blobs(width=4 + index, height=4),
            crop_box=BoundingBox(0.0, 0.0, 4.0 + index, 4.0),
            transform=CoordinateTransform(1.0, 1.0, 0.0, 0.0),
            emphasis=SubjectEmphasis.NONE,
        )
        for index, slot in enumerate((EvidenceSlot.MASKED_SUBJECT, EvidenceSlot.TIGHT_CROP, EvidenceSlot.CONTEXTUAL_CROP))
    )
    request = RegionReasoningRequest(
        region_id="region-a", region_box=box, image_width=32, image_height=32, views=views, scene_claims=()
    )

    assert adapter.analyze_region(request, _config()) == {"label": "door"}

    *images, prompt = models.requests[0]["contents"]
    assert len(images) == 3
    assert prompt == region_prompt(request)
    widths = [int.from_bytes(image.inline_data.data[16:20], "big") for image in images]
    assert widths == [4, 5, 6]


# Sem chave no ambiente o backend fica indisponível antes de qualquer requisição.
def test_missing_api_key_is_unavailable_without_contacting_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recusa a consulta de forma acionável quando GEMINI_API_KEY não existe."""
    monkeypatch.delenv(GEMINI_API_KEY_ENV, raising=False)
    adapter, models, _, keys = _adapter([])

    with pytest.raises(BackendUnavailableError, match=GEMINI_API_KEY_ENV):
        adapter.analyze_scene(payload_with_blobs(width=4, height=4), _config())

    assert keys == [] and models.requests == []


# Rate limit e timeout são transitórios: o adapter espera com backoff e registra
# cada tentativa na telemetria.
def test_transient_failures_are_retried_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repete após 429 e timeout e registra as falhas transitórias."""
    monkeypatch.setenv(GEMINI_API_KEY_ENV, SECRET)
    rate_limited = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    adapter, models, sleeps, _ = _adapter([rate_limited, TimeoutError("read timeout"), _response("{}")])

    assert adapter.analyze_scene(payload_with_blobs(width=4, height=4), _config()) == {}

    assert len(models.requests) == 3
    assert sleeps == [1.0, 2.0]
    (call,) = adapter.calls
    assert call.succeeded and call.attempts == 3
    assert call.transient_failures == ("http_429", "timeout")
    assert call.prompt_tokens == 1230 and call.output_tokens == 86


# Regressão do run de 2026-09-15: o timeout de leitura chega como ``ReadTimeout``,
# subclasse de ``TimeoutException``, e não era repetido.
def test_read_timeout_from_the_http_client_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconhece timeout pela hierarquia da exceção, não pelo nome da classe."""
    monkeypatch.setenv(GEMINI_API_KEY_ENV, SECRET)

    class TimeoutException(Exception):
        """Espelha a base de timeout do cliente HTTP."""

    class ReadTimeout(TimeoutException):
        """Espelha o timeout de leitura do cliente HTTP."""

    adapter, models, sleeps, _ = _adapter([ReadTimeout("The read operation timed out"), _response("{}")])

    assert adapter.analyze_scene(payload_with_blobs(width=4, height=4), _config()) == {}
    assert len(models.requests) == 2 and sleeps == [1.0]
    assert adapter.calls[0].transient_failures == ("timeout",)


# Regressão do run de 2026-09-15: o modelo respondeu no formato de detecção (lista
# com ``box_2d``) e o parser extraía o item como se fosse a resposta de cena.
def test_top_level_list_is_not_mistaken_for_the_requested_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uma lista no topo vira objeto vazio para o diagnóstico de schema."""
    monkeypatch.setenv(GEMINI_API_KEY_ENV, SECRET)
    adapter, _, _, _ = _adapter([_response('[{"box_2d": [1, 2, 3, 4], "label": "light pole"}]')])

    assert adapter.analyze_scene(payload_with_blobs(width=4, height=4), _config()) == {}


# Erro definitivo não é repetido nem trocado por outro backend.
def test_non_transient_error_fails_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Converte 400 em BackendExecutionError sem retry."""
    monkeypatch.setenv(GEMINI_API_KEY_ENV, SECRET)
    invalid = errors.ClientError(400, {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}})
    adapter, models, sleeps, _ = _adapter([invalid])

    with pytest.raises(BackendExecutionError, match="gemini_robotics_er"):
        adapter.analyze_scene(payload_with_blobs(width=4, height=4), _config())

    assert len(models.requests) == 1 and sleeps == []
    assert not adapter.calls[0].succeeded


# Quando as tentativas acabam, a falha fica explícita em vez de virar resultado vazio.
def test_exhausted_retries_raise_execution_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falha depois de max_retries + 1 tentativas com 503."""
    monkeypatch.setenv(GEMINI_API_KEY_ENV, SECRET)
    unavailable = [
        errors.ServerError(503, {"error": {"code": 503, "message": "unavailable", "status": "UNAVAILABLE"}})
        for _ in range(3)
    ]
    adapter, models, _, _ = _adapter(unavailable)

    with pytest.raises(BackendExecutionError):
        adapter.analyze_scene(payload_with_blobs(width=4, height=4), _config(max_retries=2))

    assert len(models.requests) == 3
    assert adapter.calls[0].attempts == 3 and adapter.calls[0].transient_failures == ("http_503", "http_503")


# Resposta sem JSON não é falha de transporte: application recebe o dict vazio e
# emite o diagnóstico de schema, como acontece com o Qwen.
def test_invalid_json_becomes_empty_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """Devolve objeto vazio para texto que não é JSON."""
    monkeypatch.setenv(GEMINI_API_KEY_ENV, SECRET)
    adapter, _, _, _ = _adapter([_response("não é JSON")])

    assert adapter.analyze_scene(payload_with_blobs(width=4, height=4), _config()) == {}


# A chave vive só no ambiente: não entra em config, fingerprint nem telemetria.
def test_api_key_is_never_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confere que nenhum artifact serializável carrega a credencial."""
    monkeypatch.setenv(GEMINI_API_KEY_ENV, SECRET)
    adapter, _, _, _ = _adapter([_response("{}")])
    config = ModuleConfig(multimodal_reasoning=_config())
    adapter.analyze_scene(payload_with_blobs(width=4, height=4), config.multimodal_reasoning)

    serialized = json.dumps(config.to_dict()) + config.fingerprint() + json.dumps([asdict(c) for c in adapter.calls])
    assert SECRET not in serialized


# O backend é selecionável só por configuração, sem ler chave na composição.
def test_factory_selects_gemini_without_reading_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compõe o adapter Gemini mesmo sem GEMINI_API_KEY definida."""
    monkeypatch.delenv(GEMINI_API_KEY_ENV, raising=False)
    ports = create_perception_ports(ModuleConfig(multimodal_reasoning=_config()))

    assert isinstance(ports.multimodal_reasoner, GeminiRoboticsReasoningAdapter)


# Parâmetros de transporte inválidos falham na fronteira da config.
@pytest.mark.parametrize(
    ("override", "message"),
    [({"timeout_s": 0.0}, "timeout_s"), ({"max_retries": -1}, "max_retries"), ({"thinking_budget": -1}, "thinking_budget")],
)
def test_remote_transport_config_is_validated(override: dict[str, Any], message: str) -> None:
    """Recusa timeout, retry e thinking budget inválidos."""
    with pytest.raises(ValueError, match=message):
        _config(**override)
