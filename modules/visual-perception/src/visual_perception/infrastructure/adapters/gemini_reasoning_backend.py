"""Adapter remoto de raciocínio multimodal com Gemini Robotics ER.

Issue: #276. Implementa o mesmo port ``MultimodalReasoner`` do Qwen local, com os
mesmos prompts (``reasoning_prompts``), para que a comparação entre os dois meça o
modelo e não o pipeline. Frames e crops saem da máquina: o backend é opt-in e nunca
substitui o local sem uma troca explícita de configuração.
"""

from __future__ import annotations

import io
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_reasoning import RegionReasoningRequest, RegionRelationRequest
from visual_perception.infrastructure.adapters._runtime import (
    payload_to_pil,
    require_checkpoint,
    require_module,
)
from visual_perception.infrastructure.adapters.reasoning_prompts import (
    concept_prompt,
    parse_json_object,
    region_prompt,
    relation_prompt,
    scene_prompt,
)

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_GEMINI_ROBOTICS_ER_MODEL = "gemini-robotics-er-2-preview"

#: Códigos HTTP tratados como transitórios: vale tentar de novo com backoff.
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: Instrução só de formato. Sem ela o Gemini Robotics ER responde no seu formato
#: padrão de detecção (lista com ``box_2d``) mesmo quando o prompt pede um objeto;
#: medido em 2026-09-15 numa região outdoor do corridor-02. Não acrescenta evidência
#: nem contexto, então a comparação com o Qwen continua medindo o modelo.
OUTPUT_FORMAT_INSTRUCTION = (
    "Answer with exactly one JSON object that follows the keys requested in the user prompt. "
    "Never return a JSON list and never add box_2d, point or mask fields unless the prompt asks for them."
)


# Registra custo e falhas de uma consulta remota. Existe porque latência, tokens e
# rate limit não passam pelo lifecycle de VRAM; o benchmark de backends lê daqui.
@dataclass(frozen=True)
class RemoteReasoningCall:
    """Telemetria de uma consulta ao backend remoto, sem conteúdo nem credencial."""

    operation: str
    model: str
    latency_s: float
    attempts: int
    succeeded: bool
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    finish_reason: str | None = None
    transient_failures: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None


# Implementa o port MultimodalReasoner chamando o Gemini pela Gen AI SDK oficial.
# Transporte, retry e erros da API ficam contidos aqui; application recebe o
# mesmo dict bruto que recebe do Qwen.
class GeminiRoboticsReasoningAdapter:
    """Satisfaz :class:`~visual_perception.ports.multimodal_reasoning.MultimodalReasoner` remotamente."""

    # Recebe fábrica de cliente e relógio injetáveis para que retry, timeout e
    # telemetria sejam testáveis sem rede nem chave.
    def __init__(
        self,
        client_factory: Callable[[str], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Inicializa o adapter sem ler credencial nem abrir conexão."""
        self._client_factory = client_factory
        self._sleep = sleep
        self._clock = clock
        self._client: Any = None
        self._calls: list[RemoteReasoningCall] = []

    # Expõe a telemetria acumulada das consultas feitas por esta instância.
    @property
    def calls(self) -> tuple[RemoteReasoningCall, ...]:
        """Retorna as consultas registradas, na ordem em que ocorreram."""
        return tuple(self._calls)

    # Analisa a imagem inteira com o prompt de cena compartilhado.
    def analyze_scene(self, image: ImagePayload, config: MultimodalReasoningConfig) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do Gemini para o contexto da cena."""
        return self._generate_json("scene", (image,), scene_prompt(), config)

    # Propõe conceitos concretos para o grounding (#277); satisfaz SceneConceptDiscoverer.
    def discover_concepts(
        self, image: ImagePayload, config: MultimodalReasoningConfig, *, max_concepts: int
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do Gemini com conceitos a localizar."""
        return self._generate_json("concepts", (image,), concept_prompt(max_concepts), config)

    # Analisa uma região com as mesmas views, na mesma ordem, que o Qwen recebe.
    def analyze_region(
        self, request: RegionReasoningRequest, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do Gemini para uma região."""
        return self._generate_json(
            "region", tuple(view.payload for view in request.views), region_prompt(request), config
        )

    # Julga a relação entre duas regiões com a view de par compartilhada.
    def analyze_relation(
        self, request: RegionRelationRequest, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do Gemini sobre a relação entre duas regiões."""
        return self._generate_json(
            "relation", tuple(view.payload for view in request.views), relation_prompt(request), config
        )

    # Executa a consulta com retry limitado em falhas transitórias. Falha final
    # vira BackendExecutionError; nunca há fallback para outro backend.
    def _generate_json(
        self,
        operation: str,
        images: tuple[ImagePayload, ...],
        prompt: str,
        config: MultimodalReasoningConfig,
    ) -> dict[str, Any]:
        """Envia imagens e prompt, registra telemetria e extrai o objeto JSON."""
        model = require_checkpoint(config.checkpoint, config.backend)
        types = require_module("google.genai.types", config.backend)
        errors = require_module("google.genai.errors", config.backend)
        client = self._get_client(config)
        contents = [*(self._image_part(types, image, config) for image in images), prompt]
        request_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction=OUTPUT_FORMAT_INSTRUCTION,
            temperature=config.temperature,
            max_output_tokens=config.max_new_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=config.thinking_budget),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(timeout=int(config.timeout_s * 1000)),
        )
        transient: list[str] = []
        start = self._clock()
        for attempt in range(1, config.max_retries + 2):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=request_config)
            except Exception as error:
                reason = _transient_reason(error, errors)
                if reason is not None and attempt <= config.max_retries:
                    transient.append(reason)
                    self._sleep(min(2.0 ** (attempt - 1), 30.0))
                    continue
                message = f"O backend {config.backend!r} falhou durante a consulta {operation}: {error}"
                self._record(operation, model, start, attempt, False, None, transient, message)
                raise BackendExecutionError(message) from error
            self._record(operation, model, start, attempt, True, response, transient, None)
            return _response_object(response.text or "")
        raise AssertionError("o laço de retry sempre retorna ou levanta")

    # Cria o cliente na primeira consulta, lendo a chave apenas do ambiente.
    def _get_client(self, config: MultimodalReasoningConfig) -> Any:
        """Retorna o cliente Gemini, falhando de forma acionável sem credencial."""
        if self._client is not None:
            return self._client
        api_key = os.environ.get(GEMINI_API_KEY_ENV)
        if not api_key:
            raise BackendUnavailableError(
                f"O backend {config.backend!r} exige a variável de ambiente {GEMINI_API_KEY_ENV}."
            )
        if self._client_factory is None:
            genai = require_module("google.genai", config.backend)
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = self._client_factory(api_key)
        return self._client

    # Serializa uma view como PNG sem perda, preservando a geometria canônica.
    def _image_part(self, types: Any, image: ImagePayload, config: MultimodalReasoningConfig) -> Any:
        """Converte um ImagePayload em parte de imagem da requisição."""
        buffer = io.BytesIO()
        payload_to_pil(image, config.backend).save(buffer, format="PNG")
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")

    # Acumula telemetria sem guardar prompt, imagem, resposta ou chave.
    def _record(
        self,
        operation: str,
        model: str,
        start: float,
        attempts: int,
        succeeded: bool,
        response: Any,
        transient: list[str],
        error: str | None,
    ) -> None:
        """Registra uma consulta concluída ou falha definitiva."""
        usage = getattr(response, "usage_metadata", None)
        self._calls.append(
            RemoteReasoningCall(
                operation=operation,
                model=model,
                latency_s=self._clock() - start,
                attempts=attempts,
                succeeded=succeeded,
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
                thought_tokens=getattr(usage, "thoughts_token_count", None),
                finish_reason=_finish_reason(response),
                transient_failures=tuple(transient),
                error=error,
            )
        )


# Classifica uma exceção da SDK como transitória (retry) ou definitiva. O timeout
# de leitura chega como ``httpx.ReadTimeout``, cujo nome não diz "timeout"
# sozinho, então a checagem percorre a hierarquia da exceção.
def _transient_reason(error: Exception, errors: Any) -> str | None:
    """Retorna o motivo transitório da falha, ou ``None`` quando não vale repetir."""
    if isinstance(error, errors.APIError):
        code = getattr(error, "code", None)
        return f"http_{code}" if code in _TRANSIENT_STATUS else None
    if isinstance(error, TimeoutError) or any(cls.__name__ == "TimeoutException" for cls in type(error).__mro__):
        return "timeout"
    return None


# Aceita apenas um objeto JSON no topo. Uma lista é o formato de detecção do
# modelo; extrair um objeto de dentro dela entregaria à application um item de
# detecção como se fosse a resposta pedida, então vira objeto vazio para o
# diagnóstico de schema, como qualquer resposta inválida do Qwen.
def _response_object(text: str) -> dict[str, Any]:
    """Retorna o objeto JSON da resposta, ou vazio se o topo não for um objeto."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return parse_json_object(text)
    return value if isinstance(value, dict) else {}


# Extrai o motivo de término do primeiro candidato para a telemetria.
def _finish_reason(response: Any) -> str | None:
    """Retorna o ``finish_reason`` do primeiro candidato, se houver."""
    candidates = getattr(response, "candidates", None) or ()
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    return None if reason is None else str(getattr(reason, "name", reason))
