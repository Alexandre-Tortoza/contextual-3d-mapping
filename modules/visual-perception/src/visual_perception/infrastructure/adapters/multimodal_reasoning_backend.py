"""Adapter real de backend de raciocínio multimodal.

Issue: #189. Checkpoint selecionado pelo benchmark #174 (Qwen2.5-VL-3B-Instruct
em 4-bit, ver ``benchmarks/results/benchmark-174-multimodal_reasoning-*.json``).
"""

from __future__ import annotations

import json
from typing import Any

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.infrastructure.adapters._runtime import (
    payload_to_pil,
    raise_backend_execution_error,
    require_checkpoint,
    require_module,
    resolve_device,
)

#: Limita o número de tokens visuais que encoders de resolução dinâmica (ex:
#: Qwen2.5-VL) produzem por imagem. Ver uso em ``_get_runtime``.
_MAX_PIXELS = 640 * 480


# Implementa o port MultimodalReasoner com um VLM compatível com Transformers,
# mantendo prompts e parsing de transporte locais ao adapter.
class RealMultimodalReasoningAdapter:
    """Satisfaz :class:`~visual_perception.ports.multimodal_reasoning.MultimodalReasoner`.

    Usa um VLM compatível com ``AutoModelForImageTextToText``. O adapter só
    converte transporte e JSON; a validação semântica permanece em application.
    """

    # Recebe (ou cria, se omitido) o lifecycle manager que carrega/libera o
    # VLM sob demanda. Compartilhar o mesmo manager entre os 4 adapters
    # reais (ver ``factory.py``) garante que no máximo um modelo pesado
    # fica residente por vez — inclusive entre chamadas repetidas de
    # analyze_scene/analyze_region dentro do mesmo pipeline, que reusam o
    # VLM já carregado via o cache do próprio manager.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem carregar VLM ou checkpoint."""
        self._lifecycle = lifecycle or ModelLifecycleManager()
        self._device: str | None = None

    # Analisa a imagem completa com um schema mínimo de cena. A resposta
    # permanece bruta para que scene_context faça sua própria validação.
    def analyze_scene(self, image: ImagePayload, config: MultimodalReasoningConfig) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do VLM para o contexto da cena."""
        prompt = (
            "Analyze the whole image. Respond with EXACTLY ONE JSON object (never a list/array, "
            "never markdown fences) with exactly these keys: scene_type (string), description "
            "(string), attributes (list of strings, may be empty), hazards (list of strings, may "
            "be empty), confidence (float between 0 and 1). Example of the exact shape required:\n"
            '{"scene_type": "corridor", "description": "...", "attributes": ["..."], '
            '"hazards": [], "confidence": 0.9}'
        )
        return self._generate_json(image, prompt, config)

    # Analisa um crop de região e usa o resumo de cena como contexto textual
    # opcional. A imagem completa não é reenviada para evitar custo duplicado.
    def analyze_region(
        self,
        image: ImagePayload,
        mask_crop: ImagePayload,
        scene_summary: str | None,
        config: MultimodalReasoningConfig,
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do VLM para uma região da imagem."""
        context = f" Broader scene context (do not just repeat it): {scene_summary}" if scene_summary else ""
        prompt = (
            "Describe ONLY what is visible in THIS cropped image, even if the crop is small "
            "or blurry — a plain surface (wall, floor, ceiling, door) is a valid, specific "
            "answer; do not just restate the whole-scene description. Respond with EXACTLY ONE "
            "JSON object (never a list/array, never markdown fences) with exactly these keys: "
            '"labels" (non-empty list of short strings — the key MUST be called "labels", '
            'plural, never "label"), "description" (string, optional), "attributes" (list of '
            'strings, optional), "condition" (string, optional), "material" (string, optional). '
            "Example of the exact shape required:\n"
            '{"labels": ["door"], "description": "...", "attributes": ["closed"], '
            f'"condition": "worn", "material": "wood"}}{context}'
        )
        return self._generate_json(mask_crop, prompt, config)

    # Executa a conversa multimodal e converte sua resposta textual em objeto
    # JSON. Um JSON inválido vira objeto vazio para a camada application emitir
    # o diagnóstico de schema estável que já possui.
    def _generate_json(
        self, image: ImagePayload, prompt: str, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Gera e extrai um objeto JSON de uma consulta multimodal ao VLM."""
        torch, processor, model, device = self._get_runtime(config)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": payload_to_pil(image, config.backend)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}
            generation_kwargs: dict[str, Any] = {"max_new_tokens": config.max_new_tokens}
            if config.temperature > 0.0:
                generation_kwargs.update({"do_sample": True, "temperature": config.temperature})
            else:
                generation_kwargs["do_sample"] = False
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            prompt_tokens = int(inputs["input_ids"].shape[1])
            text = processor.batch_decode(generated[:, prompt_tokens:], skip_special_tokens=True)[0]
            return _parse_json_object(text)
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "a inferência multimodal", error)

    # Carrega o VLM e seu processor apenas quando o backend real é composto,
    # delegando residência ao lifecycle manager compartilhado. O modo 4-bit
    # delega o posicionamento de layers ao Transformers.
    def _get_runtime(self, config: MultimodalReasoningConfig) -> tuple[Any, Any, Any, str]:
        """Retorna torch, processor, modelo e device para a configuração solicitada."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)

        def factory() -> tuple[Any, Any]:
            try:
                transformers = require_module("transformers", config.backend)
                # _MAX_PIXELS limita o encoder de resolução dinâmica de
                # modelos como o Qwen2.5-VL: sem isso, uma única imagem já
                # é o suficiente para estourar os ~7.6GB úteis da 3060 de
                # referência (visto na prática durante o benchmark #174).
                processor = transformers.AutoProcessor.from_pretrained(
                    checkpoint, max_pixels=_MAX_PIXELS
                )
                model_kwargs: dict[str, Any] = {}
                if config.load_in_4bit:
                    model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    model_kwargs["device_map"] = device
                model = transformers.AutoModelForImageTextToText.from_pretrained(
                    checkpoint, **model_kwargs
                )
                if not config.load_in_4bit:
                    model = model.to(device)
                model.eval()
                return processor, model
            except BackendUnavailableError:
                raise
            except Exception as error:
                raise_backend_execution_error(config.backend, "o carregamento do VLM", error)

        key = f"multimodal_reasoning:{checkpoint}:{device}:{config.load_in_4bit}"
        processor, model = self._lifecycle.get_or_load(key, factory)
        self._device = device
        return torch, processor, model, device


# Extrai um único objeto JSON de uma resposta textual, tolerando fences de
# markdown e texto residual produzido pelo modelo. Respostas inválidas ficam
# vazias para a validação de schema da camada application.
def _parse_json_object(text: str) -> dict[str, Any]:
    """Extrai um objeto JSON de texto do VLM ou retorna objeto vazio."""
    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}
