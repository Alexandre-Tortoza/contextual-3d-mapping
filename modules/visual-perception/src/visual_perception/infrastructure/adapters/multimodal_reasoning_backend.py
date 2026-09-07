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
from visual_perception.domain.region_evidence import EvidenceSlot
from visual_perception.domain.region_reasoning import RegionReasoningRequest
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
        return self._generate_json((image,), prompt, config)

    # Analisa uma região a partir das suas views mask-aware, apresentadas ao
    # VLM como imagens numeradas e rotuladas pelo seu papel, seguidas do
    # contexto de cena estruturado como texto.
    #
    # O exemplo de formato usa placeholders, e não valores plausíveis. Medido:
    # com o exemplo concreto anterior (`"label": "door"`, category `opening`,
    # alternativa `panel`, material `wood`), 26 das 27 regiões rotuladas
    # "door" em `corridor-02-000` reproduziam essa assinatura inteira,
    # inclusive a região de frame cheio. A taxa de eco saltou de 1/54 para
    # 27/54 quando a entrada passou a ser multi-imagem: um exemplo plausível
    # vira a resposta padrão quando a entrada fica mais difícil, e "door" num
    # corredor é plausível o bastante para não parecer erro.
    #
    # Mandar foreground e contexto como imagens *separadas* é o ponto da #203:
    # um recorte único pelo bounding box mistura objeto e fundo, e o modelo
    # descrevia o que ocupava mais pixels. A cena entra como claims tipadas
    # (#202) e o prompt proíbe explicitamente promovê-las a propriedade da
    # região.
    #
    # O prompt pede explicitamente que o modelo OMITA "confidence" quando não
    # souber estimá-la, em vez de chutar: a ausência é representável no contract
    # (ver parse_region_interpretation) e um score inventado destruiria a
    # capacidade de medir qualquer mudança downstream.
    def analyze_region(
        self, request: RegionReasoningRequest, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do VLM para uma região da imagem."""
        return self._generate_json(
            tuple(view.payload for view in request.views), _region_prompt(request), config
        )

    # Executa a conversa multimodal e converte sua resposta textual em objeto
    # JSON. Um JSON inválido vira objeto vazio para a camada application emitir
    # o diagnóstico de schema estável que já possui.
    def _generate_json(
        self, images: tuple[ImagePayload, ...], prompt: str, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Gera e extrai um objeto JSON de uma consulta multimodal ao VLM."""
        torch, processor, model, device = self._get_runtime(config)
        try:
            content: list[dict[str, Any]] = [
                {"type": "image", "image": payload_to_pil(image, config.backend)}
                for image in images
            ]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
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


# Monta o prompt de região a partir do request. É uma função pura: não toca
# em modelo, device nem checkpoint, o que permite testar o contrato textual do
# prompt sem GPU — o mesmo motivo pelo qual _describe_views e
# _describe_scene_claims vivem separados. Chamada por
# RealMultimodalReasoningAdapter.analyze_region.
def _region_prompt(request: RegionReasoningRequest) -> str:
    """Retorna o prompt de região correspondente a ``request``."""
    return (
        f"{_describe_views(request)}"
        "Describe ONLY what is actually visible in the subject region shown above, "
        "even if it is small or blurry — a plain surface (wall, floor, ceiling) is a "
        "valid, specific answer. Do not restate the whole-scene description, and do "
        "NOT name an object merely because this kind of scene usually contains one: "
        "if the pixels show a blank wall, the answer is a wall. Use the context "
        "image(s) only to disambiguate the subject, never to describe the surroundings "
        "instead. Respond with EXACTLY ONE JSON object (never a list/array, "
        "never markdown fences) with exactly these keys: "
        '"label" (non-empty short string, singular — the single best description), '
        '"kind" (exactly one of "thing" for a countable object, "stuff" for an '
        'uncountable surface or material, "part" for a component of a larger object, or '
        '"unknown"), "category" (short coarse category string, optional), "confidence" '
        "(number between 0 and 1 — YOUR ACTUAL CERTAINTY; omit the key entirely if you "
        'cannot estimate it, never guess 1.0), "alternatives" (list of '
        '{"label", "confidence"} objects for competing hypotheses, may be empty), '
        '"description" (string, optional), "attributes" (list of strings, optional), '
        '"condition" (string, optional), "material" (string, optional). '
        "The SHAPE below is the required format. Every value in it is a placeholder "
        "describing what to write there — never copy a placeholder or the example "
        "values into your answer:\n"
        '{"label": "<one noun naming the subject>", "kind": "thing", '
        '"category": "<coarse category>", "confidence": 0.71, '
        '"alternatives": [{"label": "<competing noun>", "confidence": 0.2}], '
        '"description": "<one short sentence>", "attributes": ["<adjective>"], '
        '"condition": "<state>", "material": "<material>"}'
        f"{_describe_scene_claims(request)}"
    )


#: Como cada slot de evidência é apresentado ao VLM. O texto diz ao modelo o
#: que ele está olhando e, para os slots de contexto, que aquilo NÃO é o
#: sujeito — sem isso o modelo tende a descrever o objeto mais saliente da
#: imagem de contexto em vez da região pedida.
_VIEW_ROLES = {
    EvidenceSlot.FOREGROUND_DENSE: (
        "the SUBJECT REGION isolated on a black background (black pixels are not part of it)"
    ),
    EvidenceSlot.TIGHT_CROP: "the SUBJECT REGION's bounding box, background included",
    EvidenceSlot.CONTEXTUAL_CROP: "CONTEXT ONLY: the subject plus its surroundings",
    EvidenceSlot.SCENE_CONDITIONED: "CONTEXT ONLY: the whole scene the subject belongs to",
}


# Descreve, em texto, o que cada imagem enviada representa. Existe porque um
# VLM que recebe várias imagens sem rótulo não sabe qual delas é o sujeito;
# esta é a metade textual da evidência distinguível exigida pela #203.
# Chamada por analyze_region ao montar o prompt.
def _describe_views(request: RegionReasoningRequest) -> str:
    """Retorna o preâmbulo que numera e rotula cada view enviada ao modelo."""
    lines = [
        f"Image {index}: {_VIEW_ROLES[view.slot]}."
        for index, view in enumerate(request.views, start=1)
    ]
    return "You are given {count} image(s) of the same subject region.\n{lines}\n".format(
        count=len(request.views), lines="\n".join(lines)
    )


# Converte as claims de cena estruturadas em texto agrupado por kind,
# preservando a confiança quando ela existe. Existe para que o contexto de
# cena chegue ao modelo inteiro (#202) em vez de achatado em uma única
# descrição, e para instruir explicitamente que ele não é verdade sobre a
# região. Chamada por analyze_region ao montar o prompt.
def _describe_scene_claims(request: RegionReasoningRequest) -> str:
    """Retorna o bloco de contexto de cena, ou string vazia se não houver claims."""
    if not request.scene_claims:
        return ""
    grouped: dict[str, list[str]] = {}
    for claim in request.scene_claims:
        score = "" if claim.confidence is None else f" (confidence {claim.confidence.value:.2f})"
        grouped.setdefault(claim.kind.value, []).append(f"{claim.value}{score}")
    rendered = "; ".join(f"{kind}: {', '.join(values)}" for kind, values in sorted(grouped.items()))
    return (
        "\nScene context, for disambiguation only — these are properties of the SCENE, "
        f"never of the subject region, and must not be repeated as the label: {rendered}"
    )


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
