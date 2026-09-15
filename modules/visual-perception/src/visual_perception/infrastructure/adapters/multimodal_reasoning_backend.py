"""Adapter real de backend de raciocínio multimodal.

Issue: #189. Checkpoint selecionado pelo benchmark #174 (Qwen2.5-VL-3B-Instruct
em 4-bit, ver ``benchmarks/results/benchmark-174-multimodal_reasoning-*.json``).
"""

from __future__ import annotations

from typing import Any

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_reasoning import RegionReasoningRequest, RegionRelationRequest
from visual_perception.infrastructure.adapters._runtime import (
    payload_to_pil,
    raise_backend_execution_error,
    require_checkpoint,
    require_module,
    resolve_device,
)
from visual_perception.infrastructure.adapters.reasoning_prompts import (
    concept_prompt,
    parse_json_object,
    region_prompt,
    relation_prompt,
    scene_prompt,
)

#: Limita o número de tokens visuais que encoders de resolução dinâmica (ex:
#: Qwen2.5-VL) produzem por imagem: sem isso, uma imagem de 640x480 já estoura
#: os ~7,6 GB úteis da 3060 de referência. Público porque o benchmark de
#: candidatos precisa medir o custo sob exatamente o mesmo limite.
MAXIMUM_VISUAL_PIXELS = 640 * 480


# Implementa o port MultimodalReasoner com um VLM compatível com Transformers,
# com prompts e parsing compartilhados em ``reasoning_prompts``.
class RealMultimodalReasoningAdapter:
    """Satisfaz :class:`~visual_perception.ports.multimodal_reasoning.MultimodalReasoner`.

    Usa um VLM compatível com ``AutoModelForImageTextToText``. O adapter só
    converte transporte e JSON; a validação semântica permanece em application.
    """

    # Recebe (ou cria, se omitido) o lifecycle manager que carrega e mantém
    # residente o VLM sob demanda. Compartilhar o mesmo manager entre os 4
    # adapters reais (ver ``factory.py``) permite reaproveitar modelos já
    # residentes entre estágios intercalados no mesmo frame — inclusive
    # entre chamadas repetidas de analyze_scene/analyze_region dentro do
    # mesmo pipeline, que reusam o VLM já carregado via o cache do próprio
    # manager.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem carregar VLM ou checkpoint."""
        self._lifecycle = lifecycle or ModelLifecycleManager()
        self._device: str | None = None

    # Analisa a imagem completa com um schema mínimo de cena. A resposta
    # permanece bruta para que scene_context faça sua própria validação.
    def analyze_scene(self, image: ImagePayload, config: MultimodalReasoningConfig) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do VLM para o contexto da cena."""
        return self._generate_json((image,), scene_prompt(), config)

    # Propõe conceitos concretos para o grounding (#277) com o mesmo VLM residente;
    # satisfaz o port SceneConceptDiscoverer.
    def discover_concepts(
        self, image: ImagePayload, config: MultimodalReasoningConfig, *, max_concepts: int
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do VLM com conceitos a localizar."""
        return self._generate_json((image,), concept_prompt(max_concepts), config)

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
            tuple(view.payload for view in request.views), region_prompt(request), config
        )

    # Julga a relação entre duas regiões a partir da view de par (#206). O
    # prompt oferece "none" como saída válida e repete a medida geométrica já
    # calculada: sem a saída de escape, um vocabulário fechado faz o modelo
    # escolher o predicado menos ruim, e a aresta inventada entra no grafo como
    # se fosse observação. O quanto essa saída é *anunciada* também importa, e
    # está medido em ``relation_prompt``.
    def analyze_relation(
        self, request: RegionRelationRequest, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do VLM sobre a relação entre duas regiões."""
        return self._generate_json(
            tuple(view.payload for view in request.views), relation_prompt(request), config
        )

    # Executa a conversa multimodal e converte sua resposta textual em objeto
    # JSON. Um JSON inválido vira objeto vazio para a camada application emitir
    # o diagnóstico de schema estável que já possui.
    def _generate_json(
        self, images: tuple[ImagePayload, ...], prompt: str, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Gera e extrai um objeto JSON de uma consulta multimodal ao VLM."""
        torch, processor, model, device, key = self._get_runtime(config)
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

            def _run() -> Any:
                with torch.inference_mode():
                    return model.generate(**inputs, **generation_kwargs)

            generated = self._lifecycle.call_with_eviction(key, _run)
            prompt_tokens = int(inputs["input_ids"].shape[1])
            text = processor.batch_decode(generated[:, prompt_tokens:], skip_special_tokens=True)[0]
            return parse_json_object(text)
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "a inferência multimodal", error)

    # Carrega o VLM e seu processor apenas quando o backend real é composto,
    # delegando residência ao lifecycle manager compartilhado. O modo 4-bit
    # delega o posicionamento de layers ao Transformers.
    def _get_runtime(self, config: MultimodalReasoningConfig) -> tuple[Any, Any, Any, str, str]:
        """Retorna torch, processor, modelo, device e a key residente para a configuração solicitada."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)

        def factory() -> tuple[Any, Any]:
            try:
                transformers = require_module("transformers", config.backend)
                # MAXIMUM_VISUAL_PIXELS limita o encoder de resolução dinâmica de
                # modelos como o Qwen2.5-VL: sem isso, uma única imagem já
                # é o suficiente para estourar os ~7.6GB úteis da 3060 de
                # referência (visto na prática durante o benchmark #174).
                processor = transformers.AutoProcessor.from_pretrained(
                    checkpoint, max_pixels=MAXIMUM_VISUAL_PIXELS
                )
                model_kwargs: dict[str, Any] = {}
                if config.load_in_4bit:
                    model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    model_kwargs["device_map"] = device
                    # Sem isso, a conversão para 4-bit materializa os pesos
                    # fp16 inteiros na GPU antes de quantizar, dobrando o
                    # pico de VRAM só durante o load (visto na prática: OOM
                    # no carregamento mesmo sem nenhum outro modelo residente).
                    model_kwargs["low_cpu_mem_usage"] = True
                model = transformers.AutoModelForImageTextToText.from_pretrained(
                    checkpoint, **model_kwargs
                )
                if not config.load_in_4bit:
                    model = model.to(device)
                model.eval()
                return processor, model
            except BackendUnavailableError:
                raise
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # Ver comentário equivalente em feature_extraction_backend.py:
                # preserva o tipo de OOM para que _load_with_eviction possa
                # liberar residentes por LRU e tentar de novo.
                raise
            except Exception as error:
                raise_backend_execution_error(config.backend, "o carregamento do VLM", error)

        key = f"multimodal_reasoning:{checkpoint}:{device}:{config.load_in_4bit}"
        processor, model = self._lifecycle.get_or_load(key, factory)
        self._device = device
        return torch, processor, model, device, key
