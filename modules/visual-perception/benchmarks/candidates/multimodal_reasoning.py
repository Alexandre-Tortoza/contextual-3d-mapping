"""Candidatos de multimodal reasoning para o benchmark da #174.

Proxy de qualidade: taxa de validade de schema contra o schema exato de
resposta de cena que ``application/scene_context.py`` valida (``scene_type``,
``description``, listas opcionais ``attributes``/``hazards``, ``confidence``
opcional), mais uma checagem de não-degenerescência no texto da descrição.
Isso é literalmente "o modelo produz uma saída que a camada de aplicação
consegue usar", que é a pergunta de qualidade relevante para a tarefa deste
port.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from PIL import Image

Candidate = tuple[str, Callable[[], object], Callable[[object], float]]

_PROMPT = (
    "You are analyzing one frame from a mobile robot's indoor corridor traversal. "
    "Describe the scene as strict JSON with exactly these keys: "
    '"scene_type" (short string), "description" (one or two sentences), '
    '"attributes" (list of short strings, may be empty), '
    '"hazards" (list of short strings describing any hazards, may be empty), '
    '"confidence" (float between 0 and 1). '
    "Respond with ONLY the JSON object, no markdown fences, no extra text."
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


# Pontua uma resposta de texto bruta do modelo contra o schema de resposta de
# cena esperado por application/scene_context.py: extrai o bloco JSON,
# valida os campos obrigatórios/opcionais e recompensa uma descrição
# não-degenerada. É o score de qualidade retornado por run_once para cada
# candidato de multimodal reasoning.
def _score_scene_response(text: str) -> float:
    match = _JSON_BLOCK.search(text)
    if match is None:
        return 0.0
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(payload, dict):
        return 0.0

    score = 0.3  # valid JSON object

    scene_type = payload.get("scene_type")
    description = payload.get("description")
    if isinstance(scene_type, str) and scene_type.strip():
        score += 0.2
    if isinstance(description, str) and description.strip():
        score += 0.2
        if len(set(description.lower().split())) >= 5:
            score += 0.15  # non-degenerate: not a one-word/repetitive answer

    optional_ok = True
    for key in ("attributes", "hazards"):
        if key in payload and not isinstance(payload[key], list):
            optional_ok = False
    confidence = payload.get("confidence", 1.0)
    if not isinstance(confidence, int | float) or not (0.0 <= float(confidence) <= 1.0):
        optional_ok = False
    if optional_ok:
        score += 0.15

    return max(0.0, min(1.0, score))


# Gera um iterador infinito que percorre ciclicamente os frames de referência,
# decodificando cada imagem como RGB sob demanda. Usada pelos candidatos deste
# módulo para alimentar run_once com uma nova imagem a cada chamada.
def _frame_cycle(frames: Iterable[Path]) -> Iterable[Image.Image]:
    for path in itertools.cycle(list(frames)):
        yield Image.open(path).convert("RGB")


#: Limita o número de tokens visuais que o encoder de resolução dinâmica do
#: Qwen2.5-VL produz por imagem. Sem isso, uma imagem de 640x480 já é o
#: suficiente para estourar os ~7.6GB úteis da 3060 só com os pesos do
#: modelo em fp16 (visto na prática: OOM real do 3B mesmo sem quantização).
_MAX_PIXELS = 640 * 480


# Constrói um candidato de benchmark (nome, factory, run_once) para um
# checkpoint da família Qwen2.5-VL, com o encoder visual limitado a
# _MAX_PIXELS: factory carrega o processor/modelo (em 4-bit, via
# quantização on-the-fly a partir de um checkpoint fp16, ou já
# pré-quantizado quando ``pre_quantized=True`` — necessário para o 7B, cuja
# quantização on-the-fly estourava VRAM só no carregamento, ver #174), e
# run_once envia um frame com o prompt de análise de cena e pontua a
# resposta via _score_scene_response. Chamada por candidates() para montar
# a lista de candidatos comparados por backend_benchmark.py.
def _qwen_vl_candidate(
    name: str, checkpoint: str, frames: list[Path], *, pre_quantized: bool = False
) -> Candidate:
    frame_iter = _frame_cycle(frames)

    # Carrega o processor (com o encoder visual limitado a _MAX_PIXELS) e o
    # modelo Qwen2.5-VL do checkpoint indicado; chamada uma única vez por
    # benchmark_candidate.
    def factory() -> object:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(checkpoint, max_pixels=_MAX_PIXELS)
        kwargs: dict[str, object] = {}
        if pre_quantized:
            # O checkpoint já é 4-bit (config.quantization_config próprio);
            # não materializa fp16 intermediário no carregamento.
            # ``device_map="cuda"`` (dispatch via accelerate) deixa o estado
            # de quantização de algumas camadas do vision tower sem
            # inicializar nesse checkpoint; carregar em CPU e mover com
            # ``.to("cuda")`` evita esse caminho de dispatch.
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, **kwargs).eval()
            model = model.to("cuda")
            return (processor, model)

        from transformers import BitsAndBytesConfig

        kwargs["device_map"] = "cuda"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, **kwargs).eval()
        return (processor, model)

    # Monta o prompt multimodal (imagem + instrução) para o próximo frame do
    # ciclo, gera a resposta do modelo e a pontua via _score_scene_response;
    # é o run_once exigido pelo contrato de benchmark_candidate.
    def run_once(bundle: object) -> float:
        processor, model = bundle  # type: ignore[misc]
        image = next(frame_iter)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": _PROMPT}],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return _score_scene_response(response)

    return (name, factory, run_once)


# Monta a lista de candidatos de multimodal reasoning avaliados pelo benchmark
# da #174: duas contagens de parâmetro da família Qwen2.5-VL, ambas em 4-bit
# (ver _qwen_vl_candidate) para caber no orçamento de VRAM de referência — o
# 7B usa o checkpoint já pré-quantizado da Unsloth, já que a quantização
# on-the-fly a partir do checkpoint fp16 oficial estourava VRAM no
# carregamento nesta GPU. Chamada por run_backend_benchmark.py ao
# selecionar o stage "multimodal_reasoning".
def candidates(frames: list[Path]) -> list[Candidate]:
    return [
        _qwen_vl_candidate(
            "qwen2.5-vl-4bit:Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct", frames
        ),
        _qwen_vl_candidate(
            "qwen2.5-vl-4bit:unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
            "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
            frames,
            pre_quantized=True,
        ),
    ]
