"""Candidatos de multimodal reasoning para o benchmark da #174.

Proxy de qualidade: taxa de validade da resposta de cena contra o contrato de
produção. Cada candidato recebe o mesmo prompt que o adapter real envia
(``scene_prompt``), a resposta passa pelo mesmo parser (``parse_json_object``) e
é validada pela mesma função que o pipeline usa (``validate_scene_response``).
Isso é literalmente "o modelo produz uma saída que a camada de aplicação
consegue usar". Até a #248 o benchmark tinha prompt, parser e schema próprios,
e pontuava justamente os campos que o pipeline recusa.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from PIL import Image

from visual_perception.application.scene_context import validate_scene_response
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.infrastructure.adapters.multimodal_reasoning_backend import MAXIMUM_VISUAL_PIXELS
from visual_perception.infrastructure.adapters.reasoning_prompts import parse_json_object, scene_prompt

Candidate = tuple[str, Callable[[], object], Callable[[object], float]]


# Pontua uma resposta de texto bruta do modelo pelo contrato de produção: 1
# quando o pipeline aceitaria a resposta, 0 quando a recusaria. A média sobre
# os frames é a taxa de validade que o benchmark compara entre candidatos.
def score_scene_response(text: str) -> float:
    """Retorna 1.0 para uma resposta de cena que o pipeline aceita e 0.0 caso contrário.

    Argumentos:
        text: saída textual bruta do modelo.
    Retorna:
        a validade da resposta contra o contrato de cena de produção.
    """
    try:
        validate_scene_response(parse_json_object(text))
    except ValueError:
        return 0.0
    return 1.0


# Gera um iterador infinito que percorre ciclicamente os frames de referência,
# decodificando cada imagem como RGB sob demanda. Usada pelos candidatos deste
# módulo para alimentar run_once com uma nova imagem a cada chamada.
def _frame_cycle(frames: Iterable[Path]) -> Iterable[Image.Image]:
    for path in itertools.cycle(list(frames)):
        yield Image.open(path).convert("RGB")


# Constrói um candidato de benchmark (nome, factory, run_once) para um
# checkpoint da família Qwen2.5-VL, com o encoder visual limitado a
# MAXIMUM_VISUAL_PIXELS: factory carrega o processor/modelo (em 4-bit, via
# quantização on-the-fly a partir de um checkpoint fp16, ou já
# pré-quantizado quando ``pre_quantized=True`` — necessário para o 7B, cuja
# quantização on-the-fly estourava VRAM só no carregamento, ver #174), e
# run_once envia um frame com o prompt de análise de cena e pontua a
# resposta via score_scene_response. Chamada por candidates() para montar
# a lista de candidatos comparados por backend_benchmark.py.
def _qwen_vl_candidate(
    name: str, checkpoint: str, frames: list[Path], *, pre_quantized: bool = False
) -> Candidate:
    frame_iter = _frame_cycle(frames)

    # Carrega o processor (com o encoder visual limitado a MAXIMUM_VISUAL_PIXELS) e o
    # modelo Qwen2.5-VL do checkpoint indicado; chamada uma única vez por
    # benchmark_candidate.
    def factory() -> object:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(checkpoint, max_pixels=MAXIMUM_VISUAL_PIXELS)
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
    # ciclo, gera a resposta do modelo e a pontua via score_scene_response;
    # é o run_once exigido pelo contrato de benchmark_candidate.
    def run_once(bundle: object) -> float:
        processor, model = bundle  # type: ignore[misc]
        image = next(frame_iter)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": scene_prompt()}],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=MultimodalReasoningConfig().max_new_tokens, do_sample=False
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return score_scene_response(response)

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
