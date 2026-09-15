"""Prompts e parsing de transporte compartilhados pelos backends de raciocínio multimodal.

Os backends (Qwen local, Gemini remoto) recebem exatamente o mesmo texto e a mesma
extração de JSON, para que uma comparação entre eles meça o modelo e não o prompt.
A versão desses prompts é ``MultimodalReasoningConfig.prompt_version``.
"""

from __future__ import annotations

import json
from typing import Any

from visual_perception.domain.region_evidence import EvidenceSlot
from visual_perception.domain.region_reasoning import RegionReasoningRequest, RegionRelationRequest
from visual_perception.domain.relations import NO_RELATION_PREDICATE, SEMANTIC_RELATION_PREDICATES


# Monta o prompt de região a partir do request. É uma função pura: não toca
# em modelo, device nem checkpoint, o que permite testar o contrato textual do
# prompt sem GPU — o mesmo motivo pelo qual describe_views e
# describe_scene_claims vivem separados. Chamada por
# RealMultimodalReasoningAdapter.analyze_region.
def region_prompt(request: RegionReasoningRequest) -> str:
    """Retorna o prompt de região correspondente a ``request``."""
    return (
        f"{describe_views(request)}"
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
        '"alternatives": [{"label": "<competing noun>", "confidence": 0.18}], '
        '"description": "<one short sentence>", "attributes": ["<adjective>"], '
        '"condition": "<state>", "material": "<material>"}'
        f"{describe_scene_claims(request)}"
        f"{describe_prior(request)}"
    )


# Monta o prompt de relação a partir do request. Função pura, pelo mesmo motivo
# que ``region_prompt``: o contrato textual precisa ser testável sem GPU.
# Chamada por RealMultimodalReasoningAdapter.analyze_relation.
#
# A primeira redação deste prompt dizia que ``none`` era "a resposta esperada
# para a maioria dos pares, e melhor que um chute plausível". Medido em
# ``corridor-02-000``: o modelo respondeu ``none`` em **16 de 16** pares, quinze
# deles com ``confidence`` exatamente 0,95 — inclusive para um ``ceiling tile``
# inteiramente contido num ``ceiling``, que é literalmente ``part_of``.
#
# É a mesma lição que a #212 já tinha registrado do outro lado: uma frase que
# tenta impedir alucinação com força demais deixa de medir qualquer coisa.
# ``none`` continua disponível e continua sendo a resposta certa quando nenhuma
# relação é visível — mas ela deixou de ser anunciada como o desfecho esperado.
def relation_prompt(request: RegionRelationRequest) -> str:
    """Retorna o prompt de relação correspondente a ``request``."""
    predicates = ", ".join(f'"{predicate}"' for predicate in sorted(SEMANTIC_RELATION_PREDICATES))
    return (
        "You are given one image showing two regions of the same scene. The FIRST region is "
        "outlined in GREEN and the SECOND region is outlined in BLUE.\n"
        f"The green region was described as: {request.subject_concept} "
        f"({request.subject_kind}).\n"
        f"The blue region was described as: {request.object_concept} ({request.object_kind}).\n"
        f"Measured geometry: {request.geometric_summary}.\n"
        "State how the GREEN region relates to the BLUE region, using ONLY what the pixels show. "
        f'Use "{NO_RELATION_PREDICATE}" when none of the listed relations is visible in the image. '
        "Do NOT infer depth, distance, or which one is in front: this is a single image and those "
        "are not visible. Respond with EXACTLY ONE JSON object (never a list/array, never markdown fences) "
        'with exactly these keys: "predicate" (exactly one of '
        f'"{NO_RELATION_PREDICATE}", {predicates}), "confidence" (number between 0 and 1 — YOUR '
        "ACTUAL CERTAINTY; omit the key entirely if you cannot estimate it, never guess 1.0). "
        "The SHAPE below is the required format; every value in it is a placeholder:\n"
        '{"predicate": "<one of the listed values>", "confidence": 0.42}'
    )


#: Como cada slot de evidência é apresentado ao VLM. O texto diz ao modelo o
#: que ele está olhando e, para os slots de contexto, que aquilo NÃO é o
#: sujeito — sem isso o modelo tende a descrever o objeto mais saliente da
#: imagem de contexto em vez da região pedida.
_VIEW_ROLES = {
    EvidenceSlot.FOREGROUND_DENSE: (
        "the SUBJECT REGION isolated on a black background (black pixels are not part of it)"
    ),
    # Cinza, e não preto: nestes frames o preto é a cor da vinheta do fisheye, e
    # dizer "black is not part of it" faria o modelo tratar a borda da lente
    # como fundo removido em vez de conteúdo ausente (#202).
    EvidenceSlot.MASKED_SUBJECT: (
        "the SUBJECT REGION isolated on a flat grey background (grey pixels are not part of it)"
    ),
    EvidenceSlot.TIGHT_CROP: "the SUBJECT REGION's bounding box, background included",
    EvidenceSlot.CONTEXTUAL_CROP: (
        "the subject plus its surroundings, with the subject outlined in green; "
        "the surroundings are CONTEXT ONLY"
    ),
    EvidenceSlot.SCENE_CONDITIONED: "CONTEXT ONLY: the whole scene the subject belongs to",
}


# Descreve, em texto, o que cada imagem enviada representa. Existe porque um
# VLM que recebe várias imagens sem rótulo não sabe qual delas é o sujeito;
# esta é a metade textual da evidência distinguível exigida pela #203.
# Chamada por analyze_region ao montar o prompt.
def describe_views(request: RegionReasoningRequest) -> str:
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
def describe_scene_claims(request: RegionReasoningRequest) -> str:
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


# Renderiza o que a observação anterior afirmou sobre a mesma área da imagem.
# Espelha deliberadamente a forma de ``describe_scene_claims``: um bloco curto,
# no fim do prompt, que declara o que a informação é e o que ela não autoriza.
# A sugestão é apresentada como observação anterior **falível** e não como
# resposta, porque o modo de falha conhecido deste prompt é o modelo copiar o
# que lhe é oferecido — foi o que a #202 mediu com o exemplo concreto de
# ``confidence`` e com o inventário de objetos da cena.
def describe_prior(request: RegionReasoningRequest) -> str:
    """Retorna o bloco de observação anterior, ou string vazia se não houver."""
    if request.prior is None:
        return ""
    category = "" if request.prior.category is None else f", category {request.prior.category}"
    return (
        "\nA PREVIOUS view of this same area was described as "
        f"\"{request.prior.concept}\"{category}. That description may be wrong, and the "
        "viewpoint has changed. Use it only to resolve a genuine ambiguity in what you "
        "see now; if the pixels disagree with it, describe what you actually see."
    )


# Monta o prompt de análise de cena. É função pura, como os prompts de região e
# de relação, para que o contrato textual seja testável sem GPU e para que o
# benchmark de candidatos meça exatamente o prompt de produção.
#
# O prompt descreve o **ambiente**, e nunca pede um inventário de objetos.
# Medido em corridor-02-002 com o contract anterior: o modelo devolveu
# ``attributes: ["fisheye lens", "carpeted floor", "suitcase"]``, onde a "mala"
# era o próprio quad que carrega a câmera — e aquele texto ia para o prompt de
# cada uma das 60 regiões. Não há frase que conserte isso: o campo que pedia
# objetos precisou sair (#202).
#
# O ``confidence`` do exemplo é deliberadamente não redondo, pela mesma razão
# que o do prompt de região virou 0,71 e o do de relação virou 0,42: um valor
# plausível no exemplo é copiado de volta.
def scene_prompt() -> str:
    """Retorna o prompt de análise de ambiente enviado com a imagem inteira."""
    return (
        "Describe the ENVIRONMENT shown in this image. Do NOT list or name individual "
        "objects. Respond with EXACTLY ONE JSON object (never a list/array, never markdown "
        "fences) with exactly these keys: scene_type (string), environment (string), layout "
        "(string), lighting (string), visibility (string), navigability (string), confidence "
        "(float between 0 and 1). Example of the exact shape required:\n"
        '{"scene_type": "<one noun naming the kind of place>", '
        '"environment": "<indoor or outdoor>", '
        '"layout": "<how the space is arranged, in one clause>", '
        '"lighting": "<how the space is lit>", '
        '"visibility": "<how far and how clearly one can see>", '
        '"navigability": "<how traversable the space is>", "confidence": 0.63}'
    )


# Monta o prompt de descoberta de conceitos (#277). É separado do prompt de cena
# de propósito: o de cena proíbe inventário de objetos porque o texto dele entra
# no prompt de cada região (#202); estes conceitos só viram consultas de
# grounding e nunca chegam ao reasoner de região. Os exemplos usam placeholders e
# confianças não redondas pela mesma razão dos outros prompts.
def concept_prompt(max_concepts: int) -> str:
    """Retorna o prompt que pede conceitos visuais concretos para grounding."""
    return (
        "List the concrete visual things in this image that are worth locating on a map of a "
        "disaster or inspection site. Use short noun phrases of 1 to 4 words that an object "
        "segmentation model can find. Put physical objects in \"entities\". Put visible damage, "
        "hazards, debris, obstructions, signage, openings and anomalies in \"contextual_features\". "
        "Do NOT list plain wall, floor or ceiling unless the phrase names a visible condition "
        "(for example a crack or a stain). Do NOT describe the scene and do NOT name the camera "
        f"rig or vehicle carrying the camera. List at most {max_concepts} phrases in total; an "
        "empty list is valid. Omit \"confidence\" when you cannot estimate it. Respond with "
        "EXACTLY ONE JSON object (never a list, never markdown fences) with exactly these keys:\n"
        '{"entities": [{"concept": "<short noun phrase>", "confidence": 0.57}], '
        '"contextual_features": [{"concept": "<short noun phrase>", "confidence": 0.44}]}'
    )


# Extrai um único objeto JSON de uma resposta textual, tolerando fences de
# markdown e texto residual produzido pelo modelo. Decodifica o primeiro objeto
# completo a partir do primeiro ``{`` em vez de recortar até o último ``}``: o
# Gemini Robotics ER devolveu o objeto de cena correto seguido de um ``}`` a mais,
# e o recorte incluía a chave sobrando e descartava a resposta inteira (medido em
# 2026-09-15). Respostas inválidas ficam vazias para a validação de schema da
# camada application. Pública porque o benchmark de candidatos precisa pontuar com
# o mesmo parser da produção.
def parse_json_object(text: str) -> dict[str, Any]:
    """Extrai um objeto JSON de texto do VLM ou retorna objeto vazio."""
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    start = stripped.find("{")
    if start < 0:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped, start)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
