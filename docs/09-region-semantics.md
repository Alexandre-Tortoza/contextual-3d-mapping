# 09. Region Semantics

## Onde estamos na pipeline

```mermaid
flowchart LR
    A["ObservedRegion"] --> E["Qwen2.5-VL Region Semantics"]:::current
    B["RegionView[]"] --> E
    C["SceneContext"] --> E
    E --> F["SemanticClaim[]"]
    classDef current stroke-width:3px,font-weight:bold;
```

## Objetivo

Adicionar interpretações semânticas auditáveis à geometria 2D da região sem alterar a própria geometria.

Depois do merge, `region_id`, máscara e bounding box estão congelados. Esta etapa apenas acrescenta claims.

```text
geometria
    -> permanece igual

semântica
    -> é anexada
```

## O que o Qwen recebe

O reasoner não recebe apenas uma bounding box crua.

Ele recebe um `RegionReasoningRequest` contendo:

```text
region_id
region_box
image_width
image_height
views[]
scene_claims[]
```

As `views[]` são derivadas da máscara do SAM e podem incluir:

```text
masked_subject
    sujeito isolado, fundo neutralizado

tight_crop
    recorte justo

contextual_crop
    região + entorno, com o sujeito contornado

scene_conditioned
    view global quando configurada
```

Assim, a máscara do SAM influencia o Qwen **indiretamente pelos pixels que são mostrados ao modelo**.

## Relação SAM -> Qwen

```text
SAM
 -> mask
 -> RegionView
 -> Qwen
 -> hipótese textual
```

A máscara booleana não é usada como label e não diz ao Qwen o que existe. Ela apenas define o sujeito visual.

Exemplo:

```text
SAM:
    "estes pixels formam uma região"

Qwen:
    "vendo essa região e o contexto, acho que é wooden panel"
```

## O DINO participa diretamente dessa decisão?

Não na implementação atual.

O Qwen recebe views em pixels e contexto de cena. O `foreground_dense` do DINO fica preservado como evidência visual da região, mas não é passado como vetor numérico ao `analyze_region()`.

```text
DINO embedding 768D
    -> evidência visual densa

Qwen
    -> interpretação multimodal dos pixels
```

Isso mantém o reasoner independente do espaço interno do DINO.

## O que o Qwen produz

O contrato atual aceita:

```text
primary label
alternatives
category
region kind
confidence opcional
atributos descritivos
condition
material
```

Exemplo real para `region-2c84165423b25fc3`:

```json
{
  "label": "wooden panel",
  "confidence": 0.9,
  "alternatives": [
    {"label": "wooden door", "confidence": 0.8}
  ],
  "category": "door",
  "kind": "thing",
  "attributes": ["smooth", "shiny"],
  "condition": "new",
  "material": "wood"
}
```

A resposta bruta é preservada em evidência para permitir auditoria posterior.

## Claims, não uma label única

A pipeline evita transformar o resultado do Qwen diretamente em um campo único:

```text
region.label = "wooden panel"
```

Em vez disso, cria claims irmãos:

```text
PRIMARY:
    wooden panel

ALTERNATIVE:
    wooden door
```

Isso é importante porque a etapa seguinte pode encontrar evidência visual favorecendo a alternativa.

## Exemplo de por que alternativas importam

Para a região de referência:

```text
Qwen primary:
    wooden panel

Qwen alternative:
    wooden door
```

Depois, o CLIP encontra:

```text
masked_subject -> wooden door combina mais
contextual_crop -> wooden panel combina mais
```

Se a alternativa tivesse sido descartada, o pipeline não teria uma hipótese concorrente explícita para comparar.

## Confidence do Qwen não é qualidade geométrica

Na mesma região:

```text
geometric_confidence: 0.97944176197052
semantic confidence:  0.9
```

Esses sinais medem coisas diferentes.

```text
geometric_confidence
    -> qualidade/confiança da região visual proposta

semantic confidence
    -> score declarado pelo Qwen para a interpretação
```

Eles não devem ser combinados como se fossem probabilidades equivalentes.

## Um problema real da reference run

No frame completo:

```text
semantic_confidence
count: 40
min: 0.9
max: 0.9
distinct_values: 1
stddev: 0.0
degenerate: true
```

O Qwen devolveu `0.9` para todas as regiões.

Isso torna esse score pouco informativo para distinguir uma região confiável de uma região duvidosa.

Por isso a pipeline possui um canal independente de suporte com CLIP.

## O que acontece com ausência de score

Se o modelo omitir `confidence`, o sistema preserva:

```text
confidence = None
```

Não converte para:

```text
confidence = 1.0
```

Essa distinção evita inventar certeza.

## Saída

```text
SemanticClaim[]
    |
    +--> Hypothesis Support
    +--> calibration
    +--> selective refinement
    +--> reconciliation
```

As claims continuam vinculadas à mesma `ObservedRegion` e preservam proveniência do modelo, checkpoint, estágio e prompt.

## Próxima leitura

- [10. Hypothesis Support](./10-hypothesis-support.md)
- [Política de semântica contextual](../modules/visual-perception/docs/contextual-semantics.md)
- [Pipeline detalhada de `visual-perception`](../modules/visual-perception/docs/pipeline.md)