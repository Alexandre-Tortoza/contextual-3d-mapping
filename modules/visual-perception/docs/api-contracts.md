# API pública e contracts

Consumidores devem importar a superfície estável a partir de `visual_perception`, não de
`application/`, `domain/` ou `infrastructure/`. A lista autoritativa de exports está em
[`src/visual_perception/__init__.py`](../src/visual_perception/__init__.py).

## Entrada e composição

| Símbolo público | Papel |
| --- | --- |
| `ImageObservation` | Metadados da imagem: dimensões, encoding, artifact de origem e `ObservationReference`. |
| `ImagePayload` | Pixels RGB resolvidos em `numpy.ndarray` com shape `(H, W, 3)`. |
| `ModuleConfig` | Configuração validada dos estágios, backends e orçamento de memória. |
| `QualityProfile` | Intenção de execução: `research_quality` ou `reduced_cost`. |
| `PerceptionPorts` | Os quatro backends que o pipeline usa; normalmente criado por `create_perception_ports`. |
| `create_perception_ports` | Compõe fakes ou adapters reais conforme `ModuleConfig`. |

`ImageObservation` é o contract de metadados e `ImagePayload` é o dado de inferência;
ambos são necessários porque o primeiro não acopla a API a uma biblioteca de imagens.
O payload precisa ter as mesmas dimensões declaradas na observação. Os encodings aceitos
e demais validações estão em
[`domain/image_observation.py`](../src/visual_perception/domain/image_observation.py) e
[`domain/image_payload.py`](../src/visual_perception/domain/image_payload.py).

`ObservationReference` e `SourceArtifactReference` são exports compartilhados do
repositório. Consulte [`contracts/observations`](../../../contracts/observations/README.md)
para o significado de identidade, timestamp, frame e artifact.

## Execução e saída

```python
from visual_perception import (
    ImageObservation,
    ImagePayload,
    ModuleConfig,
    create_perception_ports,
    run_canonical_pipeline,
)

config = ModuleConfig()
result = run_canonical_pipeline(image, payload, config, create_perception_ports(config))
```

`run_canonical_pipeline(...) -> PipelineResult` devolve:

- `observation: VisualObservation`, a saída canônica consumida downstream;
- `region_interpretation_failures`, falhas recuperáveis e locais à interpretação de uma
  região;
- `audit: AuditResult`, com `passed`, `errors` e `warnings` para decisão do consumidor.

Falhas de infraestrutura, configuração ou backend usam a hierarquia de erros definida
em [`domain/errors.py`](../src/visual_perception/domain/errors.py). A integração com um
runtime pode traduzi-las para seu próprio diagnóstico, como demonstrado em
[integration.md](integration.md).

## Semântica da saída

`VisualObservation` contém o `source` compartilhado, dimensões, contexto de cena,
regiões, relações, versão de schema e convenção de coordenadas. Seus contracts são:

| Conceito | Significado para o consumidor | Código dono |
| --- | --- | --- |
| `ObservedRegion` | Região com máscara, box, confiança geométrica, propostas contribuintes, claims e referências de embedding. | [`domain/regions.py`](../src/visual_perception/domain/regions.py) |
| `SemanticClaim` | Hipótese auditável com tipo, valor, confiança, evidência e proveniência de modelo. | [`domain/semantics.py`](../src/visual_perception/domain/semantics.py) |
| `CandidateRelation` | Relação entre regiões válida apenas como candidata 2D. | [`domain/relations.py`](../src/visual_perception/domain/relations.py) |
| `AuditResult` | Achados determinísticos; `ERROR` invalida a observação, `WARNING` preserva contradição visível. | [`domain/audit.py`](../src/visual_perception/domain/audit.py) |
| `ModelProvenance` | Identidade do produtor de claim ou relação; não substitui a proveniência compartilhada. | [`domain/references.py`](../src/visual_perception/domain/references.py) |

Não suponha que claims de mesmo tipo concordam, que `geometric_confidence` é confiança
semântica, ou que uma relação é confirmada em 3D. Antes de enviar a saída a outro módulo,
trate `audit.passed` como o critério de validade estrutural e preserve warnings,
evidências e proveniência.

## Ports de extensão

Uma implementação alternativa deve satisfazer um dos `Protocol` em
[`ports/`](../src/visual_perception/ports/): `RegionDiscoverer`,
`DenseFeatureExtractor`, `LanguageAlignedEncoder` ou `MultimodalReasoner`. Ela deve
devolver apenas os contracts do módulo, sem vazar tensors ou erros específicos do
framework. Para usar o port, injete-o em `PerceptionPorts`; não acrescente uma nova
estratégia à assinatura do pipeline.

As invariantes por port e suas responsabilidades exatas estão nas docstrings dos quatro
arquivos. [model-backends.md](model-backends.md) explica os adapters concretos já
fornecidos.
