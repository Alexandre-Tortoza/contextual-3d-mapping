# Visual Perception

Esta é a documentação local de `visual-perception`. O diretório raiz [`docs/`](../../../docs/README.md) descreve a pipeline completa entre módulos; aqui ficam arquitetura interna, backends, política semântica, pontos de extensão e evidências específicas deste módulo.

A visão completa do repositório, incluindo todas as relações de `visual-perception` com `state-estimation`, `geometric-map`, `sensor-association`, `semantic-fusion`, `semantic-map`, adapters e aplicações, está em [`docs/system-flow.md`](../../../docs/system-flow.md).

## Papel no sistema

`visual-perception` transforma um frame RGB canônico em `VisualObservation`, preservando regiões, evidências densas, evidência alinhada à linguagem, hipóteses semânticas, contexto de cena, relações e proveniência.

```text
ImageObservation + ImagePayload
    -> image-area masks
    -> scene concept discovery
    -> concept grounding
    -> generic region discovery
    -> proposal filtering
    -> cross-scale merge
    -> geometry freeze
    -> multi-context evidence
    -> scene context
    -> optional temporal prior
    -> region semantics
    -> language embeddings
    -> hypothesis support
    -> calibration
    -> selective refinement
    -> geometric relations
    -> reconciliation
    -> semantic relations
    -> contextual publication
    -> final audit
    -> VisualObservation
```

A saída ainda é 2D. Projeção para LiDAR e geometria persistente pertence a `sensor-association`.

## Diagramas canônicos do módulo

Use estes documentos em conjunto:

- [Pipeline canônica completa](./pipeline-flow.md), ordem de execução, bifurcações, `geometry freeze`, entradas, saídas e falhas;
- [Region discovery flow](./region-discovery-flow.md), relação entre discovery genérico, tiling, scene concept discovery e SAM3 PCS;
- [Model flow](./model-flow.md), relação entre SAM3, DINOv2, Qwen/Gemini e CLIP;
- [Semantic flow](./semantic-flow.md), SceneContext, claims, suporte, calibração, refinamento, reconciliação e relações;
- [Pipeline detalhada em prosa](./pipeline.md), explicação dos dados e modelos estágio por estágio;
- [Fluxo completo do repositório](../../../docs/system-flow.md), relação deste módulo com toda a cadeia 3D.

## Como os modelos se relacionam

A configuração de referência atual usa quatro papéis diferentes:

```text
SAM
    encontra regiões e masks

DINOv2
    transforma a imagem em uma grade de patches com embeddings visuais

Qwen2.5-VL / Gemini
    interpreta cena e regiões e produz hipóteses textuais

CLIP
    coloca imagens e textos no mesmo espaço para medir suporte às hipóteses
```

O fluxo mais importante é:

```text
SAM mask
   |
   +--> views em pixels --> Qwen/Gemini --> primary + alternatives
   |
   +--> views em pixels --> CLIP image embeddings
                                  ^
                                  |
VLM labels --> CLIP text embeddings
                                  |
                                  v
                         hypothesis support
```

Em paralelo:

```text
RGB --> DINOv2 --> FeatureMap [Hf, Wf, C]
SAM mask ---------> mask-aware pooling
                         |
                         v
                 dense visual evidence
```

O vetor DINO não é enviado diretamente ao VLM no pipeline atual. DINO e CLIP são canais independentes de evidência, e CLIP é o canal usado para medir compatibilidade imagem-texto das hipóteses propostas pelo reasoner.

A explicação completa, com shapes, exemplos de vetores e contratos, está em [Pipeline detalhada de Visual Perception](./pipeline.md).

## Organização do código

```text
src/visual_perception/
├── domain/          # contracts e invariantes do domínio visual
├── ports/           # interfaces dos backends substituíveis
├── application/     # composição da pipeline e políticas
├── infrastructure/
│   ├── adapters/    # backends reais e factories
│   ├── fakes/       # implementações determinísticas para testes
│   ├── integration/ # fronteiras com runtime, persistência e outros módulos
│   └── serialization.py
└── config.py        # configuração versionável da pipeline
```

### Domain

`domain/` contém os tipos que dão forma à evidência, incluindo regiões, geometria 2D, feature maps, embeddings, claims, hipóteses, relações, observações e sinais de suporte. O código de aplicação deve compor esses contracts em vez de trocar dicionários sem schema entre estágios.

### Application

`application/` contém o comportamento da pipeline. Os principais grupos são:

- descoberta de conceitos de cena e grounding dirigido por conceito;
- discovery genérico, tiling, filtragem e merge de regiões;
- views multi-contexto, features densas e embeddings visuais;
- contexto e semântica de cena/região;
- prior temporal opcional;
- suporte independente de hipóteses e calibração;
- refinement e reconciliation;
- relações geométricas e semânticas;
- publicação contextual;
- auditoria, diagnósticos, cache e lifecycle.

[`pipeline.py`](../src/visual_perception/application/pipeline.py) é o ponto central para entender a composição do caminho canônico.

### Infrastructure

`infrastructure/adapters/` implementa backends reais atrás dos ports. `infrastructure/fakes/` fornece backends determinísticos sem GPU para testes de contract. `infrastructure/integration/` mantém dependências de runtime e persistência fora do domínio.

## Documentos especializados

- [Pipeline canônica completa](./pipeline-flow.md)
- [Pipeline detalhada](./pipeline.md)
- [Region discovery flow](./region-discovery-flow.md)
- [Model flow](./model-flow.md)
- [Semantic flow](./semantic-flow.md)
- [Backends de modelos](./model-backends.md)
- [Semantic grounding](./semantic-grounding.md)
- [Política de semântica contextual](./contextual-semantics.md)
- [Pipeline completa do repositório](../../../docs/system-flow.md)
- [Índice end-to-end do repositório](../../../docs/README.md)

A documentação histórica mais extensa continua preservada em [`.old-docs/modules/visual-perception/docs/`](../../../.old-docs/modules/visual-perception/docs/) e não deve ser tratada automaticamente como descrição do código atual.

## Mapeamento para a documentação principal

As etapas 1 a 14 da documentação raiz pertencem majoritariamente a este módulo:

1. [`01-input-rgb.md`](../../../docs/01-input-rgb.md)
2. [`02-region-discovery.md`](../../../docs/02-region-discovery.md)
3. [`03-region-merge.md`](../../../docs/03-region-merge.md)
4. [`04-dense-features.md`](../../../docs/04-dense-features.md)
5. [`05-mask-aware-pooling.md`](../../../docs/05-mask-aware-pooling.md)
6. [`06-region-evidence.md`](../../../docs/06-region-evidence.md)
7. [`07-language-aligned-evidence.md`](../../../docs/07-language-aligned-evidence.md)
8. [`08-scene-context.md`](../../../docs/08-scene-context.md)
9. [`09-region-semantics.md`](../../../docs/09-region-semantics.md)
10. [`10-hypothesis-support.md`](../../../docs/10-hypothesis-support.md)
11. [`11-selective-refinement.md`](../../../docs/11-selective-refinement.md)
12. [`12-reconciliation.md`](../../../docs/12-reconciliation.md)
13. [`13-relations.md`](../../../docs/13-relations.md)
14. [`14-visual-observation.md`](../../../docs/14-visual-observation.md)

## Benchmarks e artifacts

`benchmarks/` não é código de produção da pipeline. Ele contém harnesses, candidatos de pesquisa, probes, ferramentas de inspeção e artifacts usados para comparar mudanças.

Resultados de execução são regenerados sob demanda e não são contracts permanentes. Uma nova reference run real deve ser linkada em [`docs/README.md`](../../../docs/README.md#reference-run) quando houver artifacts que devam servir de exemplo documentado.

## Regra de manutenção

Documentação local deve acompanhar comportamento implementado e contracts públicos. Resultados medidos devem apontar para artifacts versionados. Arquitetura pretendida, implementação atual e comportamento observado devem permanecer explicitamente separados.

Mudanças que alterem uma relação entre módulos devem atualizar também [`docs/system-flow.md`](../../../docs/system-flow.md).

## Próxima leitura

- [Pipeline canônica completa](./pipeline-flow.md)
- [Fluxo completo do repositório](../../../docs/system-flow.md)
- [Pipeline detalhada de Visual Perception](./pipeline.md)
- [02. Region Discovery](../../../docs/02-region-discovery.md)
- [Backends de modelos](./model-backends.md)
