# Pipeline canônico

Este documento é o ponto de entrada para entender **o que o módulo `visual-perception`
faz, como o pipeline funciona e em quais arquivos alterar cada comportamento**.

## Visão geral do módulo

`visual-perception` é responsável por transformar uma observação RGB canônica em uma
representação visual 2D estruturada, semântica e auditável.

A entrada não é um arquivo de dataset, uma ROS bag ou uma imagem sem contexto. O módulo
recebe uma `ImageObservation`, que identifica a observação, sensor, timestamp, frame e
proveniência, junto com um `ImagePayload`, que contém os pixels RGB validados.

A saída principal é uma `VisualObservation`. Ela reúne, para a mesma imagem:

- contexto global da cena;
- regiões 2D com `mask`, `bounding box` e confiança geométrica;
- embeddings visuais obtidos a partir de features densas;
- embeddings alinhados à linguagem;
- claims semânticos sobre as regiões, como label, atributos, material e condição;
- relações 2D candidatas entre regiões;
- proveniência dos modelos e configurações que produziram cada evidência.

O pipeline retorna essa observação dentro de `PipelineResult`, junto com falhas isoladas
de interpretação de regiões e o resultado da auditoria de qualidade.

Em termos simples:

```text
imagem RGB canônica
        |
        v
propostas geométricas 2D
        |
        v
regiões visuais consolidadas
        |
        +--> features e embeddings
        |
        +--> contexto global da cena
        |
        +--> interpretação semântica por região
        |
        +--> relações 2D candidatas
        |
        v
VisualObservation
        |
        v
auditoria de qualidade
```

Uma distinção importante é que **uma proposta de região não é automaticamente uma
entidade semântica do mundo**. O backend de region discovery é class-agnostic: ele
propõe geometria visual. Uma superfície pode gerar várias propostas, e um objeto pode
ser dividido em partes. A semântica é adicionada posteriormente pelo pipeline.

Portanto, o fluxo conceitual é:

```text
RegionProposal
    = evidência geométrica candidata

ObservedRegion
    = região 2D consolidada

SemanticClaim
    = interpretação semântica anexada à região
```

O módulo termina no domínio visual 2D. Ele não é responsável por projetar regiões no
LiDAR, associar pixels a pontos 3D, estimar pose, construir o mapa persistente ou fundir
semântica entre observações espaciais. Essas responsabilidades pertencem aos módulos
downstream, especialmente `sensor-association`, `semantic-fusion` e `semantic-map`.

## Ponto de entrada do pipeline

O ponto de entrada de produção é:

[`application/pipeline.py`](../src/visual_perception/application/pipeline.py)

A função principal é:

```python
run_canonical_pipeline(
    image: ImageObservation,
    payload: ImagePayload,
    config: ModuleConfig,
    ports: PerceptionPorts,
) -> PipelineResult
```

Ela é a orquestradora do módulo. Alterações na **ordem dos estágios**, na forma como os
resultados de um estágio alimentam o próximo ou na construção final da
`VisualObservation` devem começar nesse arquivo.

`PerceptionPorts` agrupa os quatro backends substituíveis usados pelo pipeline:

- `RegionDiscoverer`;
- `DenseFeatureExtractor`;
- `LanguageAlignedEncoder`;
- `MultimodalReasoner`.

O pipeline conhece apenas essas interfaces. Escolha de biblioteca, checkpoint, modelo
real ou fake fica fora da orquestração.

## Como o código está dividido

| Local | Responsabilidade |
| --- | --- |
| [`domain/`](../src/visual_perception/domain/) | Tipos e invariantes do módulo: geometria, regiões, embeddings, claims, relações, observações, erros e auditoria. |
| [`application/`](../src/visual_perception/application/) | Fluxo e regras do pipeline: tiling, merge, pooling, interpretação, relações, auditoria, cache, lifecycle e extensões. |
| [`ports/`](../src/visual_perception/ports/) | Interfaces mínimas que os backends de ML precisam implementar. |
| [`infrastructure/adapters/`](../src/visual_perception/infrastructure/adapters/) | Implementações reais dos ports e composição dos modelos selecionados. |
| [`infrastructure/fakes/`](../src/visual_perception/infrastructure/fakes/) | Implementações determinísticas sem GPU usadas em testes e desenvolvimento. |
| [`infrastructure/integration/`](../src/visual_perception/infrastructure/integration/) | Fronteiras entre `visual-perception` e adapters, mapping runtime, persistência e consumidores downstream. |
| [`config.py`](../src/visual_perception/config.py) | Configuração validada de todos os estágios, backends e checkpoints. |
| [`tests/`](../tests/) | Testes unitários e de integração do módulo. |
| [`benchmarks/`](../benchmarks/) | Benchmarks de backends e execução end-to-end da pipeline de referência. |

A regra prática é:

```text
conceito e invariantes       -> domain/
regra ou transformação       -> application/
interface de modelo          -> ports/
modelo/biblioteca concreta   -> infrastructure/adapters/
integração com outro módulo  -> infrastructure/integration/
```

## Quero mudar X: em qual arquivo mexo?

Esta tabela é o índice operacional do módulo. Use-a antes de procurar pelo código inteiro.

| Quero alterar | Arquivo principal | Arquivos relacionados / testes |
| --- | --- | --- |
| Ordem ou composição do pipeline | [`application/pipeline.py`](../src/visual_perception/application/pipeline.py) | [`tests/test_pipeline.py`](../tests/test_pipeline.py) |
| Como a imagem é dividida em tiles | [`application/tiling.py`](../src/visual_perception/application/tiling.py) | [`tests/test_tiling.py`](../tests/test_tiling.py), [`config.py`](../src/visual_perception/config.py) |
| Como regiões candidatas são descobertas | [`ports/region_discovery.py`](../src/visual_perception/ports/region_discovery.py) | [`infrastructure/adapters/region_discovery_backend.py`](../src/visual_perception/infrastructure/adapters/region_discovery_backend.py), [`tests/test_region_discovery_fake.py`](../tests/test_region_discovery_fake.py) |
| Modelo real usado para region discovery | [`infrastructure/adapters/region_discovery_backend.py`](../src/visual_perception/infrastructure/adapters/region_discovery_backend.py) | [`infrastructure/adapters/factory.py`](../src/visual_perception/infrastructure/adapters/factory.py), [`config.py`](../src/visual_perception/config.py), [`tests/test_real_adapters_gpu.py`](../tests/test_real_adapters_gpu.py) |
| Como propostas sobrepostas são consolidadas | [`application/region_merge.py`](../src/visual_perception/application/region_merge.py) | [`domain/regions.py`](../src/visual_perception/domain/regions.py), [`tests/test_region_merge.py`](../tests/test_region_merge.py) |
| Contrato de `mask`, `bounding box` ou coordenadas | [`domain/geometry.py`](../src/visual_perception/domain/geometry.py) | [`domain/regions.py`](../src/visual_perception/domain/regions.py), [`tests/test_geometry.py`](../tests/test_geometry.py) |
| Como features visuais densas são extraídas | [`ports/feature_extraction.py`](../src/visual_perception/ports/feature_extraction.py) | [`infrastructure/adapters/feature_extraction_backend.py`](../src/visual_perception/infrastructure/adapters/feature_extraction_backend.py), [`domain/feature_map.py`](../src/visual_perception/domain/feature_map.py) |
| Como uma mask vira embedding visual | [`application/pooling.py`](../src/visual_perception/application/pooling.py) | [`domain/embeddings.py`](../src/visual_perception/domain/embeddings.py), [`tests/test_pooling.py`](../tests/test_pooling.py) |
| Como uma região recebe embedding alinhado à linguagem | [`application/language_embedding.py`](../src/visual_perception/application/language_embedding.py) | [`ports/language_embedding.py`](../src/visual_perception/ports/language_embedding.py), [`infrastructure/adapters/language_embedding_backend.py`](../src/visual_perception/infrastructure/adapters/language_embedding_backend.py) |
| Como o contexto global da cena é produzido | [`application/scene_context.py`](../src/visual_perception/application/scene_context.py) | [`ports/multimodal_reasoning.py`](../src/visual_perception/ports/multimodal_reasoning.py), [`infrastructure/adapters/multimodal_reasoning_backend.py`](../src/visual_perception/infrastructure/adapters/multimodal_reasoning_backend.py), [`tests/test_scene_context.py`](../tests/test_scene_context.py) |
| Labels, descrições, atributos, material ou condição de uma região | [`application/region_semantics.py`](../src/visual_perception/application/region_semantics.py) | [`domain/semantics.py`](../src/visual_perception/domain/semantics.py), [`ports/multimodal_reasoning.py`](../src/visual_perception/ports/multimodal_reasoning.py), [`infrastructure/adapters/multimodal_reasoning_backend.py`](../src/visual_perception/infrastructure/adapters/multimodal_reasoning_backend.py), [`tests/test_region_semantics.py`](../tests/test_region_semantics.py) |
| Formato da resposta do VLM ou prompts de cena/região | [`infrastructure/adapters/multimodal_reasoning_backend.py`](../src/visual_perception/infrastructure/adapters/multimodal_reasoning_backend.py) | [`ports/multimodal_reasoning.py`](../src/visual_perception/ports/multimodal_reasoning.py), [`application/scene_context.py`](../src/visual_perception/application/scene_context.py), [`application/region_semantics.py`](../src/visual_perception/application/region_semantics.py) |
| Relações 2D entre regiões | [`application/relation_generation.py`](../src/visual_perception/application/relation_generation.py) | [`domain/relations.py`](../src/visual_perception/domain/relations.py), [`tests/test_relation_generation.py`](../tests/test_relation_generation.py) |
| Estrutura final de `VisualObservation` | [`domain/visual_observation.py`](../src/visual_perception/domain/visual_observation.py) | [`tests/test_visual_observation.py`](../tests/test_visual_observation.py), [`application/pipeline.py`](../src/visual_perception/application/pipeline.py) |
| Regras de qualidade e inconsistências | [`application/quality_audit.py`](../src/visual_perception/application/quality_audit.py) | [`domain/audit.py`](../src/visual_perception/domain/audit.py), [`tests/test_quality_audit.py`](../tests/test_quality_audit.py) |
| Escolher entre backend fake e real | [`infrastructure/adapters/factory.py`](../src/visual_perception/infrastructure/adapters/factory.py) | [`config.py`](../src/visual_perception/config.py), [`tests/test_real_adapters.py`](../tests/test_real_adapters.py) |
| Checkpoints, thresholds e parâmetros dos estágios | [`config.py`](../src/visual_perception/config.py) | [`tests/test_config.py`](../tests/test_config.py) |
| Carregamento, descarregamento e uso de VRAM dos modelos | [`application/lifecycle.py`](../src/visual_perception/application/lifecycle.py) | [`infrastructure/adapters/_runtime.py`](../src/visual_perception/infrastructure/adapters/_runtime.py), [`tests/test_lifecycle.py`](../tests/test_lifecycle.py) |
| Cache e fingerprint de estágios caros | [`application/cache.py`](../src/visual_perception/application/cache.py) | [`application/support.py`](../src/visual_perception/application/support.py), [`tests/test_cache.py`](../tests/test_cache.py) |
| Converter entrada RGB dos adapters para o módulo | [`infrastructure/integration/rgb_adapter_boundary.py`](../src/visual_perception/infrastructure/integration/rgb_adapter_boundary.py) | [`tests/test_integration_boundaries.py`](../tests/test_integration_boundaries.py) |
| Integrar a execução ao mapping runtime | [`infrastructure/integration/mapping_runtime_integration.py`](../src/visual_perception/infrastructure/integration/mapping_runtime_integration.py) | [`tests/test_integration_boundaries.py`](../tests/test_integration_boundaries.py) |
| Contrato de saída para `sensor-association` | [`infrastructure/integration/sensor_association_contract.py`](../src/visual_perception/infrastructure/integration/sensor_association_contract.py) | [`docs/integration.md`](integration.md) |
| Persistir ou serializar a observação | [`infrastructure/integration/persistence_integration.py`](../src/visual_perception/infrastructure/integration/persistence_integration.py) | [`infrastructure/serialization.py`](../src/visual_perception/infrastructure/serialization.py), [`tests/test_serialization.py`](../tests/test_serialization.py) |
| Rodar a pipeline real sobre frames de referência | [`benchmarks/validate_reference_pipeline.py`](../benchmarks/validate_reference_pipeline.py) | [`benchmarks/render_overlay.py`](../benchmarks/render_overlay.py) |
| Comparar candidatos de backend | [`benchmarks/run_backend_benchmark.py`](../benchmarks/run_backend_benchmark.py) | [`benchmarks/backend_benchmark.py`](../benchmarks/backend_benchmark.py), [`benchmarks/candidates/`](../benchmarks/candidates/) |

## Backends usados pelo pipeline

A seleção dos backends acontece em
[`infrastructure/adapters/factory.py`](../src/visual_perception/infrastructure/adapters/factory.py),
não em `pipeline.py`.

A composição atual aceita:

| Capability | Port | Backend real configurável |
| --- | --- | --- |
| Region discovery | `RegionDiscoverer` | `sam` |
| Dense feature extraction | `DenseFeatureExtractor` | `dinov2` |
| Language-aligned embedding | `LanguageAlignedEncoder` | `clip` |
| Multimodal reasoning | `MultimodalReasoner` | `qwen_vl` |

Cada capability também possui um backend `fake` para testes determinísticos sem GPU.

Os adapters reais compartilham um `ModelLifecycleManager`. Isso permite que os ports
estejam todos compostos ao mesmo tempo sem exigir que todos os modelos pesados fiquem
residentes simultaneamente na VRAM.

## Fluxo canônico

`run_canonical_pipeline` recebe `ImageObservation`, `ImagePayload`, `ModuleConfig` e
`PerceptionPorts`, e devolve um `PipelineResult`. A assinatura e os tipos públicos estão
em [api-contracts.md](api-contracts.md).

```mermaid
flowchart TD
    Input[ImageObservation + ImagePayload] --> Tiling[Tiling]
    Tiling --> Discovery[Region discovery]
    Discovery --> Merge[Cross-scale merge]
    Merge --> Features[Dense features]
    Features --> Pooling[Mask-aware pooling]
    Merge --> Language[Language embedding]
    Merge --> Scene[Scene context]
    Scene --> Semantics[Region semantics]
    Merge --> Semantics
    Semantics --> Relations[Candidate relations]
    Pooling --> Output[VisualObservation]
    Language --> Output
    Relations --> Output
    Scene --> Output
    Output --> Audit[AuditResult]
```

A execução concreta segue esta ordem:

1. `build_tiles` divide a imagem conforme a configuração de tiling.
2. `RegionDiscoverer.discover` produz `RegionProposal` para cada tile.
3. `remap_to_global` converte a geometria local de cada tile para as coordenadas da imagem original.
4. `merge_regions` consolida propostas sobrepostas em `ObservedRegion`.
5. Se existirem regiões, o `DenseFeatureExtractor` produz o feature map da imagem.
6. `pool_regions` usa as masks para obter um embedding visual por região.
7. `encode_regions` produz embeddings alinhados à linguagem e anexa suas referências às regiões.
8. `analyze_scene` interpreta o contexto global da imagem.
9. `interpret_regions` interpreta cada região individualmente usando a imagem, o crop pelo bounding box e o contexto da cena.
10. `generate_relations` produz relações 2D candidatas entre as regiões resultantes.
11. O pipeline monta a `VisualObservation` canônica.
12. `audit_observation` verifica consistência e contradições sem modificar a observação.

## Estágios

| Estágio | Entrada principal | Saída | Código dono |
| --- | --- | --- | --- |
| Tiling | `ImagePayload` | tiles com origem conhecida | [`application/tiling.py`](../src/visual_perception/application/tiling.py) |
| Region discovery | tile RGB | `RegionProposal` local | [`ports/region_discovery.py`](../src/visual_perception/ports/region_discovery.py), [`infrastructure/adapters/region_discovery_backend.py`](../src/visual_perception/infrastructure/adapters/region_discovery_backend.py) |
| Remapeamento | proposta local + tile | `RegionProposal` em coordenadas globais | [`application/tiling.py`](../src/visual_perception/application/tiling.py) |
| Cross-scale merge | propostas globais | `ObservedRegion` | [`application/region_merge.py`](../src/visual_perception/application/region_merge.py) |
| Dense features | imagem RGB | `FeatureMap` | [`ports/feature_extraction.py`](../src/visual_perception/ports/feature_extraction.py), [`infrastructure/adapters/feature_extraction_backend.py`](../src/visual_perception/infrastructure/adapters/feature_extraction_backend.py) |
| Mask-aware pooling | regiões + feature map | embedding visual por região | [`application/pooling.py`](../src/visual_perception/application/pooling.py) |
| Language embedding | regiões + imagem | embedding alinhado à linguagem por região | [`application/language_embedding.py`](../src/visual_perception/application/language_embedding.py) |
| Scene context | imagem completa | `SceneContext` | [`application/scene_context.py`](../src/visual_perception/application/scene_context.py) |
| Region semantics | região + imagem + scene context | `SemanticClaim` anexado à região | [`application/region_semantics.py`](../src/visual_perception/application/region_semantics.py) |
| Relations | regiões interpretadas | relações 2D candidatas | [`application/relation_generation.py`](../src/visual_perception/application/relation_generation.py) |
| Observação final | todos os resultados anteriores | `VisualObservation` | [`application/pipeline.py`](../src/visual_perception/application/pipeline.py), [`domain/visual_observation.py`](../src/visual_perception/domain/visual_observation.py) |
| Auditoria | `VisualObservation` | `AuditResult` | [`application/quality_audit.py`](../src/visual_perception/application/quality_audit.py) |

## Comportamento em falhas

Sem regiões, o pipeline ainda analisa o contexto global da cena, produz uma
`VisualObservation` válida e executa a auditoria.

A interpretação semântica é isolada por região. Se o reasoner retornar uma resposta
inválida para uma região, a execução não descarta as demais regiões nem perde a
geometria já produzida. A falha aparece em
`PipelineResult.region_interpretation_failures`.

Isso significa que:

```text
falha de semântica em uma região
        !=
falha da observação inteira
```

A geometria, embeddings e evidências válidas produzidas pelos outros estágios continuam
disponíveis para inspeção e auditoria.

## Extensões pós-pipeline

As capacidades abaixo compõem sobre a saída canônica e não mudam a ordem nem a API de
`run_canonical_pipeline`:

- [`application/fusion.py`](../src/visual_perception/application/fusion.py) funde propostas e claims de múltiplas fontes, preservando proveniência e contradições;
- [`application/refinement.py`](../src/visual_perception/application/refinement.py) reinterpreta seletivamente regiões incertas;
- [`application/execution_profile.py`](../src/visual_perception/application/execution_profile.py) seleciona um candidato de pesquisa sujeito ao orçamento de memória.

Elas são APIs de capability, não etapas obrigatórias. Um consumidor que apenas precisa
de uma observação visual deve chamar o pipeline canônico e decidir explicitamente se
alguma extensão é necessária.

## Pipelines legadas

Os pipelines do laboratório histórico `image-context` não fazem parte deste módulo e
não são selecionáveis em produção. Seu status e as condições para uma comparação futura
estão em [`../benchmarks/legacy/README.md`](../benchmarks/legacy/README.md).

A ausência deliberada de um seletor de estratégia evita que aplicações escolham uma
baseline que não é reproduzível neste repositório. O código de produção deve convergir
para `run_canonical_pipeline` como caminho único de percepção visual.