# Visual Perception

Esta é a documentação local de `visual-perception`. O diretório raiz [`docs/`](../../../docs/README.md) descreve a pipeline completa entre módulos; aqui ficam arquitetura interna, backends, política semântica, pontos de extensão e evidências específicas deste módulo.

## Papel no sistema

`visual-perception` transforma um frame RGB canônico em `VisualObservation`, preservando regiões, evidências densas, evidência alinhada à linguagem, hipóteses semânticas, contexto de cena, relações e proveniência.

```text
ImageObservation + ImagePayload
    -> region discovery
    -> region merge
    -> dense features
    -> mask-aware pooling
    -> region evidence
    -> language-aligned evidence
    -> scene context
    -> region semantics
    -> hypothesis support
    -> selective refinement
    -> reconciliation
    -> relations
    -> VisualObservation
```

A saída ainda é 2D. Projeção para LiDAR e geometria persistente pertence a `sensor-association`.

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

- descoberta, merge, views e tiling de regiões;
- extração e pooling de evidência densa;
- embeddings alinhados à linguagem;
- contexto e semântica de cena/região;
- suporte de hipóteses e calibração;
- refinement e reconciliation;
- relações semânticas e geométricas;
- auditoria, diagnósticos, cache e lifecycle.

[`pipeline.py`](../src/visual_perception/application/pipeline.py) é o ponto central para entender a composição do caminho canônico.

### Infrastructure

`infrastructure/adapters/` implementa backends reais atrás dos ports. `infrastructure/fakes/` fornece backends determinísticos sem GPU para testes de contract. `infrastructure/integration/` mantém dependências de runtime e persistência fora do domínio.

## Documentos especializados

- [Backends de modelos](./model-backends.md)
- [Política de semântica contextual](./contextual-semantics.md)
- [Pipeline end-to-end do repositório](../../../docs/README.md)

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

`benchmarks/` não é código de produção da pipeline. Ele contém harnesses, candidatos de pesquisa, probes, ferramentas de inspeção e artifacts versionados usados para comparar mudanças.

A reference run indicada pela documentação raiz é `20260910T115810Z`. Ela serve como evidência observada de uma revisão específica, não como ground truth.

## Regra de manutenção

Documentação local deve acompanhar comportamento implementado e contracts públicos. Resultados medidos devem apontar para artifacts versionados. Arquitetura pretendida, implementação atual e comportamento observado devem permanecer explicitamente separados.