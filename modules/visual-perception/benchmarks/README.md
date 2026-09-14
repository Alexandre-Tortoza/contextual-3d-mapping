# Visual Perception Benchmarks

Este diretório contém avaliação, probes, candidatos de pesquisa e artifacts de `visual-perception`. Ele não faz parte da API de produção do módulo.

## Organização

```text
benchmarks/
├── candidates/        # implementações candidatas comparadas antes de promoção
├── results/           # relatórios e artifacts versionados
├── harness.py         # infraestrutura compartilhada de execução
├── compare_runs.py    # comparação entre runs
├── frame_artifacts.py # leitura/escrita de artifacts por frame
├── render_*.py        # inspeção visual
├── *_benchmark.py     # benchmarks de uma capacidade específica
└── *_probe.py         # experimentos diagnósticos focados
```

Scripts como `derive_sequence_masks.py` e `prepare_corridor02_frames.py` preparam fixtures ou metadados específicos de sequências. Eles devem registrar a proveniência necessária para que o artifact gerado continue auditável.

## Candidates

`candidates/` existe para testar alternativas sem tornar a alternativa parte do caminho canônico. Há candidatos para capacidades como feature extraction, language embedding, multimodal reasoning e region discovery.

Um candidato só deve ser promovido para `src/visual_perception/infrastructure/adapters/` quando houver uma decisão explícita baseada em contracts e evidência de benchmark.

## Results

`results/` guarda relatórios e, quando necessário, amostras de execução. Um resultado deve ser interpretado junto com:

- revisão do Git usada;
- configuração da pipeline;
- modelos/checkpoints;
- dataset ou sequência;
- frames avaliados;
- perfil de qualidade;
- métricas e limitações do benchmark.

A reference run usada pela documentação principal é `20260910T115810Z`. Ela representa comportamento observado de uma revisão específica, não ground truth.

## Regra de uso

Use benchmarks para responder perguntas mensuráveis, por exemplo:

- um backend melhora a qualidade de região?
- elevar a resolução das dense features preserva melhor bordas e objetos pequenos?
- uma mudança reduz falsos positivos sem destruir cobertura?
- uma política de semântica contextual reduz labels estruturais triviais?
- o custo de memória e tempo cabe no orçamento de execução?

Não use o diretório para esconder lógica necessária ao runtime. Comportamento canônico deve viver em `src/`; o benchmark apenas o exercita ou compara com candidatos.

## Documentação relacionada

- [`../docs/README.md`](../docs/README.md), arquitetura local do módulo;
- [`../docs/model-backends.md`](../docs/model-backends.md), backends canônicos e pontos de substituição;
- [`../docs/contextual-semantics.md`](../docs/contextual-semantics.md), política semântica a ser avaliada;
- [`../../../docs/README.md`](../../../docs/README.md), reference run e pipeline global.