# Baselines legadas

Issue: #176.

Os dois pipelines de produção históricos do laboratório standalone `image-context` —

- **baseline**: VLM de três passes -> Grounding DINO -> SAM2;
- **region-first**: SAM2 automático -> DINOv2 -> Qwen global/local;

foram removidos do codebase antes da migração para este módulo (veja
`docs/architecture.md`, que documenta sua remoção e linka de volta para o histórico de
issues original do `image-context`). Eles não estão presentes aqui como código
executável.

## Por que não são reproduzidos como código em execução

Reintroduzi-los exigiria a stack de modelos pesados original (Grounding DINO, SAM2,
DINOv2, Qwen) e seus prompts/configs originais, nenhum dos quais está disponível neste
ambiente. Recriá-los do zero sem os artifacts originais não seria uma baseline
científica fiel, então este módulo não fabrica um substituto.

## O que é preservado em vez disso

- Ambos os pipelines permanecem totalmente recuperáveis a partir do histórico git no
  repositório `image-context` original (veja o commit referenciado por
  `docs/architecture.md`).
- O pipeline canônico (`application/pipeline.py`, #169) não expõe nenhum seletor de
  estratégia legada: existe exatamente um ponto de entrada de produção, satisfazendo o
  requisito da #176 de que o uso canônico nunca exija escolher uma variante legada.
- Se uma futura comparação científica precisar reexecutar uma baseline legada, seu
  harness viveria sob este diretório, isolado de `application/pipeline.py` e usando seu
  próprio namespace de run-artifact, para que não colida com run artifacts canônicos
  (os tipos `DatasetReference`/`BenchmarkReport` de `benchmarks/harness.py` são
  reutilizáveis para essa comparação assim que o código legado for reintroduzido).
