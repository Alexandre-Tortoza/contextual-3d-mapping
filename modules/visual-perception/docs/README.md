# Image Context Documentation

Esta pasta documenta a arquitetura, os fluxos de execução e os artefatos do projeto `image-context`.

As pipelines de produção `baseline` e `region-first` foram removidas (issue
[#4](https://github.com/Alexandre-Tortoza/image-context/issues/4)). O projeto está em
transição para um único módulo canônico, desenhado pelo epic
[#3](https://github.com/Alexandre-Tortoza/image-context/issues/3) e suas issues filhas
(#5 a #14) — veja o diagrama de arquitetura-alvo e o contrato `ImageContext` diretamente
nessas issues.

## Conteúdo

- [architecture.md](architecture.md), arquitetura mínima hoje (amostragem) e link para a
  arquitetura-alvo.
- [pipelines.md](pipelines.md), nota sobre a remoção das pipelines legadas e onde a
  arquitetura nova está sendo desenhada.
- [execution.md](execution.md), comando `sample` disponível hoje.
- [artifacts.md](artifacts.md), estrutura de saída atual.
