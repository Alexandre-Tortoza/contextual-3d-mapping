# Pipelines

As pipelines de produção `baseline` (VLM-first: três passagens Qwen -> Grounding DINO ->
SAM2) e `region-first` (geometry-first: SAM2 automático -> DINOv2 -> Qwen global/local)
foram removidas do repositório na issue
[#4](https://github.com/Alexandre-Tortoza/image-context/issues/4). Elas existiam como
dois experimentos independentes e comparáveis via `image-context analyze --strategy all`;
esse comando não existe mais.

O código permanece recuperável no histórico do git para comparação científica futura
(issue [#12](https://github.com/Alexandre-Tortoza/image-context/issues/12)), mas deixou
de ser a arquitetura suportada.

A pipeline canônica que substitui as duas está sendo desenhada issue a issue sob o epic
[#3](https://github.com/Alexandre-Tortoza/image-context/issues/3):

- [#5](https://github.com/Alexandre-Tortoza/image-context/issues/5) — schema `ImageContext`
- [#6](https://github.com/Alexandre-Tortoza/image-context/issues/6) — benchmark de modelos sob 8 GB de VRAM
- [#7](https://github.com/Alexandre-Tortoza/image-context/issues/7) — percepção panoptic open-vocabulary
- [#8](https://github.com/Alexandre-Tortoza/image-context/issues/8) — features densas e embeddings
- [#9](https://github.com/Alexandre-Tortoza/image-context/issues/9) — raciocínio semântico/contextual
- [#10](https://github.com/Alexandre-Tortoza/image-context/issues/10) — relações, incerteza e proveniência
- [#11](https://github.com/Alexandre-Tortoza/image-context/issues/11) — orquestração de VRAM/cache
- [#14](https://github.com/Alexandre-Tortoza/image-context/issues/14) — composição da pipeline canônica end-to-end

Esta página será reescrita para descrever o fluxo real quando a issue #14 estiver
concluída.
