# Rastreabilidade de pesquisa

Esta página separa famílias de técnicas que orientam os ports do módulo de escolhas de
engenharia próprias. Ela não afirma que um checkpoint seja recomendado: essa decisão
depende de benchmark reproduzível e permanece pendente.

## Fronteiras inspiradas por técnicas estabelecidas

| Capacidade | Família de técnicas | Fronteira no código |
| --- | --- | --- |
| Region discovery | Segmentação class-agnostic e promptable, como SAM. | [`ports/region_discovery.py`](../src/visual_perception/ports/region_discovery.py) |
| Dense features | Backbones self-supervised com grid espacial, como DINOv2. | [`ports/feature_extraction.py`](../src/visual_perception/ports/feature_extraction.py) |
| Language embedding | Encoders contrastivos imagem-texto, como CLIP. | [`ports/language_embedding.py`](../src/visual_perception/ports/language_embedding.py) |
| Multimodal reasoning | VLMs para respostas estruturadas de cena e região. | [`ports/multimodal_reasoning.py`](../src/visual_perception/ports/multimodal_reasoning.py) |

As fronteiras são independentes de implementação para permitir fakes, adapters concretos
e candidatos futuros sob o mesmo contract. O status de cada adapter disponível está em
[model-backends.md](model-backends.md).

## Decisões de engenharia do projeto

- O cache usa fingerprints encadeados em
  [`application/cache.py`](../src/visual_perception/application/cache.py): mudar um
  estágio invalida seus dependentes, não toda a execução.
- A semântica é um conjunto de claims auditáveis, não um label vencedor; veja
  [`domain/semantics.py`](../src/visual_perception/domain/semantics.py).
- Relações geométricas são derivadas deterministicamente de masks e boxes e preservam
  uma fonte distinta de relações inferidas; veja
  [`application/relation_generation.py`](../src/visual_perception/application/relation_generation.py).
- O pooling mask-aware possui um caminho de alta resolução para representar regiões
  menores que uma célula de feature map; veja
  [`application/pooling.py`](../src/visual_perception/application/pooling.py).
- Fusão multi-fonte reaproveita o merge de regiões e mantém claims discordantes; veja
  [`application/fusion.py`](../src/visual_perception/application/fusion.py).

## Questões em aberto

- Qual checkpoint por estágio maximiza qualidade dentro do orçamento de memória de
  referência ainda precisa ser decidido por benchmark.
- Os thresholds do refinamento seletivo não foram calibrados contra a distribuição de
  incerteza de um backend real; consulte
  [`application/refinement.py`](../src/visual_perception/application/refinement.py).
- As baselines do laboratório histórico `image-context` são recuperáveis no histórico,
  mas não são executáveis neste repositório; veja
  [`../benchmarks/legacy/README.md`](../benchmarks/legacy/README.md).

Os scripts de benchmark locais não são evidência suficiente por si só: um resultado que
oriente configuração de referência precisa registrar dataset, versão, checkpoint,
configuração, hardware e métrica.
