# Model Backends

## Estado e seleção

O módulo roda por default com `backend="fake"`. Os fakes são determinísticos, sensíveis
ao conteúdo e livres de GPU; eles exercitam contracts, pipeline, cache e fronteiras sem
baixar modelos. Adapters reais também estão implementados e foram validados na GPU de
referência (RTX 3060 8GB) — os checkpoints abaixo foram escolhidos por benchmark real
(#174), não por conveniência de integração. `research_quality_config(real_backends=True)`
(em [`application/execution_profile.py`](../src/visual_perception/application/execution_profile.py))
retorna a `ModuleConfig` de referência com os 4 backends reais já configurados.

| Port | Fake default | Adapter real selecionado (#174) | Identificador |
| --- | --- | --- | --- |
| `RegionDiscoverer` | `FakeRegionDiscoverer` | SAM ViT-H via Transformers (`facebook/sam-vit-huge`) | `sam` |
| `DenseFeatureExtractor` | `FakeDenseFeatureExtractor` | DINOv2-base via Transformers (`facebook/dinov2-base`) | `dinov2` |
| `LanguageAlignedEncoder` | `FakeLanguageAlignedEncoder` | CLIP ViT-L/14 via Transformers (`openai/clip-vit-large-patch14`) | `clip` |
| `MultimodalReasoner` | `FakeMultimodalReasoner` | Qwen2.5-VL-3B-Instruct em 4-bit (`Qwen/Qwen2.5-VL-3B-Instruct`) | `qwen_vl` |

Os detalhes de runtime ficam isolados em
[`infrastructure/adapters/`](../src/visual_perception/infrastructure/adapters/); o
pipeline continua conhecendo apenas seus ports. Consulte
[api-contracts.md](api-contracts.md#ports-de-extensão) se precisar fornecer outra
implementação.

## Por que estes checkpoints

Metodologia completa e resultados brutos em `../benchmarks/results/benchmark-174-*.json`
(candidatos, quality score, peak VRAM, latência, `benchmarks/candidates/*.py` para os
proxies de qualidade usados por estágio — todos medidos com o conjunto representativo de
18 frames do corridor-02, ver `../benchmarks/prepare_corridor02_frames.py`).

- **Region discovery**: SAM ViT-H bateu SAM2.1-hiera-large e FastSAM-x no IoU previsto
  médio (0.956 vs 0.937 vs 0.585); todos cabem no budget de 8GB, então venceu por
  qualidade.
- **Feature extraction**: DINOv2-base bateu DINOv2-large (0.966 vs 0.958) na coerência
  espacial do primeiro componente PCA — o modelo maior não melhorou a estruturação das
  features para este dataset, e ainda usa mais VRAM.
- **Language embedding**: CLIP ViT-L/14 bateu SigLIP-base (0.024 vs 0.009) na margem
  top1/top2 de similaridade de cosseno zero-shot contra um vocabulário indoor genérico.
- **Multimodal reasoning**: Qwen2.5-VL-3B-Instruct em 4-bit (bitsandbytes nf4) — o
  candidato 7B (tanto quantização on-the-fly quanto o checkpoint pré-quantizado da
  Unsloth) falhou em 3 tentativas diferentes nesta GPU (OOM no carregamento e um bug de
  compatibilidade `bitsandbytes`/Transformers no vision tower fundido), documentado no
  próprio JSON de resultado. O 3B-4bit rodou com score de qualidade perfeito (1.0) e
  grande folga de VRAM (2.5GB de 8GB), então venceu por eliminação com evidência real, não
  por ausência de alternativa testada.

## Orçamento de VRAM: liberação sequencial obrigatória

Cada um dos 4 backends reais cabe individualmente no budget de 8GB, mas a **soma** dos
picos (SAM 4.6GB + DINOv2 0.3GB + CLIP 1.6GB + Qwen 2.5GB ≈ 9GB) o estoura. Por isso os 4
adapters reais recebem (ou criam, se omitido) um
[`ModelLifecycleManager`](../src/visual_perception/application/lifecycle.py) compartilhado
via [`create_perception_ports`](../src/visual_perception/infrastructure/adapters/factory.py):
no máximo um modelo pesado fica residente em VRAM por vez, mesmo com os 4 ports já
construídos para uma única chamada de `run_canonical_pipeline`. Isso foi confirmado na
prática: sem o manager compartilhado, o pipeline real estourava VRAM ao tentar carregar o
VLM depois de SAM+DINOv2+CLIP ficarem residentes.

`ModelLifecycleManager.metrics` também funciona como log de auditoria por estágio (nome do
backend/checkpoint, tempo de load, pico de VRAM real via `torch.cuda.max_memory_allocated`)
— usado pelas amostras de validação em `../benchmarks/results/samples/`.

## Instalação e configuração

O ambiente fake-only requer apenas a instalação base. Para executar todos os adapters
reais, instale o extra `ml` no diretório do módulo:

```bash
pip install -e ".[ml]"
```

Cada configuração real deve declarar o backend, um checkpoint explícito e o device
desejado. `device="auto"` seleciona CUDA somente quando disponível; `device="cuda"`
falha se não houver uma GPU utilizável. Um checkpoint `"none"`, dependência ausente ou
GPU indisponível gera `BackendUnavailableError`. Falhas no carregamento ou na inferência
são encapsuladas em `BackendExecutionError`. Nenhum adapter substitui silenciosamente o
backend por um fake.

Use os campos específicos em [`config.py`](../src/visual_perception/config.py), não
configuração de aplicação para parâmetros do algoritmo. O fingerprint de `ModuleConfig`
deve acompanhar artifacts e resultados que dependam de um backend real.

## Limites atuais

- SAM produz propostas geométricas class-agnostic; merge e semântica continuam sendo
  responsabilidade dos estágios canônicos. Em superfícies repetitivas (ex: teto em
  ladrilhos) o gerador automático over-segmenta — cada ladrilho pode virar sua própria
  proposta, já que o merge atual (`region_merge`) só funde por IoU/sobreposição, não por
  similaridade semântica entre regiões vizinhas não sobrepostas.
- DINOv2 expõe um `FeatureMap` espacial; tensors e tokens internos não cruzam o port.
- CLIP mantém embeddings de imagem e texto no mesmo espaço, com dimensão configurada
  (768 para ViT-L/14).
- O VLM devolve JSON bruto; parsing e validação semântica pertencem a `application/`. Em
  crops de região pequenos ou ambíguos, o Qwen2.5-VL-3B às vezes falha em produzir JSON
  parseável (isolado por região via `RegionInterpretationFailure`, não derruba o resto) ou
  responde com o contexto geral da cena em vez de descrever o conteúdo específico do crop.

## Validação end-to-end (#190)

`../benchmarks/validate_reference_pipeline.py` roda o pipeline canônico real sobre o
conjunto representativo do corridor-02, gerando por frame: a `VisualObservation`
serializada (JSON), um overlay das máscaras/labels sobre a imagem original, e um
`manifest.json` com git revision, configuração completa e o log de estágios do
`ModelLifecycleManager`. Saída em `../benchmarks/results/samples/<run-id>/`.
