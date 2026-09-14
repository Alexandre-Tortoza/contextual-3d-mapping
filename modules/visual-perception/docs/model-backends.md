# Backends de modelos

`visual-perception` separa capacidades de percepção dos modelos concretos. A aplicação depende dos ports; os adapters em `infrastructure/adapters/` escolhem e carregam os backends reais.

## Capacidades substituíveis

| Capacidade | Responsabilidade | Backend da reference run |
| --- | --- | --- |
| Region discovery | propor masks e regiões sem exigir classe | SAM ViT-H |
| Dense feature extraction | produzir feature map visual denso | DINOv2-base |
| Language embedding | alinhar evidência visual a conceitos textuais | CLIP ViT-L/14 |
| Multimodal reasoning | interpretar cena e regiões em linguagem estruturada | Qwen2.5-VL-3B-Instruct, 4-bit |

Esses modelos descrevem a reference run `20260910T115810Z` documentada em [`docs/README.md`](../../../docs/README.md). Outras configurações podem selecionar backends diferentes sem alterar os contracts de domínio.

## Onde cada backend entra

```text
RGB
├── RegionDiscoverer        -> RegionProposal[]
├── FeatureExtractor       -> FeatureMap
├── LanguageEncoder        -> evidência alinhada à linguagem
└── MultimodalReasoner     -> contexto e claims semânticos
```

Factories e adapters concretos ficam em [`infrastructure/adapters/`](../src/visual_perception/infrastructure/adapters/). Fakes determinísticos ficam em [`infrastructure/fakes/`](../src/visual_perception/infrastructure/fakes/).

## Regra arquitetural

O domínio não deve importar bibliotecas específicas de SAM, DINO, CLIP ou Qwen. Dependências pesadas, detalhes de device, quantização, carregamento de checkpoints e transformações específicas de backend ficam na infraestrutura.

Isso permite:

- testar a pipeline sem GPU;
- trocar modelos em benchmarks sem reescrever o domínio;
- comparar backends por capacidade;
- registrar proveniência do modelo utilizado em cada observação;
- manter o payload entre estágios estável mesmo quando a implementação muda.

## Relação entre DINO, SAM, CLIP e Qwen

Os modelos não executam a mesma função.

- SAM encontra regiões e masks. Ele não é a fonte final da identidade semântica.
- DINO produz representações visuais densas dos patches. Essas features descrevem aparência e estrutura visual, mas não são labels por si só.
- CLIP fornece alinhamento entre evidência visual e linguagem, permitindo medir proximidade com conceitos textuais.
- Qwen é o reasoner multimodal usado para produzir contexto e hipóteses semânticas estruturadas a partir das evidências que a pipeline disponibiliza.

A decisão semântica final de uma região é, portanto, resultado da composição da pipeline, e não um simples `argmax` de um único backend.

## Feature upsampling

Existe uma fronteira específica de feature upsampling em `infrastructure/adapters/feature_upsampling_backend.py`. Ela deve ser tratada como um passo opcional de resolução da evidência densa, não como uma nova fonte de labels. Benchmarks de elevação de resolução permanecem em `benchmarks/` até que uma configuração seja promovida ao caminho canônico.

## Como documentar uma troca de modelo

Ao mudar um backend, registre no mínimo:

- capacidade substituída;
- nome e versão/checkpoint do modelo;
- parâmetros que alteram o contract observável;
- resolução ou shape relevante;
- precisão/quantização e device quando afetarem comportamento;
- impacto medido em benchmark ou artifact versionado;
- configuração usada para reproduzir o resultado.

Não atualize esta página com um modelo apenas planejado. O backend deve existir no código ou em um benchmark explicitamente identificado como candidato.