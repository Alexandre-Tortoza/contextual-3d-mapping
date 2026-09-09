# Evidência densa em alta resolução

Issues: #191 (contract), #192 (amostragem) e #208 (elevação real).

Este documento descreve o contract de evidência densa pixel-aligned e o que
o benchmark mediu sobre ele na GPU de referência.

## O contract (#191)

`domain/feature_map.py` define `FeatureMap` como uma grade espacial de
features **mais** a metadata necessária para amarrar cada valor de volta à
sua coordenada na imagem original:

| campo | o que responde |
| --- | --- |
| `representation` | é uma grade de patches, uma grade elevada ou um mapa pixel-aligned? |
| `generation` | valores nativos, reamostrados ou produzidos por upsampler aprendido? |
| `interpolation` | qual regra de amostragem vale para ler este mapa |
| `stride_x`/`stride_y` + `origin_x`/`origin_y` | o `CoordinateTransform` invertível grade → imagem |
| `valid_support` | quais células têm suporte real |
| `model_id`, `checkpoint`, `preprocessing`, `upsampling_method` | proveniência numérica |
| `source_grid_*`, `upsampler_checkpoint`, `fallback_reason` | origem da elevação e fallback observável |

`sample_feature_map(feature_map, xs, ys)` é a única forma pública de ler o
mapa. Ela devolve `(values, valid)`: coordenadas fora da área coberta, sem
vizinhança completa (bilinear) ou sem suporte voltam marcadas como inválidas
com vetor zero. Nada é extrapolado em silêncio.

O array denso nunca é serializado inline. `feature_map_spec_to_dict`
persiste só a metadata; os valores viajam por referência de artifact.

### Materializar ou amostrar

`application/dense_evidence.py::upsample_feature_map` materializa o mapa
pixel-aligned completo, linha a linha, atrás de um teto explícito de
memória. Ele existe porque a #191 exige que as duas resoluções sejam
representações distinguíveis, e porque o benchmark precisa medir o custo
real desse caminho.

Em produção, o pooling **não** materializa: ele amostra sob demanda com
`sample_feature_map`. Uma região cobre uma fração pequena da imagem, e os
dois caminhos produzem os mesmos valores para as mesmas coordenadas — a
diferença é só de memória. Quando a materialização não cabe no budget, o
erro diz isso e aponta para a amostragem.

## O que o benchmark mediu (#192)

Execução real: RTX 3060 (8,2 GB), DINOv2-base, SAM-huge para descoberta de
regiões, 6 frames do `corridor-02` a 640x480, 115 regiões descobertas.
Relatório completo em
`benchmarks/results/benchmark-192-dense-upsampling-20260906T140720Z.md`.

Sem conjunto de referência anotado revisado (#197), **nenhuma métrica de
acerto semântico foi calculada**. As medidas abaixo não dependem de
anotação, e as conclusões estão limitadas ao que elas cobrem.

### 1. A grade de patches não representa 54% das regiões pequenas

| caminho | representáveis | pequenas (≤1024 px²) |
| --- | --- | --- |
| `patch_grid` | 94/115 | **17/37** |
| `nearest` | 115/115 | 37/37 |
| `bilinear` | 112/115 | 34/37 |

O baseline rejeita qualquer região sem centro de célula dentro da máscara.
Isso não é uma perda de qualidade: é uma falha dura, em que a região não
recebe embedding nenhum. É o resultado mais forte do benchmark.

Os 3 casos que o `bilinear` perde são regiões pequenas encostadas na borda,
onde não existe vizinhança completa de quatro células para interpolar.

### 2. A amostragem pixel-aligned representa melhor os próprios pixels

Sobre as 94 regiões que **todos** os caminhos representam (comparar médias
sobre conjuntos diferentes creditaria ao baseline justamente as regiões em
que ele falha):

| caminho | consistência intra-região | suporte médio |
| --- | --- | --- |
| `patch_grid` | 0,7760 | 1,0000 |
| `nearest` | 0,7976 | 1,0000 |
| `bilinear` | **0,8831** | 0,8188 |

O ganho do `bilinear` é maior justamente nas regiões pequenas (0,8805 →
0,9665). Em compensação, 18% dos pixels de máscara ficam sem suporte por
caírem na meia-célula de borda.

### 3. Resolução mais alta *reduz* a separação entre regiões vizinhas

| caminho | separação inter-região |
| --- | --- |
| `patch_grid` | **0,6405** |
| `nearest` | 0,6043 |
| `bilinear` | 0,5879 |

Resultado negativo, e o mais importante para calibrar expectativa: amostrar
cada pixel de uma máscara a partir de uma grade grosseira faz duas regiões
adjacentes puxarem as *mesmas* células. A evidência fica mais fiel aos
pixels de cada região e, ao mesmo tempo, um pouco menos discriminativa
entre vizinhas.

### 4. O gargalo real é o resize do backbone, não a regra de pooling

A proveniência registrada pelo contract da #191 mostra o motivo:

```
"preprocessing": "BitImageProcessor:224x224",
"stride_x": 40.0, "stride_y": 30.0, "grid_width": 16, "grid_height": 16
```

O processor do DINOv2 redimensiona a imagem de 640x480 para 224x224 antes
de qualquer coisa. Com patch 14, sobram 16x16 células, e **uma célula cobre
40x30 pixels da imagem original**. Nenhuma regra de leitura recupera
detalhe que o resize já descartou.

Aumentar a resolução de entrada do backbone tem, portanto, mais potencial do
que qualquer upsampler aplicado depois. Isso não estava visível antes da
#191: sem `preprocessing` e `stride` no contract, o `16` da configuração
parecia ser a resolução efetiva.

### 5. Custo

| caminho | pooling de 115 regiões (s) | pico de VRAM |
| --- | --- | --- |
| `patch_grid` | 0,12 | 0,35 GB |
| `nearest` | 6,12 | 0,35 GB |
| `bilinear` | 24,99 | 0,35 GB |

A regra de leitura não muda a residência de VRAM — o custo é de CPU. Já
materializar o mapa pixel-aligned completo de um frame ocupa 0,94 GB
(640x480x768 em float32), o que só é viável um frame por vez dentro do
budget de 8 GB.

## Candidato justificado para o quality profile

**`nearest`.** Ele elimina a falha dura de representabilidade (94/115 →
115/115, e 17/37 → 37/37 nas regiões pequenas) sem perder suporte de pixel
e a um custo de CPU aceitável. É por isso que ele é o default de
`FeatureExtractionConfig.upsampling`.

`bilinear` fica disponível, e é a escolha certa quando fidelidade de borda
importa mais que cobertura — mas quem o usar precisa contar com a perda de
18% de suporte e com a invalidação da meia-célula de borda.

`patch_grid` permanece como baseline reproduzível da ablation, não como
opção de produção.

## Elevação real (#208)

A #192 mostrou que `nearest` e `bilinear` apenas leem uma grade existente:
eles melhoram representabilidade da máscara, mas não criam detalhe visual. A #208 mantém
esses caminhos como baselines e acrescenta duas produções efetivamente distintas:

- `dinov2` com `input_resolution=448`: o backbone recebe o frame em resize
  aspect-preserving e produz uma grade nativa maior (32x24 em frames 640x480), em vez da
  grade 16x16 do processor 224x224;
- `featup`: adapter isolado que usa o JBU pré-treinado do FeatUp sobre DINOv2-small e
  produz `FeatureGeneration.LEARNED_UPSAMPLER`. O checkpoint tem 384 canais e não é
  comparado diretamente ao espaço DINOv2-base de 768 canais.

O teto `max_feature_map_mb` é aplicado antes da inferência FeatUp: a resolução de entrada
é reduzida por múltiplos do patch quando o mapa float32 previsto não cabe. Um fallback só
existe quando `fallback_backend` e `fallback_checkpoint` são declarados; o mapa retornado
carrega `fallback_reason`, portanto uma indisponibilidade nunca parece uma execução FeatUp
bem-sucedida.

O benchmark reproduzível compara os cinco caminhos nos mesmos três frames e nas mesmas
regiões:

```bash
python benchmarks/feature_elevation_benchmark.py
```

Ele grava `benchmark-208-feature-elevation-<run-id>.json` e `.md`, com resolução efetiva,
proveniência, representabilidade, cobertura, consistência, separação, latência e VRAM.
Resultados só devem orientar o perfil depois de o artifact real ser produzido; a presença
do adapter, isoladamente, não demonstra ganho de qualidade.

## Candidatos ainda não medidos

Duas alternativas modernas foram investigadas nesta rodada e **nenhuma** foi medida, por
razões diferentes:

- **um backbone mais recente da mesma família** trata exatamente a fraqueza que a #208
  identificou — a degradação do mapa denso em resolução alta — e a variante de mesma
  classe de tamanho do DINOv2-base (86 M params) está disponível na versão de
  `transformers` instalada. O checkpoint, porém, está sob licença *gated* no Hub e exige
  aceite manual na conta do usuário. Issue #219;
- **um upsampler aprendido baseado em coordenadas**, alternativo ao FeatUp já integrado.
  Não foi avaliado nesta rodada por prioridade: a auditoria de contexto visual mediu que o
  ganho restante maior estava na composição da evidência, não na resolução dela. Issue
  #221.

Nenhuma das duas muda o default. A regra da #208 continua valendo: a presença de um
adapter não demonstra ganho, e o benchmark é que decide.

### O que estes números **não** sustentam

- que a evidência de alta resolução melhora acerto semântico: isso exige o
  conjunto de referência anotado (#197), ainda em `pending_review`;
- que a separação inter-região menor prejudica a tarefa final: a medida é um
  proxy geométrico, não uma métrica de tarefa;
- qualquer conclusão fora de 640x480, DINOv2-base e este ambiente interno.
