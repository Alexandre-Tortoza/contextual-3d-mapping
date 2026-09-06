# Benchmark de evidência densa em alta resolução (#192)

- revisão de código: `ee71fa9`
- hardware: NVIDIA GeForce RTX 3060 (8.2 GB)
- backbone: `dinov2` / checkpoint `facebook/dinov2-base`
- fingerprint de configuração: `6ccc623f1f22b02d8332cf87dd41c3a8cf8e5b31fd8f283ae398efbbe11d9f05`
- frames: 6 — regiões descobertas: 115
- resolução de entrada: 640x480
- grade de features: 16x16
- regiões representáveis por todos os caminhos: 94

## Representabilidade (conjunto completo)

Quantas regiões cada caminho consegue representar. A grade de patches
rejeita qualquer região sem centro de célula dentro da máscara.

| caminho | representáveis | pequenas | médias | grandes |
| --- | --- | --- | --- | --- |
| `patch_grid` | 94/115 | 17/37 | 46/47 | 31/31 |
| `nearest` | 115/115 | 37/37 | 47/47 | 31/31 |
| `bilinear` | 112/115 | 34/37 | 47/47 | 31/31 |

## Qualidade de evidência (conjunto comum)

| caminho | consistência intra | separação inter | suporte médio | concordância baseline |
| --- | --- | --- | --- | --- |
| `patch_grid` | 0.7760 | 0.6405 | 1.0000 | não medida |
| `nearest` | 0.7976 | 0.6043 | 1.0000 | 0.9733 |
| `bilinear` | 0.8831 | 0.5879 | 0.8188 | 0.9595 |

## Por estrato de tamanho projetado (conjunto comum)

| caminho | estrato | regiões | consistência intra | suporte médio | concordância baseline |
| --- | --- | --- | --- | --- | --- |
| `patch_grid` | large | 31 | 0.6923 | 1.0000 | não medida |
| `patch_grid` | medium | 46 | 0.7937 | 1.0000 | não medida |
| `patch_grid` | small | 17 | 0.8805 | 1.0000 | não medida |
| `nearest` | large | 31 | 0.6947 | 1.0000 | 0.9973 |
| `nearest` | medium | 46 | 0.8258 | 1.0000 | 0.9598 |
| `nearest` | small | 17 | 0.9091 | 1.0000 | 0.9659 |
| `bilinear` | large | 31 | 0.7864 | 0.8677 | 0.9938 |
| `bilinear` | medium | 46 | 0.9192 | 0.8025 | 0.9439 |
| `bilinear` | small | 17 | 0.9665 | 0.7736 | 0.9390 |

## Custo

| caminho | pooling (s) | pico de VRAM (GB) |
| --- | --- | --- |
| `patch_grid` | 0.120 | 0.35 |
| `nearest` | 6.119 | 0.35 |
| `bilinear` | 24.994 | 0.35 |

### Materialização do mapa pixel-aligned completo

- `nearest`: 0.94 GB em 0.287 s
- `bilinear`: 0.94 GB em 1.004 s
