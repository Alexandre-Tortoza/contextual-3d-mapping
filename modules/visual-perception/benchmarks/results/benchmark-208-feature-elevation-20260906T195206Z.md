# Benchmark de elevação de features — 20260906T195206Z

Frames: `corridor-02-000, corridor-02-008, corridor-02-017`
Hardware: NVIDIA GeForce RTX 3060 (8.2 GB)

| caminho | geração | grade efetiva | representáveis | extração (s) | pico VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| patch_grid | backbone_native | 16x16 | 138/168 | 1.203 | 0.66 GiB |
| nearest | backbone_native | 16x16 | 168/168 | 0.000 | 0.00 GiB |
| bilinear | backbone_native | 16x16 | 164/168 | 0.000 | 0.00 GiB |
| dinov2_native_448 | backbone_native | 32x24 | 168/168 | 1.032 | 0.69 GiB |
| featup_dinov2_small_jbu | learned_upsampler | 512x384 | 168/168 | 3.873 | 2.90 GiB |

## Qualidade da evidência

| caminho | suporte médio | consistência intra | separação inter |
| --- | ---: | ---: | ---: |
| patch_grid | 1.0000 | 0.8191 | 0.4317 |
| nearest | 1.0000 | 0.8470 | 0.3681 |
| bilinear | 0.8298 | 0.9230 | 0.3441 |
| dinov2_native_448 | 1.0000 | 0.8408 | 0.4295 |
| featup_dinov2_small_jbu | 1.0000 | 0.9590 | 0.3581 |

## Custo de pooling e materialização

| caminho | pooling (s) | mapa materializado | materialização (s) |
| --- | ---: | ---: | ---: |
| patch_grid | 0.181 | n/a | n/a |
| nearest | 3.429 | 900.0 MiB | 0.3334 |
| bilinear | 14.311 | 900.0 MiB | 1.1208 |
| dinov2_native_448 | 3.404 | 900.0 MiB | 0.3458 |
| featup_dinov2_small_jbu | 1.857 | 450.0 MiB | 0.2045 |

A concordância vetorial só é calculada dentro do espaço DINOv2-base. 
FeatUp usa DINOv2-small (384 dimensões), portanto sua qualidade é comparada por 
representabilidade, cobertura, consistência, separação e custo — nunca por produto 
escalar direto contra o backbone de 768 dimensões.
