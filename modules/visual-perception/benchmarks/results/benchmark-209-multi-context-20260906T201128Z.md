# Ablation de evidência multi-contexto — #209

Revisão: `ebe211f`  
Hardware: NVIDIA GeForce RTX 3060 (8,2 GB)  
Frames, na ordem vinculante: `corridor-02-000`, `corridor-02-008`, `corridor-02-017`

Runs-fonte:

- baseline: `samples/20260906T200203Z/manifest.json`;
- full: `samples/20260906T195310Z/manifest.json`.

Os dois runs usaram os mesmos hashes de entrada e a mesma configuração fora de
`multi_context`. A geometria canônica foi comparada por SHA-256 de
`region_id + mask + box` em cada frame e permaneceu idêntica:

| frame | regiões | digest da geometria |
| --- | ---: | --- |
| `corridor-02-000` | 54 | `9e484002d5292a463b57ebcfc243085ebaef3d03abf34a22b4877988a04e02cc` |
| `corridor-02-008` | 22 | `d804bf97bfc929e14b90b243d77ac3160230cfd68cc887ea48326f7503e96578` |
| `corridor-02-017` | 50 | `b4d2a774d20737dbfb6e7cc9d80e782786e2463a6a799877e2bf488c4bba4b42` |

## Cobertura e falhas

| perfil | slot | available | missing | failed | model calls | latência do slot (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | foreground dense | 126 | 0 | 0 | 0 | 4,056 |
| baseline | tight crop | 126 | 0 | 0 | 126 | 17,549 |
| baseline | contextual crop | 0 | 126 | 0 | 0 | 0,000 |
| baseline | scene conditioned | 0 | 126 | 0 | 0 | 0,000 |
| full | foreground dense | 126 | 0 | 0 | 0 | 4,004 |
| full | tight crop | 126 | 0 | 0 | 126 | 6,458 |
| full | contextual crop | 126 | 0 | 0 | 126 | 6,415 |
| full | scene conditioned | 126 | 0 | 0 | 3 | 12,248 |

No perfil full, os quatro slots foram produzidos para as 126 regiões, sem falha de
evidência, interpretação ou calibração. Os `model_calls` de foreground são zero porque
o slot reutiliza a única extração DINO do frame; o custo é contabilizado como pooling.

## Custo operacional

| perfil | latência total (s) | model calls | pico de VRAM |
| --- | ---: | ---: | ---: |
| baseline | 503,492 | 261 | 4,57 GiB |
| full | 508,981 | 390 | 4,57 GiB |
| delta full−baseline | +5,489 (+1,09%) | +129 | +0,00 GiB |

A latência de cada slot inclui carga lazy quando ela ocorre naquele slot; por isso o
tight crop do baseline absorve o primeiro load do encoder, enquanto no perfil full esse
load aparece antes em scene conditioned. O wall-clock total é a comparação operacional
correta entre runs; as latências de slot servem para localizar custo, não para somar
loads compartilhados como se fossem independentes.

Esta ablation demonstra cobertura, isolamento geométrico, custo e saúde operacional.
Qualidade semântica calibrada continua dependendo das anotações humanas da #210 e não é
inferida destes proxies.
