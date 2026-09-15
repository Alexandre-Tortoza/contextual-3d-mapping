# Run integrada de grounding em 6 segundos

## Resultado

A run `20260912T050915Z` reexecutou a percepção completa nos **25 frames** da
janela de 6,029 s. Todos terminaram sem falhas de pipeline, OOM ou erros de
audit. Grounding removeu a associação extensa do pallet à parede adjacente e
preservou associações sobre o pallet e a porta nos exemplos revisados.

A aceitação de qualidade continua **parcial**. Os vãos entre as ripas ainda
entram na máscara do pallet, permanecem pontos 3D distantes e nenhum label de
dano obteve footprint forte. A cobertura contextual final é bastante restrita:
1.111 pontos com quatro labels exatos. Não houve expansão para 30 segundos.

## Método e proveniência

Esta execução sucede a [validação com reconhecimento congelado](semantic-grounding-validation.md).
Discovery, embeddings, interpretação, refinamento, relações e grounding foram
executados novamente nos frames `corridor-02-04210` a `corridor-02-04354`.
Os hashes das 25 imagens são idênticos aos da run `20260911T024436Z`; os IDs e
máscaras das 860 regiões de discovery também coincidem entre as duas runs.
A geometria e a odometria existentes foram reutilizadas.

O perfil atual usa `local_first`; o anterior usava `context_assisted`. Portanto,
a comparação histórica inclui uma mudança de reconhecimento e não isola o
efeito do grounding. As quatro ablações abaixo usam as mesmas observações da
**nova** percepção, preservando geometria, calibração, exclusão do rig e
política de oclusão. Não houve ajuste de threshold por frame ou conceito.

O diretório [da execução](../../../artifacts/old/integrated-grounding-6s-20260912T050652Z/README.md)
preserva código, hashes, patch do checkout, checkpoints, configuração,
entradas, logs e manifests. O HEAD de referência é
`3596e90f975472d3a521d59005e556c7376f044c`, com alterações locais arquivadas.
Os modelos foram carregados do cache com `HF_HUB_OFFLINE=1`.

O harness visual ainda publica referências sintéticas de teste em
`observation.source`. Na composição espacial, `window.json` resolve os
timestamps reais de header e gravação, o sensor e o índice RGB. A limitação de
metadados do harness permanece explícita no manifest da execução.

## Contagens e ablações

A tabela conta pontos únicos após fusão pelo **label exato vencedor**, antes
de atenuações do viewer. Variantes de porta permanecem separadas.

| Braço | `door` | `dark brown door` | `wooden door` | `wooden pallet` | `wooden slats` | Total contextual |
|---|---:|---:|---:|---:|---:|---:|
| Discovery + nearest | 2.610 | 147 | 35 | 1.933 | 11 | 13.511 |
| Grounding + nearest | 0 | 466 | 31 | 612 | 25 | 1.134 |
| Grounding + boundary + nearest | 0 | 430 | 28 | 599 | 19 | 1.076 |
| Grounding + boundary + interpolação | 0 | 447 | 21 | 623 | 20 | 1.111 |

Redução de contagem não mede acurácia. Excluir contribuições concorrentes
também pode aumentar quantos pontos vencem com outro label, como ocorre com
`dark brown door`. Os 1.111 pontos finais incluem 723 com estado `corroborated`
e 388 `uncorroborated`; nenhum recebe estado `weak`. Esses estados não
constituem ground truth.

Grounding preserva os 34.242 pontos com associação RGB do braço nearest.
A interpolação altera esse conjunto para 34.135. A boundary separa 2.085
contribuições fortes e 133 tentativas com nearest; com interpolação, são 2.122
fortes e 142 tentativas. Contribuições contam pares ponto/observação, enquanto
a tabela conta pontos únicos.

Dados completos: [métricas de revisão](../../../artifacts/old/integrated-grounding-6s-20260912T050652Z/review-metrics.json)
e [comparação entre braços](../../../artifacts/old/integrated-grounding-6s-20260912T050652Z/comparison.json).

## Revisão visual

No frame `04288`, o pallet `region-b062138834f2b1d6` conserva
**25.614 → 25.591 pixels**. A região de parede `region-b99bcbe3220f5df9`, também
chamada `wooden pallet`, perde o footprint forte: **48.326 → 0 pixels**, com
`semantic_mask_inconsistent`. O segmentador localiza o pallet separadamente da
parede. As contribuições `wooden pallet` desse frame passam de 1.464 para 394
com grounding, 382 com boundary e 379 com interpolação.

![Parede rejeitada como footprint do pallet](../../../artifacts/old/integrated-grounding-6s-20260912T050652Z/masks/corridor-02-04288/region-b99bcbe3220f5df9/comparison.png)

![Pallet preservado e redução da projeção sobre a parede](../../../artifacts/old/integrated-grounding-6s-20260912T050652Z/projections/corridor-02-04288/projection-02.png)

No frame `04258`, a porta corretamente localizada permanece associada como
`wooden door`: 380 contribuições em discovery, 379 com grounding, 347 com
boundary e 371 com interpolação. Os pontos distantes em 3D continuam visíveis
nas figuras e requerem revisão da associação/oclusão.

![Porta preservada nos quatro braços](../../../artifacts/old/integrated-grounding-6s-20260912T050652Z/projections/corridor-02-04258/projection-01.png)

A região `region-af034aba16490177` do frame `04246`, antes chamada `door`, é
agora interpretada como `ceiling`, com alternativa `wall`, e permanece em
structural context. A ausência dessa claim de porta na nova run inclui uma
mudança de reconhecimento; não pode ser atribuída apenas ao grounding.

As duas candidatas `broken tile` terminaram em `semantic_mask_inconsistent`;
`cracked tile` terminou em `mask_fragmented`, e `cracked concrete` também foi
inconsistente. Esses labels tinham respectivamente 82, 16 e 104 pontos
vencedores no braço discovery e nenhum no braço final. Preservar suas claims
não demonstra cobertura espacial de dano. No frame `04282`, o segmentador de
`cracked tile` cobre uma faixa extensa do chão, sem produzir suporte semântico
aceito para a região de origem.

![Ausência de footprint forte para cracked tile](../../../artifacts/old/integrated-grounding-6s-20260912T050652Z/masks/corridor-02-04282/region-f4b8fe9f282e4845/comparison.png)

## Falhas e custo

Foram preservadas 77 regiões candidatas à publicação e 783 em structural
context. Das 77 tentativas de grounding, seis terminaram em `refined`, 17 em
`semantic_mask_inconsistent`, 13 em `mask_fragmented` e 41 em `grounding_failed`.
As falhas e claims permanecem nos artifacts; não há fallback forte de discovery.

Audit: **zero erros e 1.109 warnings**, contra 968 warnings da percepção
anterior. Não houve falha nas etapas de interpretação, evidência, sinais,
relações ou calibração semântica. A nova inferência pode mudar as claims e seus
warnings, mesmo com discovery idêntica.

Na RTX 3060 8 GB, o pico CUDA alocado foi **4,57 GiB**. A soma das latências de
percepção foi 117,2 minutos; a execução inteira, incluindo a composição e as
figuras, levou **121,3 minutos**. Grounding registrou 123 chamadas de modelo.
Essa medição cobre a run integrada e não estima isoladamente o custo adicional
do grounding nem desempenho em tempo real.

## Publicação e verificação

O viewer oferece quatro pastas independentes: original de 11/09, replay de
grounding sobre o reconhecimento anterior, discovery da nova run e a
[run integrada](http://localhost:5173/?artifact=/runs/20260912T050915Z-integrated/context.json).
Os quatro braços científicos completos continuam no diretório da execução.
O [fluxo de CLI](../../../apps/cli/README.md#salvar-e-comparar-runs-com-contexto)
documenta como publicar, listar e abrir resultados posteriores.

`make verify PYTHON=.venv-verify/bin/python`: 806 testes passaram; o módulo de
testes GPU foi ignorado nesse ambiente sem PyTorch. `ruff` e `mypy` passaram.
O frontend passou em 23 testes e no build. A inspeção em Chromium isolado
confirmou o catálogo contextual, a seleção inicial da run integrada e a
preservação da câmera ao alternar versões da mesma geometria.
