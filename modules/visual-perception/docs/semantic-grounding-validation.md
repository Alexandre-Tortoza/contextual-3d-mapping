# Validação curta de grounding semântico

A execução posterior com percepção completa está na
[validação integrada de 12/09](semantic-grounding-integrated-validation.md).
Este relatório preserva os resultados do replay com reconhecimento congelado.

## Resultado e limite de aceitação

A separação entre discovery e footprint semântico removeu as grandes regiões
de parede chamadas `door` e `wooden pallet` nos exemplos inspecionados. Ambos os
conceitos continuam reconhecidos e associados ao objeto em outros frames do
mesmo trecho. O pipeline agora exige evidência de localização/segmentação e
registra a incerteza de fronteira para publicar um label forte.

A aceitação é **parcial**: o SAM ainda inclui vãos entre as ripas do pallet;
há alguns pontos distantes em 3D apesar de projetados no footprint grounded;
e nenhuma claim `broken tile` obteve footprint forte nesta janela. As claims
e o contexto foram preservados, mas preservar reconhecimento não demonstra
cobertura espacial suficiente para danos. Não se afirma que todos os pixels
ou pontos remanescentes estejam corretos.

## Evidência e método

HEAD analisado: `3596e90f975472d3a521d59005e556c7376f044c`, com a implementação
local desta mudança. [Diagnóstico e contracts](semantic-grounding.md).

Foram reprocessados **25 frames em 6,029 s**, aproximadamente 4 FPS, de
`corridor-02-04210` a `corridor-02-04354`. Os artifacts existentes de
`20260911T024436Z` congelam discovery, interpretação, relações e structural
context. Não houve nova inferência de VLM nem aumento de evidência temporal.
O mapa geométrico existente foi reutilizado; não foi executado FAST-LIO novamente.

Os quatro braços usam os mesmos frames, geometria, calibração MEI, exclusão de
ego, valid area e buffer de oclusão. `before` é uma recomposição explícita do
footprint de discovery, não uma cópia do antigo arquivo final. A correção da
leitura de suporte da claim e a exclusão do rig são comuns a todos os braços.
Grounding não muda o ranking de fusão ou os filtros do viewer.

Artifacts completos: [manifest](../../../artifacts/semantic-grounding-6s-20260911/manifest.json),
[métricas](../../../artifacts/semantic-grounding-6s-20260911/metrics.json),
[mapa anterior](../../../artifacts/semantic-grounding-6s-20260911/before.json),
[mapa com os três fatores](../../../artifacts/semantic-grounding-6s-20260911/pose.json).
São arquivos locais; os dados volumosos ficam em `artifacts/`, ignorado pelo Git.

## Pontos após fusão

Contagens de pontos únicos com o **label exato** vencedor, sem remover pontos
de suporte fraco no viewer. `dark brown door` permanece um label separado:
não houve normalização específica para os exemplos.

| Braço | `door` | `wooden pallet` | `dark brown door` | Total contextual |
|---|---:|---:|---:|---:|
| Discovery + nearest | 2.555 | 664 | 290 | 20.384 |
| Grounding + nearest | 242 | 276 | 285 | 3.294 |
| Grounding + boundary + nearest | 225 | 281 | 257 | 3.087 |
| Grounding + boundary + interpolação | 235 | 268 | 250 | 3.160 |

Em relação ao braço de discovery, a configuração final reduz `door` em 90,8%
e `wooden pallet` em 59,6%. Redução de contagem não é, isoladamente, acurácia.
Os 281 pontos de pallet no terceiro braço podem superar os 276 do segundo:
retirar contribuições de boundary concorrentes também muda qual label vence
a fusão. A geometria semântica permanece igual entre esses braços.

| Label / braço | Suporte espacial médio | Pontos weak | Agreement médio | Pontos com ≥2 contribuições |
|---|---:|---:|---:|---:|
| door / discovery | 0,742 | 14 | 0,919 | 1.650 |
| door / grounding | 0,422 | 0 | 1,000 | 109 |
| door / boundary | 0,440 | 0 | 1,000 | 109 |
| door / interpolação | 0,501 | 0 | 1,000 | 134 |
| wooden pallet / discovery | 0,331 | 284 | 0,440 | 630 |
| wooden pallet / grounding | 0,648 | 0 | 0,960 | 99 |
| wooden pallet / boundary | 0,670 | 0 | 0,963 | 104 |
| wooden pallet / interpolação | 0,624 | 0 | 0,949 | 95 |

Agreement só é calculado onde há múltiplas contribuições. A parede incorretamente
rotulada como porta tinha suporte espacial alto por ser extensa e coerente
consigo mesma; a queda desse agregado não contradiz a melhora visual. Essas
métricas não substituem referência anotada nem tornam confiança VLM calibrada.

## Inspeção da porta

No frame `04246`, a região `region-af034aba16490177` chamada `door` tem **24.252
pixels de parede**. O detector e SAM localizam a porta separadamente. Não há
interseção utilizável entre o segmento da porta e aquela discovery; o resultado
é `semantic_mask_inconsistent`, claim preservada e footprint ausente. Aquele
frame deixa de fornecer **553 associações `door`** sobre a parede/entorno.

![Discovery de parede chamada door e segmentação localizada da porta](../../../artifacts/semantic-grounding-6s-20260911/frames/corridor-02-04246/masks/region-af034aba16490177/comparison.png)

![Projeção antes/depois da região incorreta de porta](../../../artifacts/semantic-grounding-6s-20260911/projections/corridor-02-04246/projection-01.png)

O frame `04258` contém discovery sobre a própria porta: **7.807 → 7.786 pixels**,
um componente, 535 pixels de boundary e 7.251 de interior (93,1%). A porta
permanece associada: 380 contribuições anteriores, 379 com grounding, 347 com
boundary e 371 com interpolação. O refinamento não apaga uma região boa para
obter redução artificial de área.

![Porta preservada com projeção grounded](../../../artifacts/semantic-grounding-6s-20260911/projections/corridor-02-04258/projection-02.png)

Nos exemplos inspecionados, a grande superfície de parede deixa de receber
`door`. Permanecem pontos 3D distantes isolados, visíveis nas figuras; a
segmentação correta não comprova que a superfície 3D projetada seja a visível.
Esse residual envolve a associação/oclusão e não foi ocultado ou corrigido
por ajuste do buffer nesta mudança.

## Inspeção do pallet

No frame `04288`, duas regiões carregam a mesma claim:

| Região | Discovery | Máscara semântica | Componentes antes/depois | Resultado |
|---|---:|---:|---|---|
| `region-b062138834f2b1d6` — pallet | 25.614 | 25.591 | 1 / 1 | refined |
| `region-b99bcbe3220f5df9` — parede | 48.326 | 0 | 4 / 0 | semantic_mask_inconsistent |

A região da parede já tinha quatro componentes e mantinha quatro depois do
ownership histórico. Todos herdavam `wooden pallet`. Agora nenhum fornece
footprint forte. O pallet tem 818 pixels de boundary e 24.773 de interior
(96,8%). As contribuições desse frame passam de **1.462 para 394** apenas com
grounding, 382 com boundary e 379 com interpolação.

![Pallet preservado na comparação das máscaras](../../../artifacts/semantic-grounding-6s-20260911/frames/corridor-02-04288/masks/region-b062138834f2b1d6/comparison.png)

![Projeção do pallet e parede antes/depois](../../../artifacts/semantic-grounding-6s-20260911/projections/corridor-02-04288/projection-03.png)

A grande faixa de parede lateral deixa de receber `wooden pallet`. A máscara
ainda preenche vãos entre ripas, pelos quais pode haver parede visível. Não foi
medida precisão por pixel nesses vãos. A verificação visual sustenta a redução
do vazamento adjacente, não sua eliminação completa.

As figuras por frame incluem todas as suas contribuições fortes, inclusive as
que não venceram a fusão final. Por isso suas contagens diferem da tabela de
pontos únicos. As escalas e orientações 3D são comuns aos quatro braços; todos
os pontos rotulados, inclusive os distantes, permanecem desenhados. As
proporções visuais dos eixos são fixas, com coordenadas em metros.

## Boundary e pose como fatores separados

A boundary tem margem de `sqrt(0.5)` pixels, referente ao arredondamento nos
dois eixos; sigma de registro adicional é zero por ausência de estimativa
medida. Não é uma margem calibrada de erro de pose/calibração. A máscara não é
erodida para publicação; a distância e a força semântica ficam nos diagnostics.

Com nearest, há **8.183 pixels de boundary** e **153.684 pixels de interior**
somados entre frames, dos 161.867 pixels após grounding/ownership. Das
associações, **4.026** ficam fortes e **293** são tentativas de boundary:
69 relacionadas a `door` e 26 a `wooden pallet`. Com pose interpolada, são
**4.134 fortes e 268 tentativas**, incluindo 68 de porta e 28 de pallet.
Esses números contam pares ponto/observação, não pontos únicos após fusão.

Grounding e boundary preservam os mesmos 34.242 pontos com associação RGB.
A interpolação altera esse conjunto para 34.135. O delta nearest→RGB chegou
a **50,07 ms**; a interpolação usa o timestamp RGB exato, sem extrapolação,
com intervalo máximo observado entre poses de **120,10 ms**. O delta
RGB/LiDAR permaneceu igual nos braços, com máximo de **50,09 ms** dentro
da tolerância de 100 ms já usada na composição.

## Falhas, contexto e custo

Nas 121 regiões candidatas: 11 groundings aceitos, 59 falhas de localização,
26 inconsistências espaciais e 25 perdas/ambiguidades do suporte principal.
`mask_fragmented` também cobre suporte principal que não sobrevive a clipping,
mesmo quando resta apenas um componente. Não há fallback forte de discovery.

Todas as claims e todo structural context foram preservados nos 25 frames.
Audit errors: **0 → 0**; warnings: **968 → 968**. As predições incorretas antigas
não são apagadas para reduzir o audit. `broken tile`, damage e outras evidências
continuam inspecionáveis, mas a cobertura espacial de dano não passou na
aceitação: nenhum `broken tile` ficou forte. O comportamento de abstenção é
correto; a capacidade do backend para esses conceitos ainda precisa melhorar.

Na RTX 3060 8 GB, grounding adicionou **115,34 s** de carregamento, inferência e validação ao
trecho: **4,61 s/frame** em média (22 frames com candidatas, três sem), mediana
5,26 s e máximo 8,86 s. O pico CUDA alocado foi **4,57 GiB**. Detector e SAM
foram carregados em sequência, nunca simultaneamente. Houve **136 chamadas**
adicionais, média 5,44/frame; cada região preserva seu custo alocado e
chamadas compartilhadas são contabilizadas uma vez. Geração de PNGs e composição
dos quatro mapas não entram nessa latência adicional do estágio.

## Verificação e reprodução

Suite final em CPU: **792 testes passaram**, um módulo de GPU ignorado nesse
ambiente. A suíte inicial do ambiente ML executou também os adapters reais:
619 testes passaram. A validação acima executou o novo backend real nos 25
frames. `ruff`, `mypy` (84 arquivos) e `git diff --check` passaram.

Os testes incluem discovery de porta com parede, refinamento menor, ownership
fragmentado, perda da âncora, thing com componentes distantes, stuff extenso,
falhas de backend, múltiplas boxes, serialização, fluxo entre módulos até
associação, interior/boundary/exterior, buracos, valid area/MEI, ego e SLERP.

O manifest registra os argumentos exatos da execução. Seu comando é
`python -m visual_perception_experiments.grounding_validation`, com as raízes
de `contracts`, módulos, runtime e `experiments` no `PYTHONPATH`; os checkpoints
usados já estavam em cache e a execução usou `HF_HUB_OFFLINE=1`. O diretório
`code/` e `code-sha256.json` preservam o código da execução. O pós-processamento
das figuras e métricas corrigiu a qualificação de IDs e fixou eixos comuns,
sem reexecutar modelos, associação ou fusão.

Não foi executado um novo run de 30 segundos, não houve tracking e nenhum
threshold foi ajustado aos frames ou labels deste trecho. A próxima limitação
a avaliar é cobertura de grounding para evidências contextuais e estruturas
com vazios, separadamente do residual de oclusão 3D.
