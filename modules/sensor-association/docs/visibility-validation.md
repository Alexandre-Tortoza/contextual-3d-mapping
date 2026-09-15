# Validação de visibilidade e suporte de regiões — corridor-02

A comparação remove o vazamento observado de `wooden pallet` para a parede
distante do corredor em L. O novo caminho também exige suporte geométrico
para o vínculo semântico, mas preserva menos pontos próximos que a ablação
com células densas. A retenção próxima continua uma limitação mensurada.

## Entradas congeladas

- 25 frames, de `corridor-02-04210` a `corridor-02-04354`, aproximadamente 6 s;
- percepção em `artifacts/old/integrated-grounding-6s-20260912T050652Z/perception/samples/20260912T050915Z`;
- mapa `corridor-02-176s-30s`, frame `map`, metros;
- PCD com 1.093.011 registros; viewer com 136.627 pontos (stride 8);
- SHA-256 do PCD: `50ea48916331e5e5061b0993c353eca7bf90ec5507079637e2d83eef7e923760`;
- odometria e calibrações existentes, pose interpolada no instante RGB;
- mesmas máscaras, labels, grounding, boundary policy e fusão nos três braços.

O replay executa apenas associação e fusão. Não chama FAST-LIO, VLM ou SAM.
Os braços históricos são ablações explícitas desta comparação ativa; o default
de composição é `measured_surfaces`.

A avaliação registra IDs explícitos dos 623 pontos que tinham `wooden pallet`
na baseline: 593 no grupo próximo e 30 no grupo distante. Os grupos foram
auditados antes da mudança e têm uma separação superior a 32 m em x. Esses
IDs e a separação não são usados pelo algoritmo de associação. O grupo próximo
é uma referência de retenção, não uma anotação de ground truth.

## Resultado dos três braços

| Medida | Células na amostra | Células no PCD completo | Superfícies medidas + vínculo |
| --- | ---: | ---: | ---: |
| IDs distantes ainda com `wooden pallet` | 30 | 0 | 0 |
| IDs próximos retidos, dos 593 | 593 | 378 | 236 |
| IDs próximos sem esse label | 0 | 215 | 357 |
| Novos pontos com esse label | 0 | 0 | 1 |
| Total `wooden pallet` | 623 | 378 | 237 |
| Total `dark brown door` | 447 | 431 | 316 |
| Total `wooden door` | 21 | 14 | 32 |
| Total `wooden slats` | 20 | 12 | 20 |
| Total de pontos com label forte | 1.111 | 835 | 605 |

A baseline recomposta preservou exatamente o mapeamento ID→label da run
congelada. As duas variantes com PCD completo removeram os 30 vazamentos.
O método de superfícies é mais conservador para os pontos próximos: não se
pode apresentar a redução de labels como ganho global de acurácia.

Seis das 77 regiões publicadas possuem componentes com suporte 3D na nova
composição. As demais conservam claims e diagnostics de incerteza, inclusive
quando não têm footprint semântico aceito. Os 605 pontos fortes continuam
ancorados nos mesmos IDs do mapa; nenhuma geometria de objeto foi criada.

## Inspeção das perdas

Na visualização x/z do palete, as perdas não se limitam à parede distante:
atingem amostras do topo, das ripas e da extremidade direita do grupo próximo.
O conjunto final ainda ocupa a região próxima do palete, mas com cobertura
menor. A comparação mostra todos os pontos fortes, sem filtros de confiança
ou de legenda do viewer.

Nos 357 IDs próximos que perderam `wooden pallet`, o contexto primário final
aparece como: 96 ocluídos, 197 RGB fora do footprint de sua observação primária,
42 sem suporte de superfície, 16 interiores com outro label e 6 de boundary.
Esses estados resumem a fusão final; um estado `outside` não prova que todas
as observações rejeitaram a máscara. Os arquivos de auditoria por frame
permitem consultar a decisão no instante em que a baseline via o palete.

Ruído do mapa acumulado e patches de superfícies próximas podem bloquear
pontos que a aproximação por células mantinha. A regra de contorno acrescenta
abstenções em componentes sem âncora. Estes resultados não demonstram que
cada uma dessas perdas seja correta; isso exige referência geométrica ou
anotação adicional.

Na auditoria das mesmas observações, as perdas se distribuem assim:

| Frame | Próximos com label antes | Retidos no mesmo frame | Ocluídos | Sem superfície confirmada | Sem vínculo semântico |
| --- | ---: | ---: | ---: | ---: | ---: |
| 04282 | 258 | 120 | 117 | 5 | 16 |
| 04288 | 357 | 124 | 218 | 7 | 8 |
| 04294 | 458 | 135 | 304 | 11 | 8 |

As diferenças medianas entre ponto próximo rejeitado e primeira superfície
foram 0,422 m, 0,544 m e 0,394 m, respectivamente. A revisão RGB confirma que
as perdas atingem pixels sobre ripas e vãos do palete. A decisão de primeira
superfície responde pela maior parte das perdas, antes da regra de vínculo.
`near-loss-review.png`, `near-loss-audit.json` e seu script preservam essa
inspeção. As contagens por frame incluem IDs repetidos entre observações.

## Janela, tangência e amostragem

As regressões determinísticas cobrem primeira superfície entre dois planos,
limite espacial de um patch, plano inclinado, cilindro com duas interseções e
incidência próxima da tangência, geometria insuficiente, fundo angularmente
mais denso e invariância de candidatos comuns ao subamostrar o viewer.

Fixtures de janela verificam RGB visível no fundo sem label `window`, moldura
medida com label, ausência total de moldura e fundo majoritário. Não houve
janela anotada nesta sequência real; a validação específica de janela é
sintética. A máscara e o claim 2D são preservados quando falta suporte 3D.

## Artifacts e reprodução

A run histórica fica em `artifacts/old/visibility-validation-20260912-v2/`:

```text
manifest.json                         hashes de entradas/código, política e tempos
regression-ids.json                   593 IDs próximos + 30 distantes auditados
metrics.json                         contagens, IDs perdidos e evidências
comparison.png                       escalas compartilhadas do mapa e do palete
legacy_cells/context.json            baseline recomposta
dense_cells/context.json            PCD integral com regra de células
measured_surfaces/context.json       primeira superfície + vínculo da região
*/context-assets/                    previews RGB congelados
*/context-DEBUG/sensor-association/   decisões por ID e frame, com pose
```

Com os pacotes locais no `PYTHONPATH`, o ponto de entrada é:

```bash
modules/visual-perception/.venv/bin/python -m visual_perception_experiments.visibility_validation \
  --geometry artifacts/old/corridor-02-176s-30s.json \
  --bag datasets/raw/corridor-02/corridor-02.bag \
  --intrinsics datasets/raw/corridor-02/corridor-02-Intrinsics.yaml \
  --extrinsics datasets/raw/corridor-02/corridor-02-extrinsics.yaml \
  --odometry artifacts/old/corridor-02-176s-30s-odometry.csv \
  --window artifacts/old/integrated-grounding-6s-20260912T050652Z/window.json \
  --visual-run artifacts/old/integrated-grounding-6s-20260912T050652Z/perception/samples/20260912T050915Z \
  --baseline artifacts/old/integrated-grounding-6s-20260912T050652Z/pose.json \
  --regression-ids artifacts/old/visibility-validation-20260912-v2/regression-ids.json \
  --output artifacts/visibility-validation-NOVA-RUN
```

O destino precisa ser novo. O processo falha se faltam frames congelados, se o
hash do PCD diverge, se a baseline recomposta muda labels ou se algum dos 30
IDs distantes conserva `wooden pallet` no braço de superfícies.

## Execução e viewer

No replay final, células na amostra levaram 71,1 s; células densas, 70,3 s;
superfícies medidas, 289,1 s. Esses tempos incluem composição, serialização e
auditoria; não são benchmark isolado do kernel. O modelo produziu 743.341
patches a partir da geometria integral.

As três composições estão publicadas como `20260912-visibility-legacy`,
`20260912-visibility-dense` e `20260912-visibility-surfaces`. A última é o
default mais recente do catálogo. O inspector distingue a hipótese 2D do
label forte e mostra primeira superfície, tolerância e motivo por ponto.

A verificação de implementação passou em 824 testes Python, com um módulo GPU
ignorado por ausência de `torch` na venv de verificação, além de Ruff, mypy e
24 testes da interface. O build do viewer foi concluído. A validação adicional
dos frames da calibração e a propriedade pública de contagem de patches não
alteraram status, label ou evidência dos 623 IDs rechecados no frame 04288.
O código exato usado no replay foi preservado em `code/` e conferido contra
os hashes do manifest.
