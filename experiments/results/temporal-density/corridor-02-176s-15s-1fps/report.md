# Densidade temporal — relatório de dados

Gerado por `visual_perception_experiments.temporal_density`. Estritamente descritivo: 
apresenta o que os runs registraram sobre si mesmos, sem afirmar correção semântica — 
o conjunto não tem anotação humana revisada, e usar a saída do próprio pipeline como 
ground truth produziria uma acurácia sem significado.

## Proveniência

- **B · 16 frames @ 1 FPS**: `modules/visual-perception/benchmarks/results/samples/20260910T211622Z`
- **A · 16 keyframes @ 2 s**: `modules/visual-perception/benchmarks/results/samples/20260910T183839Z`

## Por frame

### B · 16 frames @ 1 FPS

| # | t (s) | frame | proposals | regiões | publicadas | estrutural | conceitos publicados |
|---|---|---|---|---|---|---|---|
| 00 |  -0.01 | `corridor-02-04234` | 69 | 35 | 1 | 34 | door (14608 px) |
| 01 |   1.00 | `corridor-02-04259` | 76 | 34 | 7 | 27 | ceiling vent (60676 px); wooden door (12935 px); black rectangular object (6776 px); black rectangular object (9212 px) ⚠; ceiling light fixture (41140 px); black rectangular object (6417 px); curved wall (75330 px) |
| 02 |   1.98 | `corridor-02-04282` | 75 | 37 | 4 | 33 | wooden pallet (210456 px); broken tile (63162 px); wooden pallet (19698 px); broken tile (23660 px) |
| 03 |   3.00 | `corridor-02-04307` | 73 | 39 | 0 | 39 | — |
| 04 |   4.02 | `corridor-02-04331` | 67 | 32 | 3 | 29 | rock (1704 px); broken concrete floor (57600 px); broken floor tile (169058 px) ⚠ |
| 05 |   5.00 | `corridor-02-04354` | 80 | 44 | 4 | 40 | wall (84762 px) ⚠; broken floor (22000 px) ⚠; wall (3286 px) ⚠; broken tile (50880 px) |
| 06 |   6.02 | `corridor-02-04379` | 65 | 36 | 9 | 27 | red door (24795 px); wall (2610 px) ⚠; white door (39336 px); arch-shaped wall section (41208 px) ⚠; smooth surface (3212 px) ⚠; door (270080 px); red door (62370 px); door (13940 px); white curved object (16705 px) |
| 07 |   6.99 | `corridor-02-04402` | 65 | 32 | 3 | 29 | wall (91155 px) ⚠; pink wall (42280 px); wall (3325 px) ⚠ |
| 08 |   7.97 | `corridor-02-04426` | 67 | 39 | 3 | 36 | red door (15678 px); door (196480 px); beige carpet (8694 px) |
| 09 |   8.99 | `corridor-02-04450` | 58 | 31 | 2 | 29 | ceiling fixture (52521 px); grid pattern (9512 px) |
| 10 |  10.01 | `corridor-02-04475` | 60 | 29 | 2 | 27 | wall (30464 px) ⚠; ceiling fan (57812 px) |
| 11 |  10.99 | `corridor-02-04498` | 50 | 25 | 1 | 24 | ceiling (78596 px) ⚠ |
| 12 |  12.00 | `corridor-02-04523` | 48 | 24 | 4 | 20 | ceiling light fixture (79194 px); ceiling light fixture (84665 px); ceiling light fixture (46944 px); ceiling light fixture (16632 px) |
| 13 |  12.98 | `corridor-02-04546` | 51 | 25 | 3 | 22 | ceiling light fixture (100450 px); ceiling light fixture (78997 px); ceiling light fixture (53592 px) |
| 14 |  14.00 | `corridor-02-04570` | 50 | 25 | 3 | 22 | ceiling light fixture (78997 px); ceiling light fixture (100450 px); ceiling (53592 px) ⚠ |
| 15 |  15.02 | `corridor-02-04595` | 42 | 18 | 3 | 15 | ceiling vent (87150 px); blue textured surface (7068 px) ⚠; black glove (1677 px) |

## Totais

| eixo | A · 16 keyframes @ 2 s (completo) | A · 16 keyframes @ 2 s (mesmo trecho) | B · 16 frames @ 1 FPS |
|---|---|---|---|
| frames | 16 | 8 | 16 |
| trecho coberto (s) | 30.0 | 14.0 | 15.0 |
| proposals | 947 | 501 | 996 |
| regiões observadas | 462 | 257 | 505 |
| regiões publicadas | 36 | 29 | 52 |
| contexto estrutural | 426 | 228 | 453 |
| publicadas por frame | 2.25 | 3.62 | 3.25 |
| conceitos distintos | 19 | 16 | 26 |
| frames sem evidência contextual | 3/16 | 0/8 | 1/16 |
| maior sequência cega | 2 | 0 | 1 |
| regiões com primárias divergentes | 8 | 6 | 14 |

## Conceitos publicados

| conceito | regiões |
|---|---|
| ceiling light fixture | 10 |
| wall | 6 |
| door | 4 |
| black rectangular object | 3 |
| broken tile | 3 |
| red door | 3 |
| ceiling | 2 |
| ceiling vent | 2 |
| wooden pallet | 2 |
| arch-shaped wall section | 1 |
| beige carpet | 1 |
| black glove | 1 |
| blue textured surface | 1 |
| broken concrete floor | 1 |
| broken floor | 1 |
| broken floor tile | 1 |
| ceiling fan | 1 |
| ceiling fixture | 1 |
| curved wall | 1 |
| grid pattern | 1 |
| pink wall | 1 |
| rock | 1 |
| smooth surface | 1 |
| white curved object | 1 |
| white door | 1 |
| wooden door | 1 |

## Persistência temporal

`persistência` é `frames / span`: 1.00 significa presença em todos os frames entre a 
primeira e a última publicação. **Isto é co-ocorrência de conceito, não identidade de 
objeto** — nenhum tracking temporal foi executado, então `n` frames não são `n` objetos 
nem garantidamente o mesmo.

| conceito | 1º visto (s) | último (s) | frames | span | persistência | regiões | maior área (px) |
|---|---|---|---|---|---|---|---|
| ceiling light fixture | 1.00 | 14.00 | 4 | 14 | 0.29 | 10 | 100450 |
| wall | 5.00 | 10.01 | 4 | 6 | 0.67 | 6 | 91155 |
| door | -0.01 | 7.97 | 3 | 9 | 0.33 | 4 | 270080 |
| broken tile | 1.98 | 5.00 | 2 | 4 | 0.50 | 3 | 63162 |
| red door | 6.02 | 7.97 | 2 | 3 | 0.67 | 3 | 62370 |
| ceiling | 10.99 | 14.00 | 2 | 4 | 0.50 | 2 | 78596 |
| ceiling vent | 1.00 | 15.02 | 2 | 15 | 0.13 | 2 | 87150 |
| black rectangular object | 1.00 | 1.00 | 1 | 1 | 1.00 | 3 | 9212 |
| wooden pallet | 1.98 | 1.98 | 1 | 1 | 1.00 | 2 | 210456 |
| arch-shaped wall section | 6.02 | 6.02 | 1 | 1 | 1.00 | 1 | 41208 |
| beige carpet | 7.97 | 7.97 | 1 | 1 | 1.00 | 1 | 8694 |
| black glove | 15.02 | 15.02 | 1 | 1 | 1.00 | 1 | 1677 |
| blue textured surface | 15.02 | 15.02 | 1 | 1 | 1.00 | 1 | 7068 |
| broken concrete floor | 4.02 | 4.02 | 1 | 1 | 1.00 | 1 | 57600 |
| broken floor | 5.00 | 5.00 | 1 | 1 | 1.00 | 1 | 22000 |
| broken floor tile | 4.02 | 4.02 | 1 | 1 | 1.00 | 1 | 169058 |
| ceiling fan | 10.01 | 10.01 | 1 | 1 | 1.00 | 1 | 57812 |
| ceiling fixture | 8.99 | 8.99 | 1 | 1 | 1.00 | 1 | 52521 |
| curved wall | 1.00 | 1.00 | 1 | 1 | 1.00 | 1 | 75330 |
| grid pattern | 8.99 | 8.99 | 1 | 1 | 1.00 | 1 | 9512 |
| pink wall | 6.99 | 6.99 | 1 | 1 | 1.00 | 1 | 42280 |
| rock | 4.02 | 4.02 | 1 | 1 | 1.00 | 1 | 1704 |
| smooth surface | 6.02 | 6.02 | 1 | 1 | 1.00 | 1 | 3212 |
| white curved object | 6.02 | 6.02 | 1 | 1 | 1.00 | 1 | 16705 |
| white door | 6.02 | 6.02 | 1 | 1 | 1.00 | 1 | 39336 |
| wooden door | 1.00 | 1.00 | 1 | 1 | 1.00 | 1 | 12935 |

## Controle de não-determinismo

Frames presentes nos dois runs — mesmos pixels, mesma configuração. A divergência 
aqui é ruído do pipeline, e calibra quanto de qualquer diferença entre os braços é real.

| frame | publicadas no baseline | publicadas no candidato | só no baseline | só no candidato |
|---|---|---|---|---|
| `corridor-02-04234` | 1 | 1 | — | — |
| `corridor-02-04282` | 4 | 4 | — | — |
| `corridor-02-04331` | 3 | 3 | — | — |
| `corridor-02-04379` | 9 | 9 | — | — |
| `corridor-02-04426` | 3 | 3 | — | — |
| `corridor-02-04475` | 2 | 2 | — | — |
| `corridor-02-04523` | 4 | 4 | — | — |
| `corridor-02-04570` | 3 | 3 | — | — |

## Frames de interesse

| motivo | frame | publicadas |
|---|---|---|
| primeiro frame | `corridor-02-04234` | 1 |
| último frame | `corridor-02-04595` | 3 |
| mais evidências publicadas | `corridor-02-04379` | 9 |
| maior região publicada (candidato a falso positivo) | `corridor-02-04379` | 9 |
| door na maior área | `corridor-02-04379` | 9 |
| mais dano/detrito | `corridor-02-04282` | 4 |

## Notas estruturais

- confiança semântica degenerada em 16/16 frames;
- 14 regiões publicadas carregam claims `primary` divergentes 
  resolvidas pela reconciliação intra-frame (marcadas com ⚠ na tabela por frame);
- 67 hipóteses de entidade formadas ao 
  longo da sequência.
