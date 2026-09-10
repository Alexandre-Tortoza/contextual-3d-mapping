# Sensor Association

`sensor-association` estabelece a correspondência geométrica e temporal entre uma
observação de imagem e a geometria 3D, com proveniência de calibração explícita.

## Responsabilidades

- projetar pontos 3D no modelo de câmera calibrado (pinhole, fisheye equidistante, MEI);
- filtrar visibilidade: hemisfério frontal, limites da imagem, suporte óptico válido;
- resolver oclusão de forma determinística quando vários pontos disputam o mesmo pixel;
- atribuir cor e região visual ao ponto visível;
- declarar explicitamente o motivo de cada rejeição.

## Não-responsabilidades

- descoberta de regiões e semântica de imagem (`visual-perception`);
- construção do mapa geométrico (`geometric-map`);
- decisão entre classificações concorrentes de observações diferentes (`semantic-fusion`).

## Duas fronteiras de associação

O módulo expõe duas entradas, que diferem no frame de partida dos pontos:

```text
associate_points      scan LiDAR recém-chegado -> frame RGB
associate_map_points  pontos do mapa persistente -> frame RGB
```

`associate_map_points` existe porque colorir o scan e anexá-lo ao mapa produzia duas
amostragens da mesma superfície a poucos centímetros uma da outra — a do mapa, sempre sem
contexto, e a do scan, colorida. No viewer isso aparecia como pontos cinzas "fantasmas"
encostados em pontos coloridos, quando na verdade eram pontos diferentes de fontes
diferentes. Ancorando a associação no mapa, cada ponto persistido passa a ter ou não um
label, e a ausência passa a declarar seu motivo.

Ela recebe o transform mapa→câmera derivado da pose estimada no instante da observação.
Isso desloca a exigência de precisão: com o contexto ancorado no mapa, um erro de pose
vira pixel errado e, portanto, label errado — e não apenas um ponto deslocado.

### Oclusão em um mapa esparso

As duas entradas usam regras de oclusão diferentes, e essa é a diferença que importa.

Um scan isolado é naturalmente livre de oclusão a partir do seu próprio viewpoint, então
o z-buffer por pixel exato basta. Um mapa acumulado e subamostrado, não: entre as amostras
da superfície da frente sobram vazios de pixels, e um ponto de outro cômodo passa por eles
e recebe o label do que está na imagem. Medido no corridor-02, o efeito era classificar o
mapa **através das paredes** — pontos na altura da parede recebendo o label `ceiling`.

`associate_map_points` agrega profundidade em células de pixels e consulta a vizinhança
3×3 da célula do ponto, aproximando a área que cada amostra do mapa realmente cobre.
A profundidade comparada é a do **eixo óptico**, não a distância radial: em um campo de
visão largo a distância radial cresce em direção às bordas mesmo sobre uma superfície
frontal, e compará-la faria a periferia ocluir a si mesma.

Os parâmetros (`occlusion_cell_px`, `occlusion_relative_tolerance`,
`occlusion_absolute_margin_m`) são explícitos porque dependem da densidade do mapa;
célula zero volta ao z-buffer por pixel exato.

A regra é conservadora por escolha: em descontinuidades de profundidade ela perde uma
borda de pontos em vez de rotular o fundo com o label da frente.

## Invariantes de substituição

As duas entradas compartilham projeção, filtro de visibilidade, regra de oclusão e o
contract de saída. Um ponto físico alcançado pelos dois caminhos produz o mesmo pixel e a
mesma classificação; isso é verificado por teste.

- unidades em metros, pixels inteiros por arredondamento determinístico;
- frames validados contra a calibração antes de qualquer projeção;
- LiDAR e RGB precisam compartilhar `clock_id` e respeitar `max_time_delta_ns`;
- rejeições nunca carregam evidência visual residual.
