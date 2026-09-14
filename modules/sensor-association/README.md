# Sensor Association

`sensor-association` estabelece a correspondência geométrica e temporal entre uma observação de imagem e a geometria 3D, com proveniência de calibração explícita.

> Documentação detalhada: [`docs/`](docs/README.md).

## Responsabilidades

- projetar pontos 3D no modelo de câmera calibrado, incluindo pinhole, fisheye equidistante e MEI;
- filtrar visibilidade, hemisfério frontal, limites da imagem e suporte óptico válido;
- resolver oclusão de forma determinística quando vários pontos disputam suporte visual;
- atribuir cor e região visual ao ponto visível;
- declarar explicitamente o motivo de cada rejeição.

## Não-responsabilidades

- descoberta de regiões e semântica de imagem (`visual-perception`);
- construção do mapa geométrico (`geometric-map`);
- decisão entre classificações concorrentes de observações diferentes (`semantic-fusion`).

## Duas fronteiras de associação

```text
associate_points      scan LiDAR recém-chegado -> frame RGB
associate_map_points  pontos do mapa persistente -> frame RGB
```

`associate_map_points` existe porque colorir o scan e anexá-lo ao mapa produzia duas amostragens da mesma superfície. Ancorando a associação no mapa, cada ponto persistido passa a ter ou não uma associação visual, e a ausência declara seu motivo.

A função recebe o transform mapa para câmera derivado da pose estimada no instante da observação. Com o contexto ancorado no mapa, erro de pose pode virar erro de pixel e, portanto, erro semântico downstream.

## Oclusão em mapa esparso

Um scan isolado pode usar z-buffer por pixel. Um mapa acumulado e subamostrado precisa de suporte espacial mais conservador para impedir que pontos atrás de uma superfície atravessem lacunas da amostragem.

`associate_map_points` agrega profundidade em células e consulta a vizinhança da célula do ponto. A profundidade é comparada no eixo óptico. Os parâmetros de célula e tolerância são explícitos porque dependem da densidade do mapa.

## Invariantes

- unidades em metros;
- pixels inteiros por arredondamento determinístico;
- frames validados contra a calibração antes da projeção;
- LiDAR e RGB precisam compartilhar `clock_id` e respeitar `max_time_delta_ns`;
- rejeições não carregam evidência visual residual;
- as duas fronteiras compartilham projeção, filtros e contract de saída.