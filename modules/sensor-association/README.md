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

## Fronteira pública de associação

A composição do mapa usa `MeasuredSurfaceModel` e
`associate_measured_map_points`. O modelo recebe `PointCloudGeometry` completo,
no frame do mapa, e a configuração tipada `SurfaceVisibilityConfig`. A
associação recebe os candidatos do viewer, RGB, calibração, transform
mapa→câmera, âncora temporal LiDAR e footprints visuais congelados.

`SurfaceAssociationResult` devolve uma tentativa por candidato na ordem de
entrada, diagnostics por região e contagens de visibilidade. A geometria
inteira constrói oclusores; a seleção dos candidatos apenas determina quais
resultados serão devolvidos. Subamostrar ou reordenar candidatos preserva a
decisão de cada ponto comum, inclusive quando vários caem no mesmo pixel.

A API `associate_points` atende scans LiDAR recém-chegados. As comparações
ativas usam `associate_map_points` para os braços `legacy_cells` (amostra do
viewer) e `dense_cells` (mesma regra com `visibility_geometry` integral).
Esses braços mantêm células 3×3, profundidade axial e exclusividade por pixel
para isolar o efeito de trocar o oclusor e o algoritmo. Não são fallback
implícito para ausência do PCD no caminho de superfícies.

## Visibilidade e vínculo da região

A visibilidade consulta a primeira interseção positiva do raio óptico com
patches orientados e limitados, construídos em torno de medições reais. O
modelo contém suporte local, não planos infinitos. Não usa um alcance máximo
por objeto nem regras específicas de `pallet`, `window` ou outros labels.

Uma primeira superfície compatível autoriza RGB. Uma superfície anterior
produz `occluded`; ausência de patch ou incompatibilidade sem oclusor
confirmado produz `visibility_unconfirmed`. Rejeições não carregam cor ou
label, mas podem registrar `SurfaceAssociationEvidence`, com distâncias radiais
em metros, tolerância, índice original do PCD e motivo da decisão.

A identidade do objeto exige uma decisão adicional. Componentes medidos
sob a máscara precisam de continuidade local e âncoras geométricas no
contorno. O único retorno ou o componente dominante não ganha o label por
esse motivo. Um componente sem âncora conserva `tentative_label` e
`surface_unsupported`, mesmo quando sua cor RGB é válida.

Em uma janela, a moldura medida pode receber o label. A paisagem visível
através da abertura permanece RGB sem herdar a identidade da janela. Se não
há suporte medido da janela, o claim e sua máscara permanecem como evidência
2D: o algoritmo não cria pontos, vidro ou planos virtuais. A mesma regra vale
para portas abertas e outras aberturas, independentemente do nome do claim.

## Invariantes de fronteira

- XYZ em metros; candidatos e PCD completo no mesmo `map_id` e frame;
- raios unitários no frame óptico da observação RGB;
- pixels top-left, inteiros por arredondamento determinístico;
- transform de câmera, `clock_id`, calibração e tolerância RGB/LiDAR validados;
- ausência de suporte gera abstenção; não há associação forte de fallback;
- cor visível, suporte semântico e corroboração temporal são estados distintos.

A leitura integral e a validação de SHA-256 pertencem a
[`geometric-map`](../geometric-map/README.md). Algoritmo, parâmetros, resultados
e limitações estão em [visibilidade medida](docs/measured-visibility.md) e na
[validação do trecho](docs/visibility-validation.md).

## Footprint semântico e boundary

`VisualRegionEvidence.pixels` representa a máscara após grounding, clipping à
área utilizável e ownership. `grounding_status="refined"` e uma
`grounding_reference` identificam a geometria que autoriza o conceito. O default
é `grounding_unavailable`; preencher apenas `label` não publica associação forte.

`PointVisualAssociation` separa visibilidade de força semântica. Um ponto visível
continua com cor RGB mesmo se `semantic_status` for `boundary`, `outside` ou
`ungrounded` ou `surface_unsupported`. No caminho de superfícies, `label`
exige interior grounded e componente 3D sustentado; `tentative_label`
preserva a hipótese em boundary ou footprint não grounded. A ablação histórica
exige tanto `legacy_discovery` na evidência quanto `allow_legacy_discovery` na política.

`distance_to_mask_boundary` mede distância euclidiana aos centros dos pixels
fronteiriços. A fronteira tem ao menos um vizinho 8 fora da máscara; buracos e
bordas do frame contam. A distância é zero nesses pixels e no exterior, que é
distinguido pela pertença à máscara. A máscara original não é alterada.

`BoundaryPolicy.margin_px` soma o raio máximo de arredondamento de dois eixos,
`sqrt(0.5)` pixels, a `sigma_multiplier * registration_sigma_px`. O default de
sigma zero declara ausência de estimativa adicional; não significa calibração
ou pose perfeitas. A composição deve fornecer a incerteza de registro medida
quando disponível. Pontos com distância menor ou igual à margem ficam tentativos.
`enabled=False` permite medir somente o efeito de grounding.

Diagnostics preservam margem, componentes de incerteza, contagens de boundary
e interior. Cada associação preserva distância, margem e referência ao grounding.
O buffer de oclusão, projeção MEI e tolerância RGB/LiDAR continuam independentes:
eles verificam correspondência geométrica, não a identidade dos pixels da máscara.
