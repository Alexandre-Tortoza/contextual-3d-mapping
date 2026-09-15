# Visibilidade sobre superfícies medidas

## Ownership e dados de entrada

`geometric-map.read_pcd_geometry` fornece a nuvem integral em metros, seu frame,
mapa, índices originais e digest. `MeasuredSurfaceModel` pertence a
`sensor-association` e constrói uma representação local apenas para consultar
visibilidade. Não modifica o mapa nem acrescenta geometria persistente.

`associate_measured_map_points` recebe os candidatos de renderização
separadamente. O modelo completo e o raster da máscara independem desses
candidatos; dois subconjuntos do viewer produzem a mesma decisão para cada ID
comum. A correspondência temporal é validada contra a observação LiDAR usada
como âncora, e o transform mapa→câmera corresponde ao instante RGB.

## Patches locais

A PCA usa os 16 vizinhos mais próximos de cada medição. Vizinhanças degeneradas
ou com resíduos excessivos não produzem patch. O disco fica centrado na
medição original, com orientação dada pelo menor autovetor. Seu raio é o menor
entre 0,55 vezes a distância ao 16º vizinho, a distância ao terceiro vizinho
não central e 0,15 m. O limite do terceiro vizinho evita ampliar patches de
pontos isolados usando a densidade de uma superfície distante.

A fração máxima do menor autovalor na variância total é 0,12 e o resíduo RMS
máximo é 0,02 m. Esses parâmetros toleram ruído local; não são uma estimativa
de acurácia do mapa. A versão inicial com fração 0,03 se absteve em muitos
pontos próximos de superfícies ruidosas. Os limites finais ficam registrados
em `SurfaceVisibilityConfig` e no manifest de cada experimento.

O cálculo usa blocos de 16.384 medições para limitar memória. O modelo é
construído uma vez por mapa; cada câmera transforma os centros e as normais.

## Primeira interseção

O índice da câmera agrupa discos por abertura angular. Cada consulta considera
todos os discos cujo cone pode conter o raio; não restringe a busca a um
número fixo de vizinhos angulares, que poderia favorecer um fundo denso.

Para centro `c`, normal unitária `n` e raio unitário `r`, a interseção usa
`t = dot(n, c) / dot(n, r)`. Só há hit para `t > 0`, incidência estável e
interseção dentro do disco. O menor `t` é a primeira superfície medida. A
orientação impede que uma superfície inclinada se oclua por comparação com a
profundidade de pixels vizinhos.

Incidência com cosseno absoluto menor que 0,02 não sustenta interseção. A
compatibilidade do ponto usa 0,03 m mais três vezes o resíduo RMS do plano
dividido pelo cosseno absoluto da incidência. Esse termo explicita o aumento
de incerteza em incidências rasantes. Discos não sustentados, raios inválidos
ou pontos anteriores ao suporte disponível produzem abstenção.

RGB é amostrado no pixel arredondado, mas a visibilidade usa o raio contínuo do
próprio ponto. O vínculo semântico consulta também a superfície do pixel e
reavalia seu plano sobre o raio contínuo, evitando penalizar apenas a variação
de profundidade por arredondamento em superfícies inclinadas.

## Componentes e âncoras da região

O raster usa raios calibrados de pinhole, fisheye equidistante ou MEI. Somente
as máscaras e uma faixa externa de três pixels precisam ser consultadas.
A inversão MEI verifica o resíduo da distorção antes de aceitar o raio.

Dentro da máscara, vizinhos de imagem integram o mesmo componente quando os
planos locais concordam: separação normal até 0,10 m e cosseno absoluto entre
normais de pelo menos 0,85. Esse teste usa separação em relação ao plano,
preservando continuidade de uma superfície inclinada.

Uma âncora compara a superfície interna ao plano medido logo fora do contorno,
projetado sobre o mesmo raio. O componente precisa de pelo menos quatro
âncoras, 75% favoráveis e pelo menos quatro sinais de contraste geométrico
com o exterior ou com componentes interiores mais profundos. A posição
projetada do plano externo é somente uma referência de comparação; não cria
um plano na abertura.

Componentes atrás do contorno ou sem âncora ficam incertos. O componente
mais próximo, mais numeroso ou único não ganha automaticamente a identidade
da região. A máscara e o texto do claim não são alterados.

## Janela e limites conhecidos

Uma moldura medida com suporte de contorno pode receber `window`. Pontos do
fundo vistos pela abertura continuam visíveis, porém não recebem esse label.
Sem moldura ou outra superfície medida sustentada, a janela permanece como
claim 2D. A aplicação conserva a máscara e a hipótese para inspeção.

A geometria sozinha não prova identidade semântica. Uma superfície coplanar,
uma porta reentrante, uma região parcialmente ocluída ou componentes
fragmentados podem perder o label por falta de âncora. Objetos transparentes
não observados pelo LiDAR não são reconstruídos. Ruído, desalinhamento e
superfícies dinâmicas acumuladas no mapa podem provocar oclusões falsas.

Os patches são uma aproximação local e podem cobrir espaços menores que seu
raio; não existe garantia universal de preservação de toda abertura subamostrada.
Incidências rasantes aumentam a incerteza de profundidade. Esses limites devem
ser avaliados junto da retenção de superfícies próximas, e não apenas pela
remoção de labels distantes. A validação do corridor-02 mostra essa troca em
[visibility-validation.md](visibility-validation.md).
