# Glossário

Este glossário reúne o vocabulário usado pelo módulo `visual-perception`. Os termos
descrevem os contracts e o fluxo do pipeline canônico; não definem conceitos de outros
módulos, como associação com LiDAR ou persistência de mapas.

## Termos-chave e conceitos

### Pooling

Operação que agrega vários vetores de features em um único vetor de tamanho fixo. Neste
módulo, o pooling recebe a `Mask` de uma região e um `FeatureMap` denso, seleciona as
features cobertas pela região, calcula a média por dimensão e normaliza o resultado com
norma L2. A saída é um `VisualEmbedding` que resume o conteúdo visual da região.

O pipeline canônico usa `pixel_nearest_highres`: para cada pixel da máscara, seleciona a
feature da célula mais próxima no grid. O método preserva regiões menores que uma célula
do grid. A baseline `patch_grid_baseline` considera somente células cujo centro cai
dentro da máscara e, por isso, pode não representar regiões pequenas.

### Pooling mask-aware

Pooling que usa a máscara da região, e não apenas sua box, para decidir quais features
participam da agregação. Isso reduz a contribuição de pixels do fundo ou de objetos
vizinhos incluídos pela área retangular da box.

### Feature visual densa

Vetor produzido para cada célula espacial de um grid derivado da imagem. O conjunto
desses vetores forma um `FeatureMap`, que preserva informação espacial antes do pooling.
Uma feature ainda está associada a uma posição; um embedding de região já representa a
região inteira.

### Embedding

Vetor numérico que representa propriedades relevantes de uma entrada em um espaço
vetorial. Vetores do mesmo espaço podem ser comparados por métricas compatíveis, como
similaridade de cosseno. Dimensão, modelo produtor, normalização e proveniência fazem
parte da interpretação do vetor.

### Embedding visual

Representação vetorial de uma região obtida pelo pooling de features visuais densas. O
tipo `VisualEmbedding` registra a região de origem, o método e a resolução de pooling, o
modelo produtor, a dimensão e o estado de normalização.

### Embedding alinhado à linguagem

Representação de uma região em um espaço vetorial compatível com texto, produzida por um
encoder próprio. `LanguageEmbedding` e `VisualEmbedding` são contracts separados porque
podem pertencer a espaços incompatíveis e não devem ser comparados sem uma garantia
explícita de alinhamento.

### Referência de embedding

Identificador armazenado em `ObservedRegion` que aponta para um embedding produzido para
aquela região. `visual_embedding_ref` e `language_embedding_ref` mantêm explícita a
diferença entre os dois espaços vetoriais sem duplicar os vetores dentro da região.

### Atualização imutável

Criação de uma nova instância com os campos desejados, preservando a instância original.
Como `ObservedRegion` é um dataclass congelado, `_attach_visual_refs` usa
`dataclasses.replace` para devolver novas regiões com `visual_embedding_ref`, sem mutar
as regiões recebidas. Isso torna as transições entre estágios explícitas e evita efeitos
colaterais sobre dados já compartilhados.

### Região observada

Região canônica da imagem depois do merge de propostas. `ObservedRegion` contém máscara,
box, confiança geométrica, propostas contribuintes, claims semânticos e referências para
seus embeddings. É a unidade estável consumida pelos estágios seguintes e por módulos
downstream.

### Mask e box

Uma `Mask` indica, pixel a pixel, a forma ocupada por uma região. Uma `BoundingBox` é o
retângulo mínimo usado para delimitar essa região. A mask oferece contorno preciso; a box
é uma aproximação retangular mais simples. Ambas seguem as convenções de coordenadas
definidas em [architecture.md](architecture.md#invariantes-de-coordenadas).

### Normalização L2

Divisão de um vetor por sua norma Euclidiana para que seu comprimento seja igual a um.
O pooling normaliza o vetor resultante, permitindo que comparações por similaridade
reflitam principalmente a direção do embedding, e não sua magnitude original.

### Tiling multi-scale

Divisão da imagem em tiles processados em uma ou mais escalas. O procedimento ajuda a
descobrir regiões que seriam pequenas na imagem completa. As propostas locais são
remapeadas para as coordenadas globais antes do merge.

### Merge de regiões

Consolidação de propostas sobrepostas ou equivalentes, possivelmente originadas de
tiles e escalas diferentes, em regiões canônicas. Os identificadores das propostas
contribuintes são preservados para rastreabilidade.

### Pipeline canônico

Sequência oficial de estágios coordenada por `run_canonical_pipeline`, desde a descoberta
de regiões até o audit de qualidade. “Canônico” significa que esse é o caminho de
produção suportado pelo módulo, em contraste com baselines legadas ou capacidades
opcionais executadas sobre sua saída.

### Port e backend

Um port declara o comportamento de que o pipeline precisa; um backend fornece uma
implementação concreta desse comportamento. Essa separação permite substituir modelos
reais por fakes em testes ou comparar modelos em benchmarks sem alterar a orquestração
do pipeline.

### Claim semântico

Afirmação semântica auditável sobre uma região ou cena, acompanhada de confiança e
proveniência. O módulo preserva claims como evidências independentes em vez de reduzir
cada região a uma única label definitiva.

### Proveniência

Metadados que permitem identificar de onde um resultado veio e como foi produzido. Para
embeddings, isso inclui informações como região, modelo, checkpoint, método de pooling e
resolução de features, conforme o contract específico.

### Audit de qualidade

Último estágio do pipeline canônico, responsável por verificar a consistência estrutural
da `VisualObservation` e registrar problemas detectados sem ocultar a saída que foi
avaliada.

## Relação entre os conceitos

No trecho de pooling do pipeline, o `DenseFeatureExtractor` produz um `FeatureMap` para a
imagem. `pool_regions` combina esse mapa com a `Mask` de cada `ObservedRegion`, gera um
`VisualEmbedding` normalizado e associa cada `region_id` ao respectivo `embedding_id`.
Por fim, `_attach_visual_refs` cria novas regiões contendo essas referências. Assim, o
vetor, sua proveniência e a região permanecem entidades distintas, mas rastreáveis entre
si.

O fluxo pode ser resumido como:

```text
imagem -> FeatureMap + regiões -> pooling mask-aware -> VisualEmbedding
                                              |
                                              -> referência anexada à ObservedRegion
```
