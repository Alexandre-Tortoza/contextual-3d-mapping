# Point Representation

`point-representation` converte nuvens de pontos LiDAR em embeddings 3D aprendidos por
ponto, para uso como suporte de coerência geométrica/semântica na fusão (ver o roadmap de
contexto, Fase 1 e 2) — não como substituto de labels textuais.

## Estado atual

Implementados até aqui: os contracts públicos de entrada e saída (`PointCloud`,
`PointEmbedding`, `BatchedPointEmbeddings`, `PointTrainingSample`), a validação de
fronteira de nuvens de pontos, a seleção configurável de canais de entrada do encoder
(`FeatureSelectionConfig`/`select_features`), e os três transforms de pré-processamento
determinísticos: recorte espacial (`crop_points`), downsampling por voxel
(`voxel_downsample`) e normalização reversível de coordenadas
(`normalize_coordinates`/`denormalize_coordinates`).

Também implementados: rastreamento de lineage de índice composto entre transforms
encadeados (`PointLineage`/`compose_lineage`) e colagem de nuvens de tamanho variável em
lote (`collate_point_clouds`).

Também implementado: o port `PointEncoder` (contract framework-agnóstico de encoder,
com um fake determinístico para testes) e a colagem de entradas de encoder por amostra
em lote (`collate_encoder_inputs`).

Ainda não implementados: qualquer backbone concreto, o pipeline de treino, e a API
pública de inferência (`encode`). Ver o roadmap de contexto, Fase 1.

## Contracts principais

- `PointCloud`: entrada canônica — coordenadas obrigatórias mais canais auxiliares
  opcionais por nome (cor, intensidade, normais, ...).
- `PointTrainingSample`: `PointCloud` mais o que só existe durante treino — features de
  professor alinhadas a ponto e índices de correspondência com a amostra de origem.
- `PointEmbedding`/`BatchedPointEmbeddings`: saída pública — um embedding por ponto, com
  validade e confiança explícitas, agrupado por amostra em lote.
- `EncoderInput`: entrada montada para um encoder concreto, com coordenadas e features
  aprendidas mantidas separadas.
- `AxisAlignedBounds`/`CropConfig`: região e política de recorte, com subsample
  determinístico opcional via `seed`.
- `VoxelDownsampleConfig`: tamanho de voxel para reduzir densidade; cada ponto de saída
  preserva os índices de origem que o compõem (política de centróide).
- `NormalizationConfig`/`NormalizationTransform`: centralização e escala configuráveis,
  com fallback seguro para nuvens sem extensão (ponto único ou pontos duplicados) e
  reversão exata via `denormalize_coordinates`.
- `PointLineage`: mapeamento de cada ponto de saída de um transform para os índices de
  entrada que o originaram (um-para-um ou muitos-para-um); `compose_lineage` encadeia dois
  estágios sem que o chamador precise juntar os grupos manualmente.
- `PointBatch`/`collate_point_clouds`: lote de amostras de tamanho variável concatenadas,
  com offsets estilo CSR e lineage por amostra preservados; `resolve_source` recupera a
  amostra e o ponto local de origem de qualquer linha do lote.
- `EncoderInputBatch`/`EncoderOutput`: entrada e saída batched do port `PointEncoder`; a
  saída nunca assume correspondência 1:1 com a entrada — carrega seu próprio
  `PointLineage` de volta aos pontos de entrada, para acomodar um backbone que mescle
  pontos internamente.
- `PointEncoder` (`ports/point_encoder.py`): Protocol framework-agnóstico de encoder 3D,
  com `embedding_dimension` e `required_channels` (vazio declara suporte
  somente-coordenadas) expostos antes de qualquer chamada a `encode`.

Nenhum destes tipos importa um framework de tensor específico ou acopla a um backbone
concreto — isso fica atrás de adapters que implementam `PointEncoder` (o primeiro
backbone concreto ainda não foi implementado).
