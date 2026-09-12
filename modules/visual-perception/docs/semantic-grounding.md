# Grounding semântico e geometria de discovery

Resultado medido: [validação de 6 segundos e limites de aceitação](semantic-grounding-validation.md).

## Diagnóstico anterior à implementação

Inspeção confirmada no HEAD `3596e90f975472d3a521d59005e556c7376f044c`
(a primeira inspeção havia ocorrido em `9a2a421`; os pontos abaixo permanecem
iguais no HEAD atual):

- `region_semantics.interpret_regions` usa `replace(region, claims=...)`.
  Identidade, máscara, box, confiança geométrica e proposals permanecem iguais.
  Uma classificação semântica não melhora o grounding espacial.
- `refinement.refine_observation` oferece novas views ao reasoner; é refinamento
  de interpretação, sem segmentação condicionada ao conceito.
- `region_merge._build_region` une por OR as proposals de um grupo transitivo
  de IoU/containment mútua. Isso pode ampliar a extensão, mas não explica sozinho
  o erro: no artifact `20260911T024436Z`, `corridor-02-04288` contém uma proposta
  única de 25.614 pixels chamada `wooden pallet`; `corridor-02-04246` contém uma
  proposta única de 24.252 pixels chamada `door`. O merge foi preservado.
- `mapping_runtime.corridor02_context._region_evidence` promove `region.mask`
  a `VisualRegionEvidence.pixels`. `mask & (ownership < 0)` pode fragmentá-la;
  não havia validação de componentes e todos os restos herdavam o label.
- `sensor_association.projector` resolve visibilidade e consulta
  `region_by_pixel[pixel]`. A pertença à máscara bastava para copiar o label.
  MEI, hemisfério frontal, valid area e tolerância RGB/LiDAR são proteções
  geométricas independentes. O buffer de oclusão não verifica a máscara.
- A pose usa `_nearest_pose`. Deslocamentos temporais podem afetar fronteiras,
  mas não explicam uma máscara que já cobre parede na imagem.
- `semantic_fusion` escolhe a contribuição por confiança calibrada/bruta,
  depois `visual_support` e `region_quality`. Concordância é medida após a
  escolha; não impede uma observação espacialmente errada de vencer.

## Decisão

O dono de discovery, reconhecimento e grounding 2D é `visual-perception`.
São operações diferentes. A geometria de discovery permanece em `mask` e `box`;
`grounding` registra conceito, máscara do segmentador, máscara semântica aceita,
prompt espacial, proveniência, falhas e medidas. O schema v5 lê v1–v4 sem
inventar grounding para artifacts antigos.

O grounding roda após a seleção das candidatas a publicação contextual, antes
da associação. Contexto estrutural, claims, relações e embeddings de discovery
permanecem auditáveis. Embeddings antigos não passam a descrever a nova máscara.

A menor composição disponível com localização condicionada ao conceito é um
detector de vocabulário aberto seguido de SAM promptado por box. Reutilizamos
`IDEA-Research/grounding-dino-base` e `facebook/sam-vit-huge`, completos no cache
local; Transformers, torch e scipy já estão no ambiente. Não se adiciona SAM2.
Repetir SAM com a mesma box ou escolher pontos só pela discovery não fornece
evidência de que o suporte pertence ao conceito. O VLM não produz geometria.

Referências primárias: [Grounding DINO no Transformers](https://huggingface.co/docs/transformers/model_doc/grounding-dino)
e [SAM no Transformers](https://huggingface.co/docs/transformers/model_doc/sam).
O primeiro localiza texto; o segundo segmenta prompts espaciais. Isso constitui
evidência de grounding, não garantia de acurácia nem confiança calibrada.

O lifecycle compartilhado libera o detector antes de carregar SAM. Localização
é compartilhada entre regiões com o mesmo conceito no frame, e o embedding SAM
da imagem é reutilizado entre boxes. Não há tracking ou aumento de frames.

## Políticas de fronteira

Somente máscaras explicitamente grounded autorizam associação forte. Ausência,
falha, ambiguidade e inconsistência mantêm a claim e um diagnóstico. Não há
fallback forte para discovery. Um fallback opcional de `stuff` é apenas
tentativo, identificado como tal. `thing`, `part` e natureza desconhecida não
herdam esse fallback.

Após clipping à discovery e área utilizável, e novamente após ownership, a
conectividade é validada. Para regiões contáveis, só o componente que contém o
suporte espacial principal do grounding pode permanecer forte. Se esse suporte
for removido ou não for único, a região se abstém. Área não elege o componente.
`stuff` pode manter suporte extenso e descontínuo; nenhuma lista de labels
controla essa política.

`sensor-association` é dono da distância euclidiana à fronteira e da margem
de incerteza em pixels. Cor RGB/visibilidade e força da associação semântica
serão estados separados. Uma associação de boundary permanece auditável, sem
participar como contribuição forte na fusão.

Interpolação temporal de translação e quaternion é uma alteração separada da
composição, com os dois timestamps de suporte e fator de interpolação. Nearest
permanece selecionável para comparação. Não se extrapola silenciosamente.


## API e invariantes

- `GroundingRequest`: `region_id`, conceito primário, `RegionKind` e máscara de
  discovery na resolução RGB original.
- `SemanticGrounder.ground`: port substituível que devolve uma predição por
  request; não altera o label ou a região.
- `GroundingPrediction`: máscara bruta, todas as boxes promptadas e sua extensão,
  scores do detector/SAM, proveniência, custo e falha. Boxes usam xyxy semiaberto;
  masks são booleanas H×W, no frame top-left da mesma imagem.
- `SemanticGrounding`: predição e máscara aceita, suporte principal, status e
  diagnostics. A máscara aceita é subconjunto da saída do segmentador, das
  localizações, de discovery e da área utilizável. A interseção é conservadora:
  esta versão não recupera partes do objeto ausentes em discovery.
- `ground_regions`: chamado após `partition_observation`, antes da associação.
- `build_spatial_footprints`: resolve ownership e revalida componentes. Retorna
  também regiões sem footprint. Empates de área preservam o desempate histórico
  por confiança e id; confiança nunca substitui a exigência de grounding.

O centro da box localizada orienta a escolha de suporte no segmento SAM. Entre
pixels à mesma distância, componentes distintos causam abstenção. O suporte é
fixado antes de clipping/ownership: se desaparece, não migra para outro resto.
Essa é uma aproximação explícita de suporte principal, não object tracking nem
prova de identidade. Componentes desconectados de um objeto parcialmente ocluído
podem ser conservadoramente excluídos; `stuff` pode preservar descontinuidade.

Discovery não se torna ground truth: um detector ou SAM incorreto ainda pode
produzir grounding incorreto. Há referência ao método e à geometria, não promessa
de segmentação perfeita. Conceitos abstratos e danos sutis podem não ser
localizados pelo detector escolhido. Suas claims, relações e contexto estrutural
continuam disponíveis mesmo quando o footprint espacial se abstém.

## Configuração e custo

`SemanticGroundingConfig.backend` é `unavailable` em configurações mínimas/fakes;
o perfil real de pesquisa seleciona `grounded_sam`. Thresholds de detecção 0,4 e
texto 0,3 seguem o exemplo oficial citado acima. O IoU 0,85 remove apenas boxes
quase duplicadas da mesma consulta, antes de testar ambiguidade. Não há blacklist,
limiar de área máxima ou parâmetro escolhido por conceito/frame. Área mínima
um pixel só rejeita vazios; valores maiores precisam de justificativa experimental.

Para uma região contável, múltiplas boxes distintas que sobrepõem discovery
causam abstenção; não se escolhe a mais conveniente por score. Para `stuff`,
segmentos de localizações independentes podem ser unidos, mas cada box original
é preservada e pixels fora de todas elas são rejeitados.

Por frame, o detector roda uma vez por conceito distinto. SAM computa um embedding
de imagem compartilhado e uma decodificação por request localizado; várias boxes
de stuff são decodificadas juntas. `model_calls` conta inferências (incluindo o
embedding); a primeira região cobra o trabalho compartilhado uma única vez.
A latência regional também aloca carregamento e embedding ao primeiro consumidor;
a medida por frame é a referência para comparação de performance. O lifecycle
registra pico CUDA em bytes e libera modelos inclusive após falha de carregamento.

## Diagnostics e reprodução

O JSON v5 preserva as duas máscaras e a saída bruta do modelo. `DEBUG/*-grounding.json`
e os metadados de composição expõem áreas, redução, boxes, componentes, suporte,
`region_kind`, conceito, confiança geométrica e `visual_support`, status e razão
de falha. Depois de ownership, também preservam componentes residuais, pixels
retirados e máscara final. `boundary_pixel_count`, `safe_interior_pixel_count` e
`safe_interior_ratio` são calculados pela mesma política da associação.

A comparação separa:

| Braço | Footprint | Boundary | Pose |
|---|---|---|---|
| before | discovery histórica | desativada | nearest |
| grounding | máscara semântica | desativada | nearest |
| boundary | máscara semântica | ativada | nearest |
| pose | máscara semântica | ativada | interpolada |

Reconhecimento, contexto, geometria persistente, calibração, valid area, exclusão
de ego e política de oclusão são compartilhados entre braços. As métricas de
suporte espacial/multi-view são diagnósticas; não alteram máscaras nem escondem
pontos na comparação. A tabela não mede uma interação fatorial completa entre
pose e grounding; mede os incrementos declarados na mesma evidência congelada.

O experimento é `experiments/visual_perception_experiments/grounding_validation.py`.
Ele aceita `--visual-run`, `--window`, `--geometry`, `--bag`, `--intrinsics`,
`--extrinsics`, `--odometry`, `--output` e `--review-frame` repetível. Recusa janelas
maiores que dez segundos e diretórios de saída existentes. Não executa FAST-LIO,
SAM discovery, VLM ou tracking novamente. O manifest registra HEAD, versões de
bibliotecas, snapshots de checkpoint, hashes dos artifacts de reconhecimento,
custo e contagens de auditoria por frame. Regiões abstidas continuam nos resultados.
