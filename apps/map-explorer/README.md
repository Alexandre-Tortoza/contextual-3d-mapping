# Map Explorer

`map-explorer` é a principal aplicação voltada a humanos para abrir, consultar e inspecionar visualmente mapas persistentes.

## Responsabilidades

- abrir um mapa através de contracts de aplicação estáveis;
- renderizar geometria 3D e overlays semânticos;
- submeter consultas semânticas, espaciais, relacionais e contextuais através do `query-engine`;
- focar o viewer nas entidades ou regiões retornadas;
- expor observações relacionadas, evidência, confiança e proveniência;
- visualizar relações de scene-graph sem possuir a construção do grafo.

## Estrutura inicial

```text
map-explorer/
├── README.md
├── api/
├── web/
└── configs/
```

A API e o frontend são camadas de entrega. Não devem depender diretamente de implementações privadas de módulo nem de schemas específicos de armazenamento.
`scripts/publish_map_index.py` é dono da publicação estática de runs, consumida
pela CLI, pelo Makefile e pelo frontend.

## Execução local

Instale dependências de forma determinística, produza o bundle e inicie o
servidor local:

```bash
make map-explorer-install map-explorer-test map-explorer-build
cd apps/map-explorer/web
npm run dev
```

O viewer valida a versão e os campos mínimos do artifact antes de renderizar.
O catálogo atual contém somente mapas com contexto
(`artifact_type: contextual_rgb_lidar_slice`, schema 2). A geometria é desenhada
como referência para os overlays; mapas de geometria pura não entram no seletor.

Publique um resultado existente e abra o viewer pela CLI:

```bash
apps/cli/.venv/bin/contextual-3d-mapping-cli publish \
  --artifact artifacts/grounding-validation/pose.json \
  --run-id context-run \
  --label 'Minha run com grounding'
apps/cli/.venv/bin/contextual-3d-mapping-cli serve --run-id context-run
```

Também é possível publicar e servir usando Make:

```bash
make map-explorer-serve \
  MAP_EXPLORER_ARTIFACT=artifacts/grounding-validation/pose.json \
  MAP_EXPLORER_RUN_ID=context-run
```

Sem `MAP_EXPLORER_ARTIFACT`, Make apenas atualiza o catálogo existente e serve
o viewer. `make map-explorer-publish` publica sem iniciar o servidor.

Acesse `http://localhost:5173` ou, em uma sessão SSH,
`http://<ip-do-servidor>:5173`. A primeira abertura seleciona a publicação mais
recente. O parâmetro `?artifact=/runs/<run-id>/context.json` fixa uma versão.
Sem catálogo, `/current-map.json` permanece como fallback para mapas contextuais
históricos; ele deixa de ser o destino das novas publicações.

## Pastas de runs

Cada publicação possui uma pasta independente:

```text
web/public/
├── maps/index.json
└── runs/<run-id>/
    ├── context.json
    ├── manifest.json
    └── <pastas de previews referenciadas pelo mapa>/
```

O JSON e as URLs relativas são preservados. Todas as imagens originais e
overlays referenciados são copiados para a pasta da run. A publicação rejeita
previews ausentes, externos ou fora da pasta de origem. O manifest registra
identidade, nome, instante UTC da publicação (`created_at`), origem do artifact,
frame espacial, quantidade de frames e pontos com contexto, além dos hashes
SHA-256 do mapa e das imagens. O instante da publicação não representa o início
da inferência.

O fingerprint inclui o mapa e seus previews. A mesma publicação é idempotente;
usar uma identidade existente para outro conteúdo produz erro. A pasta só entra
no catálogo depois que a cópia completa termina. Os resultados ficam ignorados
pelo Git e a exclusão da origem não quebra os previews já publicados.

`public/maps/index.json` é a fronteira de leitura compartilhada com a CLI:
cada entrada expõe `run_id`, `url`, `label`, `artifact_type`, `created_at`,
`map_id`, `frame_count` e `contextual_point_count`, ordenadas pela publicação
mais recente. O publisher aceita `--result-file` para entregar à CLI os
metadados da pasta efetivamente salva ou reutilizada.

O seletor **Run** alterna entre essas entradas. O catálogo é atualizado a cada
30 segundos e quando a janela recupera foco, preservando a seleção atual.
Ao alternar versões com o mesmo frame, origem geométrica e quantidade de pontos,
a câmera permanece na posição escolhida. Uma geometria diferente é enquadrada
automaticamente. A comparação ocorre pela alternância no mesmo viewport.

## Mapas consolidados

Depois de cada publicação, o catálogo reconstrói automaticamente um mapa
consolidado para cada grupo de runs sobre a mesma nuvem de origem (mesmo PCD,
mapa e frame). Esse artifact é derivado das pastas imutáveis em `runs/`: não
altera nenhuma run de origem e preserva a identidade, o hash, as observações e
as regiões que contribuíram.

Uma run de trecho carrega só a vizinhança das suas poses. Quando ela declara
`geometry_backdrop`, o publisher copia esse slice global uma vez para
`maps/geometry/<sha256>.json`, depois de conferir o digest e a origem. O
consolidado usa esse fundo como geometria do mapa geral e acrescenta apenas os
pontos com contexto de cada run.

O seletor separa **Runs** de **Mapas consolidados**. Cada ponto consolidado
recebe no máximo um voto por run; variantes como `pallet` e `wooden pallet`
formam a mesma família textual. Uma maioria estrita decide o label. Quando não
há maioria, o claim de maior qualidade vence, mas aparece atenuado e identificado
como desempate por qualidade no inspector. O painel permite abrir a evidência de
cada run contribuinte sem duplicar suas previews.

Runs com `map_id` igual, mas pontos ou coordenadas diferentes, não são unidas.
O viewer não faz registro espacial entre aquisições distintas.

## Cores e sustentação

O reasoner produz vocabulário aberto e cheio de quase-sinônimos: `wall`,
`plain wall`, `wall tiles` e `arch-shaped wall` descrevem a mesma superfície. O
viewer agrupa labels em famílias derivadas dos próprios dados — um label entra
na família do label mais frequente que aparece inteiro dentro dele — e colore
por família, com a legenda expansível nos labels brutos.

São **quatro matizes, e nenhum a mais**. Uma nuvem de pontos é um caso
"all-pairs": qualquer classe pode encostar em qualquer outra no espaço, então
toda combinação precisa ser distinguível, e não apenas as vizinhas de uma
legenda ordenada. Sob essa restrição, contra o fundo deste viewer, só dois
conjuntos de quatro matizes passam nos limiares de separação para visão normal e
para daltonismo, e nenhum conjunto de cinco passa. As famílias além da quarta
compartilham um neutro claro em uma linha `Outros labels`, em vez de receberem
matizes que o leitor não conseguiria separar.

A regra de agrupamento é por token, então variantes morfológicas de uma mesma
raiz (`flooring` contra `floor`) formam famílias separadas. É o limite aceito
para não embutir morfologia no viewer; a linha `Outros labels` permanece
expansível e isolável.

Pontos cujo label não se sustenta aparecem apagados. O artifact publica dois
sinais independentes por ponto — concordância entre keyframes e suporte da
vizinhança geométrica — e a legenda oferece atenuar quem falha nos dois
(ligado por default) e quem foi visto por um único frame (desligado). Nenhum
claim é reescrito: a atenuação é decisão de desenho, e o inspector mostra os
dois números e o estado.

Artifacts contextuais expõem geometria, associações RGB e claims do VLM. A
legenda separa a **evidência contextual publicada**, colorida por família, da
**cobertura visual**. Nesta última, pontos vistos sem evidência publicada
aparecem em cinza neutro e pontos que nenhum keyframe alcançou permanecem em
cinza discreto. A distinção evita tratar a ausência legítima de evidência
contextual como falha de classificação e mantém a falta de cobertura visível.
Ao selecionar um ponto observado, o painel mostra o frame de origem, o pixel
projetado, a região, a confiança, o estado de suporte e a proveniência. O frame
pode ser ampliado em um modal e alternado entre a imagem original, sem qualquer
marcação, e o overlay das regiões VLM.

Um ponto sem label declara o motivo no painel de detalhes — fora do campo de
visão, projetado fora da imagem, fora do suporte óptico válido, ou ocluído por
uma superfície mais próxima. Quando várias observações classificam o mesmo ponto,
o painel mostra o label primário, quantas observações contribuíram e a
concordância entre elas.

## Navegação

O modo `Explorar` usa botão esquerdo para orbitar, botão direito ou central
para mover câmera e alvo, e scroll para aproximar no cursor. `Modo voo` adiciona
`W/A/S/D` para movimento horizontal, `Q/E` para altura e `Shift` para acelerar.
Seleção não move a câmera; duplo clique ou `F` foca o ponto. `Home` restaura a
vista geral, enquanto `1` e `2` abrem as vistas superior e isométrica.

O clique seleciona o ponto mais próximo do cursor, e não o mais próximo da
câmera. A distinção importa: o raycaster do Three.js ordena interseções por
distância à câmera, então sem essa regra — e com o raio de captura default de um
metro — um vizinho à frente vence o ponto apontado. O raio de captura acompanha
a escala do mapa.

A sidebar `Detalhes` concentra a inspeção. A legenda contextual fica em um
painel pequeno e recolhível no canto inferior esquerdo do mapa e altera o foco
sem recarregar a geometria: a classe em foco fica com cor plena e o restante do
mapa continua desenhado, atenuado, como referência espacial. Pontos atenuados
seguem selecionáveis. O frontend consome a nuvem por uma
fronteira interna compatível com bounds e LOD futuros; o artifact local atual
continua sendo servido como um único chunk estático.

Cada linha da legenda carrega quatro leituras. O swatch diz ao mesmo tempo a cor
da classe e o estado do filtro — cheio em foco, vazado oculto, meio a meio para
a família cujos labels estão parcialmente em foco. A contagem vem acompanhada de
uma barra de proporção comparada **dentro da própria seção**, já que contra a
cobertura visual toda classe publicada viraria um traço invisível. O contador de
labels abre a família, e o alvo isola a linha. O cabeçalho resume o recorte
atual — quantos pontos estão em foco, quantas classes estão ocultas — e oferece
`Padrão`, que volta ao recorte de abertura com as estruturas genéricas ocultas,
e `Tudo`, que traz todas as classes de volta. Cada botão fica desabilitado
quando já é o estado atual.

Nomes de classe cortados pela largura do painel e ações representadas só por
ícone abrem uma dica ao passar o ponteiro ou ao receber foco de teclado. A dica
é medida a partir do próprio elemento: um nome que coube inteiro não ganha
tooltip redundante.

## Inspeção de visibilidade

Artifacts com `surface_evidence` mostram distância do ponto, distância da
primeira superfície medida, tolerância e motivo. O vínculo do rótulo aparece
separadamente da visibilidade RGB e da corroboração entre frames.

Um ponto RGB com `tentative_label` aparece como **Hipótese 2D · sem rótulo 3D**
no inspector. A preview preserva a máscara e o claim original, inclusive para
uma janela sem suporte 3D. `visibility_unconfirmed` explica a ausência de
superfície compatível; `occluded` identifica uma superfície anterior no raio.
