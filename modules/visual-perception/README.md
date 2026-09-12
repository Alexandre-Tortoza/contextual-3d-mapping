# Visual Perception

Discovery e grounding espacial têm contracts distintos. `ObservedRegion.mask`
e `box` preservam a descoberta; `ObservedRegion.grounding` contém a predição do
segmentador, a máscara semântica aceita, as boxes condicionadas ao conceito e
os diagnostics. Reconhecimento isolado preserva claims sem publicar precisão
espacial inexistente. Ver [decisão, API e ablações](docs/semantic-grounding.md).

`visual-perception` transforma uma observação de imagem RGB canônica em uma observação
visual estruturada e auditável: regiões descobertas com masks e boxes, embeddings de
região densos e alinhados à linguagem, claims semânticos em nível de cena e de região,
relações candidatas em nível de imagem, e um audit de qualidade — com proveniência de
modelo e configuração anexada em todo o processo.

O módulo **não** é uma segmentação semântica completa da cena. A capacidade que ele
entrega é **extração de evidência visual portadora de contexto** (*context-bearing visual
evidence extraction*), e a diferença aparece no que ele publica: a pergunta que ele
responde não é "que superfícies existem nesta imagem?", e sim "que informação
visualmente observável merece ser preservada como evidência contextual em um mapa 3D?".
Ver [contexto estrutural vs. observação contextual](#contexto-estrutural-vs-observação-contextual).

> Documentação detalhada: [`docs/`](docs/README.md).

O módulo é testável de forma independente e cada estágio canônico é substituível por
trás de um port, e cada estágio contextual é desligável por configuração — para que uma
comparação consiga atribuir um efeito a um estágio só. Ele vem com fakes completos, determinísticos e GPU-free para cada
backend, para que seus contracts, pipeline, cache e fronteiras de integração possam ser
totalmente exercitados sem uma GPU ou download de modelo; backends reais são rastreados
separadamente (veja [docs/model-backends.md](docs/model-backends.md)).

## Responsabilidades

- consumir observações RGB canônicas emitidas por `[adapters]` (não ler datasets/ROS
  bags diretamente);
- descobrir, dividir em tiles e mesclar regiões de imagem em regiões canônicas estáveis;
- fazer pooling de features visuais densas e produzir embeddings alinhados à linguagem
  por região;
- interpretar semântica em nível de cena e de região como claims auditáveis, não labels
  únicos;
- publicar como observação contextual apenas a evidência que discrimina algo, mantendo
  superfície estrutural genérica como contexto interno;
- medir, por um canal **independente** do reasoner, se a evidência sustenta cada hipótese
  de identidade;
- reinterpretar seletivamente as regiões que têm uma razão explícita, sempre com evidência
  nova;
- reconciliar, dentro do frame, conceitos equivalentes e regiões que provavelmente são a
  mesma superfície, sem apagar nenhuma delas;
- gerar relações 2D candidatas entre regiões, geométricas e semânticas, mantidas
  distinguíveis;
- auditar a observação resultante quanto à consistência estrutural e contradições;
- fazer cache de estágios caros e serializar a observação canônica para persistência.

## Contexto estrutural vs. observação contextual

Uma parede não é evidência contextual. Uma rachadura na parede é.

Identificar que uma região é `wall`, `floor` ou `ceiling` não produz informação
discriminativa para um mapa contextual: essas superfícies ocupam a maior parte de um
frame indoor e dominariam qualquer inventário. A informação que o mapa precisa preservar
está **sobre** elas — rachadura, graffiti, texto de aviso, tinta descascada, mancha
d'água, mofo, corrosão, dano estrutural, obstrução.

O módulo aplica então esta regra de domínio:

> Superfícies estruturais genéricas não são emitidas como observações semânticas a menos
> que contenham, delimitem, exibam ou participem de evidência contextualmente
> significativa.
>
> A ausência de uma observação contextualmente significativa é preferível a emitir um
> label estrutural genérico com pouco valor discriminativo.

A consequência prática é a partição da saída canônica em duas metades:

```text
observation.regions             evidência contextual publicada
observation.structural_context  superfície estrutural, preservada como contexto
```

**A superfície continua existindo, e continua sendo necessária.** Ela alimenta a análise
de cena, a reconciliação intra-frame que agrupa fragmentos da mesma parede, as relações
candidatas, a coerência conceito↔natureza e o audit de qualidade — o modelo precisa
entender que aquilo é uma parede para concluir que a marca sobre ela é uma rachadura
estrutural. Nada é apagado: a região suprimida mantém máscara, box, claims, embeddings e
evidência, continua alcançável por `region_by_id`, e continua sendo alvo válido de
relação e de grupo. O que muda é o **contract do que o módulo publica**.

Quando a evidência publicada nomeia a superfície que a hospeda, ela ganha uma claim
derivada `host_surface`:

```text
label:        cracked wall     (o conceito do produtor, preservado)
host_surface: wall             (derivado da própria identidade afirmada)
```

`host_surface` nunca é forçado. Ausência da claim significa host desconhecido, e é um
desfecho legítimo: decidir que uma evidência 2D pertence a uma superfície física do mundo
exige geometria 3D, e pertence a `sensor-association`, `semantic-fusion` e ao mapa.

Uma cena inteiramente normal pode publicar poucas regiões, ou nenhuma. Esse é o
comportamento esperado.

Três coisas que a política deliberadamente **não** faz:

- **não substitui label.** Uma região que o produtor não decidiu entre `wall` e `door`
  sai do output público; ela nunca é promovida a `door` porque `wall` deixou de ser
  publicável. A política reduz falso positivo, não troca de classe;
- **não usa posição na imagem.** Nada de "embaixo é piso, em cima é teto": estes frames
  são fisheye, e o projeto também mira UAV e orientação arbitrária de câmera;
- **não fecha o vocabulário.** A decisão é sobre o **núcleo nominal** de um conceito
  livre, não sobre uma lista de classes permitidas. O sistema continua open-vocabulary.

A política mora em [`domain/contextual_evidence.py`](src/visual_perception/domain/contextual_evidence.py)
como Specification pura, é aplicada por
[`application/contextual_publication.py`](src/visual_perception/application/contextual_publication.py),
e é desligável por `contextual_publication.enabled`.

## Não-responsabilidades

- sampling e transporte de dataset/ROS bag (`[adapters]`);
- calibração, projeção cross-sensor, associação com LiDAR (`sensor-association`);
- construção de mapa geométrico/semântico persistente, scene graphs;
- decidir a que superfície física do mundo uma evidência 2D pertence — o módulo observa o
  dano, e a associação com uma superfície 3D é resolvida depois:

```text
visual-perception:   structural crack #7
geometric-map:       planar surface #31
sensor-association:  crack #7 belongs_to surface #31
semantic-fusion:     surface #31 likely wall
semantic-map:        Wall #31 → structural_crack #7
```

## Estrutura

```text
visual-perception/
├── README.md
├── docs/
├── benchmarks/
├── src/
│   └── visual_perception/
│       ├── domain/
│       ├── ports/
│       ├── application/
│       ├── infrastructure/
│       │   ├── fakes/
│       │   ├── adapters/
│       │   └── integration/
│       └── config.py
└── tests/
```

## Desenvolvimento

```bash
mise trust ../../.mise.toml  # uma vez, se o ambiente usa mise
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,bench]"
cd ../..
make verify
```

O alvo `verify` da raiz é a fonte única para a suíte ampliada: testes unitários, de
integração e de contract dos benchmarks, `ruff` no repositório inteiro e `mypy` em
modo estrito. Dependências pesadas dos backends reais continuam no extra `ml` e não
são necessárias para a suíte determinística em CPU.

`contextual_mapping_contracts` (e, apenas para os testes de integração,
`contextual_mapping_adapters`/`contextual_mapping_datasets`) são resolvidos a partir de
suas source trees via `pythonpath` do pytest no `pyproject.toml`, até que esses pacotes
tenham seu próprio build instalável.
