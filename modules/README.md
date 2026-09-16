# Módulos

`modules/` contém as capacidades centrais do sistema. Cada módulo deve ser independentemente compreensível, testável e substituível através de uma fronteira pública pequena.

A estrutura física do repositório acompanha o código real. Capacidades apenas planejadas não precisam existir como diretórios vazios.

## Módulos implementados

| Módulo | Responsabilidade | Documentação local |
| --- | --- | --- |
| `visual-perception` | RGB para evidência visual e semântica estruturada | [`visual-perception/docs/`](./visual-perception/docs/README.md) |
| `state-estimation` | LiDAR/IMU para pose, trajetória e contexto de movimento | [`state-estimation/docs/`](./state-estimation/docs/README.md) |
| `geometric-map` | geometria persistente e referências estáveis | [`geometric-map/docs/`](./geometric-map/docs/README.md) |
| `sensor-association` | projeção e associação entre geometria 3D e evidência visual | [`sensor-association/docs/`](./sensor-association/docs/README.md) |
| `semantic-fusion` | fusão multi-view e suporte espacial de claims por ponto | [`semantic-fusion/docs/`](./semantic-fusion/docs/README.md) |
| `semantic-map` | consolidação de runs contextuais publicadas sobre geometria compartilhada | [`semantic-map/docs/`](./semantic-map/docs/index.md) |
| `point-representation` | contracts e transforms para embeddings 3D por ponto (sem backbone concreto ainda) | [`point-representation/README.md`](./point-representation/README.md) |

A reconstrução geométrica persistente pertence a `geometric-map`. Implementações concretas de odometria pertencem a `state-estimation`. Percepção visual não deve assumir ownership de geometria 3D persistente.

## Capacidades planejadas

As capacidades abaixo continuam fazendo parte da arquitetura pretendida, mas não possuem implementação suficiente para justificar um diretório próprio neste momento:

- `semantic-memory`;
- `scene-graph`;
- `context-reasoning`;
- `query-engine`.

Elas devem ser materializadas em `modules/<capability>/` quando houver uma issue implementando contract, código, teste ou documentação concreta. Não use `.gitkeep` apenas para representar arquitetura futura.

## Fronteira pública

Consumidores devem depender de:

- tipos de dados públicos;
- funções ou classes públicas;
- pontos de entrada estáveis;
- protocols apenas quando existe substituição real.

Caches, layout de armazenamento, estruturas de treino, objetos específicos de modelo e detalhes de backend permanecem internos ao módulo dono.

Contracts específicos de uma capacidade pertencem ao módulo que os define, mesmo quando são consumidos por outro módulo.

## Estrutura

Use a menor estrutura que represente o código existente. Um módulo pode evoluir para:

```text
modules/<module>/
├── README.md
├── docs/
│   └── README.md
├── src/
│   └── <package>/
├── tests/
├── configs/
└── benchmarks/
```

Nenhum desses diretórios é obrigatório por antecipação. Crie `configs/`, `benchmarks/`, subcamadas internas ou adapters apenas quando houver conteúdo concreto.

## Documentação

A documentação é dividida em dois níveis:

```text
docs/
    visão end-to-end e decisões que atravessam módulos

modules/<module>/docs/
    contracts, algoritmos, backends, limitações e decisões locais
```

A documentação principal deve apontar para a documentação especializada em vez de duplicar detalhes internos. A documentação local, por sua vez, deve apontar para o estágio correspondente da pipeline quando o assunto atravessar módulos.

Comece por [`../docs/README.md`](../docs/README.md).

A documentação histórica anterior à reorganização está preservada em [`../.old-docs/`](../.old-docs/) e não deve ser tratada automaticamente como descrição do código atual.

## Sequência de implementação

Ao materializar ou ampliar um módulo:

1. declare sua responsabilidade no `README.md`;
2. defina entradas, saídas, unidades, frames, timestamps e proveniência;
3. implemente o caminho funcional mínimo;
4. introduza interfaces somente em pontos reais de variação;
5. teste contracts e comportamento local;
6. adicione benchmarks quando performance ou qualidade precisarem ser comparadas;
7. documente decisões não óbvias em `docs/`;
8. atualize [`../docs/README.md`](../docs/README.md) quando a mudança afetar a pipeline global.

Antes de alterar fronteiras de módulo ou arquitetura do repositório, leia [`../AGENTS.md`](../AGENTS.md) e a documentação ativa em [`../docs/`](../docs/README.md).