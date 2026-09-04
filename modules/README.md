# Módulos

Cada módulo representa uma capacidade independentemente compreensível, desenvolvível, testável, avaliável (benchmarkable) e substituível.

O repositório é orientado a capacidades. Prefira a menor estrutura interna que deixe clara a capacidade, suas entradas, saídas e pontos de variação.

## Ownership de capacidades

- `state-estimation`: observações LiDAR/IMU convertidas em pose, trajetória e frames LiDAR corrigidos por movimento.
- `geometric-map`: geometria persistente do mundo construída a partir de poses e observações geométricas.
- `visual-perception`: observações de imagem convertidas em features visuais e semânticas estruturadas.
- `point-representation`: pontos LiDAR convertidos em representações 3D aprendidas.
- `sensor-association`: associação geométrica e temporal entre observações de sensor.
- `semantic-fusion`: fusão multi-fonte, temporal e multi-view em observações 3D semânticas.
- `semantic-map`: informação semântica persistente de vocabulário aberto vinculada à geometria do mundo.
- `semantic-memory`: recuperação semântica e espacial sobre a informação mapeada.
- `scene-graph`: entidades, hierarquia e relações extraídas do mapa.
- `context-reasoning`: inferência contextual com proveniência explícita.
- `query-engine`: interface unificada de consulta semântica, espacial e contextual.

A reconstrução geométrica persistente é de posse de `geometric-map`.

Implementações concretas de odometria são de posse de `state-estimation`. Módulos semânticos referenciam a geometria através de tipos públicos documentados, para que o sistema mantenha uma única representação geométrica autoritativa.

## Fronteira pública

Cada módulo expõe uma pequena API pública documentada.

Consumidores dependem de:

- tipos de dados públicos;
- funções ou classes públicas;
- pontos de entrada estáveis;
- protocolos que representam pontos de variação reais.

Objetos de modelo específicos de implementação, caches, layout de armazenamento, estruturas de treino e representações de backend permanecem locais ao módulo dono.

Contracts específicos de capacidade pertencem à capacidade que os define.

Por exemplo, um contract de representação de ponto aprendida pertence a `point-representation`, mesmo quando `sensor-association` o consome.

## Estrutura interna

Um módulo pode evoluir para:

```text
modules/<module>/
├── README.md
├── src/
│   └── <package>/
│       ├── __init__.py
│       ├── models.py
│       ├── config.py
│       └── <arquivos de capacidade>.py
├── tests/
├── configs/
├── benchmarks/
└── docs/
```

Crie cada diretório quando ele tiver uma responsabilidade e conteúdo concretos.

Mantenha integrações externas específicas de capacidade próximas do módulo que as possui. Por exemplo, um backend de odometria LiDAR-inercial pertence a `state-estimation`, e um backend de point-cloud usado apenas para reconstrução pertence a `geometric-map`.

## Sequência de implementação

Ao desenvolver um módulo:

1. declare a responsabilidade da capacidade no `README.md`;
2. defina entradas e saídas públicas;
3. documente unidades, frames, timestamps, shapes, proveniência e outros invariantes de fronteira;
4. implemente o comportamento funcional mais simples localmente;
5. introduza protocolos apenas para pontos reais de substituição ou comparação;
6. teste o comportamento local e os contracts públicos;
7. adicione benchmarks para comportamento sensível a performance;
8. documente decisões algorítmicas e de pesquisa não óbvias em `docs/`.

## Abstrações

Use protocolos, interfaces, strategies, factories ou registries quando representarem um ponto de variação concreto, como:

- múltiplas implementações;
- substituibilidade intencional;
- comparações de pesquisa;
- isolamento de dependência de terceiros;
- estabilidade de fronteira de módulo;
- testes em nível de contract com substitutos.

Para comportamento local direto com uma única implementação, construção e chamadas diretas são preferidas porque tornam o caminho do código mais fácil de ler.

## Documentação

Comportamento detalhado de módulo, algoritmos, justificativa de implementação, escolhas de modelo, benchmarks, limitações e referências de pesquisa pertencem a:

```text
modules/<module>/docs/
```

Decisões arquiteturais de nível de repositório pertencem ao `docs/` raiz.

Antes de implementar um módulo, leia o [`AGENTS.md`](../AGENTS.md) raiz, [`docs/architecture.md`](../docs/architecture.md) e [`docs/engineering-principles.md`](../docs/engineering-principles.md).
