# Guia para Agentes

Este arquivo define as regras de implementação válidas para todo o repositório, voltadas a agentes de código e contribuidores.

Leia este arquivo antes de criar ou alterar código, pastas, interfaces, issues ou documentação.

## Idioma da documentação

Toda documentação em prosa — `README.md`, arquivos em `docs/`, docstrings e comentários no código — é escrita em **português do Brasil**.

Permanecem em inglês:

- identificadores de código (nomes de classes, funções, variáveis, módulos, arquivos e diretórios);
- mensagens de commit e títulos/descrições de PR;
- jargão técnico e arquitetural já consagrado (`pipeline`, `backend`, `adapter`, `port`, `framework`, `dataset`, `benchmark`, `checkpoint`, etc.) — usado dentro da prosa em português para não divergir dos nomes usados no código;
- marcadores convencionais como `NOTE:`, `TODO:`, `FIXME:`, `WARNING:` (o texto que os segue vai em português).

### Formato de docstring de função e método

Toda função, método e classe é documentada em duas partes:

1. um bloco de comentário `#` logo acima da assinatura, em português, explicando **o que ela faz, por que existe** e, quando for relevante, **onde é usada**;
2. a docstring `"""..."""` logo abaixo, em português (`Argumentos`, `Retorna`, `Levanta`, etc.), mantendo o jargão técnico em inglês.

```python
# Converte a nuvem de pontos bruta no embedding denso usado pelo pipeline
# de fusão. Existe porque o backend de fusão espera vetores densos, não
# pontos esparsos; chamada por FusionPipeline.run() a cada frame.
def encode_points(points: PointCloud) -> Embedding:
    """Encode a point cloud into a dense embedding.

    Argumentos:
        points: nuvem de pontos bruta capturada pelo sensor.
    Retorna:
        embedding denso pronto para o backend de fusão.
    """
```

Docstrings de módulo (topo do arquivo) só são traduzidas — já cumprem o papel de contexto para o arquivo inteiro.

## Direção arquitetural

Use uma **arquitetura modular orientada a capacidades**.

Organize o sistema em torno de capacidades claras, como estimação de estado, mapeamento geométrico, percepção visual, representação de pontos, associação de sensores, fusão semântica, memória semântica, grafos de cena, raciocínio contextual e consulta.

O objetivo de implementação é simples: um leitor deve conseguir abrir um único módulo e entender a maior parte daquela capacidade sem explorar diretórios não relacionados.

Aplique os princípios SOLID nas fronteiras relevantes de código e módulo, mantendo a estrutura direta e fácil de navegar.

## Prioridades de design

Ao tomar decisões arquiteturais, use esta ordem de prioridade:

1. legibilidade e compreensibilidade local;
2. responsabilidade e posse (ownership) explícitas;
3. baixo acoplamento entre capacidades;
4. testabilidade e substituibilidade;
5. extensibilidade em torno de pontos de variação reais;
6. abstração quando torna uma das propriedades anteriores mais clara ou segura.

Prefira designs cuja intenção seja visível a partir de nomes de arquivos, tipos, assinaturas de função e fluxo de dados.

## Fluxo de trabalho de implementação

Para toda nova capacidade ou mudança:

1. identifique o módulo dono do comportamento;
2. defina as entradas, saídas, invariantes, unidades, frames e proveniência que cruzam a fronteira;
3. implemente o comportamento dentro do módulo dono;
4. exponha apenas os tipos e operações públicas que os consumidores precisam;
5. componha módulos a partir de `apps/`, experimentos, ou código de orquestração explícito;
6. adicione testes no nível mais estreito que seja útil;
7. documente decisões não óbvias ao lado do código que as possui.

Se a posse (ownership) não estiver clara, resolva isso primeiro. Ownership é uma decisão arquitetural.

## Papéis do repositório

Use os diretórios do repositório da seguinte forma:

```text
apps/         composições executáveis e pontos de entrada voltados ao usuário
modules/      capacidades de pesquisa e do sistema
datasets/     manifests, schemas, splits e suporte a nível de dataset
evaluation/   métricas e lógica de avaliação reutilizáveis
experiments/  comparações, ablations e orquestração de experimentos
docs/         arquitetura e decisões de nível de repositório
tests/        testes de integração ou end-to-end de nível de repositório
```

Quando primitivas de nível de repositório se tornarem necessárias, mantenha-as pequenas e estáveis em um local compartilhado. Bons exemplos são timestamps, frames espaciais, poses, transforms, identificadores de mapa e primitivas de proveniência.

Integrações, contracts, configuração e helpers de persistência específicos de capacidade, e model runtimes, ficam com o módulo dono.

## Formato de módulo

Comece cada módulo com a estrutura mínima útil. Um módulo pode evoluir para:

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

Crie diretórios quando eles tiverem uma responsabilidade e conteúdo concretos.

Um módulo deve permanecer compreensível pelo vocabulário da sua capacidade, e não por nomes genéricos de camada arquitetural.

## Fronteira pública do módulo

Cada módulo expõe uma pequena API pública documentada.

Consumidores devem depender de:

- tipos de dados públicos;
- funções ou classes públicas;
- protocolos explícitos para pontos de variação reais;
- pontos de entrada estáveis do módulo.

Mantenha detalhes de implementação locais ao módulo produtor, incluindo objetos de modelo internos, caches, estruturas de treino, tipos específicos de backend, layout de armazenamento e objetos de runtime de terceiros.

Um formato de dependência útil é:

```text
consumidor -> API pública do produtor -> implementação do produtor
```

## Regras SOLID

### Princípio da Responsabilidade Única

Dê a cada módulo, classe, função e serviço um motivo coerente para mudar.

Prefira componentes focados como:

```text
TemporalFusion
ObservationMatcher
PointEncoder
PoseEstimator
SemanticMapReader
SemanticMapWriter
```

Divida comportamento quando as responsabilidades evoluem independentemente.

### Princípio Aberto/Fechado

Represente pontos de variação genuínos com contracts estáveis.

Exemplos típicos neste projeto incluem:

- múltiplos pose estimators;
- múltiplas estratégias de fusão;
- múltiplos point encoders;
- múltiplos backends de persistência;
- variantes de pesquisa comparadas por experimentos.

Consumidores devem permanecer estáveis enquanto implementações variam por trás do mesmo comportamento público.

### Princípio da Substituição de Liskov

Implementações do mesmo contract público devem ser seguramente intercambiáveis.

Documente e teste invariantes relevantes para substituição, como:

- unidades;
- frames de coordenadas;
- semântica de timestamp;
- garantias de ordenação;
- dimensões de tensor ou embedding;
- comportamento de ciclo de vida;
- comportamento de erro.

### Princípio da Segregação de Interface

Modele interfaces em torno da necessidade de um único consumidor.

Por exemplo, leitura e escrita de mapa podem ser contracts separados quando os consumidores precisam apenas de um dos lados.

Mantenha as interfaces estreitas o suficiente para que uma implementação as satisfaça sem responsabilidades não relacionadas.

### Princípio da Inversão de Dependência

Orquestração de alto nível depende de comportamento de capacidade estável.

Por exemplo:

```text
mapping runtime -> PoseEstimator <- concrete LiDAR-inertial estimator
```

O runtime depende da capacidade de que precisa. O estimator concreto fornece essa capacidade.

Use esse padrão em fronteiras que sejam substituíveis, caras, externas, ou intencionalmente variadas por experimentos de pesquisa.

## Quando criar uma interface ou protocolo

Crie uma quando uma fronteira de comportamento estável for útil porque:

- múltiplas implementações existem;
- uma implementação é intencionalmente substituível;
- um experimento de pesquisa compara implementações;
- uma dependência de terceiros deve ficar contida;
- uma fronteira de módulo precisa de um contract estável;
- testes precisam de um substituto leve para um componente caro ou externo.

Mantenha construção direta e código concreto local para comportamento sem ponto de variação relevante.

## Tipos e contracts compartilhados

Coloque um tipo em código compartilhado de nível de repositório quando ambas as condições forem verdadeiras:

1. o conceito é de posse do sistema como um todo;
2. múltiplos módulos precisam concordar com exatamente o mesmo significado estável.

Bons candidatos:

```text
Timestamp
FrameId
Pose
RigidTransform
MapId
ArtifactId
Provenance
```

Tipos específicos de capacidade ficam com sua capacidade. Uma representação de ponto aprendida, por exemplo, é de posse de `point-representation` mesmo quando outro módulo a consome.

## Integrações externas

Mantenha cada integração externa próxima da capacidade que a possui.

Exemplos:

```text
modules/state-estimation/
    integração concreta de odometria LiDAR-inercial

modules/geometric-map/
    backend de point-cloud ou reconstrução usado pelo mapeamento geométrico

modules/visual-perception/
    model runtime usado pela percepção visual
```

Crie infraestrutura de integração de nível de repositório apenas para comportamento genuinamente compartilhado por capacidades não relacionadas com a mesma semântica.

## Aplicações

`apps/` é dono da composição executável.

Aplicações podem:

- selecionar implementações concretas;
- conectar entradas e saídas de módulos;
- configurar workflows;
- expor pontos de entrada de CLI, GUI, TUI ou serviço;
- carregar e persistir estado de mapa composto;
- coordenar o ciclo de vida do runtime.

Algoritmos de pesquisa permanecem de posse do módulo que implementa a capacidade.

## Implementações de pesquisa

Projete a arquitetura pública em torno das responsabilidades do projeto.

Um paper, modelo, framework, dataset ou repositório externo pode informar uma implementação concreta. Registre essa proveniência científica na documentação do módulo relevante, mantendo os nomes públicos baseados em capacidade.

Por exemplo:

```text
capacidade pública: PoseEstimator
implementação concreta: estimator baseado em FAST_LIO
```

Issues devem descrever a capacidade, o comportamento observável, testes, critérios de aceitação e restrições relevantes. Referências de pesquisa pertencem à documentação do módulo quando ajudam a explicar a implementação ou comparação.

## Fluxo de dados multimodal explícito

Prefira transformações visíveis com tipos explícitos do projeto:

```text
observação
    -> validação de fronteira
    -> transformação da capacidade
    -> saída explícita
    -> próxima capacidade
```

Preserve os metadados necessários para interpretar ou reproduzir resultados. Dependendo da fronteira, isso inclui:

- timestamp;
- identidade do sensor;
- frame de coordenadas;
- unidades;
- proveniência de calibração ou transform;
- identidade da observação de origem;
- confiança ou incerteza;
- proveniência de modelo/checkpoint.

Valide essas propriedades quando os dados cruzam uma fronteira de módulo.

## Configuração

Mantenha a configuração de algoritmo com o módulo dono do algoritmo.

Mantenha a configuração de composição com a aplicação ou experimento que seleciona e conecta implementações.

Prefira configuração tipada e explícita, com defaults reprodutíveis para parâmetros que afetam experimentos.

## Tratamento de erros

Valide invariantes de fronteira cedo e retorne erros acionáveis.

Alvos importantes de validação incluem:

- compatibilidade de frame de coordenadas;
- sincronização de timestamp;
- validade de transform;
- dimensões de tensor e embedding;
- atributos de ponto;
- compatibilidade de mapa/versão;
- proveniência obrigatória.

## Expectativas de teste

Use o teste mais estreito que proteja o comportamento necessário:

- testes unitários para comportamento local determinístico;
- testes de contract para implementações intercambiáveis;
- testes de integração para fronteiras de módulo;
- testes end-to-end representativos para composição de aplicação;
- benchmarks para componentes de pesquisa sensíveis a runtime, memória e acurácia;
- testes de regressão para modos de falha observados anteriormente.

Teste comportamento observável e garantias de fronteira, para que implementações internas possam evoluir com segurança.

## Nomenclatura

Nomeie componentes pela sua capacidade e responsabilidade.

Prefira:

```text
PointEncoder
PoseEstimator
ObservationMatcher
TemporalFusion
SemanticMapReader
SemanticMapWriter
```

Use nomes genéricos como `Manager`, `Helper`, `Utils`, `Processor` ou `Service` apenas quando forem genuinamente o termo de domínio mais claro.

## Documentação

Arquitetura e decisões de engenharia de nível de repositório pertencem a `docs/`.

Comportamento de módulo, algoritmos, escolhas de modelo, justificativa de implementação, benchmarks, limitações e referências de pesquisa pertencem a `modules/<module>/docs/`.

Atualize a documentação na mesma mudança quando uma fronteira, regra de ownership, contract público ou direção de dependência mudar.

## Teste de decisão antes de adicionar complexidade

Antes de adicionar uma camada, pacote global, registry, factory, classe base ou interface, responda:

1. Que responsabilidade concreta ela representa?
2. Que dependência ou ponto de variação ela torna explícito?
3. Que consumidor fica mais simples ou mais seguro por causa dela?
4. Que cenário de teste ou substituição se beneficia dela?
5. É a menor estrutura que comunica a intenção com clareza?

Use a abstração quando essas respostas forem concretas. Caso contrário, mantenha a implementação local e direta.

## Fonte da verdade

Para arquitetura e princípios de engenharia do repositório, leia também:

- `docs/architecture.md`
- `docs/engineering-principles.md`
- `docs/documentation-policy.md`
- `modules/README.md`

Quando código e documentação divergirem, trate a inconsistência como parte da mudança e restaure uma única fonte da verdade arquitetural clara.
