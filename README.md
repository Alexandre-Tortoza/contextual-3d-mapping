# contextual-3d-mapping

Framework de pesquisa para construir mapas 3D semânticos e contextuais de vocabulário aberto a partir de RGB, LiDAR e estimativas de movimento, combinando geometria persistente do mundo, features visual-language, representações de pontos aprendidas, memória semântica, grafos de cena e raciocínio espacial.

## Arquitetura

O repositório usa uma **arquitetura modular simples orientada a capacidades**.

Os principais objetivos são legibilidade, responsabilidade explícita, baixo acoplamento, testabilidade, substituibilidade, manutenibilidade e reprodutibilidade científica. Os princípios SOLID guiam o design de código e dependências nas fronteiras relevantes.

A regra central é simples:

```text
uma capacidade -> um módulo dono claro -> API pública pequena -> composição explícita
```

Integrações e detalhes de implementação específicos de capacidade ficam com o módulo dono. Aplicações compõem módulos em workflows executáveis. Primitivas compartilhadas permanecem pequenas e estáveis.

## Fluxo de alto nível

```text
RGB + LiDAR + IMU
        -> state-estimation
        -> geometric-map
        -> visual-perception / point-representation
        -> sensor-association
        -> semantic-fusion
        -> semantic-map
        -> semantic-memory / scene-graph / context-reasoning
        -> query-engine
        -> applications
```

`apps/mapping-runtime` compõe workflows de construção de mapa. `apps/map-explorer` abre mapas persistidos para visualização 3D, consulta, inspeção de evidências e exploração de grafo. `apps/cli` fornece acesso scriptável a operações de nível de aplicação.

Agentes de código e contribuidores devem ler [`AGENTS.md`](./AGENTS.md) antes de criar ou alterar código, pastas, interfaces, ou arquitetura de nível de repositório.

## Documentação

A documentação de arquitetura e integração de nível de repositório está disponível em [`docs/README.md`](./docs/README.md).

Decisões arquiteturais importantes são documentadas em:

- [`docs/architecture.md`](./docs/architecture.md)
- [`docs/system-flow.md`](./docs/system-flow.md)
- [`docs/applications.md`](./docs/applications.md)
- [`docs/map-lifecycle.md`](./docs/map-lifecycle.md)
- [`docs/engineering-principles.md`](./docs/engineering-principles.md)
- [`docs/documentation-policy.md`](./docs/documentation-policy.md)

Documentação detalhada de implementação vive dentro de cada módulo, em `modules/<module>/docs/`, à medida que esses módulos são desenvolvidos.
