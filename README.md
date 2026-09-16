# contextual-3d-mapping

Framework de pesquisa para construir mapas 3D semânticos e contextuais de vocabulário aberto a partir de RGB, LiDAR e estimativas de movimento, combinando geometria persistente do mundo, features visual-language, representações de pontos aprendidas, memória semântica, grafos de cena e raciocínio espacial.

## Arquitetura

O repositório usa uma arquitetura modular simples orientada a capacidades.

```text
uma capacidade -> um módulo dono claro -> API pública pequena -> composição explícita
```

Fluxo de alto nível:

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

A visão **completa** do repositório, incluindo adapters, contracts, `mapping-runtime`, todos os módulos implementados, relações entre payloads, proveniência, persistência, aplicações e capacidades planejadas, está em [`docs/system-flow.md`](./docs/system-flow.md).

A documentação estágio por estágio está organizada em [`docs/README.md`](./docs/README.md). Ela separa explicitamente arquitetura pretendida, implementação atual e comportamento observado.

A documentação anterior foi preservada em [`.old-docs/`](./.old-docs/).

`apps/mapping-runtime` compõe workflows de construção de mapa. `apps/map-explorer` abre mapas persistidos para visualização 3D, consulta, inspeção de evidências e exploração de grafo. `apps/cli` fornece acesso scriptável a operações de nível de aplicação.

Agentes de código e contribuidores devem ler [`AGENTS.md`](./AGENTS.md) antes de criar ou alterar código, pastas, interfaces ou arquitetura de nível de repositório.

## Verificação de desenvolvimento

O módulo executável atual requer Python 3.12. Um ambiente reproduzível para a suíte completa pode ser preparado com:

```bash
mise trust .mise.toml
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "modules/visual-perception[dev,bench]"
make verify
```

O primeiro slice RGB-LiDAR também possui smoke tests executáveis sem ROS, GPU ou dataset externo:

```bash
make m1-test
make m1-demo
make map-explorer-install map-explorer-build
cd apps/map-explorer/web && npm run dev
```

## Documentação

Comece por [`docs/system-flow.md`](./docs/system-flow.md) para entender a arquitetura completa e todas as relações entre capacidades. Depois use [`docs/README.md`](./docs/README.md) para seguir a informação estágio por estágio e a documentação local em `modules/<module>/docs/` para detalhes internos.

Artifacts de execução são regeneráveis e não são tratados como contracts permanentes. Quando uma página usa valores históricos, ela deve identificá-los explicitamente como contexto ou exemplo conceitual, e não como comportamento garantido da revisão atual.
