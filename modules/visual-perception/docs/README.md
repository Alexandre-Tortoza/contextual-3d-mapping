# Documentação de Visual Perception

`visual-perception` transforma uma observação RGB canônica em uma `VisualObservation`
estruturada, auditável e pronta para consumidores downstream. Estas páginas explicam
como consumir e operar o módulo; as regras locais de cada tipo, função e classe ficam
documentadas ao lado do código.

## Por onde começar

- Para integrar o módulo a uma aplicação ou adapter, comece por
  [integration.md](integration.md).
- Para entender os tipos que atravessam a API pública, consulte
  [api-contracts.md](api-contracts.md).
- Para compreender responsabilidades e fluxo interno, leia
  [architecture.md](architecture.md) e [pipelines.md](pipelines.md).

## Referência por necessidade

| Necessidade | Página |
| --- | --- |
| Chamar `run_canonical_pipeline` e tratar sua saída | [api-contracts.md](api-contracts.md) |
| Adaptar uma observação RGB ou compor o runtime | [integration.md](integration.md) |
| Entender estágios e extensões opcionais | [pipelines.md](pipelines.md) |
| Escolher, instalar ou diagnosticar um backend | [model-backends.md](model-backends.md) |
| Entender cache, auditoria e verificações locais | [execution.md](execution.md) |
| Persistir ou recarregar uma observação | [artifacts.md](artifacts.md) |
| Distinguir engenharia de hipóteses de pesquisa | [research-traceability.md](research-traceability.md) |
| Consultar vocabulário do módulo | [glossary.md](glossary.md) |

## Estado do módulo

O pipeline canônico, os fakes determinísticos e os adapters reais estão implementados.
O uso padrão continua sendo `backend="fake"`, portanto desenvolvimento, testes e
integração inicial não exigem GPU nem download de modelo. A seleção de checkpoints de
referência e a validação quantitativa em hardware real continuam pendentes; consulte
[model-backends.md](model-backends.md) antes de tratar um backend real como uma
configuração de produção validada.

Os benchmarks locais existem em `../benchmarks/`, mas seus resultados não fazem parte
desta documentação até que sejam versionados com o ambiente e dataset de referência.
