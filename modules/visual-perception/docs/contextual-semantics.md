# Política de semântica contextual

O objetivo de `visual-perception` não é transformar cada superfície visível em uma classe. O produto do módulo deve priorizar evidências que acrescentem contexto útil ao mapa.

## Regra principal

Superfícies estruturais genéricas como `wall`, `floor` e `ceiling` podem continuar existindo como regiões geométricas descobertas pela pipeline, mas não devem dominar os claims semânticos apenas por serem superfícies visíveis.

A semântica deve priorizar entidades, sinais e anomalias com valor contextual, por exemplo:

- portas, janelas, placas, objetos e equipamentos;
- texto, símbolos, sinalização e grafite;
- rachaduras, danos estruturais, umidade e tinta descascada;
- obstruções, hazards e alterações relevantes da superfície;
- entidades cuja presença ou estado ajude navegação, busca, inspeção ou raciocínio posterior.

## Estrutura não é o mesmo que contexto

A pipeline precisa preservar geometria suficiente para associação e raciocínio espacial. Isso não implica promover toda região estrutural a uma identidade semântica de alto nível.

Exemplo:

```text
região geométrica: parede
    -> sem evidência contextual relevante
       -> permanece como suporte geométrico

região geométrica: parede
    -> contém rachadura extensa
       -> claim contextual: structural_crack
       -> relação: crack ON wall
```

O segundo caso acrescenta informação ao mapa. O primeiro, isoladamente, tende a apenas repetir informação estrutural que a geometria já representa.

## Labels e claims

Uma região pode possuir múltiplas hipóteses e evidências. A pipeline não deve forçar uma região a receber uma label de objeto quando a evidência é insuficiente.

Em particular:

- ausência de identidade forte não deve ser convertida automaticamente em `wall`, `floor` ou `ceiling`;
- claims alternativos devem permanecer auditáveis quando houver ambiguidade;
- confiança e suporte devem refletir evidência disponível, não uma certeza artificial;
- region geometry e semantic claims têm papéis diferentes e não devem ser colapsados no mesmo conceito.

## Cena e região

O contexto global da cena deve ajudar a interpretar regiões, mas não substituir evidência local. Da mesma forma, uma interpretação local não deve reescrever o contexto global sem suporte.

O reasoner de região recebe contexto ambiental relevante para desambiguar observações, enquanto sinais de suporte independentes devem poder contestar uma hipótese quando a evidência visual não a sustenta.

## Falsos positivos estruturais

Confusões como uma parede identificada como `refrigerator` não devem ser corrigidas simplesmente adicionando `wall` como classe vencedora. O objetivo é distinguir:

```text
região sem identidade contextual forte
```

de:

```text
região com evidência suficiente para uma entidade ou anomalia específica
```

Isso reduz o incentivo para a pipeline substituir um erro de objeto por uma classificação estrutural pouco informativa.

## Estado de adoção

Esta política é normativa para a direção atual do módulo, mas o comportamento implementado ainda está em transição. Runs anteriores continuam válidas como baselines reproduzíveis e podem conter labels estruturais que esta política pretende reduzir.

Ao avaliar uma mudança, separe:

- política pretendida;
- comportamento implementado na revisão testada;
- comportamento observado nos artifacts da run.

## Critérios de avaliação

Além de acurácia de identidade, benchmarks devem observar:

- taxa de claims estruturais triviais;
- falsos positivos de objetos em superfícies;
- cobertura de entidades e anomalias relevantes;
- estabilidade entre frames;
- qualidade da confiança e do suporte de hipóteses;
- preservação de ambiguidade quando a evidência não permite decisão forte.

A reference run e os artifacts usados pela documentação principal estão indexados em [`docs/README.md`](../../../docs/README.md).