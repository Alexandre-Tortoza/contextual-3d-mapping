# Contracts

`contracts/` contém primitivas compartilhadas, independentes de implementação, usadas por múltiplas capacidades e aplicações.

Grupos iniciais:

```text
contracts/
├── spatial/
├── temporal/
├── observations/
└── maps/
```

Contracts de nível de repositório podem definir shape de dados, invariantes, proveniência, confiança, frames de coordenadas, timestamps, identificadores, referências de artifact e fronteiras de serialização.

Contracts específicos de capacidade permanecem de posse do módulo que define a capacidade. Por exemplo, contracts de point embedding aprendido pertencem a `point-representation`; eles não devem ser movidos para cá apenas porque outro módulo os consome.

O pacote de contracts raiz deve permanecer pequeno e estável o suficiente para dar suporte a módulos e aplicações que evoluem de forma independente.
