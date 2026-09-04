# Temporal Contracts

Primitivas temporais de nível de repositório, como timestamps, intervalos de tempo, metadados de sincronização e identidade de clock/origem, pertencem aqui quando compartilhadas por múltiplas capacidades.

`contextual_mapping_contracts.Timestamp` armazena nanossegundos inteiros não negativos com um identificador de clock explícito; conversão entre clocks nunca é implícita.
