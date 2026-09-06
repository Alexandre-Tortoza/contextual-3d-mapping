# Dataset Splits

Seleções reproduzíveis de sequência de treino, validação, teste, benchmark e experimento pertencem aqui quando são artifacts de nível de repositório, e não dados de treino privados de módulo.

`corridor-02-visual-reference-1.json` fixa os IDs dos splits de development,
calibration e test do conjunto visual. O arquivo registra o digest do manifest
estrutural que originou a partição; qualquer alteração desse manifest exige regenerar e
revisar a partição, em vez de aceitar drift silencioso.
