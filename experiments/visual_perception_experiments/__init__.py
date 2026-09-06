"""Orquestração de experimentos de percepção visual.

Issues: #192 (benchmark de upsampling denso), #197 (preparação do conjunto de
referência), #199 (avaliação de calibração e abstenção), #200 (matriz de
ablation).

Este pacote é a única camada que depende ao mesmo tempo do módulo avaliado
(``visual_perception``) e das métricas reutilizáveis
(``visual_perception_evaluation``). Manter essa dependência dupla aqui é o
que permite que a avaliação permaneça independente da representação interna
do módulo, conforme AGENTS.md ("experiments/ comparações, ablations e
orquestração de experimentos").
"""

from .paths import ensure_import_roots

# Prepara as source trees pertencentes à composição antes que qualquer CLI
# do pacote importe datasets, evaluation ou visual_perception.
ensure_import_roots()
