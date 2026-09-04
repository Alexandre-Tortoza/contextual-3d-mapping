"""Fingerprints de stage e um cache de artifacts reutilizável e retomável.

Issue: #170.

O fingerprint de um stage é encadeado: ele faz hash da própria versão e
configuração do stage junto com o fingerprint de cada stage upstream.
Mudar a configuração de um stage, portanto, muda apenas o fingerprint
daquele stage e todos os fingerprints computados a partir dele (seus
dependentes downstream), enquanto stages irmãos que não consomem sua
saída mantêm uma entrada de cache válida.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

#: Incrementar quando o formato do registro de artifact em disco mudar, para
#: rejeitar entradas de cache obsoletas gravadas por uma versão incompatível
#: do módulo.
CACHE_SCHEMA_VERSION = 1


# Calcula o fingerprint de cache de um stage combinando sua própria versão e
# configuração com os fingerprints de todos os stages upstream, formando a
# cadeia que StageCache usa para decidir hits/misses de cache.
def compute_fingerprint(
    stage_name: str,
    stage_version: str,
    config_fingerprint: str,
    upstream_fingerprints: tuple[str, ...] = (),
) -> str:
    """Calcula o fingerprint de cache de um stage."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage_name": stage_name,
        "stage_version": stage_version,
        "config_fingerprint": config_fingerprint,
        "upstream_fingerprints": list(upstream_fingerprints),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# Armazena em disco os resultados de stages já concluídos, permitindo
# retomar um pipeline sem reprocessar stages cujo fingerprint já bate com
# uma entrada existente no cache.
class StageCache:
    """Um cache retomável, baseado em disco, de resultados de stage concluídos."""

    # Guarda o diretório raiz onde os registros de cache deste StageCache
    # são lidos e gravados.
    def __init__(self, cache_directory: Path) -> None:
        self.cache_directory = cache_directory

    # Consulta o cache por um resultado de stage já computado; usado pelo
    # pipeline para pular reprocessamento quando o fingerprint bate.
    def get(self, stage_name: str, fingerprint: str) -> dict[str, Any] | None:
        """Retorna o registro de artifact em cache, ou ``None`` em caso de cache miss."""
        path = self._record_path(stage_name, fingerprint)
        if not path.exists():
            return None
        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        artifacts: dict[str, Any] = record["artifacts"]
        return artifacts

    # Persiste o resultado de um stage concluído em disco, escrevendo
    # primeiro em um arquivo temporário e renomeando de forma atômica para
    # evitar registros corrompidos em caso de falha no meio da escrita.
    def put(self, stage_name: str, fingerprint: str, artifacts: dict[str, Any]) -> None:
        """Registra de forma durável que ``stage_name`` foi concluído para ``fingerprint``."""
        path = self._record_path(stage_name, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"schema_version": CACHE_SCHEMA_VERSION, "artifacts": artifacts}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(record, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    # Atalho booleano sobre `get`: True quando já existe um registro de
    # cache válido para esse stage e fingerprint.
    def is_complete(self, stage_name: str, fingerprint: str) -> bool:
        return self.get(stage_name, fingerprint) is not None

    # Resolve o caminho em disco onde o registro de cache de um stage e
    # fingerprint específicos fica armazenado.
    def _record_path(self, stage_name: str, fingerprint: str) -> Path:
        return self.cache_directory / stage_name / f"{fingerprint}.json"
