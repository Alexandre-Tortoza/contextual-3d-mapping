"""Fake em memória de :class:`EvidencePersistencePort` para testes de integração (#180)."""

from __future__ import annotations

from typing import Any


# Store fake que satisfaz o port de persistência de evidência sem tocar
# armazenamento real. Existe para testar a integração com persistência
# (ver infrastructure/integration/persistence_integration.py) isoladamente
# de qualquer banco de dados ou object store concreto.
class InMemoryEvidenceStore:
    """Um store trivial usado para testar a integração de persistência sem armazenamento real."""

    # Inicializa o dicionário em memória que representa o "armazenamento".
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    # Grava um payload e devolve uma referência estável, no mesmo formato de
    # referência esperado por um backend real (esquema memory://).
    def put(self, artifact_id: str, payload: dict[str, Any]) -> str:
        reference = f"memory://{artifact_id}"
        self._records[reference] = payload
        return reference

    # Recupera o payload previamente gravado por put(), a partir da referência.
    def get(self, reference: str) -> dict[str, Any]:
        return self._records[reference]
