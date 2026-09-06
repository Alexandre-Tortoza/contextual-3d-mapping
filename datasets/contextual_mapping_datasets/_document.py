"""Regras compartilhadas de leitura de documentos JSON de manifest.

Existe para que os dois manifests versionados do pacote — o de dataset
(`manifest_io.py`) e o de anotação de referência
(`annotation_manifest_io.py`) — apliquem exatamente as mesmas regras de
fronteira: nada de coerção implícita, nada de caminho absoluto, nada de
traversal. Duplicar essas regras faria os dois manifests divergirem
silenciosamente na primeira correção aplicada a só um deles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit


# Obtém um objeto JSON do documento sem aceitar listas ou valores escalares.
# Centralizar essa checagem mantém os erros de schema precisos e consistentes.
def mapping(value: object, source: str) -> Mapping[str, Any]:
    """Retorna ``value`` como objeto JSON, ou falha com o caminho do campo."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be a JSON object.")
    return value


# Obtém uma lista JSON e recusa strings, que tecnicamente também são
# sequências em Python mas nunca representam listas de itens de manifest.
def sequence(
    document: Mapping[str, Any], field: str, source: str, default: Sequence[object] | None = None
) -> Sequence[object]:
    """Retorna ``document[field]`` como lista JSON, recusando strings."""
    value = document.get(field, default)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{source}.{field} must be a JSON array.")
    return value


# Obtém um campo string obrigatório, ou um default explícito quando o schema
# oferece valor estável. Evita coerção implícita de números e valores nulos.
def string(document: Mapping[str, Any], field: str, source: str, default: str | None = None) -> str:
    """Retorna um campo string não vazio, sem coerção de outros tipos."""
    value = document.get(field, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}.{field} must be a non-empty string.")
    return value


# Obtém um campo string opcional sem transformar valores nulos em strings.
# Isso preserva a diferença entre metadata ausente e uma referência declarada.
def optional_string(document: Mapping[str, Any], field: str, source: str) -> str | None:
    """Retorna um campo string opcional, preservando ``None`` como ausência."""
    value = document.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}.{field} must be a non-empty string when provided.")
    return value


# Obtém um campo booleano sem aceitar 0/1 ou strings, para que um flag de
# manifest nunca dependa de coerção.
def boolean(document: Mapping[str, Any], field: str, source: str, default: bool) -> bool:
    """Retorna um campo booleano estrito, com default explícito."""
    value = document.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"{source}.{field} must be a boolean.")
    return value


# Obtém um campo inteiro positivo, recusando booleanos (que são int em
# Python) e floats. Usado por resoluções e contagens de manifest.
def positive_integer(document: Mapping[str, Any], field: str, source: str) -> int:
    """Retorna um campo inteiro estritamente positivo."""
    value = document.get(field)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{source}.{field} must be a positive integer.")
    return value


# Valida uma URI relativa preservada no manifest. Existe para impedir que o
# manifest rastreado aponte para caminhos absolutos, URLs remotas ou segmentos
# de escape, mantendo todo artifact sob `datasets/raw/<dataset_id>/`.
def relative_artifact_uri(value: str, source: str) -> str:
    """Rejeita URIs absolutas, remotas ou com traversal, devolvendo a relativa válida."""
    parsed = urlsplit(value)
    path = PurePosixPath(parsed.path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or path.is_absolute():
        raise ValueError(f"{source}.artifact_uri must be a relative local path.")
    if not parsed.path or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{source}.artifact_uri must not contain path traversal.")
    return value
