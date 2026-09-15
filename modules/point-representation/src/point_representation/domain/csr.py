"""Validação compartilhada de offsets estilo CSR de fronteira de amostra.

``BatchedPointEmbeddings``, ``PointBatch`` e os contracts do port
``PointEncoder`` (#19) todos particionam uma sequência flat de pontos em
amostras da mesma forma — offsets crescentes começando em 0 e terminando na
contagem total. Esta é a única implementação dessa regra, para que as três
nunca divirjam sobre o que conta como um offset válido.
"""

from __future__ import annotations


# Valida que ``offsets`` descreve uma partição CSR válida de ``total``
# elementos: ao menos uma amostra, começando em 0, terminando em ``total``,
# e não-decrescente.
def validate_csr_offsets(offsets: tuple[int, ...], total: int, *, field: str) -> None:
    """Valida que ``offsets`` particiona ``total`` elementos em amostras CSR.

    Argumentos:
        offsets: offsets propostos, com ``len(offsets) == num_samples + 1``.
        total: contagem total de elementos que ``offsets`` deve cobrir.
        field: nome usado nas mensagens de erro.
    Levanta:
        ValueError: se ``offsets`` não descrever uma partição válida.
    """
    if len(offsets) < 2:
        raise ValueError(f"{field} must describe at least one sample (>= 2 entries).")
    if offsets[0] != 0:
        raise ValueError(f"{field} must start at 0.")
    if offsets[-1] != total:
        raise ValueError(f"{field} must end at {total}, got {offsets[-1]}.")
    if any(b < a for a, b in zip(offsets, offsets[1:], strict=False)):
        raise ValueError(f"{field} must be non-decreasing.")
