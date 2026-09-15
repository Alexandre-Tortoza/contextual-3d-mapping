"""Execução observável e cancelável de processos externos."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


# Sinaliza falha de um comando mantendo código e argv para diagnóstico da UI.
class ProcessFailure(RuntimeError):
    """Falha acionável de um processo externo.

    Argumentos:
        command: argv executado.
        returncode: código POSIX devolvido.
    """

    # Preserva os dados estruturados usados pelos testes e mensagens finais.
    def __init__(self, command: Sequence[str], returncode: int) -> None:
        """Inicializa a falha com comando e código de saída."""
        self.command = tuple(command)
        self.returncode = returncode
        super().__init__(f"comando falhou com código {returncode}: {' '.join(command)}")


# Executa comandos longos em foreground herdando stdout/stderr. A classe é
# pequena para permitir substituição por fake nos testes dos workflows.
class ProcessRunner:
    """Executor de subprocessos em foreground."""

    # Executa e aguarda o processo, encerrando o filho ao receber Ctrl+C.
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        """Executa um comando e levanta ``ProcessFailure`` quando necessário.

        Argumentos:
            command: argv sem shell.
            cwd: diretório de trabalho.
            environment: variáveis adicionais.
        Levanta:
            ProcessFailure: se o processo terminar com código diferente de zero.
            KeyboardInterrupt: depois de encerrar um filho cancelado pelo usuário.
        """
        merged_environment = os.environ.copy()
        if environment is not None:
            merged_environment.update(environment)
        process = subprocess.Popen(tuple(command), cwd=cwd, env=merged_environment)
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if returncode != 0:
            raise ProcessFailure(command, returncode)
