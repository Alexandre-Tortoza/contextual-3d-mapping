"""Publicação do índice de mapas servidos localmente pelo map-explorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INDEX_NAME = "index.json"


# Descreve um artifact publicado pelo nome que o usuário lê no seletor. O rótulo
# é derivado do próprio arquivo — e não de constantes no frontend — para que um
# trecho novo apareça corretamente sem alterar código do viewer.
def describe(artifact: Path) -> str:
    """Deriva o rótulo apresentado no seletor de mapas.

    Argumentos:
        artifact: arquivo JSON publicado no diretório servido.
    Retorna:
        rótulo legível que distingue geometria pura de artifact contextual.
    Levanta:
        ValueError: se o arquivo não for um slice de mapa reconhecível.
    """
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    map_id = payload.get("map_id")
    if not isinstance(map_id, str):
        raise ValueError(f"{artifact} does not declare a map_id.")
    if payload.get("artifact_type") != "contextual_rgb_lidar_slice":
        return f"Geometria · {map_id}"
    frames = payload.get("context_summary", {}).get("visual_observation_count", 0)
    unit = "frame" if frames == 1 else "frames"
    return f"Contexto · {frames} {unit} · {map_id}"


# Varre o diretório servido e escreve o índice consumido pelo seletor. Existe
# porque o conjunto de mapas publicados muda a cada trecho gerado, e manter essa
# lista no frontend faria o viewer mentir sobre o que está disponível.
def publish(public_directory: Path) -> Path:
    """Grava o índice dos mapas presentes no diretório servido.

    Argumentos:
        public_directory: diretório estático servido pelo cliente web.
    Retorna:
        caminho do índice publicado.
    Levanta:
        FileNotFoundError: se o diretório servido não existir.
    """
    if not public_directory.is_dir():
        raise FileNotFoundError(public_directory)
    maps_directory = public_directory / "maps"
    maps_directory.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    current = public_directory / "current-map.json"
    if current.is_file():
        entries.append({"url": "/current-map.json", "label": describe(current)})
    for artifact in sorted(maps_directory.glob("*.json")):
        if artifact.name == INDEX_NAME:
            continue
        entries.append({"url": f"/maps/{artifact.name}", "label": describe(artifact)})
    destination = maps_directory / INDEX_NAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


# Ponto de entrada usado pelo alvo `map-explorer-serve` do Makefile.
def main() -> None:
    """Publica o índice de mapas do diretório informado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_directory", type=Path, help="diretório estático servido")
    arguments = parser.parse_args()
    print(publish(arguments.public_directory))


if __name__ == "__main__":
    main()
