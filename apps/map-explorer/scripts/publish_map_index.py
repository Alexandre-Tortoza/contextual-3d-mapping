"""Publicação de runs contextuais independentes para comparação no map-explorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for _source_root in (
    _REPOSITORY_ROOT / "modules" / "semantic-map" / "src",
    _REPOSITORY_ROOT / "modules" / "semantic-fusion" / "src",
):
    if _source_root.is_dir() and str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from semantic_map import (
    CONSOLIDATED_ARTIFACT_TYPE,
    PublishedContextRun,
    consolidate_context_runs,
    geometry_fingerprint,
)

INDEX_NAME = "index.json"
CONTEXT_TYPE = "contextual_rgb_lidar_slice"
GEOMETRY_DIRECTORY = "geometry"


# Valida o tipo antes de copiar arquivos para que mapas de geometria pura não
# apareçam como runs contextuais nem substituam uma comparação existente.
def read_context(artifact: Path) -> dict:
    """Lê um mapa contextual com observações e geometria explícitas.

    Argumentos:
        artifact: arquivo JSON de contexto que será publicado.
    Retorna:
        conteúdo validado do artifact.
    Levanta:
        ValueError: se não houver um mapa contextual completo.
    """
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("artifact_type") != CONTEXT_TYPE:
        raise ValueError(f"{artifact}: selecione um mapa com contexto.")
    if (payload.get("schema_version") != 2 or not isinstance(payload.get("map_id"), str)
            or not isinstance(payload.get("map_frame"), str)):
        raise ValueError(f"{artifact}: mapa contextual sem schema ou identidade válidos.")
    if not isinstance(payload.get("points"), list) or not isinstance(payload.get("regions"), list):
        raise ValueError(f"{artifact}: mapa contextual sem points ou regions.")
    if not isinstance(payload.get("observations"), list) or not payload["observations"]:
        raise ValueError(f"{artifact}: mapa contextual sem observações.")
    # Uma run estruturalmente válida mas sem regiões ou pontos contextuais indica
    # falha silenciosa da percepção ou da associação; o viewer só recebe contexto real.
    if not payload["regions"]:
        raise ValueError(f"{artifact}: mapa contextual sem regiões; a percepção não produziu contexto.")
    contextual_points = payload.get("context_summary", {}).get("contextual_point_count")
    if not isinstance(contextual_points, int) or contextual_points <= 0:
        raise ValueError(f"{artifact}: mapa contextual sem pontos contextuais associados.")
    return payload


# Reúne somente os previews que o inspector consome. Mantém as URLs relativas
# intactas e exige arquivos locais para que cada pasta seja autossuficiente.
def preview_files(artifact: Path, payload: dict) -> dict[Path, Path]:
    """Resolve imagens relativas ao mapa, rejeitando arquivos fora da sua pasta.

    Retorna:
        mapeamento de paths relativos publicados para arquivos de origem.
    Levanta:
        ValueError: se um preview sair da pasta de origem ou usar URL externa.
        FileNotFoundError: se uma imagem referenciada estiver ausente.
    """
    previews = {}
    source_directory = artifact.parent.resolve()
    for observation in payload["observations"]:
        debug_assets = observation.get("debug_assets", {})
        if not isinstance(debug_assets, dict) or not all(isinstance(uri, str) for uri in debug_assets.values()):
            raise ValueError(f"{artifact}: debug_assets inválido.")
        for key, uri in (
            *( (key, observation.get(key)) for key in ("raw_image_uri", "overlay_image_uri") ),
            *( (f"debug_assets.{name}", value) for name, value in debug_assets.items() ),
        ):
            if not isinstance(uri, str) or not uri:
                raise ValueError(f"{artifact}: observação sem {key}.")
            parsed = urlsplit(uri)
            relative = Path(unquote(parsed.path))
            source = (source_directory / relative).resolve()
            if (not relative.parts or parsed.scheme or parsed.netloc or relative.is_absolute() or ".." in relative.parts
                    or not source.is_relative_to(source_directory)):
                raise ValueError(f"{artifact}: preview deve permanecer na pasta da run: {uri}")
            if relative.parts[0] in {"context.json", "manifest.json"}:
                raise ValueError(f"{artifact}: preview usa nome reservado da run: {uri}")
            if not source.is_file():
                raise FileNotFoundError(source)
            previews[relative] = source
    return previews


# Publica o fundo global declarado pela run em ``maps/geometry/<sha256>.json``. Existe
# porque cada trecho carrega só a própria vizinhança e o mapa geral precisa da
# geometria do corredor inteiro; o arquivo é compartilhado pelas runs da mesma origem
# e só é aceito com o digest declarado, a mesma origem e o tipo de slice geométrico.
def publish_backdrop(public_directory: Path, payload: dict) -> str | None:
    """Copia o fundo global da run para o catálogo e retorna seu digest.

    Argumentos:
        public_directory: raiz estática do viewer.
        payload: artifact contextual já validado.
    Retorna:
        sha256 do fundo publicado, ou ``None`` quando a run não declara fundo.
    Levanta:
        ValueError: se o fundo divergir do digest, do tipo ou da origem da run.
        FileNotFoundError: se o arquivo declarado não existir.
    """
    declared = payload.get("geometry_backdrop")
    if declared is None:
        return None
    if not isinstance(declared, dict) or not isinstance(declared.get("sha256"), str) or not isinstance(declared.get("artifact_uri"), str):
        raise ValueError("geometry_backdrop deve declarar artifact_uri e sha256.")
    digest = declared["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("geometry_backdrop.sha256 inválido.")
    destination = public_directory / "maps" / GEOMETRY_DIRECTORY / f"{digest}.json"
    if destination.is_file():
        return digest
    source = Path(declared["artifact_uri"])
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"{source}: fundo global diferente do digest declarado pela run.")
    backdrop = json.loads(content)
    if backdrop.get("artifact_type") != "geometric_pcd_slice" or geometry_fingerprint(backdrop) != geometry_fingerprint(payload):
        raise ValueError(f"{source}: fundo global não é um slice da mesma origem da run.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return digest


# Publica mapa e previews juntos, sem sobrescrever a identidade de uma run.
# O fingerprint inclui as imagens: mudar um preview também cria outro resultado.
def save_run(
    public_directory: Path, artifact: Path, *, run_id: str | None = None, label: str | None = None,
) -> Path:
    """Salva uma run em runs/<run-id>/ com mapa, imagens e manifest.

    Argumentos:
        public_directory: raiz estática do viewer.
        artifact: mapa contextual completo de origem.
        run_id: identidade opcional; por default, data UTC e fingerprint.
        label: nome legível opcional para distinguir execução ou ablação.
    Retorna:
        pasta publicada, reutilizada se o mesmo conteúdo já estiver salvo.
    Levanta:
        FileExistsError: se a identidade solicitada já contiver outro resultado.
    """
    payload = read_context(artifact)
    previews = preview_files(artifact, payload)
    backdrop_sha256 = publish_backdrop(public_directory, payload)
    artifact_bytes = artifact.read_bytes()
    asset_hashes = {str(path): hashlib.sha256(source.read_bytes()).hexdigest()
                    for path, source in sorted(previews.items())}
    fingerprint = hashlib.sha256(
        artifact_bytes + json.dumps(asset_hashes, sort_keys=True).encode(),
    ).hexdigest()
    runs_directory = public_directory / "runs"
    runs_directory.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        for previous in sorted(runs_directory.glob("*/manifest.json")):
            if previous.parent.name.startswith("."):
                continue
            saved = json.loads(previous.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == fingerprint:
                return previous.parent
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{fingerprint[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id deve conter apenas letras, números, ponto, hífen ou underscore.")
    destination = runs_directory / run_id
    if destination.exists():
        previous = destination / "manifest.json"
        if previous.is_file() and json.loads(previous.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
            return destination
        raise FileExistsError(f"A run {run_id!r} já existe com outro conteúdo; escolha outro run-id.")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=runs_directory))
    try:
        (staging / "context.json").write_bytes(artifact_bytes)
        for relative, source in previews.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        manifest = {
            "schema_version": 1, "artifact_type": CONTEXT_TYPE, "run_id": run_id,
            "label": label or run_id, "created_at": datetime.now(UTC).isoformat(),
            "map_id": payload["map_id"], "map_frame": payload.get("map_frame"),
            "frame_count": len(payload["observations"]),
            "contextual_point_count": payload.get("context_summary", {}).get("contextual_point_count"),
            "source_artifact": str(artifact.resolve()), "artifact": "context.json",
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "assets_sha256": asset_hashes, "fingerprint": fingerprint,
            "geometry_backdrop_sha256": backdrop_sha256,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


# Lê as runs estáticas já validadas pelo publisher. Existe para que o mapa
# consolidado use somente conteúdo imutável e previews que o viewer consegue
# abrir, em vez de varrer artifacts de trabalho ainda mutáveis.
def published_context_runs(public_directory: Path) -> tuple[PublishedContextRun, ...]:
    """Retorna as runs contextuais completas disponíveis no catálogo estático."""
    runs = []
    for manifest_path in sorted((public_directory / "runs").glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        artifact = manifest_path.parent / "context.json"
        if not artifact.is_file():
            continue
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if metadata.get("artifact_type") != CONTEXT_TYPE or payload.get("artifact_type") != CONTEXT_TYPE:
            continue
        run_id = metadata.get("run_id")
        if not isinstance(run_id, str) or run_id != manifest_path.parent.name:
            continue
        digest = metadata.get("artifact_sha256")
        if not isinstance(digest, str):
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        runs.append(PublishedContextRun(
            run_id=run_id,
            label=str(metadata.get("label") or run_id),
            artifact_url=f"/runs/{quote(run_id)}/context.json",
            artifact_sha256=digest,
            payload=payload,
        ))
    return tuple(runs)


# Escreve um documento derivado sem expor ao viewer um arquivo parcial durante
# a reconstrução do catálogo. O conteúdo pode ser regenerado integralmente das
# runs imutáveis de origem.
def _write_json_atomic(destination: Path, payload: dict) -> None:
    """Grava JSON em arquivo temporário e o promove atomicamente."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)


# Escolhe o fundo global de um grupo: o que todas as runs do grupo declaram. Runs
# sem fundo, ou com fundos diferentes, consolidam pela união dos próprios pontos.
def _group_backdrop(public_directory: Path, runs: list[PublishedContextRun]) -> dict | None:
    """Retorna o slice global compartilhado pelo grupo, ou ``None``."""
    digests = {
        (run.payload.get("geometry_backdrop") or {}).get("sha256") for run in runs
    }
    if len(digests) != 1 or None in digests:
        return None
    path = public_directory / "maps" / GEOMETRY_DIRECTORY / f"{digests.pop()}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# Reconstrói um mapa derivado por fingerprint geométrico. A separação por
# fingerprint impede que duas runs de mesmo nome lógico, mas nuvens diferentes,
# sejam fundidas por acidente.
def publish_consolidated_maps(public_directory: Path) -> list[dict]:
    """Atualiza os mapas consolidados derivados das runs publicadas.

    Retorna:
        entradas prontas para o índice estático do viewer.
    """
    grouped: dict[str, list[PublishedContextRun]] = {}
    for run in published_context_runs(public_directory):
        fingerprint = geometry_fingerprint(run.payload)
        grouped.setdefault(fingerprint, []).append(run)
    entries = []
    for fingerprint, runs in sorted(grouped.items()):
        consolidated = consolidate_context_runs(runs, backdrop=_group_backdrop(public_directory, runs))
        destination = public_directory / "maps" / "consolidated" / fingerprint / "context.json"
        _write_json_atomic(destination, consolidated.to_payload())
        source_metadata = [
            json.loads((public_directory / "runs" / run.run_id / "manifest.json").read_text(encoding="utf-8"))
            for run in runs
        ]
        created_at = max(str(item.get("created_at", "")) for item in source_metadata)
        manifest = {
            "schema_version": 1,
            "artifact_type": CONSOLIDATED_ARTIFACT_TYPE,
            "geometry_fingerprint": fingerprint,
            "map_id": consolidated.payload["map_id"],
            "map_frame": consolidated.payload["map_frame"],
            "source_run_ids": [run.run_id for run in sorted(runs, key=lambda item: item.run_id)],
            "source_run_count": len(runs),
            "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
        _write_json_atomic(destination.with_name("manifest.json"), manifest)
        entries.append({
            "url": f"/maps/consolidated/{quote(fingerprint)}/context.json",
            "label": f"Mapa consolidado · {consolidated.payload['map_id']} · {len(runs)} {'run' if len(runs) == 1 else 'runs'}",
            "artifact_type": CONSOLIDATED_ARTIFACT_TYPE,
            "map_id": consolidated.payload["map_id"],
            "created_at": created_at,
            "source_run_count": len(runs),
            "geometry_fingerprint": fingerprint,
            "frame_count": sum(int(item.get("frame_count", 0)) for item in source_metadata),
            "contextual_point_count": consolidated.payload["context_summary"]["contextual_point_count"],
        })
    return entries


# Indexa apenas as pastas publicadas de contexto. Arquivos antigos de geometria
# podem continuar no disco sem ocupar o seletor de comparação de runs.
def publish(public_directory: Path) -> Path:
    """Grava o índice das runs contextuais, da mais recente para a mais antiga.

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
    entries = []
    for manifest_path in (public_directory / "runs").glob("*/manifest.json"):
        if manifest_path.parent.name.startswith("."):
            continue
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("artifact_type") != CONTEXT_TYPE or not (manifest_path.parent / "context.json").is_file():
            continue
        frames = metadata["frame_count"]
        entries.append({
            "url": f"/runs/{quote(manifest_path.parent.name)}/context.json",
            "label": f"{metadata['run_id']} · {frames} {'frame' if frames == 1 else 'frames'}",
            "artifact_type": CONTEXT_TYPE, "run_id": metadata["run_id"],
            "created_at": metadata["created_at"], "map_id": metadata["map_id"],
            "frame_count": frames, "contextual_point_count": metadata.get("contextual_point_count"),
        })
    entries.sort(key=lambda item: (item["created_at"], item["run_id"]), reverse=True)
    entries.extend(publish_consolidated_maps(public_directory))
    destination = maps_directory / INDEX_NAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


# Ponto de entrada usado pelo alvo `map-explorer-serve` do Makefile.
def main() -> None:
    """Salva uma run opcional e atualiza o índice servido pelo viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_directory", type=Path, help="diretório estático servido")
    parser.add_argument("--artifact", type=Path, help="mapa com contexto a salvar em uma pasta própria")
    parser.add_argument("--run-id", help="identidade da run; nunca sobrescreve outro resultado")
    parser.add_argument("--label", help="nome legível da execução ou ablação")
    parser.add_argument("--result-file", type=Path, help="arquivo de resposta JSON para a CLI")
    arguments = parser.parse_args()
    if arguments.result_file is not None and arguments.artifact is None:
        parser.error("--result-file exige --artifact")
    arguments.public_directory.mkdir(parents=True, exist_ok=True)
    if arguments.artifact is not None:
        saved = save_run(arguments.public_directory, arguments.artifact, run_id=arguments.run_id, label=arguments.label)
        print(saved)
    print(publish(arguments.public_directory))
    if arguments.result_file is not None:
        metadata = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
        metadata.update(path=str(saved.resolve()), url=f"/runs/{quote(saved.name)}/context.json")
        arguments.result_file.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
