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
    _REPOSITORY_ROOT / "contracts",
):
    if _source_root.is_dir() and str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from semantic_map import (
    CONSOLIDATED_ARTIFACT_TYPE,
    PublishedContextRun,
    consolidate_context_runs,
    geometry_fingerprint,
)

from contextual_mapping_contracts import next_run_id

INDEX_NAME = "index.json"
CONTEXT_TYPE = "contextual_rgb_lidar_slice"
GEOMETRY_DIRECTORY = "geometry"
_MANUAL_RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# Acima deste teto, o mapa consolidado é servido em partições espaciais para que o
# viewer pinte a cena progressivamente em vez de baixar um único JSON gigante.
CHUNK_POINT_THRESHOLD = 60_000


# Valida os eixos de uma comparação antes de a run ser publicada. Existe para
# que o viewer possa montar uma matriz por backend a partir de dados estáveis,
# sem tentar deduzir a experiência pelo nome livre da run.
def _comparison_metadata(payload: dict, artifact: Path) -> dict | None:
    """Retorna metadados opcionais de comparação já validados.

    Argumentos:
        payload: artifact contextual de origem.
        artifact: caminho usado nas mensagens de validação.
    Retorna:
        ``None`` sem comparação, ou ``group_id`` e eixos textuais.
    Levanta:
        ValueError: se a declaração de comparação for incompleta ou inválida.
    """
    comparison = payload.get("comparison")
    if comparison is None:
        return None
    if not isinstance(comparison, dict):
        raise ValueError(f"{artifact}: comparison precisa ser um objeto.")
    group_id = comparison.get("group_id")
    axes = comparison.get("axes")
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError(f"{artifact}: comparison.group_id precisa ser texto não vazio.")
    if not isinstance(axes, dict) or not axes:
        raise ValueError(f"{artifact}: comparison.axes precisa conter ao menos um eixo.")
    if any(not isinstance(name, str) or not name.strip() or not isinstance(value, str) or not value.strip()
           for name, value in axes.items()):
        raise ValueError(f"{artifact}: comparison.axes precisa mapear nomes e valores textuais não vazios.")
    return {"group_id": group_id, "axes": dict(axes)}


# Particiona os pontos do mapa consolidado ao longo do eixo de maior extensão, em
# grupos contíguos de tamanho aproximadamente igual. Existe porque um mapa
# consolidado grande precisa ser servido em pedaços espacialmente coerentes para
# que o viewer possa pintar a cena progressivamente; ordenar pelo eixo mais longo
# aproxima bem a coerência espacial em corredores sem exigir uma estrutura
# espacial completa (k-d tree, octree) para este volume de dados.
def _spatial_chunks(points: list[dict], chunk_size: int) -> list[dict]:
    """Divide os pontos em grupos contíguos ao longo do eixo de maior extensão.

    Argumentos:
        points: pontos do mapa consolidado, já com coordinates_m.
        chunk_size: teto aproximado de pontos por grupo.
    Retorna:
        lista de ``{"points": [...], "bounds_m": {"min_m": [...], "max_m": [...]}}``.
    """
    coordinates = [point["coordinates_m"] for point in points]
    ranges = [max(c[axis] for c in coordinates) - min(c[axis] for c in coordinates) for axis in range(3)]
    axis = ranges.index(max(ranges))
    order = sorted(range(len(points)), key=lambda index: coordinates[index][axis])
    group_count = -(-len(points) // chunk_size)
    group_size = -(-len(points) // group_count)
    chunks = []
    for start in range(0, len(points), group_size):
        indices = order[start:start + group_size]
        chunk_points = [points[index] for index in indices]
        chunk_coordinates = [point["coordinates_m"] for point in chunk_points]
        chunks.append({
            "points": chunk_points,
            "bounds_m": {
                "min_m": [min(c[axis] for c in chunk_coordinates) for axis in range(3)],
                "max_m": [max(c[axis] for c in chunk_coordinates) for axis in range(3)],
            },
        })
    return chunks


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
    if payload.get("schema_version") == 2:
        raise ValueError(f"{artifact}: schema_version 2 não é mais aceito; re-rode o pipeline de composição.")
    if (payload.get("schema_version") != 3 or not isinstance(payload.get("map_id"), str)
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
    _comparison_metadata(payload, artifact)
    return payload


# Coleta toda folha string de um nó de ``debug_manifest``, recursivamente.
# Existe porque o manifest tem profundidade e forma variáveis por etapa
# (dicts aninhados, listas de dicts como ``discovery_tiles``/``region_views``),
# e uma etapa desconhecida precisa ser aceita sem que o publisher precise
# conhecer seu formato — só o tipo do valor importa, nunca o nome da chave.
def _debug_manifest_uri_candidates(node: object) -> list[str]:
    """Retorna toda folha string de um nó de ``debug_manifest``.

    Argumentos:
        node: ``debug_manifest`` completo, ou qualquer sub-nó dele.
    Retorna:
        strings-folha encontradas, na ordem de travessia; nem toda folha é
        necessariamente uma URI de arquivo (identificadores como ``id`` e
        ``region_id`` também são strings) — quem resolve decide isso pela
        existência do arquivo, não por aqui.
    """
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [uri for value in node.values() for uri in _debug_manifest_uri_candidates(value)]
    if isinstance(node, list):
        return [uri for item in node for uri in _debug_manifest_uri_candidates(item)]
    return []


# Reúne somente os previews que o inspector consome. Mantém as URLs relativas
# intactas e exige arquivos locais para que cada pasta seja autossuficiente.
def preview_files(artifact: Path, payload: dict) -> dict[Path, Path]:
    """Resolve imagens relativas ao mapa, rejeitando arquivos fora da sua pasta.

    Argumentos:
        artifact: mapa contextual de origem, usado só para mensagens de erro.
        payload: conteúdo já validado por ``read_context``.
    Retorna:
        mapeamento de paths relativos publicados para arquivos de origem.
    Levanta:
        ValueError: se um preview tentar sair da pasta de origem, usar URL
            externa ou usar um nome reservado da run.
        FileNotFoundError: se ``raw_image_uri``/``overlay_image_uri`` da
            observação estiver ausente — essas duas são sempre exigidas. Uma
            folha de ``debug_manifest`` que não resolve para um arquivo
            existente é ignorada em silêncio: nem toda folha string é uma URI
            (``id``, ``region_id``), e todo o resto do manifest é best-effort.
    """
    previews: dict[Path, Path] = {}
    source_directory = artifact.parent.resolve()

    # Valida o sandboxing de um candidato a preview e, quando o arquivo
    # existe, registra-o; quando não existe, levanta apenas se `required`
    # (as duas URIs sempre exigidas por observação) — o resto é best-effort.
    def _register(uri: str, *, description: str, required: bool) -> None:
        if not isinstance(uri, str) or not uri:
            if required:
                raise ValueError(f"{artifact}: observação sem {description}.")
            return
        parsed = urlsplit(uri)
        relative = Path(unquote(parsed.path))
        source = (source_directory / relative).resolve()
        if (not relative.parts or parsed.scheme or parsed.netloc or relative.is_absolute() or ".." in relative.parts
                or not source.is_relative_to(source_directory)):
            raise ValueError(f"{artifact}: preview deve permanecer na pasta da run: {uri}")
        if relative.parts[0] in {"context.json", "manifest.json"}:
            raise ValueError(f"{artifact}: preview usa nome reservado da run: {uri}")
        if not source.is_file():
            if required:
                raise FileNotFoundError(source)
            return
        previews[relative] = source

    for observation in payload["observations"]:
        for key in ("raw_image_uri", "overlay_image_uri"):
            _register(observation.get(key), description=key, required=True)
        debug_manifest = observation.get("debug_manifest", {})
        if debug_manifest and not isinstance(debug_manifest, dict):
            raise ValueError(f"{artifact}: debug_manifest inválido.")
        for uri in _debug_manifest_uri_candidates(debug_manifest):
            _register(uri, description="debug_manifest", required=False)

    payload_debug_manifest = payload.get("debug_manifest", {})
    if payload_debug_manifest and not isinstance(payload_debug_manifest, dict):
        raise ValueError(f"{artifact}: debug_manifest inválido.")
    for uri in _debug_manifest_uri_candidates(payload_debug_manifest):
        _register(uri, description="debug_manifest", required=False)

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
        name = label or "manual"
        if not _MANUAL_RUN_NAME.fullmatch(name):
            raise ValueError(
                "label deve conter apenas letras minúsculas, números, ponto, hífen ou "
                "underscore para nomear uma run publicada sem --run-id."
            )
        existing_run_ids = (path.name for path in runs_directory.glob("*") if not path.name.startswith("."))
        run_id = str(next_run_id(existing_run_ids, today=datetime.now(UTC).date(), name=name))
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
        comparison = _comparison_metadata(payload, artifact)
        if comparison is not None:
            manifest["comparison"] = comparison
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
        payload = consolidated.to_payload()
        geometry_directory = destination.parent / GEOMETRY_DIRECTORY
        if len(payload["points"]) > CHUNK_POINT_THRESHOLD:
            if geometry_directory.is_dir():
                shutil.rmtree(geometry_directory)
            geometry_directory.mkdir(parents=True, exist_ok=True)
            manifest_entries = []
            for index, chunk in enumerate(_spatial_chunks(payload["points"], CHUNK_POINT_THRESHOLD)):
                chunk_id = f"chunk-{index:04d}"
                _write_json_atomic(geometry_directory / f"{chunk_id}.json", {"points": chunk["points"]})
                manifest_entries.append({
                    "chunk_id": chunk_id,
                    "bounds_m": chunk["bounds_m"],
                    "point_count": len(chunk["points"]),
                    "url": f"/maps/consolidated/{quote(fingerprint)}/{GEOMETRY_DIRECTORY}/{chunk_id}.json",
                })
            payload = {**payload, "points": [], "geometry_manifest": manifest_entries}
        elif geometry_directory.is_dir():
            shutil.rmtree(geometry_directory)
        _write_json_atomic(destination, payload)
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
            **({"comparison": metadata["comparison"]} if metadata.get("comparison") is not None else {}),
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
