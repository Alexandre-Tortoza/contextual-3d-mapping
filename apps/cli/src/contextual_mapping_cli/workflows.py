"""Workflows operacionais compartilhados por menus e subcomandos."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from urllib.parse import urlencode
from urllib.request import urlopen

from .models import CostEstimate, DatasetProfile, ExtractionRequest, ExtractionResult
from .processes import ProcessRunner
from .project import Project

if TYPE_CHECKING:
    from contextual_mapping_adapters import RosbagRecording


# Coordena as APIs públicas e os processos externos sem conter algoritmos de
# mapeamento ou percepção. É a fronteira testável consumida pela UI e pelo CLI.
class Workflows:
    """Orquestra extração, percepção, composição e publicação.

    Argumentos:
        project: estado local e convenções do checkout.
        runner: executor substituível de processos longos.
    """

    # Guarda dependências explícitas e ativa os pacotes do checkout somente
    # quando a aplicação é construída.
    def __init__(self, project: Project, runner: ProcessRunner | None = None) -> None:
        """Inicializa os workflows para um projeto local."""
        self.project = project
        self.runner = runner or ProcessRunner()
        self.project.activate_source_packages()

    # Inspeciona a rosbag por meio do adapter dono da integração.
    def inspect_bag(self, bag: Path) -> RosbagRecording:
        """Retorna os metadados públicos da rosbag informada.

        Argumentos:
            bag: arquivo a inspecionar.
        Retorna:
            ``RosbagRecording`` publicado pelo adapter de datasets.
        """
        from contextual_mapping_adapters import inspect_rosbag

        return inspect_rosbag(bag)

    # Estima a cardinalidade antes da resolução completa para que o modo caro
    # exija confirmação antes de desserializar ou gravar milhares de imagens.
    def estimate_extraction(self, request: ExtractionRequest) -> CostEstimate:
        """Estima quantidade e armazenamento dos PNGs de uma extração.

        Argumentos:
            request: seleção temporal e política de frames.
        Retorna:
            estimativa conservadora baseada no stream RGB.
        """
        from contextual_mapping_adapters import select_rgb_topic

        recording = self.inspect_bag(request.bag)
        topic = select_rgb_topic(recording, request.camera_topic)
        duration = recording.duration_s if request.duration_s is None else request.duration_s
        if request.all_frames:
            ratio = min(max(duration / recording.duration_s, 0.0), 1.0)
            count = max(1, round(topic.message_count * ratio))
        else:
            count = int(duration // request.keyframe_interval_s) + 1
        return CostEstimate(frame_count=count, storage_bytes=count * 1_000_000)

    # Resolve a janela com mapping-runtime e pede ao adapter apenas a conversão
    # dos frames selecionados. Paths finais só aparecem depois da extração completa.
    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """Resolve uma janela e extrai seus frames RGB.

        Argumentos:
            request: seleção validada da rosbag.
        Retorna:
            paths da janela e do diretório de frames.
        Levanta:
            FileExistsError: se a identidade já possuir artifacts.
            ValueError: se os limites ou intervalo forem inválidos.
        """
        from mapping_runtime.bag_window import export_bag_window, resolve_bag_window

        from contextual_mapping_adapters import extract_rosbag_frames, select_rgb_topic

        segment_id = self.project.validate_segment_id(request.segment_id)
        if request.keyframe_interval_s <= 0 and not request.all_frames:
            raise ValueError("keyframe-interval-s deve ser positivo")
        if request.duration_s is not None and request.duration_s <= 0:
            raise ValueError("duration-s deve ser positiva")
        recording = self.inspect_bag(request.bag)
        selected_topic = select_rgb_topic(recording, request.camera_topic)
        artifacts = self.project.root / "artifacts"
        window_path = artifacts / f"{segment_id}-window.json"
        frames_path = artifacts / f"{segment_id}-frames"
        temporary_frames = artifacts / f".{segment_id}-frames.tmp"
        if window_path.exists() or frames_path.exists() or temporary_frames.exists():
            raise FileExistsError(f"já existem artifacts para o segment-id {segment_id!r}")
        window = resolve_bag_window(
            request.bag,
            start_s=request.start_s,
            duration_s=request.duration_s,
            keyframe_interval_s=request.keyframe_interval_s,
            camera_topic=selected_topic.name,
            all_frames=request.all_frames,
            recording_id=request.bag.stem,
        )
        required_bytes = len(window.keyframes) * 1_000_000
        if shutil.disk_usage(artifacts).free < required_bytes:
            raise OSError(
                f"espaço insuficiente: estimados {required_bytes} bytes para {len(window.keyframes)} frames"
            )
        try:
            written = extract_rosbag_frames(
                request.bag,
                temporary_frames,
                topic=window.camera_topic,
                frame_indices_by_timestamp={
                    item.bag_timestamp_ns: item.sequence_index for item in window.keyframes
                },
                frame_id_prefix=window.frame_id_prefix,
            )
            temporary_frames.replace(frames_path)
        except Exception:
            if temporary_frames.exists():
                shutil.rmtree(temporary_frames)
            raise
        export_bag_window(window, window_path)
        return ExtractionResult(window_path, frames_path, len(written))

    # Deriva uma estimativa de GPU de manifests anteriores, usando a mediana
    # simples por frame para não depender de um benchmark fixo da máquina.
    def estimate_perception(self, frame_count: int) -> CostEstimate:
        """Estima duração e armazenamento de visual-perception.

        Argumentos:
            frame_count: quantidade que será processada.
        Retorna:
            estimativa baseada em runs locais, ou campos desconhecidos.
        """
        latencies: list[float] = []
        bytes_per_frame: list[int] = []
        for run in self.project.available_runs():
            try:
                payload = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                latencies.extend(
                    float(item["latency_s"])
                    for item in payload.get("frames", [])
                    if not item.get("failed") and item.get("latency_s") is not None
                )
                count = int(payload.get("frame_count", 0))
                if count:
                    size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
                    bytes_per_frame.append(size // count)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        latency = _median(latencies)
        output_size = _median(bytes_per_frame)
        return CostEstimate(
            frame_count=frame_count,
            duration_s=None if latency is None else latency * frame_count,
            storage_bytes=None if output_size is None else int(output_size * frame_count),
        )

    # Executa o harness canônico com argumentos explícitos e descobre o run que
    # ele publicou. O subprocesso preserva logs e ciclo de vida dos modelos.
    def run_perception(
        self,
        frames_dir: Path,
        sequence_masks: Path | None,
        frame_ids: tuple[str, ...] = (),
    ) -> Path:
        """Executa visual-perception sobre todos os PNGs de um diretório.

        Argumentos:
            frames_dir: imagens extraídas.
            sequence_masks: geometria do rig ou ``None`` para bag genérico.
            frame_ids: seleção explícita ou tupla vazia para todos.
        Retorna:
            diretório do novo run.
        Levanta:
            FileNotFoundError: se não houver frames ou ambiente Python.
        """
        frames = tuple(frames_dir.glob("*.png"))
        if not frames:
            raise FileNotFoundError(f"nenhum frame PNG encontrado em {frames_dir}")
        module = self.project.root / "modules" / "visual-perception"
        python = module / ".venv" / "bin" / "python"
        if not python.exists():
            raise FileNotFoundError(f"ambiente Python de visual-perception ausente: {python}")
        results = module / "benchmarks" / "results"
        before = set(self.project.available_runs())
        command = [
            str(python),
            "benchmarks/validate_reference_pipeline.py",
            "--frames-dir",
            str(frames_dir.resolve()),
            "--results-dir",
            str(results.resolve()),
        ]
        if sequence_masks is None:
            command.append("--no-sequence-masks")
        else:
            command.extend(("--sequence-masks", str(sequence_masks.resolve())))
        for frame_id in frame_ids:
            command.extend(("--frame-id", frame_id))
        self.runner.run(command, cwd=module)
        created = set(self.project.available_runs()) - before
        candidates = created or set(self.project.available_runs())
        if not candidates:
            raise FileNotFoundError("visual-perception terminou sem publicar um run")
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    # Executa o alvo existente de FAST-LIO com as variáveis do segmento. Só é
    # chamado quando o bag possui um DatasetProfile completo.
    def run_mapping(self, profile: DatasetProfile, result: ExtractionResult, segment_id: str) -> Path:
        """Gera o mapa geométrico do segmento configurado.

        Argumentos:
            profile: perfil compatível com a rosbag.
            result: extração cuja window será reproduzida.
            segment_id: identidade validada dos artifacts.
        Retorna:
            slice geométrico JSON esperado pela composição.
        """
        python = self.project.root / "modules" / "visual-perception" / ".venv" / "bin" / "python"
        command = [
            "make",
            profile.map_target,
            f"PYTHON={python}",
            f"SEGMENT_ID={segment_id}",
            f"M1_SEGMENT_WINDOW={result.window}",
        ]
        self.runner.run(command, cwd=self.project.root)
        destination = self.project.root / "artifacts" / f"{segment_id}.json"
        if not destination.is_file():
            raise FileNotFoundError(f"mapeamento não publicou {destination}")
        return destination

    # Compõe pelo alvo já usado no repositório e valida previamente todos os
    # artifacts para evitar carregar a rosbag com entradas incompatíveis.
    def compose(
        self,
        profile: DatasetProfile,
        *,
        segment_id: str,
        window: Path,
        visual_run: Path,
        visibility_mode: str = "measured_surfaces",
    ) -> Path:
        """Compõe o artifact contextual de um segmento conhecido.

        Argumentos:
            profile: configuração do dataset.
            segment_id: identidade dos artifacts geométricos.
            window: janela resolvida.
            visual_run: run correspondente de percepção.
            visibility_mode: superfícies medidas ou uma ablação de células explícita.
        Retorna:
            artifact contextual publicado.
        """
        if visibility_mode not in {"measured_surfaces", "dense_cells", "legacy_cells"}:
            raise ValueError("visibility-mode deve ser measured_surfaces, dense_cells ou legacy_cells.")
        segment_id = self.project.validate_segment_id(segment_id)
        geometry = self.project.root / "artifacts" / f"{segment_id}.json"
        for required in (window, visual_run / "manifest.json", geometry):
            if not required.exists():
                raise FileNotFoundError(f"entrada de composição ausente: {required}")
        python = self.project.root / "modules" / "visual-perception" / ".venv" / "bin" / "python"
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{segment_id}"
        destination = self.project.root / "artifacts" / "runs" / run_id / "context.json"
        command = [
            "make",
            profile.context_target,
            f"PYTHON={python}",
            f"SEGMENT_ID={segment_id}",
            f"M1_SEGMENT_WINDOW={window}",
            f"M1_VISUAL_RUN={visual_run}",
            f"M1_CONTEXT_ARTIFACT={destination}",
            f"M1_VISIBILITY_MODE={visibility_mode}",
        ]
        self.runner.run(command, cwd=self.project.root)
        if not destination.is_file():
            raise FileNotFoundError(f"composição não publicou {destination}")
        shutil.copyfile(window, destination.parent / "window.json")
        shutil.copyfile(visual_run / "manifest.json", destination.parent / "perception-manifest.json")
        (destination.parent / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "created_at": datetime.now(UTC).isoformat(),
            "segment_id": segment_id, "artifact": "context.json",
            "visual_run": str(visual_run.resolve()), "geometry": str(geometry),
            "window": "window.json", "perception_manifest": "perception-manifest.json",
            "visibility_mode": visibility_mode,
        }, indent=2), encoding="utf-8")
        self.publish_context(destination, run_id=run_id)
        return destination

    # Delega a cópia atômica ao publisher dono do formato servido. A mesma
    # operação é usada pelo comando publish e depois da composição contextual.
    def publish_context(
        self, artifact: Path, *, run_id: str | None = None, label: str | None = None,
    ) -> dict:
        """Salva mapa, imagens e proveniência em uma pasta própria do viewer.

        Retorna:
            metadados públicos e URL da run publicada.
        """
        command = [
            sys.executable, str(self.project.root / "apps/map-explorer/scripts/publish_map_index.py"),
            str(self.project.root / "apps/map-explorer/web/public"), "--artifact", str(artifact.resolve()),
        ]
        if run_id is not None:
            command.extend(("--run-id", run_id))
        if label is not None:
            command.extend(("--label", label))
        # A resposta do produtor identifica inclusive uma publicação idempotente
        # cuja origem atual é uma cópia do mesmo conteúdo em outro diretório.
        with TemporaryDirectory(prefix="contextual-publication-") as directory:
            result = Path(directory) / "result.json"
            command.extend(("--result-file", str(result)))
            self.runner.run(command, cwd=self.project.root)
            return json.loads(result.read_text(encoding="utf-8"))

    # Usa o catálogo contextual sem depender de um artifact geométrico separado.
    # Reutiliza o servidor deste catálogo quando ele já estiver disponível.
    def serve(
        self, artifact: Path | None = None, segment_id: str | None = None, *, run_id: str | None = None,
    ) -> str:
        """Abre o catálogo ou uma run com contexto, iniciando o servidor se preciso.

        Argumentos:
            artifact: mapa a publicar, quando ainda não estiver no catálogo.
            segment_id: opção histórica mantida por compatibilidade.
            run_id: identidade a publicar ou selecionar no catálogo existente.
        Retorna:
            URL compartilhável da run escolhida.
        """
        if segment_id is not None:
            self.project.validate_segment_id(segment_id)
        if artifact is not None:
            run_id = self.publish_context(artifact, run_id=run_id)["run_id"]
        entries = self.project.available_context_runs()
        if not entries:
            raise FileNotFoundError("nenhuma run com contexto publicada; use publish --artifact <mapa.json>")
        selected = next((entry for entry in entries if entry["run_id"] == run_id), None) if run_id else entries[0]
        if selected is None:
            raise ValueError(f"run desconhecida: {run_id}; use o comando runs para listar as disponíveis")
        url = "http://localhost:5173/?" + urlencode({"artifact": selected["url"]}, safe="/")
        print(f"Viewer: {url}", flush=True)
        index = self.project.root / "apps/map-explorer/web/public/maps/index.json"
        try:
            with urlopen("http://127.0.0.1:5173/maps/index.json", timeout=1) as response:
                if json.load(response) == json.loads(index.read_text(encoding="utf-8")):
                    return url
        except (OSError, ValueError):
            pass
        self.runner.run(
            ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173", "--strictPort"],
            cwd=self.project.root / "apps/map-explorer/web",
        )
        return url


# Calcula a mediana sem puxar uma dependência estatística para a aplicação.
def _median(values: list[float] | list[int]) -> float | None:
    """Retorna a mediana de valores numéricos ou ``None`` para entrada vazia."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2
