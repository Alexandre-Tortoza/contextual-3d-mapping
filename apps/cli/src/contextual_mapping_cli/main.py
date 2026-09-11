"""Menu e subcomandos da CLI de mapeamento contextual 3D."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .models import CostEstimate, DatasetProfile, ExtractionRequest, ExtractionResult
from .project import Project
from .workflows import Workflows

app = typer.Typer(
    name="contextual-3d-mapping-cli",
    help="Guia os workflows operacionais do mapeamento contextual 3D.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


# Formata bytes em unidades legíveis para resumos de custo apresentados antes
# das operações longas.
def _format_bytes(value: int | None) -> str:
    """Formata uma quantidade de bytes ou informa que ela é desconhecida."""
    if value is None:
        return "indisponível"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return "indisponível"


# Formata duração aproximada sem prometer precisão quando não existe histórico.
def _format_duration(value: float | None) -> str:
    """Formata segundos em uma duração curta ou retorna ``indisponível``."""
    if value is None:
        return "indisponível"
    if value < 60:
        return f"{value:.0f} s"
    hours, remainder = divmod(int(value), 3600)
    minutes = remainder // 60
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


# Renderiza a estimativa em um bloco comum aos modos interativo e programático.
def _show_estimate(title: str, estimate: CostEstimate) -> None:
    """Mostra cardinalidade, armazenamento e duração de uma operação."""
    table = Table(title=title)
    table.add_column("Frames")
    table.add_column("Armazenamento estimado")
    table.add_column("Tempo estimado")
    table.add_row(
        str(estimate.frame_count),
        _format_bytes(estimate.storage_bytes),
        _format_duration(estimate.duration_s),
    )
    console.print(table)


# Centraliza a regra de confirmação forte usada por subcomandos e pelo wizard.
def _validate_all_frames_confirmation(request: ExtractionRequest, confirmation: str | None) -> None:
    """Exige o segment-id exato para uma extração de todos os frames.

    Argumentos:
        request: pedido que pode selecionar todos os frames.
        confirmation: texto fornecido pelo usuário ou automação.
    Levanta:
        ValueError: quando o modo integral não foi confirmado exatamente.
    """
    if request.all_frames and confirmation != request.segment_id:
        raise ValueError("o modo --all-frames exige --confirm-all-frames com o segment-id exato")


# Constrói as dependências reais em um único lugar para manter os comandos
# estreitos e permitir que testes exercitem Workflows separadamente.
def _workflows(root: Path | None = None) -> Workflows:
    """Cria os workflows para a raiz informada ou descoberta."""
    return Workflows(Project(root))


# Converte falhas operacionais em mensagens curtas e código POSIX sem traceback
# para o usuário final; testes das camadas internas continuam vendo as exceções.
def _fail(error: Exception) -> None:
    """Mostra um erro acionável e encerra o comando com código 1."""
    console.print(f"[bold red]Erro:[/bold red] {error}")
    raise typer.Exit(1)


# Executa a parte compartilhada dos comandos extract e process.
def _extract_request(
    workflows: Workflows,
    request: ExtractionRequest,
    confirmation: str | None,
    *,
    show_estimate: bool = True,
) -> ExtractionResult:
    """Valida, resume e executa uma extração.

    Argumentos:
        workflows: orquestrador configurado.
        request: seleção temporal e de frames.
        confirmation: confirmação forte quando necessária.
        show_estimate: renderiza o custo quando o chamador ainda não o mostrou.
    Retorna:
        artifacts publicados.
    """
    _validate_all_frames_confirmation(request, confirmation)
    if show_estimate:
        estimate = workflows.estimate_extraction(request)
        _show_estimate("Extração", estimate)
    result = workflows.extract(request)
    console.print(
        f"[green]Extraídos {result.frame_count} frames[/green] em {result.frames_dir}\n"
        f"Window: {result.window}"
    )
    return result


# Expõe a extração reproduzível para scripts, incluindo trecho ou bag inteiro
# e amostragem ou todos os frames.
@app.command("extract")
def extract_command(
    bag: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Rosbag de origem.")],
    segment_id: Annotated[str, typer.Option(help="Identidade dos artifacts.")],
    start_s: Annotated[float, typer.Option(min=0.0, help="Início relativo ao primeiro RGB.")] = 0.0,
    duration_s: Annotated[float | None, typer.Option(min=0.0, help="Duração do trecho.")] = None,
    whole_bag: Annotated[bool, typer.Option(help="Usa todo o stream RGB.")] = False,
    interval_s: Annotated[float, typer.Option(min=0.001, help="Espaçamento dos keyframes.")] = 2.0,
    all_frames: Annotated[bool, typer.Option(help="Extrai todas as imagens da janela.")] = False,
    camera_topic: Annotated[str | None, typer.Option(help="Tópico RGB; detectado quando ausente.")] = None,
    confirm_all_frames: Annotated[
        str | None, typer.Option(help="Repita o segment-id no modo integral.")
    ] = None,
    root: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Extrai keyframes de um trecho ou de uma rosbag inteira."""
    try:
        if whole_bag == (duration_s is not None):
            raise ValueError("informe exatamente um de --whole-bag ou --duration-s")
        workflows = _workflows(root)
        request = ExtractionRequest(
            bag=bag.resolve(),
            segment_id=segment_id,
            start_s=start_s,
            duration_s=None if whole_bag else duration_s,
            keyframe_interval_s=interval_s,
            all_frames=all_frames,
            camera_topic=camera_topic,
        )
        _extract_request(workflows, request, confirm_all_frames)
    except Exception as error:
        _fail(error)


# Executa a percepção sobre todos ou alguns frames já materializados.
@app.command("perception")
def perception_command(
    frames_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, help="Diretório de PNGs.")],
    sequence_masks: Annotated[Path | None, typer.Option(help="Geometria de área da sequência.")] = None,
    frame_id: Annotated[list[str] | None, typer.Option(help="ID de frame; pode ser repetido.")] = None,
    root: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Roda visual-perception em foreground sobre frames extraídos."""
    try:
        workflows = _workflows(root)
        count = len(frame_id or tuple(frames_dir.glob("*.png")))
        _show_estimate("Visual-perception", workflows.estimate_perception(count))
        run = workflows.run_perception(frames_dir, sequence_masks, tuple(frame_id or ()))
        console.print(f"[green]Run publicado:[/green] {run}")
    except Exception as error:
        _fail(error)


# Resolve um perfil por identidade para impedir composição acidental com a
# calibração de outro dataset.
def _profile(workflows: Workflows, profile_id: str) -> DatasetProfile:
    """Busca um perfil conhecido e falha com uma lista acionável."""
    profile = next(
        (item for item in workflows.project.available_profiles() if item.profile_id == profile_id),
        None,
    )
    if profile is None:
        raise ValueError(f"perfil desconhecido: {profile_id}")
    return profile


# Expõe a composição contextual independente do wizard para reproduzir um run.
@app.command("compose")
def compose_command(
    segment_id: Annotated[str, typer.Option(help="Identidade do segmento.")],
    window: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Artifact de janela.")],
    visual_run: Annotated[Path, typer.Option(exists=True, file_okay=False, help="Run de percepção.")],
    profile_id: Annotated[str, typer.Option(help="Perfil configurado do dataset.")] = "corridor-02",
    root: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Compõe percepção e geometria em um artifact contextual."""
    try:
        workflows = _workflows(root)
        artifact = workflows.compose(
            _profile(workflows, profile_id),
            segment_id=segment_id,
            window=window,
            visual_run=visual_run,
        )
        console.print(f"[green]Artifact contextual:[/green] {artifact}")
    except Exception as error:
        _fail(error)


# Inicia o viewer como processo foreground para manter logs e Ctrl+C visíveis.
@app.command("serve")
def serve_command(
    artifact: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Mapa contextual.")],
    segment_id: Annotated[str, typer.Option(help="Identidade da geometria.")],
    root: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Publica um artifact e serve o viewer web em foreground."""
    try:
        console.print("Viewer: [link=http://localhost:5173]http://localhost:5173[/link]")
        _workflows(root).serve(artifact, segment_id)
    except Exception as error:
        _fail(error)


# Mostra a integridade local e opcionalmente repara apenas diretórios gerados.
@app.command("validate")
def validate_command(
    repair: Annotated[bool, typer.Option(help="Cria diretórios locais seguros ausentes.")] = False,
    root: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Valida requisitos e estrutura local do projeto."""
    try:
        project = Project(root)
        if repair:
            for path in project.repair():
                console.print(f"[green]Criado:[/green] {path}")
        issues = project.validate()
        if not issues:
            console.print("[green]Estrutura válida.[/green]")
            return
        for issue in issues:
            color = "red" if issue.level == "erro" else "yellow"
            suffix = " (reparável)" if issue.repairable else ""
            console.print(f"[{color}]{issue.level.upper()}[/{color}]: {issue.message}{suffix}")
        if any(issue.level == "erro" for issue in issues):
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as error:
        _fail(error)


# Encadeia o fluxo completo quando há perfil e termina após percepção para uma
# bag genérica, tornando a limitação explícita em vez de usar calibração errada.
@app.command("process")
def process_command(
    bag: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Rosbag de origem.")],
    segment_id: Annotated[str, typer.Option(help="Identidade dos artifacts.")],
    start_s: Annotated[float, typer.Option(min=0.0)] = 0.0,
    duration_s: Annotated[float | None, typer.Option(min=0.0)] = None,
    whole_bag: Annotated[bool, typer.Option()] = False,
    interval_s: Annotated[float, typer.Option(min=0.001)] = 2.0,
    all_frames: Annotated[bool, typer.Option()] = False,
    camera_topic: Annotated[str | None, typer.Option()] = None,
    confirm_all_frames: Annotated[str | None, typer.Option()] = None,
    serve: Annotated[bool, typer.Option(help="Inicia o viewer ao terminar.")] = False,
    root: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Executa extração, percepção e, quando possível, o workflow contextual."""
    try:
        if whole_bag == (duration_s is not None):
            raise ValueError("informe exatamente um de --whole-bag ou --duration-s")
        workflows = _workflows(root)
        profile = workflows.project.profile_for(bag)
        request = ExtractionRequest(
            bag=bag.resolve(),
            segment_id=segment_id,
            start_s=start_s,
            duration_s=None if whole_bag else duration_s,
            keyframe_interval_s=interval_s,
            all_frames=all_frames,
            camera_topic=camera_topic or (profile.camera_topic if profile else None),
        )
        extracted = _extract_request(workflows, request, confirm_all_frames)
        if profile is not None:
            workflows.run_mapping(profile, extracted, segment_id)
        estimate = workflows.estimate_perception(extracted.frame_count)
        _show_estimate("Visual-perception", estimate)
        visual_run = workflows.run_perception(
            extracted.frames_dir,
            profile.sequence_masks if profile else None,
        )
        console.print(f"[green]Run publicado:[/green] {visual_run}")
        if profile is None:
            console.print(
                "[yellow]A rosbag não possui perfil de calibração/mapeamento; "
                "o workflow termina após visual-perception.[/yellow]"
            )
            return
        artifact = workflows.compose(
            profile,
            segment_id=segment_id,
            window=extracted.window,
            visual_run=visual_run,
        )
        console.print(f"[green]Artifact contextual:[/green] {artifact}")
        if serve:
            console.print("Viewer: [link=http://localhost:5173]http://localhost:5173[/link]")
            workflows.serve(artifact, segment_id)
    except Exception as error:
        _fail(error)


# Lê um número decimal nos prompts com validação local e mensagens em português.
def _ask_float(message: str, default: str) -> float:
    """Solicita um número decimal até receber entrada válida."""
    import questionary

    while True:
        value = questionary.text(message, default=default).ask()
        try:
            return float(value)
        except (TypeError, ValueError):
            console.print("[red]Informe um número válido.[/red]")


# Executa o wizard principal usando as mesmas operações do modo não interativo.
def _interactive_process(workflows: Workflows, *, extraction_only: bool = False) -> None:
    """Guia o usuário por bag, intervalo, extração e processamento downstream.

    Argumentos:
        workflows: orquestrador configurado.
        extraction_only: encerra depois de publicar window e PNGs.
    """
    import questionary

    bags = workflows.project.available_bags()
    if not bags:
        raise FileNotFoundError("nenhuma rosbag encontrada em datasets/raw")
    bag_text = questionary.select("Escolha a rosbag:", choices=[str(path) for path in bags]).ask()
    bag = Path(bag_text)
    recording = workflows.inspect_bag(bag)
    topic = (
        questionary.select(
            "Escolha o tópico RGB:",
            choices=[f"{item.name} ({item.message_count} frames)" for item in recording.image_topics],
        )
        .ask()
        .split(" (", 1)[0]
    )
    scope = questionary.select(
        "Qual intervalo processar?", choices=["Extrair um trecho", "Bag inteiro"]
    ).ask()
    start_s = 0.0
    duration_s: float | None = None
    if scope == "Extrair um trecho":
        start_s = _ask_float("Início em segundos após o primeiro RGB:", "0")
        duration_s = _ask_float("Duração do trecho em segundos:", "30")
    policy = questionary.select(
        "Quais frames extrair?",
        choices=["Keyframes espaçados", "Todos os frames"],
    ).ask()
    all_frames = policy == "Todos os frames"
    interval_s = 2.0 if all_frames else _ask_float("Intervalo entre keyframes (s):", "2")
    default_id = f"{bag.stem}-full" if duration_s is None else f"{bag.stem}-{start_s:g}s-{duration_s:g}s"
    segment_id = questionary.text("Identidade do segmento:", default=default_id).ask()
    request = ExtractionRequest(
        bag=bag,
        segment_id=segment_id,
        start_s=start_s,
        duration_s=duration_s,
        keyframe_interval_s=interval_s,
        all_frames=all_frames,
        camera_topic=topic,
    )
    confirmation = None
    if all_frames:
        estimate = workflows.estimate_extraction(request)
        _show_estimate("Execução integral", estimate)
        confirmation = questionary.text(f"Digite {segment_id!r} para confirmar todos os frames:").ask()
    extracted = _extract_request(
        workflows,
        request,
        confirmation,
        show_estimate=not all_frames,
    )
    if extraction_only:
        return
    profile = workflows.project.profile_for(bag)
    if profile is not None:
        console.print("[cyan]Gerando geometria com FAST-LIO...[/cyan]")
        workflows.run_mapping(profile, extracted, segment_id)
    perception_estimate = workflows.estimate_perception(extracted.frame_count)
    _show_estimate("Visual-perception", perception_estimate)
    if not questionary.confirm("Iniciar visual-perception agora?", default=True).ask():
        return
    visual_run = workflows.run_perception(
        extracted.frames_dir,
        profile.sequence_masks if profile else None,
    )
    console.print(f"[green]Run publicado:[/green] {visual_run}")
    if profile is None:
        console.print("[yellow]Sem perfil, composição e viewer não estão disponíveis.[/yellow]")
        return
    artifact = workflows.compose(
        profile,
        segment_id=segment_id,
        window=extracted.window,
        visual_run=visual_run,
    )
    console.print(f"[green]Artifact contextual:[/green] {artifact}")
    if questionary.confirm("Servir o artifact no viewer agora?", default=True).ask():
        console.print("Viewer: [link=http://localhost:5173]http://localhost:5173[/link]")
        workflows.serve(artifact, segment_id)


# Oferece operações frequentes sem exigir memorização dos subcomandos.
def _interactive_menu(root: Path | None = None) -> None:
    """Abre o menu principal e executa operações até o usuário sair."""
    import questionary

    workflows = _workflows(root)
    while True:
        console.print("\n[bold]Contextual 3D Mapping CLI[/bold]")
        choice = questionary.select(
            "Escolha uma operação:",
            choices=[
                "Processar um novo trecho ou bag",
                "Extrair keyframes de um trecho ou bag",
                "Rodar visual-perception em frames existentes",
                "Compor artifact contextual",
                "Servir viewer web",
                "Validar estrutura do projeto",
                "Sair",
            ],
        ).ask()
        if choice in {None, "Sair"}:
            return
        try:
            if choice == "Processar um novo trecho ou bag":
                _interactive_process(workflows)
            elif choice == "Extrair keyframes de um trecho ou bag":
                _interactive_process(workflows, extraction_only=True)
            elif choice == "Rodar visual-perception em frames existentes":
                directories = sorted((workflows.project.root / "artifacts").glob("*-frames"))
                if not directories:
                    raise FileNotFoundError("nenhum diretório *-frames encontrado em artifacts")
                selected = questionary.select(
                    "Escolha os frames:", choices=[str(path) for path in directories]
                ).ask()
                selected_path = Path(selected)
                frame_paths = tuple(sorted(selected_path.glob("*.png")))
                frame_ids = questionary.checkbox(
                    "Selecione os frames; deixe vazio para todos:",
                    choices=[path.stem for path in frame_paths],
                ).ask()
                segment_id = selected_path.name.removesuffix("-frames")
                window_path = workflows.project.root / "artifacts" / f"{segment_id}-window.json"
                masks = None
                if window_path.is_file():
                    payload = json.loads(window_path.read_text(encoding="utf-8"))
                    recording_id = str(payload.get("recording_id", "corridor-02"))
                    profile = next(
                        (
                            item
                            for item in workflows.project.available_profiles()
                            if item.profile_id == recording_id
                        ),
                        None,
                    )
                    masks = None if profile is None else profile.sequence_masks
                run = workflows.run_perception(selected_path, masks, tuple(frame_ids or ()))
                console.print(f"[green]Run publicado:[/green] {run}")
            elif choice == "Compor artifact contextual":
                windows = workflows.project.available_windows()
                runs = workflows.project.available_runs()
                window = Path(questionary.select("Window:", choices=[str(item) for item in windows]).ask())
                run = Path(questionary.select("Run:", choices=[str(item) for item in runs]).ask())
                payload = json.loads(window.read_text(encoding="utf-8"))
                profile = _profile(workflows, str(payload.get("recording_id", "corridor-02")))
                segment_id = window.name.removesuffix("-window.json")
                artifact = workflows.compose(profile, segment_id=segment_id, window=window, visual_run=run)
                console.print(f"[green]Artifact contextual:[/green] {artifact}")
            elif choice == "Servir viewer web":
                artifacts = sorted((workflows.project.root / "artifacts").glob("*-context.json"))
                artifact = Path(
                    questionary.select("Artifact:", choices=[str(item) for item in artifacts]).ask()
                )
                segment_id = artifact.name.removesuffix("-context.json")
                workflows.serve(artifact, segment_id)
            else:
                issues = workflows.project.validate()
                if not issues:
                    console.print("[green]Estrutura válida.[/green]")
                for issue in issues:
                    console.print(f"{issue.level.upper()}: {issue.message}")
        except Exception as error:
            console.print(f"[bold red]Erro:[/bold red] {error}")


# Abre o menu apenas quando nenhum subcomando foi informado.
@app.callback()
def callback(
    context: typer.Context,
    root: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Inicializa a CLI e abre o menu no modo sem argumentos."""
    if context.invoked_subcommand is None:
        _interactive_menu(root)


# Ponto de entrada fino usado pelo console script e por ``python -m``.
def main() -> int:
    """Executa a aplicação Typer e devolve código POSIX de sucesso."""
    app()
    return 0
