from __future__ import annotations

from pathlib import Path
from typing import Optional

import json

import typer
from rich import print as rich_print

from research_flow.config import load_provider_config
from research_flow.clarification import maybe_generate_clarification_turn_with_provider
from research_flow.orchestrator import ResearchFlowOrchestrator
from research_flow.providers import create_provider
from research_flow.prompts_loader import PromptLibrary
from research_flow.validation import validate_run_dir

app = typer.Typer(help="Research Flow CLI")
DEFAULT_PROVIDER_NAME = "deepseek"
DEFAULT_PROVIDER_CONFIG = Path("providers/deepseek.example.yaml")


def format_json_output(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@app.command()
def run(
    idea: Optional[str] = typer.Option(None, help="Idea text inline."),
    idea_file: Optional[Path] = typer.Option(None, exists=True, help="Path to an idea markdown/text file."),
    candidate_limit: int = typer.Option(5, help="Maximum candidates kept for ranking and downstream context."),
    max_papers: int = typer.Option(5, help="Maximum shortlisted papers."),
    sources: str = typer.Option("openalex,semanticscholar,arxiv", help="Comma-separated sources."),
    download_pdf: bool = typer.Option(True, help="Attempt PDF download and parsing."),
    parallel: bool = typer.Option(True, help="Parallelize per-paper brief generation."),
    outdir: Optional[Path] = typer.Option(None, help="Output directory for this run."),
    config: Path = typer.Option(Path("config.example.yaml"), exists=True, help="App config YAML."),
    provider: str = typer.Option(DEFAULT_PROVIDER_NAME, help="Provider name label."),
    provider_config: Path = typer.Option(DEFAULT_PROVIDER_CONFIG, exists=True, help="Provider config YAML."),
    clarification_history_file: Optional[Path] = typer.Option(None, exists=True, help="GUI clarification history JSON file."),
    main_model: Optional[str] = typer.Option(None, help="主 agent 模型名。"),
    main_reasoning_effort: Optional[str] = typer.Option(None, help="主 agent 思考强度。"),
    sub_model: Optional[str] = typer.Option(None, help="subagent 模型名。"),
    sub_reasoning_effort: Optional[str] = typer.Option(None, help="subagent 思考强度。"),
) -> None:
    """Run the full research workflow."""
    if not idea and not idea_file:
        raise typer.BadParameter("Provide either --idea or --idea-file.")
    idea_text = idea if idea is not None else idea_file.read_text(encoding="utf-8")
    clarification_history = []
    if clarification_history_file:
        clarification_history = json.loads(clarification_history_file.read_text(encoding="utf-8"))
    orchestrator = ResearchFlowOrchestrator(config, provider_config)
    run_dir = orchestrator.run(
        idea_text=idea_text,
        clarification_history=clarification_history,
        outdir=outdir,
        provider_name=provider,
        candidate_limit=candidate_limit,
        max_papers=max_papers,
        download_pdf=download_pdf,
        parallel=parallel,
        sources=[item.strip() for item in sources.split(",") if item.strip()],
        main_model=main_model,
        main_reasoning_effort=main_reasoning_effort,
        sub_model=sub_model,
        sub_reasoning_effort=sub_reasoning_effort,
    )
    rich_print(f"[green]Run completed:[/green] {run_dir}")


@app.command("clarify-turn")
def clarify_turn(
    idea: str = typer.Option(..., help="Raw user idea."),
    history_file: Optional[Path] = typer.Option(None, exists=True, help="Clarification history JSON path."),
    config: Path = typer.Option(Path("config.example.yaml"), exists=True, help="App config YAML."),
    provider_config: Path = typer.Option(DEFAULT_PROVIDER_CONFIG, exists=True, help="Provider config YAML."),
    main_model: Optional[str] = typer.Option(None, help="主模型名。"),
    main_reasoning_effort: Optional[str] = typer.Option(None, help="主模型思考强度。"),
) -> None:
    app_cfg = load_provider_config(provider_config)
    default_main_model = app_cfg.default_main_model or (app_cfg.supported_models[0] if app_cfg.supported_models else "")
    provider = create_provider(app_cfg, workdir=Path.cwd())
    available, detail = provider.check_available()
    if not available:
        raise typer.BadParameter(f"Provider unavailable: {detail}")

    prompt_lib = PromptLibrary("prompts")
    history = []
    if history_file:
        history = json.loads(history_file.read_text(encoding="utf-8"))
    turn, provider_result = maybe_generate_clarification_turn_with_provider(
        raw_idea=idea,
        history=history,
        provider=provider,
        prompt_library=prompt_lib,
        timeout=45,
        runtime_options={
            "model": (main_model or "").strip() or default_main_model,
            "reasoning_effort": main_reasoning_effort or "",
            "thinking_enabled": "true",
        },
    )
    typer.echo(
        format_json_output(
            {
                "turn": turn.model_dump(),
                "provider_result": provider_result or {},
            }
        )
    )


@app.command("validate")
def validate(run_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Run directory to validate.")) -> None:
    """Validate a completed run directory."""
    report = validate_run_dir(run_dir)
    rich_print(report.model_dump_json(indent=2))


@app.command("provider-check")
def provider_check(
    provider_config: Path = typer.Option(DEFAULT_PROVIDER_CONFIG, exists=True, help="Provider config YAML."),
) -> None:
    """Check whether a provider CLI is callable."""
    config = load_provider_config(provider_config)
    provider = create_provider(config, workdir=Path.cwd())
    ok, detail = provider.check_available()
    rich_print({"available": ok, "detail": detail, "capabilities": provider.describe_provider_capabilities().model_dump()})


@app.command()
def ui(
    output_root: Path = typer.Option(Path("outputs"), help="Output root to inspect."),
) -> None:
    """Print the Electron UI path and expected output root."""
    rich_print({"ui_dir": str(Path("ui").resolve()), "output_root": str(output_root.resolve())})


if __name__ == "__main__":
    app()
