import json
from pathlib import Path

import click

from .demo import main as run_demo
from .tracking import get_run, get_run_metrics, get_run_params, list_runs
from .tracking.entities import MetricRecord, Run

DEFAULT_DATA_PATH = Path("ml_platform/data/raw/SMSSpamCollection")


def _run_as_dict(run: Run) -> dict[str, str | None]:
    return {
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "run_name": run.run_name,
        "status": run.status.value,
        "started_at": run.started_at.isoformat(),
        "ended_at": run.ended_at.isoformat() if run.ended_at is not None else None,
    }


def _metric_as_dict(metric: MetricRecord) -> dict[str, str | float | int]:
    return {
        "key": metric.key,
        "value": metric.value,
        "step": metric.step,
        "timestamp": metric.timestamp.isoformat(),
    }


def _echo_run_table(runs: list[Run]) -> None:
    headers = ("RUN ID", "STATUS", "NAME", "STARTED")
    rows = [
        (
            run.run_id,
            run.status.value,
            run.run_name or "-",
            run.started_at.isoformat(),
        )
        for run in runs
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    click.echo(format_row(headers))
    click.echo("  ".join("-" * width for width in widths))
    for row in rows:
        click.echo(format_row(row))


@click.group(no_args_is_help=True)
def cli() -> None:
    """Track and inspect local ML experiments."""


@cli.command()
def demo() -> None:
    """Run the built-in Naive Bayes classifier demo."""
    run_demo()


@cli.group()
def runs() -> None:
    """Inspect tracked runs."""


@runs.command("list")
@click.option("--experiment", required=True, help="Exact experiment name.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("text", "json")),
    default="text",
    show_default=True,
)
def list_runs_command(experiment: str, output_format: str) -> None:
    """List runs belonging to an experiment."""
    experiment_runs = list_runs(experiment)
    if output_format == "json":
        click.echo(
            json.dumps(
                [_run_as_dict(run) for run in experiment_runs],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not experiment_runs:
        click.echo(f"No runs found for experiment {experiment!r}.")
        return

    _echo_run_table(experiment_runs)


@runs.command("show")
@click.argument("run_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("text", "json")),
    default="text",
    show_default=True,
)
def show_run_command(run_id: str, output_format: str) -> None:
    """Show a run, its parameters, and metric history."""
    try:
        run = get_run(run_id)
        params = get_run_params(run_id)
        metrics = get_run_metrics(run_id)
    except KeyError as error:
        raise click.ClickException(f"Run {run_id!r} was not found") from error

    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "run": _run_as_dict(run),
                    "params": params,
                    "metrics": [_metric_as_dict(metric) for metric in metrics],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    click.echo(f"Run ID:       {run.run_id}")
    click.echo(f"Experiment:   {run.experiment_id}")
    click.echo(f"Name:         {run.run_name or '-'}")
    click.echo(f"Status:       {run.status.value}")
    click.echo(f"Started:      {run.started_at.isoformat()}")
    click.echo(f"Ended:        {run.ended_at.isoformat() if run.ended_at else '-'}")
    click.echo("Parameters:")
    if params:
        for key, value in params.items():
            rendered_value = json.dumps(value, ensure_ascii=False)
            click.echo(f"  {key}: {rendered_value}")
    else:
        click.echo("  (none)")
    click.echo("Metrics:")
    if metrics:
        for metric in metrics:
            click.echo(
                f"  {metric.key}={metric.value} "
                f"(step={metric.step}, timestamp={metric.timestamp.isoformat()})"
            )
    else:
        click.echo("  (none)")


@cli.group()
def baseline() -> None:
    """Run bundled baseline workflows."""


@baseline.command("run")
@click.option(
    "--data-path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=DEFAULT_DATA_PATH,
    show_default=True,
)
@click.option(
    "--report-path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Report output path; defaults to ml_platform/reports/baseline.json.",
)
def run_baseline_command(data_path: Path, report_path: Path | None) -> None:
    """Train and track the SMS Spam Collection baseline."""
    try:
        from .scripts.train_baseline import run_baseline
    except ModuleNotFoundError as error:
        if error.name != "sklearn":
            raise
        raise click.ClickException(
            "The baseline extra is required; install it with `uv sync --extra baseline`."
        ) from error

    kwargs = {"data_path": data_path}
    if report_path is not None:
        kwargs["report_path"] = report_path
    report = run_baseline(**kwargs)
    click.echo(
        "Baseline completed. "
        f"Selected alpha: {report['model_selection']['selected_alpha']}; "
        "runs are available with `ml-platform runs list --experiment sms-baseline`."
    )


@cli.command()
@click.option("--api-port", type=click.IntRange(1, 65535), default=8000, show_default=True)
@click.option("--ui-port", type=click.IntRange(1, 65535), default=8501, show_default=True)
@click.option("--no-browser", is_flag=True, help="Do not open the dashboard automatically.")
def ui(api_port: int, ui_port: int, no_browser: bool) -> None:
    """Start the local FastAPI service and Streamlit dashboard."""
    if api_port == ui_port:
        raise click.BadParameter("API and UI ports must be different", param_hint="--ui-port")

    from .ui.runner import serve_ui

    try:
        serve_ui(api_port, ui_port, open_browser=not no_browser)
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
