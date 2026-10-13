import builtins
import json
import sys

import pytest
from click.testing import CliRunner

from ml_platform.cli import cli
from ml_platform.tracking import list_runs, log_metric, log_param, start_run
from ml_platform.tracking.api import _store_for_path
from ml_platform.ui import runner as ui_runner


@pytest.fixture(autouse=True)
def tracking_database(tmp_path, monkeypatch):
    path = tmp_path / "tracking" / "cli.db"
    monkeypatch.setenv("ML_PLATFORM_TRACKING_PATH", str(path))
    _store_for_path.cache_clear()
    yield path
    _store_for_path.cache_clear()


def test_root_help_and_demo_command():
    runner = CliRunner()

    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0
    assert "runs" in help_result.output
    assert "baseline" in help_result.output
    assert "ui" in help_result.output
    assert "demo" in help_result.output

    no_args_result = runner.invoke(cli, [])
    assert "Usage:" in no_args_result.output

    demo_result = runner.invoke(cli, ["demo"])
    assert demo_result.exit_code == 0
    assert "Класс:" in demo_result.output


def test_runs_list_text_json_and_empty_results():
    runner = CliRunner()
    with start_run("sms-baseline", "tracked-candidate") as run:
        run_id = run.run_id
        log_param("alpha", 0.5)
        log_metric("validation_spam_f1", 0.91)

    text_result = runner.invoke(
        cli, ["runs", "list", "--experiment", "sms-baseline"]
    )
    assert text_result.exit_code == 0
    assert run_id in text_result.output
    assert "tracked-candidate" in text_result.output

    json_result = runner.invoke(
        cli,
        ["runs", "list", "--experiment", "sms-baseline", "--format", "json"],
    )
    assert json_result.exit_code == 0
    runs = json.loads(json_result.output)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "FINISHED"
    assert runs[0]["started_at"].endswith("+00:00")

    empty_text = runner.invoke(
        cli, ["runs", "list", "--experiment", "missing-experiment"]
    )
    assert empty_text.exit_code == 0
    assert "No runs found" in empty_text.output

    empty_json = runner.invoke(
        cli,
        [
            "runs",
            "list",
            "--experiment",
            "missing-experiment",
            "--format",
            "json",
        ],
    )
    assert empty_json.exit_code == 0
    assert json.loads(empty_json.output) == []


def test_runs_show_text_json_and_unknown_id():
    runner = CliRunner()
    with start_run("show-test", "details") as run:
        run_id = run.run_id
        log_param("alpha", 0.5)
        log_metric("score", 0.75, step=2)

    text_result = runner.invoke(cli, ["runs", "show", run_id])
    assert text_result.exit_code == 0
    assert "Status:       FINISHED" in text_result.output
    assert 'alpha: 0.5' in text_result.output
    assert "score=0.75 (step=2" in text_result.output

    json_result = runner.invoke(cli, ["runs", "show", run_id, "--format", "json"])
    assert json_result.exit_code == 0
    run_details = json.loads(json_result.output)
    assert set(run_details) == {"run", "params", "metrics"}
    assert run_details["run"]["status"] == "FINISHED"
    assert run_details["params"] == {"alpha": 0.5}
    assert run_details["metrics"][0]["timestamp"].endswith("+00:00")

    unknown_result = runner.invoke(cli, ["runs", "show", "missing-run"])
    assert unknown_result.exit_code != 0
    assert "was not found" in unknown_result.output


def test_baseline_run_with_temporary_paths(tmp_path):
    pytest.importorskip("sklearn")
    runner = CliRunner()
    data_path = tmp_path / "SMSSpamCollection"
    report_path = tmp_path / "reports" / "baseline.json"
    rows = []
    for index in range(30):
        rows.append(f"ham\tnormal message number {index}")
        rows.append(f"spam\tclaim prize offer number {index}")
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "baseline",
            "run",
            "--data-path",
            str(data_path),
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Baseline completed" in result.output
    assert report_path.is_file()
    assert len(list_runs("sms-baseline")) == 5


def test_baseline_command_explains_missing_optional_extra(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "SMSSpamCollection"
    data_path.write_text("dataset placeholder", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "ml_platform.scripts.train_baseline", raising=False)
    real_import = builtins.__import__

    def import_without_sklearn(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sklearn)
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    list_result = runner.invoke(
        cli, ["runs", "list", "--experiment", "empty"]
    )
    assert help_result.exit_code == 0
    assert list_result.exit_code == 0

    result = runner.invoke(
        cli,
        ["baseline", "run", "--data-path", str(data_path)],
    )

    assert result.exit_code != 0
    assert "uv sync --extra baseline" in result.output


def test_ui_command_passes_options_and_rejects_duplicate_ports(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ui_runner,
        "serve_ui",
        lambda api_port, ui_port, open_browser: calls.append(
            (api_port, ui_port, open_browser)
        ),
    )
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["ui", "--api-port", "8100", "--ui-port", "8600", "--no-browser"],
    )
    assert result.exit_code == 0
    assert calls == [(8100, 8600, False)]

    duplicate_port = runner.invoke(
        cli, ["ui", "--api-port", "8100", "--ui-port", "8100"]
    )
    assert duplicate_port.exit_code != 0
    assert "must be different" in duplicate_port.output


def test_ui_command_explains_missing_optional_extra(monkeypatch):
    monkeypatch.setattr(ui_runner, "missing_ui_dependencies", lambda: ["fastapi", "streamlit"])
    result = CliRunner().invoke(cli, ["ui", "--no-browser"])

    assert result.exit_code != 0
    assert "uv sync --extra ui" in result.output
