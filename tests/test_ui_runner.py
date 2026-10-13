from pathlib import Path

import pytest

from ml_platform.ui import runner


class FakeProcess:
    def __init__(self, args, env, return_code=None):
        self.args = args
        self.env = env
        self.return_code = return_code
        self.terminated = False
        self.waited = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True
        self.return_code = 0

    def wait(self, timeout=None):
        self.waited = True
        return self.return_code

    def kill(self):
        self.return_code = -9


def _install_fake_processes(monkeypatch, return_codes=None):
    processes = []
    queued_codes = iter(return_codes or [])

    def popen(args, env):
        process = FakeProcess(args, env, next(queued_codes, None))
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "_port_is_available", lambda port: True)
    monkeypatch.setattr(runner, "missing_ui_dependencies", lambda: [])
    monkeypatch.setattr(runner, "_wait_for_api", lambda process, url, timeout: None)
    monkeypatch.setattr(runner, "_wait_for_ui", lambda process, port, timeout: None)
    return processes


def test_serve_ui_uses_loopback_and_stops_both_on_keyboard_interrupt(monkeypatch):
    processes = _install_fake_processes(monkeypatch)
    opened_urls = []
    monkeypatch.setattr(runner.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt))
    monkeypatch.setattr(
        runner.webbrowser,
        "open_new_tab",
        opened_urls.append,
    )

    runner.serve_ui(8100, 8600, open_browser=True)

    assert len(processes) == 2
    assert processes[0].args[2:6] == ["uvicorn", "ml_platform.ui.api:app", "--host", "127.0.0.1"]
    assert processes[1].args[2:7] == [
        "streamlit",
        "run",
        str(Path(__file__).resolve().parents[1] / "ml_platform" / "ui" / "streamlit_app.py"),
        "--server.address",
        "127.0.0.1",
    ]
    assert processes[0].env["ML_PLATFORM_API_URL"] == "http://127.0.0.1:8100"
    assert opened_urls == ["http://127.0.0.1:8600"]
    assert all(process.terminated and process.waited for process in processes)


def test_serve_ui_cleans_up_api_if_backend_startup_fails(monkeypatch):
    processes = _install_fake_processes(monkeypatch)

    def fail_api_startup(process, url, timeout):
        raise RuntimeError("API startup failed")

    monkeypatch.setattr(runner, "_wait_for_api", fail_api_startup)

    with pytest.raises(RuntimeError, match="API startup failed"):
        runner.serve_ui(8100, 8600, open_browser=False)

    assert len(processes) == 1
    assert processes[0].terminated


def test_serve_ui_stops_other_service_if_a_child_exits(monkeypatch):
    processes = _install_fake_processes(monkeypatch, return_codes=[3, None])

    with pytest.raises(RuntimeError, match="FastAPI exited with code 3"):
        runner.serve_ui(8100, 8600, open_browser=False)

    assert len(processes) == 2
    assert not processes[0].terminated
    assert processes[1].terminated


def test_serve_ui_rejects_a_port_that_is_already_in_use(monkeypatch):
    processes = _install_fake_processes(monkeypatch)
    monkeypatch.setattr(runner, "_port_is_available", lambda port: port != 8600)

    with pytest.raises(RuntimeError, match="port 8600 is already in use"):
        runner.serve_ui(8100, 8600, open_browser=False)

    assert processes == []
