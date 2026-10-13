import importlib.util
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

LOCAL_HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 30.0
PROCESS_STOP_TIMEOUT_SECONDS = 5.0
UI_DEPENDENCIES = ("fastapi", "streamlit", "uvicorn")


def missing_ui_dependencies() -> list[str]:
    return [name for name in UI_DEPENDENCIES if importlib.util.find_spec(name) is None]


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        try:
            server_socket.bind((LOCAL_HOST, port))
        except OSError:
            return False
    return True


def _wait_for_api(process: subprocess.Popen, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"FastAPI exited during startup with code {return_code}")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise RuntimeError("FastAPI did not become ready before the startup timeout")


def _wait_for_ui(process: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Streamlit exited during startup with code {return_code}")
        try:
            with socket.create_connection((LOCAL_HOST, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("Streamlit did not become ready before the startup timeout")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def serve_ui(api_port: int, ui_port: int, open_browser: bool = True) -> None:
    missing = missing_ui_dependencies()
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"UI dependencies are missing ({missing_list}); install them with `uv sync --extra ui`."
        )
    if api_port == ui_port:
        raise RuntimeError("The API and UI ports must be different")
    for label, port in (("API", api_port), ("Streamlit", ui_port)):
        if not _port_is_available(port):
            raise RuntimeError(f"The {label} port {port} is already in use")

    api_url = f"http://{LOCAL_HOST}:{api_port}"
    ui_url = f"http://{LOCAL_HOST}:{ui_port}"
    app_path = Path(__file__).with_name("streamlit_app.py")
    child_environment = os.environ.copy()
    child_environment["ML_PLATFORM_API_URL"] = api_url
    processes: list[subprocess.Popen] = []

    try:
        api_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "ml_platform.ui.api:app",
                "--host",
                LOCAL_HOST,
                "--port",
                str(api_port),
            ],
            env=child_environment,
        )
        processes.append(api_process)
        _wait_for_api(
            api_process,
            f"{api_url}/health",
            STARTUP_TIMEOUT_SECONDS,
        )

        streamlit_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.address",
                LOCAL_HOST,
                "--server.port",
                str(ui_port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            env=child_environment,
        )
        processes.append(streamlit_process)
        _wait_for_ui(streamlit_process, ui_port, STARTUP_TIMEOUT_SECONDS)

        print(f"FastAPI:   {api_url}")
        print(f"Streamlit: {ui_url}")
        if open_browser:
            webbrowser.open_new_tab(ui_url)
        print("Press Ctrl+C to stop both services.")

        while True:
            for name, process in (("FastAPI", api_process), ("Streamlit", streamlit_process)):
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(f"{name} exited with code {return_code}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping UI services...")
    finally:
        for process in reversed(processes):
            _stop_process(process)
