import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")

from fastapi.testclient import TestClient

from ml_platform.tracking import log_metric, log_param, start_run
from ml_platform.tracking.api import _store_for_path
from ml_platform.ui.api import app


@pytest.fixture(autouse=True)
def tracking_database(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_PLATFORM_TRACKING_PATH", str(tmp_path / "tracking.db"))
    _store_for_path.cache_clear()
    yield
    _store_for_path.cache_clear()


def test_health_and_experiment_routes():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/experiments").json() == []

        with start_run("z-experiment", "latest") as run:
            run_id = run.run_id
            log_param("alpha", 0.5)
            log_metric("validation_f1", 0.9)
        with start_run("a-experiment", "other"):
            pass

        experiments = client.get("/api/experiments")
        assert [item["name"] for item in experiments.json()] == [
            "a-experiment",
            "z-experiment",
        ]
        assert set(experiments.json()[1]) == {
            "experiment_id",
            "name",
            "created_at",
            "updated_at",
        }
        assert experiments.json()[1]["experiment_id"] == run.experiment_id
        assert experiments.json()[1]["created_at"].endswith("+00:00")

        runs = client.get("/api/experiments/z-experiment/runs")
        assert runs.status_code == 200
        assert len(runs.json()) == 1
        assert runs.json()[0]["run_id"] == run_id
        assert runs.json()[0]["status"] == "FINISHED"
        assert client.get("/api/experiments/unknown-experiment/runs").json() == []


def test_run_details_and_unknown_run_status():
    with start_run("details", "with-data") as populated:
        populated_id = populated.run_id
        log_param("seed", 42)
        log_metric("f1", 0.91, step=3)
    with start_run("details", "empty") as empty:
        empty_id = empty.run_id

    with TestClient(app) as client:
        detail_response = client.get(f"/api/runs/{populated_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert set(detail) == {"run", "params", "metrics"}
        assert detail["params"] == {"seed": 42}
        assert len(detail["metrics"]) == 1
        metric = detail["metrics"][0]
        assert (metric["key"], metric["value"], metric["step"]) == ("f1", 0.91, 3)
        assert metric["timestamp"].endswith("+00:00")

        empty_detail = client.get(f"/api/runs/{empty_id}").json()
        assert empty_detail["params"] == {}
        assert empty_detail["metrics"] == []

        missing = client.get("/api/runs/not-a-run")
        assert missing.status_code == 404
        assert missing.json()["detail"] == "Run not found"
