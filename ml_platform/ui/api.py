from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException

from ml_platform.tracking import api as tracking_api

app = FastAPI(title="ML Platform Tracking API")


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _serialize(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/experiments")
def experiments() -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "created_at": experiment.created_at.isoformat(),
            "updated_at": experiment.updated_at.isoformat(),
        }
        for experiment in tracking_api.list_experiments()
    ]


@app.get("/api/experiments/{experiment_name}/runs")
def experiment_runs(experiment_name: str) -> list[dict[str, Any]]:
    runs = tracking_api.list_runs(experiment_name)
    return [
        _serialize(run)
        for run in sorted(runs, key=lambda item: (item.started_at, item.run_id), reverse=True)
    ]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    try:
        run = tracking_api.get_run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Run not found") from error

    return {
        "run": _serialize(run),
        "params": _serialize(tracking_api.get_run_params(run_id)),
        "metrics": _serialize(tracking_api.get_run_metrics(run_id)),
    }
