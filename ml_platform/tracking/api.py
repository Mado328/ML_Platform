import math
import os
from functools import lru_cache
from pathlib import Path

from .context import RunContext, require_active_run, require_active_store
from .entities import Experiment, MetricRecord, Run
from .sqlite_store import SQLiteStore


@lru_cache(maxsize=None)
def _store_for_path(path: str) -> SQLiteStore:
    return SQLiteStore(path)


def get_store() -> SQLiteStore:
    configured_path = os.environ.get("ML_PLATFORM_TRACKING_PATH", "./ml_platform.db")
    path = str(Path(configured_path).expanduser().resolve())
    return _store_for_path(path)


def start_run(experiment_name: str, run_name: str | None = None) -> RunContext:
    return RunContext(get_store(), experiment_name, run_name)


def log_param(key: str, value: str | int | float | bool) -> None:
    run = require_active_run()
    if not isinstance(key, str) or not key:
        raise ValueError("Parameter key cannot be empty")
    require_active_store().log_param(run.run_id, key, value)


def log_metric(key: str, value: float, step: int = 0) -> None:
    run = require_active_run()

    if not isinstance(key, str) or not key:
        raise ValueError("Metric key cannot be empty")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Metric value must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("Metric value must be finite")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Metric step must be a non-negative integer")

    require_active_store().log_metric(run.run_id, key, float(value), step)


def get_run(run_id: str) -> Run:
    return get_store().get_run(run_id)


def list_runs(experiment_name: str) -> list[Run]:
    return get_store().list_runs(experiment_name)


def list_experiments() -> list[Experiment]:
    return get_store().list_experiments()


def get_run_params(run_id: str) -> dict[str, str | int | float | bool]:
    return get_store().get_run_params(run_id)


def get_run_metrics(run_id: str) -> list[MetricRecord]:
    return get_store().get_run_metrics(run_id)
