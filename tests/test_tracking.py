from datetime import datetime, timezone

import pytest

from ml_platform.tracking import (
    get_run,
    get_run_metrics,
    get_run_params,
    list_experiments,
    list_runs,
    log_metric,
    log_param,
    start_run,
)
from ml_platform.tracking.api import _store_for_path
from ml_platform.tracking.entities import RunStatus
from ml_platform.tracking.sqlite_store import SQLiteStore


@pytest.fixture(autouse=True)
def tracking_database(tmp_path, monkeypatch):
    path = tmp_path / "tracking" / "local.db"
    monkeypatch.setenv("ML_PLATFORM_TRACKING_PATH", str(path))
    _store_for_path.cache_clear()
    return path


def test_runs_reuse_experiment_and_persist_terminal_status():
    with start_run("sms-baseline", "first") as first:
        first_id = first.run_id
        assert first.status is RunStatus.RUNNING
        assert get_run(first_id).status is RunStatus.RUNNING

    persisted_first = get_run(first_id)
    assert persisted_first.status is RunStatus.FINISHED
    assert persisted_first.ended_at is not None
    assert persisted_first.started_at.tzinfo is not None
    assert persisted_first.ended_at.tzinfo is not None

    with start_run("sms-baseline", "second") as second:
        assert second.run_id != first_id
        assert second.experiment_id == first.experiment_id

    with start_run("SMS-baseline") as case_variant:
        assert case_variant.experiment_id != first.experiment_id


def test_failed_run_propagates_and_clears_active_context():
    with pytest.raises(LookupError, match="body failed"):
        with start_run("failed") as run:
            raise LookupError("body failed")

    assert get_run(run.run_id).status is RunStatus.FAILED
    with pytest.raises(RuntimeError, match="active run"):
        log_param("outside", 1)


def test_nested_runs_are_rejected():
    with start_run("outer"):
        with pytest.raises(RuntimeError, match="Nested runs"):
            with start_run("inner"):
                pass


def test_logging_outside_run_fails():
    with pytest.raises(RuntimeError, match="active run"):
        log_param("parameter", 1)
    with pytest.raises(RuntimeError, match="active run"):
        log_metric("metric", 1.0)


def test_parameters_round_trip_types_and_reject_conflicts():
    with start_run("params") as run:
        log_param("bool", True)
        log_param("integer", 3)
        log_param("string", "3")
        log_param("integer", 3)

        with pytest.raises(ValueError, match="another value"):
            log_param("integer", True)

    assert get_run_params(run.run_id) == {
        "bool": True,
        "integer": 3,
        "string": "3",
    }
    assert type(get_run_params(run.run_id)["bool"]) is bool
    assert type(get_run_params(run.run_id)["integer"]) is int


@pytest.mark.parametrize("key", ["", None])
def test_parameter_key_must_be_non_empty_string(key):
    with start_run("invalid-param-key"):
        with pytest.raises(ValueError, match="Parameter key"):
            log_param(key, 1)


@pytest.mark.parametrize("value", [None, [], {}, float("nan"), float("inf"), -float("inf")])
def test_parameter_values_must_be_finite_json_scalars(value):
    with start_run("invalid-param-value"):
        with pytest.raises(ValueError, match="Parameter value"):
            log_param("value", value)


def test_metrics_keep_duplicate_key_and_step_in_deterministic_order():
    with start_run("metrics") as run:
        log_metric("z-score", 9.0, step=1)
        log_metric("score", 2.0, step=1)
        log_metric("score", 1.0, step=0)
        log_metric("score", 3.0, step=1)
        assert len(get_run_metrics(run.run_id)) == 4

    history = get_run_metrics(run.run_id)
    assert [(item.key, item.step, item.value) for item in history] == [
        ("score", 0, 1.0),
        ("score", 1, 2.0),
        ("score", 1, 3.0),
        ("z-score", 1, 9.0),
    ]
    assert all(item.timestamp.tzinfo is not None for item in history)


@pytest.mark.parametrize("key", ["", None])
def test_metric_key_must_be_non_empty_string(key):
    with start_run("invalid-metric-key"):
        with pytest.raises(ValueError, match="Metric key"):
            log_metric(key, 1.0)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -float("inf")])
def test_metric_value_must_be_finite_numeric_and_not_bool(value):
    with start_run("invalid-metric-value"):
        with pytest.raises(ValueError, match="Metric value"):
            log_metric("value", value)


@pytest.mark.parametrize("step", [-1, 1.5, True])
def test_metric_step_must_be_non_negative_integer(step):
    with start_run("invalid-metric-step"):
        with pytest.raises(ValueError, match="Metric step"):
            log_metric("value", 1.0, step=step)


def test_empty_logs_and_unknown_run_lookups():
    with start_run("empty") as run:
        assert get_run_params(run.run_id) == {}
        assert get_run_metrics(run.run_id) == []

    for lookup in (get_run, get_run_params, get_run_metrics):
        with pytest.raises(KeyError):
            lookup("missing-run")


def test_list_runs_filters_by_exact_experiment_and_orders_newest_first(monkeypatch):
    timestamps = iter(
        datetime(2025, 1, 1, 0, minute=minute, tzinfo=timezone.utc)
        for minute in range(6)
    )
    monkeypatch.setattr(SQLiteStore, "_now", staticmethod(lambda: next(timestamps)))

    with start_run("sms-baseline", "older") as older:
        older_id = older.run_id
    with start_run("other-experiment", "unrelated"):
        pass
    with start_run("sms-baseline", "newer") as newer:
        newer_id = newer.run_id

    runs = list_runs("sms-baseline")
    assert [run.run_id for run in runs] == [newer_id, older_id]
    assert [run.run_name for run in runs] == ["newer", "older"]
    assert list_runs("SMS-baseline") == []
    assert list_runs("missing-experiment") == []


def test_list_experiments_returns_sorted_entities():
    assert list_experiments() == []

    with start_run("z-experiment"):
        pass
    with start_run("a-experiment"):
        pass

    experiments = list_experiments()
    assert [experiment.name for experiment in experiments] == [
        "a-experiment",
        "z-experiment",
    ]
    assert all(experiment.experiment_id for experiment in experiments)
    assert all(experiment.created_at.tzinfo == timezone.utc for experiment in experiments)
    assert all(experiment.updated_at.tzinfo == timezone.utc for experiment in experiments)


def test_active_run_keeps_store_selected_when_started(monkeypatch, tmp_path):
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"
    monkeypatch.setenv("ML_PLATFORM_TRACKING_PATH", str(first_path))
    with start_run("bound-store") as run:
        monkeypatch.setenv("ML_PLATFORM_TRACKING_PATH", str(second_path))
        log_param("source", "first")

    monkeypatch.setenv("ML_PLATFORM_TRACKING_PATH", str(first_path))
    assert get_run_params(run.run_id) == {"source": "first"}
    monkeypatch.setenv("ML_PLATFORM_TRACKING_PATH", str(second_path))
    with pytest.raises(KeyError):
        get_run(run.run_id)


def test_run_can_be_read_from_fresh_store(tracking_database):
    with start_run("fresh-store") as run:
        log_param("answer", 42)
        log_metric("accuracy", 0.75)

    fresh_store = SQLiteStore(tracking_database)
    assert fresh_store.get_run(run.run_id).status is RunStatus.FINISHED
    assert fresh_store.get_run_params(run.run_id) == {"answer": 42}
    metrics = fresh_store.get_run_metrics(run.run_id)
    assert [(metric.key, metric.value, metric.step) for metric in metrics] == [
        ("accuracy", 0.75, 0)
    ]
    assert isinstance(metrics[0].timestamp, datetime)
    assert metrics[0].timestamp.tzinfo == timezone.utc


def test_active_context_resets_when_finalization_fails(monkeypatch):
    def fail_finish(self, run_id, status):
        raise OSError("database unavailable")

    monkeypatch.setattr(SQLiteStore, "finish_run", fail_finish)
    with pytest.raises(OSError, match="database unavailable"):
        with start_run("finish-error"):
            pass

    with pytest.raises(RuntimeError, match="active run"):
        log_param("outside", 1)


def test_finalization_failure_chains_body_exception(monkeypatch):
    def fail_finish(self, run_id, status):
        raise OSError("database unavailable")

    monkeypatch.setattr(SQLiteStore, "finish_run", fail_finish)
    with pytest.raises(OSError, match="database unavailable") as error:
        with start_run("both-errors"):
            raise LookupError("body failed")

    assert isinstance(error.value.__context__, LookupError)
