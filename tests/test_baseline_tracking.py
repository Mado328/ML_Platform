import json

import pytest

pytest.importorskip("sklearn")

from ml_platform.scripts.train_baseline import run_baseline
from ml_platform.tracking import get_run_metrics, get_run_params, list_runs
from ml_platform.tracking.entities import RunStatus


def test_baseline_logs_candidate_and_final_runs(tmp_path, monkeypatch):
    data_path = tmp_path / "SMSSpamCollection"
    report_path = tmp_path / "baseline.json"
    tracking_path = tmp_path / "tracking" / "runs.db"
    rows = []
    for index in range(30):
        rows.append(f"ham\tnormal message number {index}")
        rows.append(f"spam\tclaim prize offer number {index}")
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("ML_PLATFORM_TRACKING_PATH", str(tracking_path))

    report = run_baseline(data_path=data_path, report_path=report_path)

    runs = list_runs("sms-baseline")
    assert len(runs) == 5
    assert all(run.status is RunStatus.FINISHED for run in runs)
    assert runs[0].run_name == "multinomial-nb-final"

    candidate_runs = runs[1:]
    expected_validation_metrics = {
        "validation_accuracy",
        "validation_spam_precision",
        "validation_spam_recall",
        "validation_spam_f1",
    }
    for run in candidate_runs:
        params = get_run_params(run.run_id)
        metric_keys = {metric.key for metric in get_run_metrics(run.run_id)}
        assert params["model"] == "MultinomialNB"
        assert params["alpha"] in {0.1, 0.5, 1.0, 2.0}
        assert params["seed"] == 42
        assert metric_keys == expected_validation_metrics

    final_run = runs[0]
    final_params = get_run_params(final_run.run_id)
    final_metric_keys = {metric.key for metric in get_run_metrics(final_run.run_id)}
    assert final_params["alpha"] == report["model_selection"]["selected_alpha"]
    assert "validation_spam_f1" in final_metric_keys
    assert {
        "test_naive_bayes_accuracy",
        "test_naive_bayes_spam_precision",
        "test_naive_bayes_spam_recall",
        "test_naive_bayes_spam_f1",
        "test_majority_baseline_accuracy",
        "test_majority_baseline_spam_precision",
        "test_majority_baseline_spam_recall",
        "test_majority_baseline_spam_f1",
    } <= final_metric_keys

    assert set(report) == {"dataset", "split", "model_selection", "test"}
    assert set(report["test"]) == {"naive_bayes", "majority_baseline"}
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
