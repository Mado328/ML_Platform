import pytest

pytest.importorskip("streamlit")

from ml_platform.ui.streamlit_app import _comparison_rows, _metric_history_rows


def test_metric_comparison_uses_latest_step_for_each_run():
    details = {
        "run-a": {
            "metrics": [
                {"key": "f1", "value": 0.8, "step": 0, "timestamp": "2025-01-01T00:00:00+00:00"},
                {"key": "f1", "value": 0.9, "step": 1, "timestamp": "2025-01-01T00:01:00+00:00"},
            ]
        },
        "run-b": {
            "metrics": [
                {"key": "f1", "value": 0.85, "step": 0, "timestamp": "2025-01-01T00:00:00+00:00"},
            ]
        },
        "run-c": {"metrics": []},
    }

    assert _comparison_rows(details, "f1") == [
        {"run": "run-a", "value": 0.9},
        {"run": "run-b", "value": 0.85},
    ]


def test_metric_history_is_sorted_and_collapses_duplicate_steps():
    detail = {
        "metrics": [
            {"key": "loss", "value": 0.6, "step": 2, "timestamp": "2025-01-01T00:02:00+00:00"},
            {"key": "loss", "value": 0.7, "step": 1, "timestamp": "2025-01-01T00:01:00+00:00"},
            {"key": "loss", "value": 0.5, "step": 2, "timestamp": "2025-01-01T00:03:00+00:00"},
        ]
    }

    assert _metric_history_rows(detail, "loss") == [
        {"step": 1, "value": 0.7},
        {"step": 2, "value": 0.5},
    ]
    assert _metric_history_rows(detail, "unknown") == []
    assert _metric_history_rows(
        {"metrics": [{"key": "score", "value": 0.9, "step": 0, "timestamp": "2025-01-01T00:00:00+00:00"}]},
        "score",
    ) == []
