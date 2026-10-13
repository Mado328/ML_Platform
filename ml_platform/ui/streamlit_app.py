import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import streamlit as st


API_BASE_URL = os.environ.get("ML_PLATFORM_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _get_json(path: str):
    url = f"{API_BASE_URL}{path}"
    try:
        with urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8")), None
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        return None, f"Service returned HTTP {error.code} for {path}: {detail}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        return None, f"Could not reach the tracking service at {url}: {error}"


def _show_request_error(error: str) -> None:
    st.error(error)


def _comparison_rows(
    details: dict[str, dict[str, Any]], metric_name: str
) -> list[dict[str, str | float]]:
    comparison = []
    for run_id, detail in details.items():
        matching = [
            metric
            for metric in detail.get("metrics", [])
            if metric["key"] == metric_name
        ]
        if matching:
            latest = max(
                matching,
                key=lambda metric: (metric["step"], metric["timestamp"]),
            )
            comparison.append({"run": run_id, "value": latest["value"]})
    return comparison


def _metric_history_rows(
    detail: dict[str, Any], metric_name: str
) -> list[dict[str, int | float]]:
    latest_by_step: dict[int, dict[str, Any]] = {}
    for metric in detail.get("metrics", []):
        if metric["key"] != metric_name:
            continue
        step = metric["step"]
        previous = latest_by_step.get(step)
        if previous is None or metric["timestamp"] > previous["timestamp"]:
            latest_by_step[step] = metric

    if len(latest_by_step) < 2:
        return []
    return [
        {"step": step, "value": latest_by_step[step]["value"]}
        for step in sorted(latest_by_step)
    ]


def main() -> None:
    st.set_page_config(page_title="ML Platform Dashboard", layout="wide")
    st.title("ML Platform Dashboard")

    experiments, error = _get_json("/api/experiments")
    if error:
        _show_request_error(error)
        return
    if not experiments:
        st.info("No experiments are available.")
        return

    experiment_names = [experiment["name"] for experiment in experiments]
    experiment_name = st.selectbox("Experiment", experiment_names)
    runs, error = _get_json(f"/api/experiments/{quote(experiment_name, safe='')}/runs")
    if error:
        _show_request_error(error)
        return
    if not runs:
        st.info("This experiment has no runs.")
        return

    st.subheader("Runs")
    st.dataframe(runs, use_container_width=True, hide_index=True)

    run_ids = [run["run_id"] for run in runs]
    selected_run_ids = st.multiselect("Compare runs", run_ids)
    details = {}
    for run_id in selected_run_ids:
        detail, error = _get_json(f"/api/runs/{quote(run_id, safe='')}")
        if error:
            _show_request_error(error)
            continue
        details[run_id] = detail

    if details:
        st.subheader("Selected Run Details")
        for run_id, detail in details.items():
            st.markdown(f"**{detail['run'].get('run_name') or run_id}** (`{run_id}`)")
            st.caption("Parameters")
            params = detail.get("params", {})
            st.dataframe(
                [{"parameter": key, "value": value} for key, value in params.items()],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Metrics")
            st.dataframe(detail.get("metrics", []), use_container_width=True, hide_index=True)

        metric_names = sorted(
            {
                metric["key"]
                for detail in details.values()
                for metric in detail.get("metrics", [])
            }
        )
        if metric_names:
            metric_name = st.selectbox("Metric comparison", metric_names)
            comparison = _comparison_rows(details, metric_name)
            if comparison:
                st.subheader(f"{metric_name} Comparison")
                st.bar_chart(comparison, x="run", y="value")

            history_run_id = st.selectbox("Metric history run", list(details))
            history_metrics = sorted(
                {
                    metric["key"]
                    for metric in details[history_run_id].get("metrics", [])
                }
            )
            if history_metrics:
                history_metric_name = st.selectbox("Metric history", history_metrics)
                history = _metric_history_rows(
                    details[history_run_id], history_metric_name
                )
                if history:
                    st.line_chart(
                        history,
                        x="step",
                        y="value",
                    )


if __name__ == "__main__":
    main()
