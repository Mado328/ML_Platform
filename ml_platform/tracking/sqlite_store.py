import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from .entities import Experiment, MetricRecord, Run, RunStatus


class SQLiteStore:
    """SQLite-backed persistence for local tracking data."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    run_name TEXT,
                    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'FINISHED', 'FAILED')),
                    started_at TEXT NOT NULL,
                    ended_at TEXT
                );

                CREATE TABLE IF NOT EXISTS parameters (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, key)
                );

                CREATE TABLE IF NOT EXISTS metric_records (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    key TEXT NOT NULL,
                    value REAL NOT NULL,
                    step INTEGER NOT NULL CHECK (step >= 0),
                    timestamp TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _datetime(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value is not None else None

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> Run:
        started_at = SQLiteStore._datetime(row["started_at"])
        assert started_at is not None
        return Run(
            run_id=row["run_id"],
            experiment_id=row["experiment_id"],
            status=RunStatus(row["status"]),
            started_at=started_at,
            run_name=row["run_name"],
            ended_at=SQLiteStore._datetime(row["ended_at"]),
        )

    def create_run(self, experiment_name: str, run_name: str | None = None) -> Run:
        if not isinstance(experiment_name, str):
            raise ValueError("Experiment name must be a string")

        now = self._now()
        timestamp = now.isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, description, created_at, updated_at
                ) VALUES (?, ?, '', ?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (str(uuid4()), experiment_name, timestamp, timestamp),
            )
            experiment = connection.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?",
                (experiment_name,),
            ).fetchone()
            assert experiment is not None
            connection.execute(
                "UPDATE experiments SET updated_at = ? WHERE experiment_id = ?",
                (timestamp, experiment["experiment_id"]),
            )

            run = Run(
                run_id=str(uuid4()),
                experiment_id=experiment["experiment_id"],
                status=RunStatus.RUNNING,
                started_at=now,
                run_name=run_name,
            )
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, experiment_id, run_name, status, started_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (
                    run.run_id,
                    run.experiment_id,
                    run.run_name,
                    run.status.value,
                    run.started_at.isoformat(),
                ),
            )
        return run

    def list_experiments(self) -> list[Experiment]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY name COLLATE BINARY"
            ).fetchall()

        experiments: list[Experiment] = []
        for row in rows:
            created_at = self._datetime(row["created_at"])
            updated_at = self._datetime(row["updated_at"])
            assert created_at is not None
            assert updated_at is not None
            experiments.append(
                Experiment(
                    experiment_id=row["experiment_id"],
                    name=row["name"],
                    description=row["description"],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return experiments

    def log_param(self, run_id: str, key: str, value: str | int | float | bool) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("Parameter key cannot be empty")
        if type(value) not in (str, int, float, bool):
            raise ValueError("Parameter value must be a string, integer, float, or boolean")
        try:
            value_json = json.dumps(value, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError("Parameter value must be a finite JSON scalar") from error

        with self._connection() as connection:
            existing = connection.execute(
                "SELECT value_json FROM parameters WHERE run_id = ? AND key = ?",
                (run_id, key),
            ).fetchone()
            if existing is not None:
                if existing["value_json"] != value_json:
                    raise ValueError(f"Parameter {key!r} is already logged with another value")
                return
            connection.execute(
                "INSERT INTO parameters (run_id, key, value_json) VALUES (?, ?, ?)",
                (run_id, key, value_json),
            )

    def log_metric(self, run_id: str, key: str, value: float, step: int) -> None:
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

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO metric_records (run_id, key, value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, key, float(value), step, self._now().isoformat()),
            )

    def finish_run(self, run_id: str, status: RunStatus) -> None:
        if status not in (RunStatus.FINISHED, RunStatus.FAILED):
            raise ValueError("A run can only be finished or failed")

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status = ?, ended_at = ?
                WHERE run_id = ? AND status = ?
                """,
                (status.value, self._now().isoformat(), run_id, RunStatus.RUNNING.value),
            )
            if cursor.rowcount == 0:
                exists = connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(run_id)
                raise ValueError(f"Run {run_id!r} is not running")

    def get_run(self, run_id: str) -> Run:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._run_from_row(row)

    def list_runs(self, experiment_name: str) -> list[Run]:
        if not isinstance(experiment_name, str):
            raise ValueError("Experiment name must be a string")

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT runs.* FROM runs
                JOIN experiments USING (experiment_id)
                WHERE experiments.name = ?
                ORDER BY runs.started_at DESC, runs.run_id DESC
                """,
                (experiment_name,),
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def get_run_params(self, run_id: str) -> dict[str, str | int | float | bool]:
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(run_id)
            rows = connection.execute(
                "SELECT key, value_json FROM parameters WHERE run_id = ? ORDER BY key",
                (run_id,),
            ).fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def get_run_metrics(self, run_id: str) -> list[MetricRecord]:
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(run_id)
            rows = connection.execute(
                """
                SELECT key, value, step, timestamp FROM metric_records
                WHERE run_id = ?
                ORDER BY key, step, timestamp, metric_id
                """,
                (run_id,),
            ).fetchall()
        metrics: list[MetricRecord] = []
        for row in rows:
            timestamp = self._datetime(row["timestamp"])
            assert timestamp is not None
            metrics.append(
                MetricRecord(
                    key=row["key"],
                    value=row["value"],
                    step=row["step"],
                    timestamp=timestamp,
                )
            )
        return metrics
