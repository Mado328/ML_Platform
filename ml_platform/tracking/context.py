from contextvars import ContextVar, Token
from typing import Any

from .entities import Run, RunStatus

_active_context: ContextVar[tuple[Run, Any] | None] = ContextVar(
    "active_tracking_context", default=None
)

class RunContext:
    def __init__(self, store, experiment_name: str, run_name: str | None = None):
        self.store = store
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run: Run | None = None
        self._token: Token | None = None

    def __enter__(self):
        if _active_context.get() is not None:
            raise RuntimeError("A run is already active. Nested runs are not supported.")

        run = self.store.create_run(self.experiment_name, self.run_name)
        self.run = run
        self._token = _active_context.set((run, self.store))
        return run

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        assert self.run is not None
        assert self._token is not None

        status = RunStatus.FAILED if exc_type is not None else RunStatus.FINISHED

        try:
            self.store.finish_run(self.run.run_id, status)
        finally:
            _active_context.reset(self._token)

        return False  # Propagate exception if any

def require_active_run() -> Run:
    active = _active_context.get()
    if active is None:
        raise RuntimeError("Logging requires an active run")
    return active[0]


def require_active_store():
    active = _active_context.get()
    if active is None:
        raise RuntimeError("Logging requires an active run")
    return active[1]
