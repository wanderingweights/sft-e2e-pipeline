"""Pipeline state + logging. Enables idempotency: a phase whose name is in
`completed` is skipped on re-run.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
STATE_PATH = PIPELINE_DIR / "state.json"
LOG_PATH = PIPELINE_DIR / "run.log"


class State(BaseModel):
    phase: str | int = 0
    completed: list[str] = Field(default_factory=list)
    notes: dict = Field(default_factory=dict)


def load_state(path: Path = STATE_PATH) -> State:
    if path.exists():
        return State.model_validate_json(path.read_text())
    return State()


def save_state(state: State, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2))


def get_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sftpipe")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        fh = logging.FileHandler(LOG_PATH)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
