"""Shared phase context. Lives in its own module so phases can import it
without a phase->run->phase import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sftpipe.config import Config
from sftpipe.state import State

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Ctx:
    cfg: Config
    state: State
    logger: object
    data_root: Path = field(default=DATA_ROOT)
