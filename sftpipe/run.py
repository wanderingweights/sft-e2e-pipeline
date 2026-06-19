"""Master control loop. Idempotent + checkpointed:

    for phase in PHASES:
        if phase in state.completed: skip
        phase.run(ctx); assert phase.check(ctx); state.completed.append(phase)

Run a single phase:   python -m sftpipe.run --only p05_decontam
Run from a phase on:  python -m sftpipe.run --from p06_format
Run everything:       python -m sftpipe.run
"""
from __future__ import annotations

import argparse
import importlib

from sftpipe.config import load_config
from sftpipe.context import Ctx
from sftpipe.phases import PHASES
from sftpipe.state import get_logger, load_state, save_state


def _import_phase(name: str):
    return importlib.import_module(f"sftpipe.phases.{name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single phase by name")
    ap.add_argument("--from", dest="from_phase", help="run from this phase onward")
    ap.add_argument("--force", action="store_true", help="ignore completed-state and re-run")
    args = ap.parse_args()

    cfg = load_config()
    state = load_state()
    logger = get_logger()
    ctx = Ctx(cfg=cfg, state=state, logger=logger)

    selected = PHASES
    if args.only:
        selected = [args.only]
    elif args.from_phase:
        selected = PHASES[PHASES.index(args.from_phase):]

    logger.info("profile=%s curate=%s phases=%s", cfg.profile, cfg.curate_level, selected)

    for name in selected:
        if name in state.completed and not args.force and not args.only:
            logger.info("skip %s (already completed)", name)
            continue
        mod = _import_phase(name)
        logger.info("=== %s: run ===", name)
        mod.run(ctx)
        if not mod.check(ctx):
            logger.error("=== %s: ACCEPTANCE CHECK FAILED — halting ===", name)
            raise SystemExit(1)
        logger.info("=== %s: check passed ===", name)
        if name not in state.completed:
            state.completed.append(name)
        state.phase = name
        save_state(state)

    logger.info("done: completed=%s", state.completed)


if __name__ == "__main__":
    main()
