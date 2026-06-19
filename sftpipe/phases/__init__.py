"""Ordered phase registry. Each phase is a module `p{NN}_{name}` exposing
`run(ctx)` and `check(ctx) -> bool`. The master loop (sftpipe.run) imports them
lazily so unimplemented phases don't break the import graph.

See RUNBOOK.md for the full contract of each phase.
"""

PHASES: list[str] = [
    "p00_env",
    "p01_acquire",
    "p02_normalize",
    "p03_filter",
    "p04_dedup",
    "p05_decontam",
    "p06_format",
    "p07_mix",
    "p08_upload",      # tokenize/loss-mask/pack are the TRAINER's job; we upload JSONL
    "p09_manifests",
    "p10_validate",
]
