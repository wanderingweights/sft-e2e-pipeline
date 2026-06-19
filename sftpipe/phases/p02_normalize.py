"""PHASE 2 — Normalize each acquired source to data/norm/<name>.jsonl in the
canonical schema. Idempotent per source. Adapters live in sftpipe.normalize and
are dispatched by SourceSpec.kind. Records in/out/retention in
manifests/normalize.json.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sftpipe.normalize import normalize_row
from sftpipe.sources import SOURCES
from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

REPORT = PIPELINE_DIR / "manifests" / "normalize.json"


def run(ctx: "Ctx") -> None:
    log = ctx.logger
    raw_root = ctx.data_root / "raw"
    norm_root = ctx.data_root / "norm"
    norm_root.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}

    for spec in SOURCES:
        name = spec.name
        raw = raw_root / name / "data.jsonl"
        norm = norm_root / f"{name}.jsonl"
        if not raw.exists():
            log.info("normalize skip %s (no raw)", name)
            continue
        if norm.exists():
            log.info("normalize skip %s (exists)", name)
            continue
        n_in = n_out = 0
        tmp = norm.with_suffix(".jsonl.tmp")
        with open(raw) as fin, open(tmp, "w") as fout:
            for idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                rec = normalize_row(spec, json.loads(line), idx)
                if rec is not None:
                    fout.write(rec.model_dump_json() + "\n")
                    n_out += 1
        tmp.rename(norm)  # atomic: only a complete file counts as done
        report[name] = {"in": n_in, "out": n_out, "retention": round(n_out / max(n_in, 1), 3),
                        "domain": spec.domain.value}
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2))
        log.info("normalized %s: %d -> %d (%.1f%%)", name, n_in, n_out, 100 * n_out / max(n_in, 1))


def check(ctx: "Ctx") -> bool:
    from sftpipe.schema import CanonicalRecord

    norm_root = ctx.data_root / "norm"
    files = sorted(norm_root.glob("*.jsonl")) if norm_root.exists() else []
    for f in files:  # spot-validate first line of each
        with open(f) as fh:
            first = fh.readline().strip()
        if first:
            CanonicalRecord.model_validate_json(first)
    ctx.logger.info("normalize: %d source files validated", len(files))
    return len(files) > 0
