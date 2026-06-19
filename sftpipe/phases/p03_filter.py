"""PHASE 3 — light quality filter: data/norm -> data/clean, per source.

Drops: empty user/assistant, absurd length, mojibake, unbalanced code fences.
Curate-light: we do NOT over-filter (question diversity > perfect solutions).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sftpipe.schema import CanonicalRecord, Domain, Role
from sftpipe.sources import SOURCES
from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

REPORT = PIPELINE_DIR / "manifests" / "filter.json"
MAX_CHARS = 200_000          # absurd-length cap (trainer packs by tokens; this just kills outliers)
MOJIBAKE = "�"


def _ok(rec: CanonicalRecord) -> bool:
    users = [m for m in rec.messages if m.role is Role.user]
    asst = [m for m in rec.messages if m.role is Role.assistant]
    if not users or not asst:
        return False
    if not any(m.content.strip() for m in users) or not any(m.content.strip() for m in asst):
        return False
    for m in rec.messages:
        text = (m.content or "") + (m.reasoning or "")
        if len(text) > MAX_CHARS or text.count(MOJIBAKE) > 3:
            return False
    if rec.domain is Domain.code:  # unbalanced ``` => broken fence
        for m in asst:
            if m.content.count("```") % 2 != 0:
                return False
    return True


def run(ctx: "Ctx") -> None:
    log = ctx.logger
    norm, clean = ctx.data_root / "norm", ctx.data_root / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for spec in SOURCES:
        src, dst = norm / f"{spec.name}.jsonl", clean / f"{spec.name}.jsonl"
        if not src.exists():
            continue
        if dst.exists():
            log.info("filter skip %s", spec.name)
            continue
        n = k = 0
        tmp = dst.with_suffix(".jsonl.tmp")
        with open(src) as fin, open(tmp, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if _ok(CanonicalRecord.model_validate_json(line)):
                    fout.write(line + "\n")
                    k += 1
        tmp.rename(dst)
        ret = round(k / max(n, 1), 3)
        report[spec.name] = {"in": n, "out": k, "retention": ret}
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2))
        log.info("filter %s: %d -> %d (%.1f%%)%s", spec.name, n, k, 100 * ret,
                 "  ⚠ >70% dropped — check adapter" if ret < 0.30 else "")


def check(ctx: "Ctx") -> bool:
    files = list((ctx.data_root / "clean").glob("*.jsonl")) if (ctx.data_root / "clean").exists() else []
    ctx.logger.info("filter: %d clean files", len(files))
    return len(files) > 0
