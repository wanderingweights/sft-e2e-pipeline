"""PHASE 10 — final validation gates. All critical gates must pass.

1. decontam re-scan == 0 (HARD; delegates to p05.check)
2. format fractions on target (reasoning-off / truncated within tolerance)
3. mix fidelity (realized domain shares vs config, logged; warn-only)
4. license audit (flag non-commercial / check-card)
5. S3 objects exist + re-signable
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

MAN = PIPELINE_DIR / "manifests"


def run(ctx: "Ctx") -> None:
    ctx.logger.info("validation: see check() gates")


def check(ctx: "Ctx") -> bool:
    from sftpipe.phases import p05_decontam
    from sftpipe.storage import Storage

    log = ctx.logger
    ok = True

    # 1. decontam hard re-scan
    if not p05_decontam.check(ctx):
        log.error("GATE 1 decontam re-scan FAILED")
        ok = False

    # 2. format fractions
    fmt = json.loads((MAN / "format.json").read_text()) if (MAN / "format.json").exists() else {}
    tot = sum(fmt.get(k, 0) for k in ("on", "off", "truncated"))
    if tot:
        off_r = fmt.get("off", 0) / tot
        trunc_r = fmt.get("truncated", 0) / tot
        log.info("format fractions: off=%.3f (tgt %.2f) trunc=%.3f (tgt %.2f)",
                 off_r, ctx.cfg.reasoning_off_fraction, trunc_r, ctx.cfg.truncated_trace_fraction)
        if abs(trunc_r - ctx.cfg.truncated_trace_fraction) > 0.02:
            log.warning("GATE 2 truncated fraction off target (warn)")

    # 3. mix fidelity (warn-only)
    mixture = json.loads((MAN / "mixture.json").read_text()) if (MAN / "mixture.json").exists() else {}
    for stage, recipe in ctx.cfg.stages.items():
        ratios = recipe.get("domain_ratios", {})
        realized = mixture.get(stage, {}).get("domain_share", {})
        for dom, want in ratios.items():
            got = realized.get(dom, 0.0)
            if abs(got - want) > 0.02:
                log.warning("GATE 3 %s/%s share %.3f vs target %.2f (supply-limited?)", stage, dom, got, want)

    # 4. license audit
    sources = json.loads((MAN / "sources.json").read_text()) if (MAN / "sources.json").exists() else {}
    flagged = [n for n, v in sources.items()
               if any(f in str(v.get("license", "")).lower() for f in ("nc", "non-commercial", "check-card"))]
    if flagged:
        log.warning("GATE 4 license: verify before commercial use: %s", flagged)

    # 5. S3 objects exist + re-signable
    s3 = json.loads((MAN / "s3_objects.json").read_text()) if (MAN / "s3_objects.json").exists() else {}
    if not s3:
        log.error("GATE 5 no S3 objects")
        ok = False
    else:
        storage = Storage(ctx.cfg.storage)
        for stage, v in s3.items():
            if not storage.object_exists(v["train_key"]) or not storage.object_exists(v["test_key"]):
                log.error("GATE 5 missing S3 object for %s", stage)
                ok = False
            else:
                storage.presigned_url(v["train_key"])  # re-sign smoke
        log.info("GATE 5 S3 objects present + re-signable")

    log.info("VALIDATION %s", "PASSED" if ok else "FAILED")
    return ok
