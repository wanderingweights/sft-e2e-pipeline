"""PHASE 6 — reasoning-format: decontam -> formatted. Assigns each record a
reasoning mode (on / off / truncated) deterministically to hit the configured
fractions, and clips long traces for the truncated set. The actual <think>
rendering happens at convert time (P8) via schema.to_conversations(mode).

  on        : <think>{reasoning}</think>\n\n{answer}
  off       : <think>\n\n</think>\n\n{answer}   (empty-think; also all non-reasoning records)
  truncated : reasoning clipped to a budget, then rendered as `on`
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from sftpipe.schema import CanonicalRecord, ReasoningMode, Role
from sftpipe.sources import Kind, SOURCES
from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

REPORT = PIPELINE_DIR / "manifests" / "format.json"
TRUNC_CHARS = 6000  # ~1.5k tokens; keeps the final answer intact
MIN_REASONING_RATE = 0.5  # CoT-bearing sources must come out at least this reasoning-present


def _expects_reasoning(spec) -> bool:
    """Sources that MUST carry chain-of-thought: explicit reasoning kinds + the
    plain-prose-CoT instruct sources. A regression here (e.g. a CoT field that
    stops parsing) silently produces empty-<think>, so we fail loud instead."""
    return spec.cot_prose or spec.reasoning_trace or spec.kind in (Kind.openr1, Kind.s1k, Kind.limo)


def _rand(rid: str, seed: int) -> float:
    return int.from_bytes(hashlib.sha256(f"{seed}:{rid}".encode()).digest()[:8], "big") / 2 ** 64


def run(ctx: "Ctx") -> None:
    log, cfg = ctx.logger, ctx.cfg
    off_f, trunc_f, seed = cfg.reasoning_off_fraction, cfg.truncated_trace_fraction, cfg.seed
    decon, fmt = ctx.data_root / "decontam", ctx.data_root / "formatted"
    fmt.mkdir(parents=True, exist_ok=True)
    counts = {"on": 0, "off": 0, "truncated": 0, "off_nonreasoning": 0}
    present: dict[str, list[int]] = {}  # spec.name -> [reasoning_present, total]
    for spec in SOURCES:
        src, dst = decon / f"{spec.name}.jsonl", fmt / f"{spec.name}.jsonl"
        if not src.exists():
            continue
        if dst.exists():
            log.info("format skip %s", spec.name)
            continue
        pc = present.setdefault(spec.name, [0, 0])
        tmp = dst.with_suffix(".jsonl.tmp")
        with open(src) as fin, open(tmp, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = CanonicalRecord.model_validate_json(line)
                pc[1] += 1
                pc[0] += int(rec.reasoning_present)
                if not rec.reasoning_present:
                    mode = ReasoningMode.off
                    counts["off_nonreasoning"] += 1
                else:
                    r = _rand(rec.id, seed)
                    if r < off_f:
                        mode = ReasoningMode.off
                        counts["off"] += 1
                    elif r < off_f + trunc_f:
                        mode = ReasoningMode.on
                        counts["truncated"] += 1
                        for m in rec.messages:
                            if m.role is Role.assistant and m.reasoning and len(m.reasoning) > TRUNC_CHARS:
                                m.reasoning = m.reasoning[:TRUNC_CHARS].rsplit(" ", 1)[0]
                    else:
                        mode = ReasoningMode.on
                        counts["on"] += 1
                rec.meta["reasoning_mode"] = mode.value
                fout.write(rec.model_dump_json() + "\n")
        tmp.rename(dst)
        log.info("format %s done", spec.name)
    by_source = {n: {"present": p, "total": t, "rate": round(p / t, 3) if t else 0.0}
                 for n, (p, t) in present.items()}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"counts": counts, "by_source": by_source}, indent=2))
    log.info("format modes: %s", counts)

    bad = []
    for spec in SOURCES:
        if not _expects_reasoning(spec):
            continue
        st = by_source.get(spec.name)
        if st and st["total"] and st["rate"] < MIN_REASONING_RATE:
            bad.append(f"{spec.name}: {st['rate']:.0%} reasoning-present ({st['present']}/{st['total']})")
    if bad:
        raise ValueError(
            "CoT-bearing sources came out mostly empty-<think> (parsing regression?):\n  "
            + "\n  ".join(bad)
        )


def check(ctx: "Ctx") -> bool:
    files = list((ctx.data_root / "formatted").glob("*.jsonl")) if (ctx.data_root / "formatted").exists() else []
    ctx.logger.info("format: %d files", len(files))
    return len(files) > 0
