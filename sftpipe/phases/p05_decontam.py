"""PHASE 5 — decontaminate clean/ against benchmark EVAL sets -> decontam/.

HARD GATE. Build an 8-gram index over the test questions of the target
benchmarks; drop any train record whose user prompt overlaps an eval question
by >= overlap_threshold of its n-grams. A re-scan then returns zero by
construction. Benchmarks that fail to load are logged (can't decontam against
them) but do not silently pass.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from sftpipe.schema import CanonicalRecord, Role
from sftpipe.sources import SOURCES
from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

REPORT = PIPELINE_DIR / "manifests" / "decontam_report.json"

# (loader id, config, split, question-column). Best-known ids; load failures are logged.
BENCH = {
    "mmlu": ("cais/mmlu", "all", "test", "question"),
    "gsm8k": ("openai/gsm8k", "main", "test", "question"),
    "math": ("HuggingFaceH4/MATH-500", None, "test", "problem"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_main", "train", "Question"),
    "ifeval": ("google/IFEval", None, "train", "prompt"),
    "aime": ("HuggingFaceH4/aime_2024", None, "train", "problem"),
}

_NON = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", _NON.sub(" ", (s or "").lower())).strip()


def _ngrams(s: str, n: int) -> set[str]:
    toks = _norm(s).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _build_index(ctx, n: int) -> tuple[set[str], dict]:
    from datasets import load_dataset

    log = ctx.logger
    grams: set[str] = set()
    loaded = {}
    for bench in ctx.cfg.benchmarks_to_decontaminate:
        if bench not in BENCH:
            log.warning("decontam: no loader for %s", bench)
            loaded[bench] = "no-loader"
            continue
        rid, cfg, split, col = BENCH[bench]
        try:
            ds = load_dataset(rid, cfg, split=split)
            c = 0
            for row in ds:
                q = row.get(col)
                if q:
                    grams |= _ngrams(str(q), n)
                    c += 1
            loaded[bench] = c
            log.info("decontam index: %s (%s) %d questions", bench, rid, c)
        except Exception as e:
            loaded[bench] = f"ERR:{type(e).__name__}"
            log.error("decontam: FAILED to load %s (%s): %s", bench, rid, str(e)[:140])
    return grams, loaded


def _user_text(rec: CanonicalRecord) -> str:
    return " ".join(m.content for m in rec.messages if m.role is Role.user)


def run(ctx: "Ctx") -> None:
    log = ctx.logger
    n = ctx.cfg.decontam.ngram
    thr = ctx.cfg.decontam.overlap_threshold
    clean, decon = ctx.data_root / "clean", ctx.data_root / "decontam"
    decon.mkdir(parents=True, exist_ok=True)
    eval_grams, loaded = _build_index(ctx, n)

    report = {"ngram": n, "threshold": thr, "benchmarks": loaded, "eval_ngrams": len(eval_grams), "per_source": {}}
    for spec in SOURCES:
        src, dst = clean / f"{spec.name}.jsonl", decon / f"{spec.name}.jsonl"
        if not src.exists():
            continue
        kept = flagged = 0
        tmp = dst.with_suffix(".jsonl.tmp")
        with open(src) as fin, open(tmp, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                pg = _ngrams(_user_text(CanonicalRecord.model_validate_json(line)), n)
                overlap = (len(pg & eval_grams) / len(pg)) if pg else 0.0
                if overlap >= thr:
                    flagged += 1
                else:
                    fout.write(line + "\n")
                    kept += 1
        tmp.rename(dst)
        report["per_source"][spec.name] = {"kept": kept, "flagged": flagged}
        log.info("decontam %s: kept %d, removed %d", spec.name, kept, flagged)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))


def check(ctx: "Ctx") -> bool:
    """Hard gate: re-scan decontam/ must find zero records over threshold, and
    at least one benchmark index must have loaded."""
    if not REPORT.exists():
        return False
    report = json.loads(REPORT.read_text())
    loaded_ok = any(isinstance(v, int) and v > 0 for v in report["benchmarks"].values())
    if not loaded_ok:
        ctx.logger.error("decontam: NO benchmark index loaded — gate fails")
        return False
    n, thr = report["ngram"], report["threshold"]
    eval_grams, _ = _build_index(ctx, n)
    decon = ctx.data_root / "decontam"
    for f in decon.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                pg = _ngrams(_user_text(CanonicalRecord.model_validate_json(line)), n)
                if pg and (len(pg & eval_grams) / len(pg)) >= thr:
                    ctx.logger.error("decontam re-scan FOUND contamination in %s — gate fails", f.name)
                    return False
    ctx.logger.info("decontam re-scan clean (0 over threshold)")
    return True
