"""PHASE 4 — dedup across all clean files: exact (content hash) + MinHash/LSH
near-dup (Jaccard ~0.8). Rewrites data/clean in place keeping survivors.

Higher-priority sources (R1-distilled) are inserted first, so on a near-dup
collision the stronger-teacher copy is the one kept.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

from sftpipe.schema import CanonicalRecord
from sftpipe.sources import SOURCES
from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

REPORT = PIPELINE_DIR / "manifests" / "dedup.json"
NUM_PERM = 64
THRESHOLD = 0.8
SHINGLE_K = 5

# R1-distilled / high-quality sources win near-dup ties.
HIGH_PRIORITY = {
    "open-r1/OpenR1-Math-220k", "open-thoughts/OpenThoughts3-1.2M",
    "nvidia/Nemotron-Post-Training-Dataset-v2", "GAIR/LIMO", "simplescaling/s1K",
}

_WORD = re.compile(r"\w+")


def _text(rec: CanonicalRecord) -> str:
    return " ".join(((m.content or "") + " " + (m.reasoning or "")) for m in rec.messages).lower()


def _shingles(text: str) -> set[str]:
    toks = _WORD.findall(text)
    if len(toks) < SHINGLE_K:
        return {text} if text else set()
    return {" ".join(toks[i:i + SHINGLE_K]) for i in range(len(toks) - SHINGLE_K + 1)}


def run(ctx: "Ctx") -> None:
    from datasketch import MinHash, MinHashLSH

    log = ctx.logger
    clean = ctx.data_root / "clean"
    specs = [s for s in SOURCES if (clean / f"{s.name}.jsonl").exists()]
    specs.sort(key=lambda s: 0 if s.id in HIGH_PRIORITY else 1)

    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    seen_exact: set[str] = set()
    keep: dict[str, set[int]] = {s.name: set() for s in specs}
    total = dropped_exact = dropped_near = 0

    for spec in specs:  # PASS 1: decide survivors (priority order)
        with open(clean / f"{spec.name}.jsonl") as fin:
            for idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                total += 1
                text = _text(CanonicalRecord.model_validate_json(line))
                h = hashlib.md5(text.encode()).hexdigest()
                if h in seen_exact:
                    dropped_exact += 1
                    continue
                shingles = _shingles(text)
                if not shingles:
                    continue
                mh = MinHash(num_perm=NUM_PERM)
                for sh in shingles:
                    mh.update(sh.encode())
                if lsh.query(mh):
                    dropped_near += 1
                    continue
                seen_exact.add(h)
                lsh.insert(f"{spec.name}#{idx}", mh)
                keep[spec.name].add(idx)

    report = {"total": total, "dropped_exact": dropped_exact, "dropped_near": dropped_near,
              "kept": total - dropped_exact - dropped_near, "per_source": {}}
    for spec in specs:  # PASS 2: rewrite survivors
        path = clean / f"{spec.name}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        keptn = 0
        with open(path) as fin, open(tmp, "w") as fout:
            for idx, line in enumerate(fin):
                if idx in keep[spec.name]:
                    fout.write(line if line.endswith("\n") else line + "\n")
                    keptn += 1
        tmp.rename(path)
        report["per_source"][spec.name] = keptn
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    log.info("dedup: %d total, -%d exact, -%d near => %d kept",
             total, dropped_exact, dropped_near, report["kept"])


def check(ctx: "Ctx") -> bool:
    ok = REPORT.exists()
    if ok:
        ctx.logger.info("dedup report: %s", json.loads(REPORT.read_text()).get("kept"))
    return ok
