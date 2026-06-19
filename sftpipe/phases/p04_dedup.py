"""PHASE 4 — dedup across all clean files: exact (content hash) + MinHash/LSH
near-dup (Jaccard ~0.8). Rewrites data/clean in place keeping survivors.

Signature computation (the expensive part) is parallelized across all cores;
the LSH insert/query is then a fast serial pass in priority order, so the
stronger-teacher (R1-distilled) copy wins a near-dup collision.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import TYPE_CHECKING

from sftpipe.sources import SOURCES
from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

REPORT = PIPELINE_DIR / "manifests" / "dedup.json"
NUM_PERM = 64
THRESHOLD = 0.8
SHINGLE_K = 5

HIGH_PRIORITY = {
    "open-r1/OpenR1-Math-220k", "open-thoughts/OpenThoughts3-1.2M",
    "nvidia/Nemotron-Post-Training-Dataset-v2", "GAIR/LIMO", "simplescaling/s1K",
}

_WORD = re.compile(r"\w+")


def _text(rec: dict) -> str:
    return " ".join(((m.get("content") or "") + " " + (m.get("reasoning") or "")) for m in rec["messages"]).lower()


def _shingles(text: str) -> set[str]:
    toks = _WORD.findall(text)
    if len(toks) < SHINGLE_K:
        return {text} if text else set()
    return {" ".join(toks[i:i + SHINGLE_K]) for i in range(len(toks) - SHINGLE_K + 1)}


def _sig(text: str):
    """Worker: (md5, minhash-digest) for one record text. Texts are streamed in
    (not held in RAM); only tiny signatures come back."""
    from datasketch import MinHash

    mh = MinHash(num_perm=NUM_PERM)
    for sh in _shingles(text):
        mh.update(sh.encode())
    return hashlib.md5(text.encode()).hexdigest(), mh.digest()


def run(ctx: "Ctx") -> None:
    from datasketch import MinHash, MinHashLSH

    log = ctx.logger
    clean = ctx.data_root / "clean"
    specs = [s for s in SOURCES if (clean / f"{s.name}.jsonl").exists()]
    specs.sort(key=lambda s: 0 if s.id in HIGH_PRIORITY else 1)

    entries: list[tuple[str, int]] = []

    def text_stream():  # yields texts in priority order, recording entries as it goes
        for spec in specs:
            with open(clean / f"{spec.name}.jsonl") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    entries.append((spec.name, idx))
                    yield _text(json.loads(line))

    n_proc = cpu_count()
    log.info("dedup: streaming signatures through %d workers", n_proc)
    with Pool(n_proc) as pool:
        sigs = list(pool.imap(_sig, text_stream(), chunksize=2000))
    log.info("dedup: %d signatures computed; serial LSH pass", len(sigs))
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    seen_exact: set[str] = set()
    keep: dict[str, set[int]] = defaultdict(set)
    dropped_exact = dropped_near = 0
    for (name, idx), (md5, digest) in zip(entries, sigs):
        if md5 in seen_exact:
            dropped_exact += 1
            continue
        mh = MinHash(num_perm=NUM_PERM, hashvalues=digest)
        if lsh.query(mh):
            dropped_near += 1
            continue
        seen_exact.add(md5)
        lsh.insert(f"{name}#{idx}", mh)
        keep[name].add(idx)

    total = len(entries)
    report = {"total": total, "dropped_exact": dropped_exact, "dropped_near": dropped_near,
              "kept": total - dropped_exact - dropped_near, "per_source": {}}
    for spec in specs:  # rewrite survivors
        path = clean / f"{spec.name}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        kept_n = 0
        with open(path) as fin, open(tmp, "w") as fout:
            for idx, line in enumerate(fin):
                if idx in keep[spec.name]:
                    fout.write(line if line.endswith("\n") else line + "\n")
                    kept_n += 1
        tmp.rename(path)
        report["per_source"][spec.name] = kept_n
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    log.info("dedup: %d total, -%d exact, -%d near => %d kept",
             total, dropped_exact, dropped_near, report["kept"])


def check(ctx: "Ctx") -> bool:
    ok = REPORT.exists()
    if ok:
        ctx.logger.info("dedup kept: %s", json.loads(REPORT.read_text()).get("kept"))
    return ok
