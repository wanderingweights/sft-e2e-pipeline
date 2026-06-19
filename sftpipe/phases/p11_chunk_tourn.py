"""Tournament packaging (NOT part of the main p00-p10 chain — invoke with
`python -m sftpipe.run --only p11_chunk_tourn`).

Slices each stage mix into `tourn_chunk_rows` (150k) chunks, each pre-split
train/test and uploaded as its own S3 object. One chunk -> one tournament task,
so tasks mirror the broad/repair/polish stages. Reads the canonical stage files
(data/mixed/stageN.jsonl) produced by P7, so it can run after the main pipeline.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sftpipe.god_prep import train_test_split, write_jsonl
from sftpipe.schema import CanonicalRecord, ReasoningMode, to_conversations
from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

MANIFEST = PIPELINE_DIR / "manifests" / "tourn_chunks.json"
STAGES = ["stage1", "stage2", "stage3"]


def run(ctx: "Ctx") -> None:
    from sftpipe.storage import Storage

    log, cfg = ctx.logger, ctx.cfg
    profile = cfg.active_profile
    size = cfg.tourn_chunk_rows
    mixed = ctx.data_root / "mixed"
    storage = Storage(cfg.storage)
    storage.ensure_bucket()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    def flush(buf: list[dict], stage: str, ci: int) -> dict:
        train, test = train_test_split(buf, cfg.val_set_size, cfg.seed)
        cdir = mixed / "tourn" / stage / f"chunk-{ci:05d}"
        tp, ep = cdir / "train.jsonl", cdir / "test.jsonl"
        write_jsonl(train, tp)
        write_jsonl(test, ep)
        base = storage.object_key(cfg.profile, "tourn-chunks", stage, f"chunk-{ci:05d}")
        tk = storage.upload_file(str(tp), f"{base}/train.jsonl")
        ek = storage.upload_file(str(ep), f"{base}/test.jsonl")
        log.info("%s chunk-%05d: %d train / %d test -> %s", stage, ci, len(train), len(test), base)
        return {"train_key": tk, "test_key": ek, "n_train": len(train), "n_test": len(test)}

    for stage in STAGES:
        src = mixed / f"{stage}.jsonl"
        if not src.exists():
            log.info("chunk skip %s (no mix)", stage)
            continue
        buf: list[dict] = []
        chunks: list[dict] = []
        ci = 0
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = CanonicalRecord.model_validate_json(line)
                mode = ReasoningMode(rec.meta.get("reasoning_mode", "on"))
                buf.append(to_conversations(rec, cfg.chat, profile, mode))
                if len(buf) >= size:
                    chunks.append(flush(buf, stage, ci))
                    ci += 1
                    buf = []
        if buf:
            chunks.append(flush(buf, stage, ci))
        manifest[stage] = {"bucket": storage.bucket, "chunk_rows": size, "n_chunks": len(chunks), "chunks": chunks}
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        log.info("%s: %d chunks of <=%d rows", stage, len(chunks), size)


def check(ctx: "Ctx") -> bool:
    return MANIFEST.exists()
