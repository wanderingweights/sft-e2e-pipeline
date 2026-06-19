"""PHASE 8 — convert each stage to the trainer's ChatTemplate schema, split
train/test, and upload to MinIO. mixed/stageN.jsonl -> mixed/stageN/{train,test}.jsonl
-> s3://<bucket>/<prefix>/<profile>/stageN/{train,test}.jsonl.

NOTE: tokenize / loss-mask / pack are the TRAINER's job (axolotl, train_on_inputs:
false). We only emit conversational JSONL + push it.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sftpipe.god_prep import guard_not_mostly_empty, train_test_split, write_jsonl
from sftpipe.schema import CanonicalRecord, ReasoningMode, to_conversations
from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

S3_MANIFEST = PIPELINE_DIR / "manifests" / "s3_objects.json"
STAGES = ["stage1", "stage2", "stage3"]


def run(ctx: "Ctx") -> None:
    from sftpipe.storage import Storage

    log, cfg = ctx.logger, ctx.cfg
    profile = cfg.active_profile
    mixed = ctx.data_root / "mixed"

    storage = Storage(cfg.storage)
    storage.ensure_bucket()
    manifest = json.loads(S3_MANIFEST.read_text()) if S3_MANIFEST.exists() else {}

    for stage in STAGES:
        src = mixed / f"{stage}.jsonl"
        if not src.exists():
            log.info("upload skip %s (no mix)", stage)
            continue
        rows = []
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = CanonicalRecord.model_validate_json(line)
                mode = ReasoningMode(rec.meta.get("reasoning_mode", "on"))
                rows.append(to_conversations(rec, cfg.chat, profile, mode))
        guard_not_mostly_empty(rows, cfg.chat)
        train, test = train_test_split(rows, cfg.val_set_size, cfg.seed)

        stage_dir = mixed / stage
        train_path, test_path = stage_dir / "train.jsonl", stage_dir / "test.jsonl"
        write_jsonl(train, train_path)
        write_jsonl(test, test_path)

        base = storage.object_key(cfg.profile, stage)
        train_key = storage.upload_file(str(train_path), f"{base}/train.jsonl")
        test_key = storage.upload_file(str(test_path), f"{base}/test.jsonl")
        manifest[stage] = {
            "bucket": storage.bucket,
            "train_key": train_key, "test_key": test_key,
            "train_url": storage.presigned_url(train_key),
            "test_url": storage.presigned_url(test_key),
            "n_train": len(train), "n_test": len(test),
        }
        S3_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        S3_MANIFEST.write_text(json.dumps(manifest, indent=2))
        log.info("uploaded %s: train=%d test=%d -> %s", stage, len(train), len(test), train_key)


def check(ctx: "Ctx") -> bool:
    if not S3_MANIFEST.exists():
        return False
    manifest = json.loads(S3_MANIFEST.read_text())
    present = [s for s in STAGES if (ctx.data_root / "mixed" / f"{s}.jsonl").exists()]
    ok = all(s in manifest and manifest[s].get("train_key") for s in present)
    ctx.logger.info("upload: %d/%d stages in S3 manifest", sum(1 for s in present if s in manifest), len(present))
    return ok and len(present) > 0
