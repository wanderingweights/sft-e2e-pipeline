"""PHASE 1 — Acquire raw datasets to data/raw/<name>/data.jsonl.

Idempotent per source (skip if .done + manifest ok). Streams + subsets to
max_rows (bounded disk). Records commit sha + retained count + columns in
manifests/sources.json. On 3 failed retries: mark `unavailable`, continue
(ratios get recomputed downstream).
"""
from __future__ import annotations

import json
from itertools import islice
from typing import TYPE_CHECKING

from sftpipe.sources import SOURCES
from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

MAX_RETRIES = 3
MANIFEST = PIPELINE_DIR / "manifests" / "sources.json"


def _load() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _save(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2))


def run(ctx: "Ctx") -> None:
    import datasets as hfds
    from datasets import load_dataset
    from huggingface_hub import HfApi

    hfds.logging.set_verbosity_error()
    log = ctx.logger
    api = HfApi()
    raw_root = ctx.data_root / "raw"
    manifest = _load()

    for spec in SOURCES:
        name = spec.name
        out_dir = raw_root / name
        out, done = out_dir / "data.jsonl", out_dir / ".done"
        if done.exists() and manifest.get(name, {}).get("status") == "ok":
            log.info("acquire skip %s (n=%s)", name, manifest[name].get("retained"))
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                try:
                    sha = api.dataset_info(spec.id).sha
                except Exception:
                    sha = None
                it = iter(load_dataset(spec.id, spec.hf_config, split=spec.split, streaming=True))
                if spec.max_rows:
                    it = islice(it, spec.max_rows)
                n, cols = 0, None
                with open(out, "w") as f:
                    for row in it:
                        cols = cols or list(row.keys())
                        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                        n += 1
                done.write_text("")
                manifest[name] = {
                    "id": spec.id, "config": spec.hf_config, "split": spec.split, "sha": sha,
                    "license": spec.license, "domain": spec.domain.value, "kind": spec.kind.value,
                    "requested_max": spec.max_rows, "retained": n, "columns": cols, "status": "ok",
                }
                _save(manifest)
                log.info("acquired %s: %d rows (sha=%s)", name, n, (sha or "?")[:10])
                break
            except Exception as e:
                last_err = e
                log.warning("acquire %s attempt %d/%d: %s", name, attempt, MAX_RETRIES, str(e)[:160])
        else:
            manifest[name] = {"id": spec.id, "config": spec.hf_config, "split": spec.split,
                              "license": spec.license, "status": "unavailable", "error": str(last_err)[:300]}
            _save(manifest)
            log.error("acquire %s UNAVAILABLE after %d retries", name, MAX_RETRIES)


def check(ctx: "Ctx") -> bool:
    m = _load()
    ok = [k for k, v in m.items() if v.get("status") == "ok"]
    bad = [k for k, v in m.items() if v.get("status") != "ok"]
    ctx.logger.info("acquire: %d ok, %d unavailable %s", len(ok), len(bad), bad or "")
    return len(ok) > 0
