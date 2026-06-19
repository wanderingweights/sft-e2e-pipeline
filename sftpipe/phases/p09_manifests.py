"""PHASE 9 — emit the dataset card from the per-phase manifests already written
(sources/normalize/filter/dedup/decontam_report/mixture/s3_objects). The pipeline
is re-derivable from these + seed + config.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sftpipe.state import PIPELINE_DIR

if TYPE_CHECKING:
    from sftpipe.context import Ctx

MAN = PIPELINE_DIR / "manifests"
CARD = MAN / "DATASET_CARD.md"
NON_COMMERCIAL_FLAGS = ("cc-by-nc", "non-commercial", "check-card")


def _load(name: str) -> dict:
    p = MAN / name
    return json.loads(p.read_text()) if p.exists() else {}


def run(ctx: "Ctx") -> None:
    sources = _load("sources.json")
    mixture = _load("mixture.json")
    decontam = _load("decontam_report.json")
    s3 = _load("s3_objects.json")

    flagged = [n for n, v in sources.items()
               if any(f in str(v.get("license", "")).lower() for f in NON_COMMERCIAL_FLAGS)]

    lines = ["# Dataset card — sft-e2e (profile: %s)" % ctx.cfg.profile, "",
             "Seed %d. Staged SFT corpus (broad -> repair -> polish), decontaminated, <think>-formatted." % ctx.cfg.seed,
             "", "## Sources (HF id, license, retained)"]
    for n, v in sorted(sources.items()):
        lines.append(f"- `{v.get('id')}`{(' [' + v['config'] + ']') if v.get('config') else ''}"
                     f"{(' split=' + v['split']) if v.get('split') not in (None, 'train') else ''}"
                     f" — {v.get('license')} — {v.get('status')} — {v.get('retained', '?')} rows (sha {str(v.get('sha'))[:10]})")
    lines += ["", "## License flags", ("⚠ verify before commercial use: " + ", ".join(flagged)) if flagged else "none flagged"]
    lines += ["", "## Decontamination", f"benchmarks: {decontam.get('benchmarks', {})}",
              f"eval n-grams indexed: {decontam.get('eval_ngrams')}"]
    lines += ["", "## Stage mixture (realized)"]
    for stage, v in mixture.items():
        lines.append(f"- {stage}: {v.get('records')} recs, {v.get('tokens')} tok, shares {v.get('domain_share')}")
    lines += ["", "## S3 objects (re-sign from key as needed)"]
    for stage, v in s3.items():
        lines.append(f"- {stage}: s3://{v.get('bucket')}/{v.get('train_key')} ({v.get('n_train')} train, {v.get('n_test')} test)")

    MAN.mkdir(parents=True, exist_ok=True)
    CARD.write_text("\n".join(lines) + "\n")
    ctx.logger.info("wrote %s (%d sources, %d flagged)", CARD.name, len(sources), len(flagged))


def check(ctx: "Ctx") -> bool:
    return CARD.exists() and (MAN / "s3_objects.json").exists()
