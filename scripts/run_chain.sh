#!/bin/bash
# Resumable, disk-bounded run of the pipeline on the build box.
# Each phase is idempotent (skips when its OUTPUT exists), so re-running after a
# failure resumes correctly even though we delete each upstream stage to bound
# peak disk to ~2 stages. Pull latest BEFORE launching (not in-script, to avoid
# bash re-reading a changed file mid-run).
cd "$(dirname "$0")/.." || exit 1
set -a; source .env; set +a
RUN=".venv/bin/python -m sftpipe.run --only"
run() { echo "=== $(date +%H:%M:%S) $1 ==="; $RUN "$1"; }
free_raw() { for d in data/raw/*/; do n=$(basename "$d"); [ -f "data/norm/$n.jsonl" ] && rm -f "$d/data.jsonl"; done; echo "freed raw"; }

rm -f pipeline/PIPELINE_DONE pipeline/PIPELINE_FAILED
{
  run p01_acquire && run p02_normalize && free_raw \
  && run p03_filter    && rm -rf data/norm \
  && run p04_dedup \
  && run p05_decontam  && rm -rf data/clean \
  && run p06_format    && rm -rf data/decontam \
  && run p07_mix       && rm -rf data/formatted \
  && run p08_upload && run p09_manifests && run p10_validate
} && touch pipeline/PIPELINE_DONE || touch pipeline/PIPELINE_FAILED
echo "=== chain end: $([ -f pipeline/PIPELINE_DONE ] && echo DONE || echo FAILED) $(date +%H:%M:%S) ==="
