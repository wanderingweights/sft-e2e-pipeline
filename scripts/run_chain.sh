#!/bin/bash
# Resumable, disk-bounded run of the pipeline. Detects the resume point from
# existing artifacts (we delete each upstream stage to bound disk, so we must
# never re-run a phase whose input is gone). Pull latest BEFORE launching.
cd "$(dirname "$0")/.." || exit 1
set -a; source .env; set +a

run() { echo "=== $(date +%H:%M:%S) $1 ==="; .venv/bin/python -m sftpipe.run --only "$1"; }
ne() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }
free_raw() { for d in data/raw/*/; do n=$(basename "$d"); [ -f "data/norm/$n.jsonl" ] && rm -f "$d/data.jsonl"; done; echo "freed raw"; }

# resume point (newest completed stage wins)
if   [ -f pipeline/manifests/s3_objects.json ]; then START=9
elif [ -f pipeline/manifests/mixture.json ];   then START=8
elif ne data/formatted;                         then START=7
elif ne data/decontam;                          then START=6
elif [ -f pipeline/manifests/dedup.json ];      then START=5
elif ne data/clean;                             then START=4
elif ne data/norm;                              then START=3
else START=1; fi
echo "=== resume START=$START ==="
skiprun() { if [ "$START" -le "$2" ]; then run "$1"; else echo "=== skip $1 (resumed past) ==="; fi; }

rm -f pipeline/PIPELINE_DONE pipeline/PIPELINE_FAILED
{
  skiprun p01_acquire 1 && skiprun p02_normalize 2 && free_raw \
  && skiprun p03_filter 3   && rm -rf data/norm \
  && skiprun p04_dedup 4 \
  && skiprun p05_decontam 5 && rm -rf data/clean \
  && skiprun p06_format 6    && rm -rf data/decontam \
  && skiprun p07_mix 7       && rm -rf data/formatted \
  && skiprun p08_upload 8 && skiprun p09_manifests 9 && skiprun p10_validate 10
} && touch pipeline/PIPELINE_DONE || touch pipeline/PIPELINE_FAILED
echo "=== chain end: $([ -f pipeline/PIPELINE_DONE ] && echo DONE || echo FAILED) $(date +%H:%M:%S) ==="
