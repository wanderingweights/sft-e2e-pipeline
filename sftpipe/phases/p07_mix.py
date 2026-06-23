"""PHASE 7 — mix to per-stage token budgets + domain ratios.
formatted/ -> mixed/stage{1,2,3}.jsonl.

- token-count each record once (Quasar tokenizer), cached per source (.tokens)
- sample without replacement to hit each stage's ratio*budget; upsample <=3x
  for under-supplied domains, else log the deviation (don't over-repeat)
- stage2 adds a replay sample of the stage-1 selection (anti-forgetting)
- stage3 = exclusive sources (LIMO/s1K) + hardest traces (longest CoT) to budget
- global shuffle within each stage (v1; curriculum is a later option)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sftpipe.sources import SOURCES
from sftpipe.state import PIPELINE_DIR

# Module-level tokenizer worker: the Quasar tokenizer is a slow custom (non-Rust)
# tokenizer, so we fan the per-record token count out across processes.
_WORKER_TOK = None


def _tok_init(path: str, trust: bool) -> None:
    global _WORKER_TOK
    from transformers import AutoTokenizer
    _WORKER_TOK = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)


def _tok_lens(texts: list[str]) -> list[int]:
    return [len(x) for x in _WORKER_TOK(texts, add_special_tokens=False)["input_ids"]]

if TYPE_CHECKING:
    from sftpipe.context import Ctx

REPORT_DIR = PIPELINE_DIR / "manifests"
MAX_UPSAMPLE = 3


@dataclass
class Item:
    name: str
    idx: int
    domain: str
    source: str
    tokens: int
    rlen: int
    difficulty: float | None = None  # pass_rate (lower = harder); for stage3 curriculum


def _record_text(rec: dict) -> str:
    return " ".join(((m.get("content") or "") + " " + (m.get("reasoning") or "")) for m in rec["messages"])


def _batches(src, size: int):
    """Stream a formatted file as bounded lists of record texts (never load whole
    file — sources reach 32GB)."""
    batch: list[str] = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(_record_text(json.loads(line)))
            if len(batch) >= size:
                yield batch
                batch = []
    if batch:
        yield batch


def _count_tokens(ctx, names) -> None:
    fmt = ctx.data_root / "formatted"
    prof = ctx.cfg.active_profile
    todo = [n for n in names if (fmt / f"{n}.jsonl").exists() and not (fmt / f"{n}.tokens").exists()]
    if not todo:
        return
    n_workers = max(1, min(24, (os.cpu_count() or 4) - 2))
    ctx.logger.info("token-count: %d sources across %d workers", len(todo), n_workers)
    # Fan tokenization out across cores; main process streams+parses, workers tokenize.
    # imap preserves order so counts[idx] stays aligned with file line order.
    pool = mp.Pool(n_workers, initializer=_tok_init,
                   initargs=(prof.tokenizer_path, prof.trust_remote_code))
    try:
        for name in todo:
            src, cache = fmt / f"{name}.jsonl", fmt / f"{name}.tokens"
            counts: list[int] = []
            for chunk_counts in pool.imap(_tok_lens, _batches(src, 1000), chunksize=1):
                counts.extend(chunk_counts)
            cache.write_text("\n".join(map(str, counts)))
            ctx.logger.info("tokens %s: %d recs, %d tok", name, len(counts), sum(counts))
    finally:
        pool.close()
        pool.join()


def _build_index(ctx, names) -> list[Item]:
    fmt = ctx.data_root / "formatted"
    items: list[Item] = []
    for name in names:
        src, cache = fmt / f"{name}.jsonl", fmt / f"{name}.tokens"
        if not src.exists() or not cache.exists():
            continue
        counts = [int(x) for x in cache.read_text().split()]
        with open(src) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rlen = sum(len(m.get("reasoning") or "") for m in rec["messages"])
                difficulty = (rec.get("meta") or {}).get("pass_rate")
                items.append(Item(name, idx, rec["domain"], rec["source"],
                                  counts[idx] if idx < len(counts) else 0, rlen, difficulty))
    return items


def _sample(pool: list[Item], target_tokens: float, rng: random.Random) -> tuple[list[Item], int, int]:
    """Sample to hit target_tokens; upsample by repeating the pool up to MAX_UPSAMPLE."""
    if not pool or target_tokens <= 0:
        return [], 0, 0
    pool = pool[:]
    rng.shuffle(pool)
    sel, tok, reps = [], 0, 0
    while tok < target_tokens and reps < MAX_UPSAMPLE:
        for it in pool:
            if tok >= target_tokens:
                break
            sel.append(it)
            tok += it.tokens
        reps += 1
    return sel, tok, reps


def _materialize_and_write(ctx, stage_sel: list[Item], out_path) -> None:
    fmt = ctx.data_root / "formatted"
    need: dict[str, set[int]] = defaultdict(set)
    for it in stage_sel:
        need[it.name].add(it.idx)
    lines: dict[tuple, str] = {}
    for name, idxs in need.items():
        with open(fmt / f"{name}.jsonl") as f:
            for idx, line in enumerate(f):
                if idx in idxs:
                    lines[(name, idx)] = line.strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as out:
        for it in stage_sel:
            out.write(lines[(it.name, it.idx)] + "\n")


def _domain_mix(by_domain, ratios, budget, rng, log, label) -> list[Item]:
    sel = []
    realized = {}
    for dom, ratio in ratios.items():
        s, tok, reps = _sample(by_domain.get(dom, []), ratio * budget, rng)
        sel += s
        realized[dom] = tok
        want = ratio * budget
        flag = f"  ⚠ {reps}x upsample, short {100*(1-tok/want):.0f}%" if tok < 0.98 * want else ""
        log.info("%s %s: %d tok (target %d, %d recs)%s", label, dom, tok, int(want), len(s), flag)
    return sel


def run(ctx: "Ctx") -> None:
    log, cfg = ctx.logger, ctx.cfg
    rng = random.Random(cfg.seed)
    names = [s.name for s in SOURCES]
    _count_tokens(ctx, names)
    items = _build_index(ctx, names)
    log.info("mix index: %d records, %d tok total", len(items), sum(it.tokens for it in items))

    st3 = cfg.stages["stage3"]
    excl = set(st3.get("exclusive_sources", []))
    hardest_from = set(st3.get("hardest_from", []))

    # ---- stage 1: broad, exclude stage-3-exclusive sources ----
    s1_pool = [it for it in items if it.source not in excl]
    by_dom1 = defaultdict(list)
    for it in s1_pool:
        by_dom1[it.domain].append(it)
    t1 = cfg.target_tokens["stage1"]
    stage1 = _domain_mix(by_dom1, cfg.stages["stage1"]["domain_ratios"], t1, rng, log, "stage1")
    rng.shuffle(stage1)

    # ---- stage 2: repair ratios + replay of stage-1 selection ----
    t2 = cfg.target_tokens["stage2"]
    replay_frac = cfg.stages["stage2"].get("replay_stage1_fraction", 0.0)
    by_dom2 = defaultdict(list)
    for it in items:
        if it.source not in excl:
            by_dom2[it.domain].append(it)
    stage2 = _domain_mix(by_dom2, cfg.stages["stage2"]["domain_ratios"], t2 * (1 - replay_frac), rng, log, "stage2")
    replay, rtok, _ = _sample(stage1, t2 * replay_frac, rng)
    stage2 += replay
    log.info("stage2 replay: %d tok (%d recs)", rtok, len(replay))
    rng.shuffle(stage2)

    # ---- stage 3: exclusive sources + hardest traces (by length, then by pass_rate) ----
    t3 = cfg.target_tokens["stage3"]
    pr_sources = set(st3.get("hardest_by_passrate", []))
    stage3 = [it for it in items if it.source in excl]
    tok3 = sum(it.tokens for it in stage3)
    hard = sorted((it for it in items if it.source in hardest_from), key=lambda it: -it.rlen)
    # pass_rate curriculum: take the LOWEST pass_rate (hardest) rows for polish
    hard_pr = sorted((it for it in items if it.source in pr_sources and it.difficulty is not None),
                     key=lambda it: it.difficulty if it.difficulty is not None else 1.0)
    # reserve up to half the remaining polish budget for the pass_rate-hardest math, so the
    # (much larger) length-sorted pool can't starve it; length-hardest fills the rest.
    pr_budget = (t3 - tok3) * 0.5
    for src, limit in ((hard_pr, tok3 + pr_budget), (hard, t3)):
        for it in src:
            if tok3 >= limit:
                break
            stage3.append(it)
            tok3 += it.tokens
    rng.shuffle(stage3)

    report = {}
    for stage, sel in [("stage1", stage1), ("stage2", stage2), ("stage3", stage3)]:
        _materialize_and_write(ctx, sel, ctx.data_root / "mixed" / f"{stage}.jsonl")
        dom_tok = defaultdict(int)
        dom_rows = defaultdict(int)        # total rows emitted (incl. upsampled repeats)
        dom_unique = defaultdict(set)      # distinct source records
        for it in sel:
            dom_tok[it.domain] += it.tokens
            dom_rows[it.domain] += 1
            dom_unique[it.domain].add((it.name, it.idx))
        total = sum(dom_tok.values())
        rows = len(sel)
        unique = sum(len(s) for s in dom_unique.values())
        report[stage] = {
            "rows": rows,                              # total trained rows (with upsampling)
            "unique_rows": unique,                     # distinct source records (real data)
            "upsample_factor": round(rows / max(unique, 1), 2),
            "tokens": total,
            "by_domain": {d: {"rows": dom_rows[d], "unique_rows": len(dom_unique[d]),
                              "tokens": dom_tok[d], "token_share": round(dom_tok[d] / max(total, 1), 3)}
                          for d in sorted(dom_tok)},
        }
        log.info("%s: %d rows (%d unique, %.2fx upsample), %d tok", stage, rows, unique,
                 report[stage]["upsample_factor"], total)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "mixture.json").write_text(json.dumps(report, indent=2))


def check(ctx: "Ctx") -> bool:
    mixed = ctx.data_root / "mixed"
    ok = all((mixed / f"{s}.jsonl").exists() for s in ("stage1", "stage2", "stage3"))
    ctx.logger.info("mix: stage files present = %s", ok)
    return ok
