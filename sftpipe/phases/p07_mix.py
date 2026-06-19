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
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sftpipe.sources import SOURCES
from sftpipe.state import PIPELINE_DIR

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


def _record_text(rec: dict) -> str:
    return " ".join(((m.get("content") or "") + " " + (m.get("reasoning") or "")) for m in rec["messages"])


def _count_tokens(ctx, names) -> None:
    from transformers import AutoTokenizer

    fmt = ctx.data_root / "formatted"
    tok = AutoTokenizer.from_pretrained(ctx.cfg.active_profile.tokenizer_path,
                                        trust_remote_code=ctx.cfg.active_profile.trust_remote_code)
    for name in names:
        src, cache = fmt / f"{name}.jsonl", fmt / f"{name}.tokens"
        if cache.exists() or not src.exists():
            continue
        texts = []
        with open(src) as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(_record_text(json.loads(line)))
        counts = []
        for i in range(0, len(texts), 1000):
            counts.extend(len(x) for x in tok(texts[i:i + 1000], add_special_tokens=False)["input_ids"])
        cache.write_text("\n".join(map(str, counts)))
        ctx.logger.info("tokens %s: %d recs, %d tok", name, len(counts), sum(counts))


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
                items.append(Item(name, idx, rec["domain"], rec["source"],
                                  counts[idx] if idx < len(counts) else 0, rlen))
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

    # ---- stage 3: exclusive sources + hardest traces ----
    t3 = cfg.target_tokens["stage3"]
    stage3 = [it for it in items if it.source in excl]
    tok3 = sum(it.tokens for it in stage3)
    hard = sorted((it for it in items if it.source in hardest_from), key=lambda it: -it.rlen)
    for it in hard:
        if tok3 >= t3:
            break
        stage3.append(it)
        tok3 += it.tokens
    rng.shuffle(stage3)

    report = {}
    for stage, sel in [("stage1", stage1), ("stage2", stage2), ("stage3", stage3)]:
        _materialize_and_write(ctx, sel, ctx.data_root / "mixed" / f"{stage}.jsonl")
        dom_tok = defaultdict(int)
        for it in sel:
            dom_tok[it.domain] += it.tokens
        total = sum(dom_tok.values())
        report[stage] = {"records": len(sel), "tokens": total,
                         "domain_share": {d: round(v / max(total, 1), 3) for d, v in dom_tok.items()}}
        log.info("%s: %d recs, %d tok, shares=%s", stage, len(sel), total, report[stage]["domain_share"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "mixture.json").write_text(json.dumps(report, indent=2))


def check(ctx: "Ctx") -> bool:
    mixed = ctx.data_root / "mixed"
    ok = all((mixed / f"{s}.jsonl").exists() for s in ("stage1", "stage2", "stage3"))
    ctx.logger.info("mix: stage files present = %s", ok)
    return ok
