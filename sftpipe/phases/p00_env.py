"""PHASE 0 — Environment, tokenizer & throughput baseline.

Goal:   reproducible env; the active model's tokenizer loads; <think> tags map
        cleanly; a measured tokens/sec to size the real token budget.
Check:  tokenizer round-trips losslessly; think_open/think_close encode as
        expected (single tokens or registered specials).

This is the contract every phase module follows: `run(ctx)` does the work,
`check(ctx) -> bool` is the acceptance gate the master loop enforces.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sftpipe.context import Ctx


def run(ctx: "Ctx") -> None:
    prof = ctx.cfg.active_profile
    log = ctx.logger
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(prof.tokenizer_path, trust_remote_code=prof.trust_remote_code)
    ctx.state.notes["vocab_size"] = tok.vocab_size

    sample = "Solve: what is 2+2? " + prof.think_open + "\nadd them\n" + prof.think_close + "\n\n4"
    ids = tok(sample, add_special_tokens=False)["input_ids"]
    roundtrip = tok.decode(ids)
    ctx.state.notes["roundtrip_ok"] = sample.strip() in roundtrip or roundtrip.strip() == sample.strip()

    for tag in (prof.think_open, prof.think_close):
        n = len(tok(tag, add_special_tokens=False)["input_ids"])
        ctx.state.notes[f"think_token_len[{tag}]"] = n
        log.info("think tag %r -> %d token(s)", tag, n)
        if n != 1:
            log.warning("%r is not a single token; add it as a special token before Phase 8.", tag)

    log.info("tokenizer=%s vocab=%s chat_template=%s", prof.tokenizer_path, tok.vocab_size, prof.chat_template)
    # TODO: measured tokens/sec smoke run on the training rig to finalize target_tokens.


def check(ctx: "Ctx") -> bool:
    return bool(ctx.state.notes.get("roundtrip_ok"))
