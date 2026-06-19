"""Per-source adapters: native row (dict) -> CanonicalRecord. Dispatched by
SourceSpec.kind. Column names come from spec.cols (verified against real
schemas 2026-06-19). CoT is split into the `reasoning` field where the source
exposes it (<think> traces, thinking_trajectories, CoT solutions).
"""
from __future__ import annotations

import re

from sftpipe.schema import CanonicalRecord, Domain, Message, Role, make_id
from sftpipe.sources import Kind, SourceSpec

_THINK = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL | re.IGNORECASE)

_FROM_MAP = {
    "human": Role.user, "user": Role.user,
    "gpt": Role.assistant, "assistant": Role.assistant, "chatgpt": Role.assistant,
    "system": Role.system,
}
_ROLE_MAP = {"user": Role.user, "assistant": Role.assistant, "system": Role.system, "tool": Role.user}

_DOMAIN_MAP = {
    "code": Domain.code, "math": Domain.math, "science": Domain.science,
    "chat": Domain.chat, "reasoning": Domain.reasoning,
}


def split_think(text: str) -> tuple[str | None, str]:
    """Return (reasoning|None, answer). If a <think>…</think> block is present,
    reasoning is its contents and answer is what follows; else reasoning=None."""
    if not text:
        return None, ""
    m = _THINK.search(text)
    if m:
        reasoning = m.group(1).strip()
        answer = m.group(2).strip()
        return (reasoning or None), (answer or text.strip())
    return None, text.strip()


def _native_id(row: dict, idx: int) -> str:
    for k in ("id", "uuid", "seq_id", "_instance_id"):
        if row.get(k) not in (None, ""):
            return str(row[k])
    return str(idx)


def _rec(spec: SourceSpec, native: str, user: str, msgs: list[Message], domain: Domain | None = None) -> CanonicalRecord | None:
    if not user.strip() or not any(m.role is Role.assistant and m.content.strip() for m in msgs):
        return None
    return CanonicalRecord(
        id=make_id(spec.id, native, user),
        source=spec.id, domain=domain or spec.domain,
        messages=msgs, license=spec.license,
        meta={"name": spec.name, **({"config": spec.hf_config} if spec.hf_config else {})},
    )


def _instruct(spec, row, idx):
    c = spec.cols
    instruction = str(row.get(c["instruction"], "") or "")
    output = str(row.get(c["output"], "") or "")
    reasoning, answer = split_think(output)
    msgs = [Message(role=Role.user, content=instruction),
            Message(role=Role.assistant, content=answer, reasoning=reasoning)]
    rec = _rec(spec, _native_id(row, idx), instruction, msgs)
    if rec and "answer" in c and row.get(c["answer"]) is not None:
        rec.meta["expected_answer"] = str(row[c["answer"]])
    return rec


def _from_turns(turns: list, role_key: str, content_key: str, role_resolver: dict) -> list[Message]:
    msgs = []
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        role = role_resolver.get(str(t.get(role_key, "")).lower())
        content = t.get(content_key)
        if role is None or content is None or not str(content).strip():
            continue  # drop empty turns (e.g. Nemotron's empty system message)
        if role is Role.assistant:
            reasoning, answer = split_think(str(content))
            msgs.append(Message(role=role, content=answer, reasoning=reasoning))
        else:
            msgs.append(Message(role=role, content=str(content)))
    return msgs


def _messages(spec, row, idx):
    msgs = _from_turns(row.get(spec.cols["messages"]), "role", "content", _ROLE_MAP)
    user = next((m.content for m in msgs if m.role is Role.user), "")
    return _rec(spec, _native_id(row, idx), user, msgs)


def _sharegpt(spec, row, idx):
    msgs = _from_turns(row.get(spec.cols["conversations"]), "from", "value", _FROM_MAP)
    user = next((m.content for m in msgs if m.role is Role.user), "")
    domain = spec.domain
    df = spec.cols.get("domain_field")
    if df and row.get(df):
        domain = _DOMAIN_MAP.get(str(row[df]).lower(), spec.domain)
    return _rec(spec, _native_id(row, idx), user, msgs, domain)


def _openr1(spec, row, idx):
    c = spec.cols
    problem = str(row.get(c["problem"], "") or "")
    gens = row.get(c.get("generations", "generations")) or []
    verify = row.get(c.get("verify", "correctness_math_verify")) or []
    text = ""
    for i, g in enumerate(gens):
        if i < len(verify) and verify[i]:
            text = str(g)
            break
    if not text and gens:
        text = str(gens[0])
    if not text:
        text = str(row.get(c.get("solution", "solution"), "") or "")
    reasoning, answer = split_think(text)
    if not answer:
        answer = str(row.get(c.get("answer", "answer"), "") or text)
    msgs = [Message(role=Role.user, content=problem),
            Message(role=Role.assistant, content=answer, reasoning=reasoning)]
    return _rec(spec, _native_id(row, idx), problem, msgs)


def _s1k(spec, row, idx):
    c = spec.cols
    question = str(row.get(c["question"], "") or "")
    tt = row.get(c["reasoning"]) or []
    reasoning = str(tt[0]) if isinstance(tt, list) and tt else (str(tt) if tt else None)
    answer = str(row.get(c.get("attempt", ""), "") or row.get(c.get("solution", ""), "") or "")
    msgs = [Message(role=Role.user, content=question),
            Message(role=Role.assistant, content=answer, reasoning=reasoning)]
    return _rec(spec, _native_id(row, idx), question, msgs)


def _limo(spec, row, idx):
    c = spec.cols
    question = str(row.get(c["question"], "") or "")
    reasoning = str(row.get(c["solution"], "") or "") or None
    answer = str(row.get(c.get("answer", ""), "") or "")
    if not answer:  # fall back to the CoT as the answer if no short answer
        answer, reasoning = (reasoning or ""), None
    msgs = [Message(role=Role.user, content=question),
            Message(role=Role.assistant, content=answer, reasoning=reasoning)]
    return _rec(spec, _native_id(row, idx), question, msgs)


_DISPATCH = {
    Kind.instruct: _instruct, Kind.messages: _messages, Kind.sharegpt: _sharegpt,
    Kind.openr1: _openr1, Kind.s1k: _s1k, Kind.limo: _limo,
}


def normalize_row(spec: SourceSpec, row: dict, idx: int) -> CanonicalRecord | None:
    return _DISPATCH[spec.kind](spec, row, idx)
