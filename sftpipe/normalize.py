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
    # tool/function-call turns -> treat tool output as a user turn (no native tool role)
    "tool": Role.user, "function": Role.user, "observation": Role.user, "function_response": Role.user,
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
    # Unterminated <think> with no closing tag (~62% of OpenThoughts3 traces are
    # truncated): treat the post-<think> body as reasoning and use its last
    # paragraph as the answer, so the record isn't hollowed out / corrupted.
    oi = text.lower().find("<think>")
    if oi != -1:
        body = text[oi + len("<think>"):].strip()
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        if len(paras) > 1:
            return "\n\n".join(paras[:-1]), paras[-1]
        return (body or None), (body or "")
    return None, text.strip()


def _extract_boxed(text: str) -> str | None:
    """Return the contents of the LAST \\boxed{...} (brace-balanced), else None."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth, out = 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip() or None
        out.append(ch)
        i += 1
    return None  # unbalanced


def split_prose_cot(text: str, expected_answer: str | None = None) -> tuple[str | None, str]:
    """For sources whose chain-of-thought is plain prose (no <think> tags), treat
    the whole solution as reasoning and synthesize a concise final-answer line.

    answer preference: explicit expected_answer -> last \\boxed{} -> last paragraph.
    If none of those yield a split, fall back to (None, text) so the record is
    rendered as a direct answer rather than fabricating reasoning."""
    text = (text or "").strip()
    if not text:
        return None, ""
    # NuminaMath et al. use sentinels ("proof", "notfound") for non-numeric answers;
    # don't box those — fall back to \boxed{} in the text, then the last paragraph.
    exp = str(expected_answer).strip() if expected_answer not in (None, "") else ""
    if exp.lower() in {"proof", "notfound", "none", "n/a"}:
        exp = ""
    ans = exp or _extract_boxed(text)
    if ans:
        return text, f"The final answer is $\\boxed{{{ans}}}$."
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) > 1:
        return "\n\n".join(paras[:-1]), paras[-1]
    return None, text


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
    expected = row.get(c["answer"]) if "answer" in c else None
    reasoning, answer = split_think(output)
    if reasoning is None and spec.cot_prose:
        # plain-prose CoT (no <think> tags): route the working into reasoning
        reasoning, answer = split_prose_cot(output, expected)
    msgs = [Message(role=Role.user, content=instruction),
            Message(role=Role.assistant, content=answer, reasoning=reasoning)]
    rec = _rec(spec, _native_id(row, idx), instruction, msgs)
    if rec and expected is not None:
        rec.meta["expected_answer"] = str(expected)
    if rec and "pass_rate" in c and row.get(c["pass_rate"]) is not None:
        try:  # difficulty signal for the curriculum (lower pass_rate = harder)
            rec.meta["pass_rate"] = float(row[c["pass_rate"]])
        except (TypeError, ValueError):
            pass
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
