"""Canonical record schema + conversion to the trainer's ChatTemplate format.

Every source, regardless of origin, is normalized to `CanonicalRecord`. The
chain-of-thought is kept in a SEPARATE `reasoning` field from the final answer,
so Phase 6 can render reasoning-on / reasoning-off / truncated variants from one
record. Formatting into <think> tags happens at render time, not here.
"""
from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, Field

from sftpipe.config import ChatCfg, Profile


class Domain(str, Enum):
    math = "math"
    code = "code"
    science = "science"
    chat = "chat"
    if_ = "if"
    tool = "tool"
    safety = "safety"
    multilingual = "multilingual"
    reasoning = "reasoning"


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str
    reasoning: str | None = None  # chain-of-thought, kept separate from content


class CanonicalRecord(BaseModel):
    id: str
    source: str
    domain: Domain
    messages: list[Message]
    reasoning_present: bool = False
    license: str = "unknown"
    meta: dict = Field(default_factory=dict)

    def model_post_init(self, _ctx) -> None:
        if not self.reasoning_present:
            object.__setattr__(
                self, "reasoning_present", any(m.reasoning for m in self.messages)
            )


def make_id(source: str, native_id: str, user_content: str) -> str:
    return hashlib.sha256(f"{source}\x00{native_id}\x00{user_content}".encode()).hexdigest()


class ReasoningMode(str, Enum):
    on = "on"          # <think>{reasoning}</think>\n\n{answer}
    off = "off"        # <think>\n\n</think>\n\n{answer}  (empty-think, direct answer)
    none = "none"      # no think tags (non-reasoning domains / passthrough)


def render_assistant(msg: Message, mode: ReasoningMode, profile: Profile) -> str:
    """Render an assistant turn's text, applying <think> formatting per `mode`."""
    answer = msg.content
    if mode is ReasoningMode.none:
        return answer
    if mode is ReasoningMode.off or not msg.reasoning:
        return f"{profile.think_open}\n\n{profile.think_close}\n\n{answer}"
    # mode is on, with reasoning present
    return f"{profile.think_open}\n{msg.reasoning}\n{profile.think_close}\n\n{answer}"


def to_conversations(
    record: CanonicalRecord,
    chat: ChatCfg,
    profile: Profile,
    mode: ReasoningMode = ReasoningMode.on,
) -> dict:
    """Convert a canonical record to one row of the trainer ChatTemplate schema:
    {<chat.column>: [{<role_field>: ..., <content_field>: ...}, ...]}.
    """
    ref = {
        Role.system: chat.system_reference,
        Role.user: chat.user_reference,
        Role.assistant: chat.assistant_reference,
    }
    turns = []
    for m in record.messages:
        value = render_assistant(m, mode, profile) if m.role is Role.assistant else m.content
        turns.append({chat.role_field: ref[m.role], chat.content_field: value})
    return {chat.column: turns}
