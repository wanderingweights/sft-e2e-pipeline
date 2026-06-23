"""The source table (P1/P2), grounded in the REAL columns of each dataset
(verified by streaming-peek on 2026-06-19). `kind` selects the normalize
adapter; `cols` maps canonical fields -> that source's column names.

max_rows subsets large sources at acquire time (mixing in P7 trims to token
budgets anyway, so we don't hoard disk). Nemotron-v2 is gated -> skipped until
HF auth is provided.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from sftpipe.schema import Domain


class Kind(str, Enum):
    instruct = "instruct"      # {instruction, output} single-turn
    messages = "messages"      # [{role, content}] chat
    sharegpt = "sharegpt"      # [{from, value}] chat
    openr1 = "openr1"          # problem + generations(<think>) + verify
    s1k = "s1k"                # question + thinking_trajectories + attempt
    limo = "limo"              # question + solution(CoT) + answer


class SourceSpec(BaseModel):
    id: str
    kind: Kind
    domain: Domain
    license: str
    stage: str = "stage1"
    hf_config: str | None = None
    split: str = "train"
    max_rows: int | None = None
    cols: dict[str, str] = Field(default_factory=dict)
    gated: bool = False
    cot_prose: bool = False       # output is plain-prose chain-of-thought (no <think> tags) -> route to reasoning
    reasoning_trace: bool = False  # output ships native <think> traces -> must come out reasoning-present (audited)

    @property
    def name(self) -> str:
        parts = [self.id.replace("/", "__")]
        if self.hf_config:
            parts.append(self.hf_config)
        if self.split and self.split != "train":
            parts.append(self.split)
        return "__".join(parts)


SOURCES: list[SourceSpec] = [
    # math — PRIMARY: hard reasoning traces (DeepSeek-R1, AoPS olympiad). Ships native
    # <think>; pass_rate_72b_tir drives the difficulty curriculum (lowest -> stage3 polish).
    SourceSpec(id="nvidia/OpenMathReasoning", kind=Kind.instruct, domain=Domain.math,
               license="CC-BY-4.0", hf_config="default", split="cot", max_rows=1_500_000, reasoning_trace=True,
               cols={"instruction": "problem", "output": "generated_solution",
                     "answer": "expected_answer", "pass_rate": "pass_rate_72b_tir"}),
    # math — SECONDARY: cleaned/deduped olympiad CoT (NuminaMath-1.5 supersedes -CoT).
    SourceSpec(id="AI-MO/NuminaMath-1.5", kind=Kind.instruct, domain=Domain.math,
               license="Apache-2.0", max_rows=300_000, cot_prose=True,
               cols={"instruction": "problem", "output": "solution", "answer": "answer"}),
    # math — BREADTH: GSM8K-level easy tier (prose answers, no traces -> cot_prose routed).
    # Keep for GSM8K coverage; NOT the reasoning primary.
    SourceSpec(id="nvidia/OpenMathInstruct-2", kind=Kind.instruct, domain=Domain.math,
               license="CC-BY-4.0", max_rows=400_000, cot_prose=True,
               cols={"instruction": "problem", "output": "generated_solution", "answer": "expected_answer"}),
    # Nemotron-v2: one config "default", multiple splits; reasoning on/off built-in.
    # Acquire English splits per-domain (skip multilingual_* for v1).
    SourceSpec(id="nvidia/Nemotron-Post-Training-Dataset-v2", kind=Kind.messages, domain=Domain.math,
               license="CC-BY-4.0", gated=True, hf_config="default", split="math", max_rows=300_000, cols={"messages": "messages"}),
    SourceSpec(id="nvidia/Nemotron-Post-Training-Dataset-v2", kind=Kind.messages, domain=Domain.code,
               license="CC-BY-4.0", gated=True, hf_config="default", split="code", max_rows=200_000, cols={"messages": "messages"}),
    SourceSpec(id="nvidia/Nemotron-Post-Training-Dataset-v2", kind=Kind.messages, domain=Domain.science,
               license="CC-BY-4.0", gated=True, hf_config="default", split="stem", max_rows=150_000, cols={"messages": "messages"}),
    SourceSpec(id="nvidia/Nemotron-Post-Training-Dataset-v2", kind=Kind.messages, domain=Domain.chat,
               license="CC-BY-4.0", gated=True, hf_config="default", split="chat", max_rows=150_000, cols={"messages": "messages"}),
    SourceSpec(id="open-thoughts/OpenThoughts3-1.2M", kind=Kind.sharegpt, domain=Domain.reasoning,
               license="check-card", max_rows=600_000,  # ~62KB/row -> cap for disk; still ample supply
               cols={"conversations": "conversations", "domain_field": "domain"}),
    SourceSpec(id="allenai/tulu-3-sft-mixture", kind=Kind.messages, domain=Domain.chat,
               license="ODC-BY", max_rows=500_000, cols={"messages": "messages"}),
    SourceSpec(id="allenai/tulu-3-sft-personas-instruction-following", kind=Kind.messages, domain=Domain.if_,
               license="ODC-BY", cols={"messages": "messages"}),
    # IF — verifier-filtered (IFEval/IFBench taxonomy). Two splits: chat constraints + structured
    # outputs (JSON/XML). messages format w/ reasoning on/off built in. Fills the IFEval supply gap.
    SourceSpec(id="nvidia/Nemotron-Instruction-Following-Chat-v1", kind=Kind.messages, domain=Domain.if_,
               license="ODC-BY", split="chat_if", max_rows=100_000, cols={"messages": "messages"}),
    SourceSpec(id="nvidia/Nemotron-Instruction-Following-Chat-v1", kind=Kind.messages, domain=Domain.if_,
               license="CC-BY-4.0", split="structured_outputs", max_rows=50_000, cols={"messages": "messages"}),
    SourceSpec(id="OpenCoder-LLM/opc-sft-stage2", kind=Kind.instruct, domain=Domain.code,
               license="MIT", hf_config="educational_instruct", max_rows=100_000,
               cols={"instruction": "instruction", "output": "output"}),
    SourceSpec(id="OpenCoder-LLM/opc-sft-stage2", kind=Kind.instruct, domain=Domain.code,
               license="MIT", hf_config="evol_instruct", max_rows=100_000,
               cols={"instruction": "instruction", "output": "output"}),
    SourceSpec(id="ise-uiuc/Magicoder-Evol-Instruct-110K", kind=Kind.instruct, domain=Domain.code,
               license="check-card", cols={"instruction": "instruction", "output": "response"}),
    # code — reasoning traces (R1, competitive programming). Ships native <think> in `output`;
    # configs are split_0 / split_1 (both needed). Targets LiveCodeBench / HumanEval / MBPP.
    SourceSpec(id="nvidia/OpenCodeReasoning", kind=Kind.instruct, domain=Domain.code,
               license="CC-BY-4.0", hf_config="split_0", split="split_0", max_rows=None, reasoning_trace=True,
               cols={"instruction": "input", "output": "output", "answer": "solution"}),
    SourceSpec(id="nvidia/OpenCodeReasoning", kind=Kind.instruct, domain=Domain.code,
               license="CC-BY-4.0", hf_config="split_1", split="split_1", max_rows=None, reasoning_trace=True,
               cols={"instruction": "input", "output": "output", "answer": "solution"}),
    # Tool/function-calling for S2 tool regressions (conversations w/ tool defs in system turn).
    SourceSpec(id="NousResearch/hermes-function-calling-v1", kind=Kind.sharegpt, domain=Domain.tool,
               license="Apache-2.0", hf_config="func_calling", cols={"conversations": "conversations"}),
    SourceSpec(id="allenai/SciRIFF", kind=Kind.instruct, domain=Domain.science,
               license="ODC-BY", hf_config="4096", max_rows=10_000,
               cols={"instruction": "input", "output": "output"}),
    SourceSpec(id="GAIR/LIMO", kind=Kind.limo, domain=Domain.math, license="check-card", stage="stage3",
               cols={"question": "question", "solution": "solution", "answer": "answer"}),
    SourceSpec(id="simplescaling/s1K", kind=Kind.s1k, domain=Domain.reasoning, license="check-card", stage="stage3",
               cols={"question": "question", "reasoning": "thinking_trajectories", "attempt": "attempt", "solution": "solution"}),
]


def available_sources() -> list[SourceSpec]:
    return [s for s in SOURCES if not s.gated]
