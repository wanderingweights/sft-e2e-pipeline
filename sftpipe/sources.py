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

    @property
    def name(self) -> str:
        parts = [self.id.replace("/", "__")]
        if self.hf_config:
            parts.append(self.hf_config)
        if self.split and self.split != "train":
            parts.append(self.split)
        return "__".join(parts)


SOURCES: list[SourceSpec] = [
    SourceSpec(id="nvidia/OpenMathInstruct-2", kind=Kind.instruct, domain=Domain.math,
               license="CC-BY-4.0", max_rows=1_000_000,
               cols={"instruction": "problem", "output": "generated_solution", "answer": "expected_answer"}),
    SourceSpec(id="open-r1/OpenR1-Math-220k", kind=Kind.openr1, domain=Domain.math,
               license="Apache-2.0",
               cols={"problem": "problem", "solution": "solution", "answer": "answer",
                     "generations": "generations", "verify": "correctness_math_verify"}),
    SourceSpec(id="AI-MO/NuminaMath-CoT", kind=Kind.instruct, domain=Domain.math,
               license="Apache-2.0", max_rows=500_000,
               cols={"instruction": "problem", "output": "solution"}),
    # Nemotron-v2: one config "default", multiple splits; reasoning on/off built-in.
    # Acquire English splits per-domain (skip multilingual_* for v1).
    SourceSpec(id="nvidia/Nemotron-Post-Training-Dataset-v2", kind=Kind.messages, domain=Domain.math,
               license="CC-BY-4.0", hf_config="default", split="math", max_rows=300_000, cols={"messages": "messages"}),
    SourceSpec(id="nvidia/Nemotron-Post-Training-Dataset-v2", kind=Kind.messages, domain=Domain.code,
               license="CC-BY-4.0", hf_config="default", split="code", max_rows=200_000, cols={"messages": "messages"}),
    SourceSpec(id="nvidia/Nemotron-Post-Training-Dataset-v2", kind=Kind.messages, domain=Domain.science,
               license="CC-BY-4.0", hf_config="default", split="stem", max_rows=150_000, cols={"messages": "messages"}),
    SourceSpec(id="nvidia/Nemotron-Post-Training-Dataset-v2", kind=Kind.messages, domain=Domain.chat,
               license="CC-BY-4.0", hf_config="default", split="chat", max_rows=150_000, cols={"messages": "messages"}),
    SourceSpec(id="open-thoughts/OpenThoughts3-1.2M", kind=Kind.sharegpt, domain=Domain.reasoning,
               license="check-card", max_rows=400_000,
               cols={"conversations": "conversations", "domain_field": "domain"}),
    SourceSpec(id="allenai/tulu-3-sft-mixture", kind=Kind.messages, domain=Domain.chat,
               license="ODC-BY", max_rows=500_000, cols={"messages": "messages"}),
    SourceSpec(id="allenai/tulu-3-sft-personas-instruction-following", kind=Kind.messages, domain=Domain.if_,
               license="ODC-BY", cols={"messages": "messages"}),
    SourceSpec(id="OpenCoder-LLM/opc-sft-stage2", kind=Kind.instruct, domain=Domain.code,
               license="MIT", hf_config="educational_instruct", max_rows=100_000,
               cols={"instruction": "instruction", "output": "output"}),
    SourceSpec(id="OpenCoder-LLM/opc-sft-stage2", kind=Kind.instruct, domain=Domain.code,
               license="MIT", hf_config="evol_instruct", max_rows=100_000,
               cols={"instruction": "instruction", "output": "output"}),
    SourceSpec(id="ise-uiuc/Magicoder-Evol-Instruct-110K", kind=Kind.instruct, domain=Domain.code,
               license="check-card", cols={"instruction": "instruction", "output": "response"}),
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
