"""Typed loader for pipeline/config.yaml.

Everything that can vary between runs/models lives in the YAML; code reads it
through these models so we never hardcode proportions, budgets or tokenizer.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "pipeline" / "config.yaml"


class Profile(BaseModel):
    base_model: str
    tokenizer_path: str
    trust_remote_code: bool = True
    chat_template: str = "chatml"
    think_open: str = "<think>"
    think_close: str = "</think>"
    adapter: str = "none"  # "none" => full fine-tune


class ChatCfg(BaseModel):
    column: str = "conversations"
    role_field: str = "from"
    content_field: str = "value"
    user_reference: str = "user"
    assistant_reference: str = "assistant"
    system_reference: str = "system"


class StorageCfg(BaseModel):
    backend: str = "minio"
    prefix: str = "sft-e2e"
    bucket_env: str = "S3_BUCKET_NAME"
    endpoint_env: str = "S3_COMPATIBLE_ENDPOINT"
    access_key_env: str = "S3_COMPATIBLE_ACCESS_KEY"
    secret_key_env: str = "S3_COMPATIBLE_SECRET_KEY"
    region_env: str = "S3_REGION"
    secure: bool = True
    # S3/MinIO SigV4 caps presigned URLs at 7 days (604800s). We default to the
    # max; for >7d access regenerate from the object key or use public_url.
    presign_expiry_secs: int = 604800


class DecontamCfg(BaseModel):
    ngram: int = 8
    overlap_threshold: float = 0.6


class Config(BaseModel):
    seed: int = 42
    curate_level: str = "light"
    profile: str
    profiles: dict[str, Profile]
    output_schema: str = "chat_template"
    chat: ChatCfg = Field(default_factory=ChatCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    target_tokens: dict[str, int]
    reasoning_off_fraction: float = 0.10
    truncated_trace_fraction: float = 0.04
    pack_len: dict[str, int]
    val_set_size: float = 0.05
    benchmarks_to_decontaminate: list[str]
    decontam: DecontamCfg = Field(default_factory=DecontamCfg)
    stages: dict = Field(default_factory=dict)
    tourn_chunk_rows: int = 150_000  # tournament packaging: rows per chunk

    @property
    def active_profile(self) -> Profile:
        return self.profiles[self.profile]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
