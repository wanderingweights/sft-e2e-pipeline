# sft-e2e-pipeline

A **model-agnostic, end-to-end SFT dataset pipeline**: it assembles a staged,
decontaminated, reasoning-formatted SFT corpus from public sources and pushes
training-ready shards to S3-compatible storage (GOD MinIO by default).

Built to post-train any chat/instruct model. The first target is
[`silx-ai/Quasar-Preview`](https://huggingface.co/silx-ai/Quasar-Preview)
(an 18B-total / 2B-active MoE, custom `quasar_long` arch), trained **full
fine-tune (no LoRA)** on 4×H100 via the G.O.D axolotl trainer — but nothing
in the pipeline is Quasar-specific; the model, tokenizer, chat template and
budgets all live in `pipeline/config.yaml`.

## Design

- **Output is what the trainer ingests**, not pre-tokenized tensors. We emit
  conversational JSONL in the G.O.D `ChatTemplateDatasetType` schema
  (`{conversations: [{from, value}]}`). The trainer (axolotl) does its own
  tokenization, chat-template application, sample-packing and loss-masking
  (`train_on_inputs: false`).
- **Reuses G.O.D prep code** (`sftpipe/god_prep.py`, `sftpipe/storage.py`) —
  column standardization, chat-row processing, train/test split and the
  MinIO uploader are lifted from `gradients-ai/G.O.D` and generalized so this
  repo is standalone (no dependency on the GOD package).
- **Curate-light by default**: normalize → dedup → **decontaminate (hard
  gate)** → `<think>`-format → mix → split → upload. Heavy quality filtering
  and curriculum ordering are deferred.

## Layout

```
pipeline/config.yaml     # all knobs: model profile, budgets, storage, benchmarks
sftpipe/                 # the package (general model-prep scripts)
  config.py              # pydantic config loader
  schema.py              # canonical record + ChatTemplate conversion
  storage.py             # S3/MinIO client (env-driven creds) — vendored from GOD
  god_prep.py            # standardize / chat-row / split / jsonl — vendored from GOD
  phases/                # one module per pipeline phase
  run.py                 # master loop (idempotent, checkpointed)
training/                # axolotl configs for the downstream full-FT run
RUNBOOK.md               # the executable build plan (phases 0–10)
```

## Storage credentials (never committed)

The MinIO/S3 client reads creds from env, matching GOD:
`S3_COMPATIBLE_ENDPOINT`, `S3_COMPATIBLE_ACCESS_KEY`, `S3_COMPATIBLE_SECRET_KEY`,
`S3_REGION`, `S3_BUCKET_NAME`. Put them in an untracked `.env`.

See `RUNBOOK.md` for the full phase-by-phase plan.
