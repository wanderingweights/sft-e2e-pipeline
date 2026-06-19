# SFT-E2E Pipeline — Build Plan (adapted)

Staged SFT corpus for **any chat/instruct model** (first target:
`silx-ai/Quasar-Preview`), assembled to maximize MMLU, GPQA, IFEval, GSM8K,
MATH, AIME via the Nemotron-style staging *pattern* (broad → repair → polish).

**This is adapted from a generic plan to the real target + trainer.** Key
deltas from the original draft, and why:

| original draft | here | why |
|---|---|---|
| tokenizer `mamba-esc` | **Quasar tokenizer** (`silx-ai/Quasar-Preview`, vocab 157,184), config-driven `profile` | real target is Quasar; pipeline is model-agnostic |
| Phase 8 emits packed Arrow (`input_ids`+`loss_mask`) | **emit conversational JSONL** (`{conversations:[{from,value}]}`) + upload to S3 | the G.O.D **axolotl** trainer does its own tokenize / pack / loss-mask (`train_on_inputs:false`); pre-packing would fight it |
| 80B / 4B / 0.5B tokens | **~1.5B / 0.3B / 0.05B** (provisional) | 4×H100 **full fine-tune** (no LoRA); 80B is multi-week. Finalize from measured tok/s (Phase 0.4) |
| generic dedup/decontam | **reuse G.O.D prep** (`god_prep.py`, `storage.py`) + add decontam/dedup/think-format | use as much GOD as possible; GOD lacks only decontam + reasoning-format + cross-source mix |
| full 10-phase heavy curation | **curate-light** for v1 | decontam + think-format are the value; heavy quality-filter + curriculum deferred |

**Output handoff:** prep ALL stages, upload each stage's train/test JSONL to
**one MinIO bucket** under per-stage keys, and record the **object keys** in the
manifest. Training is 3 sequential full-FT runs, each pointed at one stage's S3
URL (re-signed from the key on demand — presigned URLs cap at 7 days).

```
s3://$S3_BUCKET_NAME/sft-e2e/quasar/stage1/{train,test}.jsonl
                                  /stage2/{train,test}.jsonl
                                  /stage3/{train,test}.jsonl
```

Each phase is **idempotent** (skip if output exists & valid) and **checkpointed**
(`pipeline/state.json`). Master loop: `python -m sftpipe.run`.

---

## Canonical schema (`sftpipe/schema.py`)

```json
{ "id": "...", "source": "open-r1/OpenR1-Math-220k", "domain": "math",
  "messages": [{"role":"user","content":"..."},
               {"role":"assistant","content":"<final answer>","reasoning":"<CoT, optional>"}],
  "reasoning_present": true, "license": "apache-2.0", "meta": {} }
```
Rule: keep CoT in `reasoning`, SEPARATE from the final answer. `<think>` rendering
happens in Phase 6 so reasoning-on/off variants come from one record.

---

## PHASES

**P0 `p00_env`** — venv + pinned `requirements.lock`; the **Quasar tokenizer
loads**; `<think>`/`</think>` round-trip; record vocab. **+ measure tokens/sec**
on the training rig to finalize `target_tokens`.
*Done (box 185.141.218.234): tokenizer loads, vocab 156,891. Findings:*
- *`<think>`/`</think>` = **3 tokens each** → must be added as special tokens before
  training (resize embeddings; new rows train in full-FT). Done in training yaml.*
- *Quasar ships a **custom chat template** `<role>HUMAN</role>…<role>ASSISTANT</role>`
  (bos `<|startoftext|>`, eos `<|endoftext|>`, pad = eos) — use `tokenizer_default`,
  NOT chatml. Fixed in config + training yaml.*
- *tokens/sec still TODO (needs the GPU rig, not this CPU box).*

**P1 `p01_acquire`** — `snapshot_download` each source to `data/raw/<id>/`; record
commit hash + row count. Source table below. On 3 failed retries: mark
`unavailable`, continue, recompute ratios over what's available.

**P2 `p02_normalize`** — one adapter per source → `data/norm/<id>.jsonl` in the
canonical schema. Split CoT into `reasoning` vs final answer for R1/OpenThoughts/
Numina. `id = sha256(source+native_id+user)`. Schema-validate every line.

**P3 `p03_filter`** *(light)* — drop empty / over-length / mojibake / broken
fences / never-answers. Math: keep verified + capped unverified (≤20% tolerable);
prefer compact CoT. Log per-source retention; flag >70% drop.

**P4 `p04_dedup`** — exact on `id`, then MinHash/LSH (`datasketch`, Jaccard ~0.8)
within+across sources. Cross-source ties: keep stronger-teacher/higher-license.

**P5 `p05_decontam`** ⚠️ **HARD GATE** — build n-gram index over the **test/eval**
splits of `mmlu,gpqa,ifeval,gsm8k,math,aime`; drop train records with ≥0.6 8-gram
overlap or exact normalized-substring match. Re-scan must return **zero**. Write
`manifests/decontam_report.json`. GOD does NOT do this; it's the #1 reason not to
ship sources as-is.

**P6 `p06_format`** — render assistant turns with `<think>` (`schema.render_assistant`):
reasoning-on = `<think>\n{cot}\n</think>\n\n{ans}`; reasoning-off (`reasoning_off_fraction`
10%) = empty think; truncated (`truncated_trace_fraction` 4%) = clip CoT, keep answer;
non-reasoning domains = passthrough. Final answers never empty.

**P7 `p07_mix`** — token-count each record once (cache in `meta`), sample (seed=42)
to each stage's `target_tokens` + domain ratios. Without replacement; upsample
under-supplied domains ≤3×. Stage mixes:
- **Stage 1** (broad): reasoning math/sci/code 45–55%, general chat 25–30%, code
  10–15%, IF/format ~5%, science QA 5–10%.
- **Stage 2** (repair): upsample regressions (tool-calling, IFEval) + ~20% Stage-1
  replay. Small `pack_len` (protect tool/format skills).
- **Stage 3** (polish): LIMO + s1K + hardest AIME traces only.
Write `manifests/mixture_stageN.json` (source→tokens, realized ratios, seed).

**P8 `p08_upload`** — convert to ChatTemplate (`schema.to_conversations`), apply
the empty-row guard + seeded train/test split (`god_prep`, seed 42), write
`data/mixed/stageN/{train,test}.jsonl`, **upload to MinIO** per-stage
(`storage.upload_file` → object key). NOTE: tokenize / loss-mask / **pack are the
TRAINER's job** — we do not pre-pack.

**P9 `p09_manifests`** — `sources.json` (id+commit+license+count),
`mixture_stageN.json`, `decontam_report.json`, `DATASET_CARD.md` (flag any
non-commercial subset), and `s3_objects.json` (bucket + per-stage keys for
re-signing). Pipeline re-derivable from manifests + seed + config.

**P10 `p10_validate`** — gates, all must pass: (1) decontam re-scan zero *(hard)*,
(2) think-block integrity + on/off/truncated fractions on target, (3) mix fidelity
±2%, (4) license audit, (5) S3 objects exist & re-signable, (6) optional 50-step
smoke train on `stage1` on a tiny copy (loss ↓, no NaNs, tokenizer matches).

---

## Source table (P1)

| domain | HF id | role | license |
|---|---|---|---|
| math | `nvidia/OpenMathInstruct-2` | GSM8K/MATH core | CC-BY-4.0 |
| math | `open-r1/OpenR1-Math-220k` | AIME/MATH R1 traces | Apache-2.0 |
| math | `AI-MO/NuminaMath-CoT` | competition CoT | Apache-2.0 |
| math/sci/code | `nvidia/Nemotron-Post-Training-Dataset-v2` | reasoning on/off built-in | CC-BY-4.0* |
| reasoning | `open-thoughts/OpenThoughts3-1.2M` | multi-domain R1-distill | check card |
| chat/IF | `allenai/tulu-3-sft-mixture` | general + Persona-IF | ODC-BY* |
| IF | `allenai/tulu-3-sft-personas-instruction-following` | IFEval driver | ODC-BY |
| code | `OpenCoder-LLM/opc-sft-stage2` | code w/ tests | MIT |
| code | `ise-uiuc/Magicoder-Evol-Instruct-110K` | code evol | check card |
| science | `allenai/SciRIFF` (~10k) | MMLU/GPQA STEM | ODC-BY |
| polish | `GAIR/LIMO`, `simplescaling/s1K` | Stage-3 only | check cards |

`*` record per-source license; flag non-commercial subsets in the card.

---

## Downstream training (separate repo, the G.O.D trainer)

Full fine-tune, **no LoRA** — see `training/base_quasar_fullft.yml`. 3 sequential
runs (stage1 → stage2 → stage3), each `base_model` = previous stage's checkpoint,
`datasets` = that stage's S3 URL. ZeRO-3 (+ CPU optimizer offload) on 4×H100;
batch is memory-tight because full-FT state for 18B params is ~288 GB.

### Notes for later (training/eval side, not this data pipeline)

- **Checkpoint windowing + best-of selection.** Save eval checkpoints at a
  regular step window during each stage; benchmark each on the target suite
  (MMLU/GPQA/IFEval/GSM8K/MATH/AIME) and select the best checkpoint to carry into
  the next stage (consider checkpoint-averaging across the top window). Don't
  assume the last step is best.
- **WandB: online, not silent.** Remove the trainer's `wandb_mode: offline`
  (GOD's `base.yml` default) — set `wandb_mode: online` + `wandb_entity` so each
  run emits a live dashboard link. `WANDB_API_KEY` via env on the box only.
