# Heretic EXL3 Integration Handoff

## Status: rewritten against verified upstream API

The earlier scaffolding (`src/heretic/backends/`) has been removed and
replaced with a direct EXL3 model class that mirrors Heretic's existing
`Model` surface. All upstream API calls in the new code are based on
reading the actual `exllamav3` source (master branch, version 0.0.34;
PyPI's latest is 0.0.33 — API surface is stable across both).

This document describes the new architecture and lists what the user
needs to validate on EXL3-capable hardware before declaring the
integration done.

---

## Architecture

```
heretic.Settings.backend = "hf" | "exl3"            (new field)
    |
    +-- "hf"   -> heretic.model.Model                (unchanged HF/PEFT path)
    +-- "exl3" -> heretic.exl3_model.Exl3Model       (new)
```

`Exl3Model` duck-types every method `main.py` calls on `Model`:

- `get_layers()`, `get_layer_modules()`, `get_abliterable_components()`
- `get_residuals(_batched)`, `get_residuals_mean`, `get_logprobs(_batched)`
- `get_responses(_batched)`, `generate()`, `stream_chat_response()`
- `abliterate()`, `reset_model()`
- `save_adapter()`  (EXL3-only; replaces `get_merged_model()`)
- `tokenizer`  (a small wrapper that lazy-loads the HF tokenizer alongside
  the exllamav3 one — used for chat templating and batch tokenization)

`main.py` branches on `settings.backend` at:
1. model construction
2. save-to-folder (forces adapter sidecar for EXL3)
3. upload-to-hub (forces adapter sidecar; uploads via
   `huggingface_hub.upload_folder` from a temp dir)

---

## Key design decisions (and why)

### Module discovery
Walks the loaded `model` via its `__iter__` (recurses through
`Module.modules`), filters by `.key` regex
```
^model\.layers\.(\d+)\..*?(o_proj|down_proj)(?:\.slice\.\d+)?$
```
Groups by layer index and routes under `attn.o_proj` / `mlp.down_proj`
to match the names Heretic's optimizer already uses. MoE expert paths
(`...block_sparse_moe.experts.{i}.down_proj`) collapse under
`mlp.down_proj`, mirroring the HF backend.

### LoRA injection without exllamav3.LoRA
The upstream `LoRA` class only loads from a PEFT directory on disk —
not workable for a hot loop that mutates A/B every trial. Instead we
use `Linear.lora_a_tensors` and `Linear.lora_b_tensors` directly. Each
target Linear gets a pre-allocated `(in_features, 1)` / `(1, out_features)`
fp16 slot, keyed by a sentinel `object()`. `Linear.forward()` already
iterates these dicts and applies `output += x @ A @ B` for each pair, so
the math drops in for free.

### EXL3-aware abliteration math
For HF, Heretic computes `delta_W = -λ v v^T W` (shape `(out, in)`)
and stores it as a low-rank LoRA. exllamav3 stores `W_exl3` as
`(in, out)` (HF transposed) and applies LoRA additively as
`output += x @ A @ B`. The arithmetic for an HF-equivalent rank-1
abliteration update becomes:
```
A_exl3 (in, 1) = W_exl3 @ v       (== W_hf.T @ v)
B_exl3 (1, out) = -λ * v.T
```
Computed on the unpadded slice in fp32, then zero-padded to the
Linear's actual shape and cast to fp16. See
`Exl3Model._write_lora_for_module()`.

`row_normalization=pre` and `row_normalization=full` are not yet
supported on the EXL3 backend — they'd need to dequantize every target
weight per trial (PRE) and use a higher-rank slot (FULL). `abliterate`
raises `NotImplementedError` explicitly in those cases. The default
`row_normalization=full` will need to be overridden to `none` until
this is addressed.

### Per-layer residuals
`block.export_state = True` on every `TransformerBlock` captures the
block-output residual into `params["export_states"]`. To also produce
the embedding-output state (HF's `hidden_states[0]`), `Exl3Model`
monkey-patches the first block's `forward` with a wrapper that prepends
the pre-block-0 `x.half().clone()` to the same list. Captures sorted
by layer index → stacked → `(prompt, layer + 1, dim)` at the last
token position.

### PEFT-compatible adapter save
`save_adapter()` writes `adapter_config.json` + `adapter_model.safetensors`
with PEFT-shape tensors:
- `lora_A.weight`: `(rank, in_features)` (transposed from exllamav3's `(in, rank)`)
- `lora_B.weight`: `(out_features, rank)` (transposed from exllamav3's `(rank, out)`)
Keys are `base_model.model.<full_module_key>.lora_A.weight`/`.lora_B.weight`,
matching PEFT's `LoraConfig(task_type=CAUSAL_LM)` convention. The same
file is loadable by both `peft.PeftModel.from_pretrained` and
`exllamav3.model.lora.LoRA.from_directory`.

---

## What's confidently working

Verified locally in this container:
- All modules compile (`python -m compileall`).
- `ruff check` is clean.
- Imports resolve: `from heretic import config, model, exl3_model, main`.
- `Settings(model='x').backend` defaults to `Backend.HF`.
- `Settings(model='x', backend='exl3').backend == Backend.EXL3`.

---

## What needs validation on EXL3 hardware

Run these in order on a machine with the `exl3` extra installed and a
real EXL3-quantized model. The model dir must contain the HF tokenizer
files (`tokenizer.json`, `tokenizer_config.json`) — every standard
EXL3 conversion script copies them; just verify.

### 1. Module discovery
```bash
PYTHONPATH=src python scripts/inspect_exl3_modules.py /path/to/exl3-model --out artifacts/modules.json
```
Confirm the JSON has nonzero `target_module_count` and the per-layer
groupings look right. If `WARNING: no abliterable modules matched` is
printed, the architecture uses a key naming that doesn't match the
regex — paste the first 20 keys here so the regex can be widened.

### 2. LoRA injection + reset
```bash
PYTHONPATH=src python scripts/smoke_test_exl3_lora.py /path/to/exl3-model --strict --out artifacts/smoke.json
```
This synthesizes a deterministic rank-1 update, generates baseline →
adapted → reset, and asserts (a) adapted ≠ baseline, (b) reset ==
baseline. If `--strict` fails, the output JSON's `failure_category`
and `traceback` pinpoint where.

### 3. Full optimization loop
```bash
PYTHONPATH=src heretic --backend exl3 --row-normalization none /path/to/exl3-model
```
The `--row-normalization none` flag is required because PRE/FULL are
not yet implemented on the EXL3 backend. The rest of the flags work
identically to the HF path.

### Likely friction points (open questions)

These are areas where I had to make assumptions that should be
sanity-checked the first time the smoke test runs:

a) **KV cache hygiene across batches.** Each forward pass creates a
   fresh `params` dict; we don't explicitly reset cache positions.
   For Heretic's residual/logprob calls (no autoregression, fresh
   batch every time) this should be correct — exllamav3's `forward_ls`
   reads positions from `params`, not from cache state, when no
   prefill is requested. If you see corrupted residuals on the second
   batch onward, that assumption is wrong and we need to call
   `cache.reset()` (if it exists in this version) or recreate the
   cache between batches.

b) **`Generator.generate` signature.** I'm calling
   `generator.generate(prompt=[...], max_new_tokens=N, completion_only=True, add_bos=True)`.
   If the version on the user's machine uses different keyword names,
   the smoke test will fail in step "baseline_text" with a
   `TypeError` — easy to spot.

c) **`last_tokens_only=1` for logprobs.** Confirmed in the `forward_ls`
   source we read but not exercised against an actual model.

d) **`completion_only=True` returning completions only (not full
   text).** Inferred from examples/ but not verified. `get_responses`
   slices off the prompt length, so if `completion_only` is actually a
   no-op in this version we'd over-slice. Look at the first few
   baseline responses in the smoke output to confirm they're not
   truncated.

---

## Files of interest

| Path | Purpose |
|---|---|
| `src/heretic/exl3_model.py` | New `Exl3Model` class — the entire EXL3 backend |
| `src/heretic/config.py` | Added `Backend` enum + `backend` / `exl3_max_num_tokens` settings |
| `src/heretic/model.py` | Cleaned up — removed `HfBnbBackend` reference |
| `src/heretic/main.py` | Backend branching at construction, save, upload |
| `scripts/inspect_exl3_modules.py` | Rewritten against real API |
| `scripts/smoke_test_exl3_lora.py` | Rewritten — no longer requires external PEFT adapter |
| `pyproject.toml` | Added `[exl3]` optional dep |

Removed: `src/heretic/backends/` (entire directory — base.py, hf_bnb.py, exl3.py, __init__.py).

---

## If the smoke test fails

Paste the contents of `artifacts/smoke.json` (specifically
`failure_category`, `error`, `traceback`) and the first 20 keys from
`artifacts/modules.json["all_modules"]`. With that I can fix in one
pass instead of round-tripping API guesses.
