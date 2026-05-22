# Heretic EXL3 Integration Handoff (May 22, 2026)

## Executive Summary

Today’s work established a backend abstraction layer for Heretic, added early EXL3 backend integration scaffolding, and iteratively fixed EXL3 loader issues discovered during user-side runtime testing.

Most importantly, the **latest fix replaced incorrect EXL3 config construction** with the documented `from_directory` / `from_config` flow to address:

- `TypeError: Config.__init__() missing 1 required positional argument: 'model_classes'`

This is now reflected in `src/heretic/backends/exl3.py`.

---

## What We Implemented

### 1) Backend abstraction surface

Added `ModelBackend` abstract interface with methods for:

- model/tokenizer loading
- generation/logits
- module discovery
- effective weight access
- adapter lifecycle (apply/unload/reset)

File:
- `src/heretic/backends/base.py`

### 2) Backend package exports

Added backend package exports for:

- `ModelBackend`
- `HfBnbBackend`
- `Exl3Backend`

File:
- `src/heretic/backends/__init__.py`

### 3) HF/BNB facade backend

Added `HfBnbBackend` as a migration bridge delegating to existing `Model` behavior for:

- generate
- forward logits
- module listing
- target module listing
- reset adapters (via reset_model)

Unimplemented methods remain explicit `NotImplementedError` placeholders pending lifecycle refactors.

File:
- `src/heretic/backends/hf_bnb.py`

### 4) Wired backend into existing Model wrapper

`Model` now exposes a backend facade:

- `self.backend = HfBnbBackend(self)`

applied in both initial load and reload paths.

File:
- `src/heretic/model.py`

### 5) EXL3 scripts for gating and inspection

Added:

- `scripts/smoke_test_exl3_lora.py`
  - baseline → adapter → reset flow
  - structured JSON outputs
  - strict mode and failure categories

- `scripts/inspect_exl3_modules.py`
  - module dump
  - target-module extraction
  - simple Heretic role mapping

These scripts are intended as integration-gate tools before deep optimizer porting.

### 6) EXL3 backend prototype + iterative fixes

Implemented prototype `Exl3Backend` with:

- EXL3 model load
- tokenizer load
- raw text generation helper
- module listing
- target module filtering
- adapter apply/unload/reset wrappers

Then iteratively fixed runtime assumptions based on user-reported failures.

File:
- `src/heretic/backends/exl3.py`

---

## Runtime Failures Encountered and How We Responded

### Failure A

User-side error:

- `AttributeError: module 'exllamav3' has no attribute 'ExLlamaV3Config'`

Response:

- replaced hard top-level-only lookup with module-aware type resolution.

### Failure B

User feedback called out speculative API guessing.

Response:

- narrowed resolution to documented public names (`Config`, `Model`, `Generator`) with small legacy alias fallback.

### Failure C (latest and most important)

User-side error:

- `TypeError: Config.__init__() missing 1 required positional argument: 'model_classes'`

Root cause:

- direct constructor call (`Config(model_path)`) was incorrect for current ExLlamaV3 API.

Latest fix:

- switched to documented construction path:
  - `Config.from_directory(model_path)`
  - `Model.from_config(config)`
  - `Cache(model, max_num_tokens=...)`
  - `Tokenizer.from_config(config)`
  - `Generator(model=..., cache=..., tokenizer=...)`

Also expanded required type resolution to include `Cache` and `Tokenizer`.

---

## Current State of Testing

### Tests/checks executed in this environment

Performed repeatedly during edits:

- `python -m compileall -q src/heretic scripts`

This validates syntax/import structure at compile time for changed modules/scripts.

### What has **not** been fully validated end-to-end here

Due to environment differences and model/runtime availability:

- no end-to-end EXL3 load against your real model path in this container
- no full smoke run with actual adapter on GPU runtime
- no integration tests for adapter lifecycle against exllamav3 0.0.34 runtime

### User-side status focus

Your reported sequence confirms the debugging path and the latest blocker:

1. missing top-level `ExLlamaV3Config`
2. then class resolution issues
3. then incorrect `Config(...)` constructor usage

The latest patch directly targets step 3 with documented API flow.

---

## Files Added/Changed Today (Functional Scope)

- `src/heretic/backends/base.py`
- `src/heretic/backends/__init__.py`
- `src/heretic/backends/hf_bnb.py`
- `src/heretic/backends/exl3.py`
- `src/heretic/model.py`
- `scripts/smoke_test_exl3_lora.py`
- `scripts/inspect_exl3_modules.py`

---

## Recommended Immediate Next Validation Steps (On Your Runtime)

Run these in your EXL3-capable environment:

1. Module inspection:

```bash
PYTHONPATH=src python scripts/inspect_exl3_modules.py /mnt/two/Weights/Qwen_Qwen3.5-2B/ --out artifacts/exl3_module_map.json
```

2. Smoke test (strict):

```bash
PYTHONPATH=src python scripts/smoke_test_exl3_lora.py /mnt/two/Weights/Qwen_Qwen3.5-2B/ --adapter-path <adapter_path> --strict --out artifacts/exl3_smoke.json
```

If failures remain, capture exact traceback and especially:

- constructor signatures shown in error
- resolved class names/symbols from runtime
- whether `model.load_lora` / `model.reset_loras` names exist on the resolved model class

---

## Known Remaining Gaps

- `Exl3Backend.generate` HF-compatible batched return path is not implemented.
- `forward_logits` not implemented.
- `get_effective_weight` not implemented (explicitly deferred as high-risk EXL3 quantization path).
- adapter lifecycle API may still differ across exllamav3 versions and may require one more compatibility shim after live test.

---

## Suggested Next Engineering Milestone

After confirming the latest loader fix works in your runtime:

1. lock a single known-good architecture mapping (Qwen or Mistral)
2. validate LoRA apply/unload/reset behavior with strict smoke script
3. add a PEFT adapter writer bridge (`adapter_config.json` + `adapter_model.safetensors`)
4. only then connect Heretic optimization loop through backend calls

This preserves the stop/go sequencing and avoids large refactors before runtime proof.
