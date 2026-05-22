# Heretic EXL3 Integration Handoff

**Status as of latest session:** smoke test passes strictly on Qwen 3.5-2B
hybrid. First full optimization run got past load / batch-sizing / initial
refusal counting and then hit a residual-capture failure on hybrid blocks,
which has now been fixed by wrapping each block's `forward` directly
(no longer relying on `export_state`). Awaits a fresh full-loop run to
confirm the rest of the pipeline.

---

## What works (confirmed on hardware)

Smoke-tested against `/mnt/two/Weights/Qwen_Qwen3.5-2B/4` (an EXL3 quant
of a Qwen 3.5 hybrid 2B model with both `linear_attn` and `self_attn`
layers):

- **Model load** through `Config.from_directory → Model.from_config →
  Cache(model, max_num_tokens=...) → model.load(progressbar=True)`
- **Module discovery**: 24 layers × 2 components = 48 targets discovered
  (`attn.o_proj` covers both `self_attn.o_proj` and hybrid
  `linear_attn.out_proj`; `mlp.down_proj` is standard).
- **LoRA injection** via in-place mutation of
  `target.lora_a_tensors[sentinel]` / `lora_b_tensors[sentinel]`
  (bypassing exllamav3's directory-based `LoRA` class). Adapter
  application changes generated output.
- **Reset** via zeroing the same slots: produces **byte-identical**
  baseline restore.
- **Deterministic generation** via explicit `GreedySampler()` + `seed=0`
  on every `Generator.generate()` call. Matches the HF backend's
  `do_sample=False` (see `model.py:603-608`).

The smoke script output for a passing run:

```json
{
  "status": "ok",
  "adapter_changed_output": true,
  "reset_matches_baseline": true
}
```

## What is in flight (real-loop test running now)

```
PYTHONPATH=src python -m heretic --backend exl3 --row-normalization none \
    /mnt/two/Weights/Qwen_Qwen3.5-2B/4
```

Will exercise — for the first time end-to-end against an EXL3 model:

- batch-size autodetection (`get_responses`)
- per-layer residual capture (`get_residuals_batched`)
- logprob computation (`get_logprobs_batched`)
- the TPE search loop calling `abliterate` and `reset_model` hundreds
  of times
- save / upload UI (adapter-sidecar path)

User reports it appears to be working. Update this doc with the
post-run verdict on the next session.

---

## Architecture

```
heretic.Settings.backend = "hf" | "exl3"            (new field)
    |
    +-- "hf"   -> heretic.model.Model                (unchanged HF/PEFT path)
    +-- "exl3" -> heretic.exl3_model.Exl3Model       (new)
```

`Exl3Model` duck-types every method `main.py` calls on `Model`. The
old `src/heretic/backends/` abstraction (`ModelBackend`, `HfBnbBackend`,
the buggy first-pass `Exl3Backend`) was deleted — the surface
`main.py` actually needed didn't match what those base classes
exposed, and the duck-typed `Exl3Model` is simpler and easier to keep
correct.

### Branching points in main.py

1. **Model construction** (around `model = ...`): picks `Model` or
   `Exl3Model` from `settings.backend`.
2. **Save-to-folder**: EXL3 forces the adapter-sidecar path and calls
   `model.save_adapter(dir)`. Merging into quantized storage isn't
   supported upstream, so the merge prompt is skipped.
3. **Upload-to-hub**: EXL3 saves to a temp dir then uploads via
   `huggingface_hub.upload_folder`. Same constraint.

---

## Key design decisions

### Module discovery

Walks the loaded model via its `__iter__` (which recurses through
`Module.modules`), filters `.key` by regex:

```
^model(?:\.language_model)?\.layers\.(\d+)\..*?\.(o_proj|out_proj|down_proj)(?:\.slice\.\d+)?$
```

The optional `language_model` segment covers multimodal-wrapped LMs.
Both `o_proj` and `out_proj` route to `attn.o_proj` because hybrid
linear-attention layers (Qwen 3.5 GatedDeltaNet) feed the same
residual stream as standard self-attention. MoE expert paths
(`...block_sparse_moe.experts.{i}.down_proj`) and sliced MLPs
(`...mlp.down_proj.slice.{i}`) collapse under `mlp.down_proj`.

Blocks for residual capture are found by a separate key regex
(`^model(?:\.language_model)?\.layers\.\d+$`) rather than
`isinstance(TransformerBlock)`, so custom block classes are picked up.

### LoRA injection bypasses exllamav3.LoRA

The upstream `LoRA` class only loads from a PEFT directory on disk —
not workable for Heretic's hot loop. Instead we use the underlying
mechanism: `Linear.lora_a_tensors` / `lora_b_tensors` are plain dicts
keyed by any hashable; `Linear.forward()` already iterates them and
applies `output += x @ A @ B` for each pair.

We allocate one sentinel `object()` per `Exl3Model` instance, attach
`(in_features, 1)` and `(1, out_features)` fp16 slots on every target
Linear, and mutate them in place via `.copy_()` between trials.
`reset_model()` zeroes them.

### EXL3-aware abliteration math

HF stores weights as `(out, in)`. exllamav3 stores `W_exl3` as
`(in, out)` (transposed) and applies LoRA additively as
`output += x @ A @ B`. The rank-1 abliteration update becomes:

```
A_exl3 (in, 1) = W_exl3 @ v       (== W_hf.T @ v)
B_exl3 (1, out) = -λ * v.T
```

`W_exl3` is recovered via `LinearEXL3.inner.get_weight_tensor()` (fp16
dequant). Computed on the unpadded slice in fp32, then zero-padded to
the Linear's padded `(in_features, out_features)` and cast to fp16
before copying into the slot.

`row_normalization=pre` and `row_normalization=full` are **not yet
supported** on the EXL3 backend and raise `NotImplementedError`. The
default is `full`, so EXL3 runs require `--row-normalization none`.
Adding them is a known follow-up — see "Open work" below.

### Per-layer residual capture

`block.export_state = True` on every TransformerBlock captures
post-block hidden state into `params["export_states"]`. To also get
the embedding-output state (HF's `hidden_states[0]`), `Exl3Model`
monkey-patches the first block's `.forward` with a wrapper that
prepends `x.half().clone()` to the list before calling the original.
Stacked across the captured list and sliced at the last token
position to produce `(prompt, layer + 1, hidden)`, matching HF
semantics.

### Cache sizing

`Cache.max_num_tokens` is honored as set by `settings.exl3_max_num_tokens`
(default 8192), with a 2048 floor to satisfy exllamav3's autosplit
load probe (`bsz * cache_max_seq_len <= max_num_tokens` with
`max_chunk_size ≈ 2048`). Rounded up to the 256-token page size.

We do NOT auto-bump to `config.max_seq_len`. For Heretic's batched
residual/logprob passes (typically 32 prompts × ~256 tokens = 8k
tokens), 8192 is plenty. Users with unusual batch shapes can raise
`exl3_max_num_tokens`.

### Deterministic generation

`Generator.generate(sampler=None)` uses the upstream's default
sampler, which involves RNG and produces drift between repeated calls
even under identical weights. The HF backend hardcodes
`do_sample=False` (greedy) with an explicit "deterministic outputs"
comment (`model.py:603-608`); the EXL3 path mirrors that by passing
`GreedySampler()` + `seed=0` to every `generate()` call. This is what
makes the reset-matches-baseline assertion possible.

### PEFT-compatible adapter save

`save_adapter()` writes `adapter_config.json` + `adapter_model.safetensors`
with PEFT-shape tensors:

- `lora_A.weight`: `(rank, in_features)` (transposed from exllamav3's `(in, rank)`)
- `lora_B.weight`: `(out_features, rank)` (transposed from exllamav3's `(rank, out)`)

Keys are `base_model.model.<full_module_key>.lora_A.weight`/`.lora_B.weight`.
Loadable by both `peft.PeftModel.from_pretrained` and exllamav3's own
`LoRA.from_directory`. Tokenizer files (`tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`) are copied from
the base model dir into the adapter dir so the output is
self-sufficient.

---

## Open work

### `row_normalization=pre|full` for EXL3

`abliterate()` raises `NotImplementedError` on these modes. PRE
requires dequantizing every target weight per trial; FULL additionally
requires a higher-rank LoRA slot than the rank-1 we pre-allocate. Both
are doable but cost VRAM and trial-time, and aren't needed for a
first working integration. Users get `none` by passing
`--row-normalization none` (or setting it in `config.toml`).

### Save UI quality-of-life

The save prompt still shows the merge/adapter strategy chooser on
the HF path. For EXL3 we skip straight to adapter sidecar with a
print message but don't surface that to the user as an explicit
"EXL3 — saving adapter sidecar" upfront. Minor.

### Residual capture on non-`TransformerBlock` block classes — RESOLVED

The full optimization run on Qwen 3.5-2B hit exactly this:

```
RuntimeError: Expected 25 captured residuals, got 1.
```

The hybrid block class (Qwen 3.5 GatedDeltaNet) doesn't honor
`export_state=True` — setting it had no effect, so only the
pre-block-0 wrapper produced a capture (hence `got 1`). Fixed by
wrapping every decoder block's `forward` directly instead of relying
on the upstream attribute. The wrapper appends the block's output to
`params["export_states"]`; block 0's wrapper additionally captures
the input (== post-embedding state). For belt-and-braces, where
`export_state` exists we explicitly set it to `False` to avoid the
chance of double-capture on architectures whose default is True.

### Cache hygiene across batches

The EXL3 forward path constructs a fresh `params = {}` per call but
doesn't touch the cache. Assumption: for non-autoregressive forwards
(prefill-only, fresh batch each call) cache state from prior batches
doesn't affect correctness, only memory residency. If the full run
shows residual drift across batches, we'd need to reset cache
positions between calls.

---

## Files of interest

| Path | Purpose |
|---|---|
| `src/heretic/exl3_model.py` | `Exl3Model` — the entire EXL3 backend in one file (~700 lines) |
| `src/heretic/config.py` | `Backend` enum + `backend` / `exl3_max_num_tokens` settings |
| `src/heretic/model.py` | HF backend, unchanged except removed `HfBnbBackend` reference |
| `src/heretic/main.py` | Backend branching at construction, save, and upload sites |
| `scripts/inspect_exl3_modules.py` | Module-tree dump (uses `inspect_only=True`, no VRAM) |
| `scripts/smoke_test_exl3_lora.py` | Baseline / adapt / reset round-trip; synthesizes a deterministic rank-1 LoRA |
| `pyproject.toml` | `[exl3]` optional extra (`exllamav3==0.0.33`, `safetensors>=0.4`) |

The directory `src/heretic/backends/` (`__init__.py`, `base.py`,
`hf_bnb.py`, `exl3.py` — the original buggy scaffolding) was deleted.

---

## Today's debugging trail (chronological)

For future-context: these were the bugs found and fixed today, all
discovered by running the scripts against a real model.

1. **`argparse` collided with `Settings`**. `Settings.__init__`
   triggers `CliSettingsSource(cli_parse_args=True)`, which re-reads
   `sys.argv` and rejected the scripts' own positional/`--out` flags.
   Fixed by saving/restoring `sys.argv` around the `Settings(...)`
   call in both scripts.

2. **`AttributeError: module 'exllamav3' has no attribute 'Config'`**.
   The user's editable install of exllamav3 0.0.34 doesn't re-export
   `Config` at top level (the master branch's `__init__.py` does, but
   their local rearrangement doesn't). Fixed by resolving each class
   from its actual submodule path (`exllamav3.model.config.Config`,
   `exllamav3.model.model.Model`, `exllamav3.cache.cache.Cache`,
   `exllamav3.tokenizer.tokenizer.Tokenizer`,
   `exllamav3.generator.generator.Generator`) with fallbacks for
   alternate layouts.

3. **`Cache too small for batch shape`**. The inspect script set
   `max_num_tokens=512`, below the autosplit probe's
   `max_chunk_size` (~2048). Two fixes:
   - Inspect script now uses `inspect_only=True` which skips
     `model.load()` and `Cache(...)` entirely. Module-tree
     enumeration only reads `.key` strings, which are populated at
     `Model.from_config()` time. Inspection is now fast and uses no
     VRAM.
   - Runtime path enforces a 2048-token floor on cache size (the
     default of 8192 already exceeds this).

4. **No modules matched (regex was too narrow)**. Qwen 3.5-2B's keys
   are `model.language_model.layers.X.linear_attn.out_proj` and
   `model.language_model.layers.X.mlp.down_proj` — both the
   `language_model` prefix (multimodal wrap) and the `out_proj`
   suffix (hybrid linear-attention) were unmatched. Widened the
   regex and routed `out_proj`/`o_proj` both to `attn.o_proj`.

5. **`reset_matches_baseline: false` even with weights zeroed**.
   Generator's default sampler involves RNG; three consecutive
   `.generate()` calls under identical weights produced subtly
   different word choices. Fixed by passing `GreedySampler()` +
   `seed=0` explicitly, mirroring HF's hardcoded `do_sample=False`.

The commit log on the `claude/cool-ptolemy-BeiwV` branch tells this
story commit-by-commit if needed.

---

## How to resume

If you're picking this up fresh:

1. Read this file.
2. Read `src/heretic/exl3_model.py` top to bottom. It's the entire
   backend in one file with extensive comments at each non-obvious
   step.
3. Check the latest smoke output: `cat artifacts/smoke.json`. If
   `status == "ok"`, the integration is at least at the level of
   "loads, injects, resets". The full optimization run on top of
   that is what closes the loop.
4. If the full run from the last session finished, the next thing to
   try is the save path. If it failed mid-loop, the failure trace
   should point at one of the four "open work" items above.
