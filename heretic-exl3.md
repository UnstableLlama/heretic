# Heretic EXL3 integration

This fork tracks Heretic **2.0.0.dev0**, upstream commit
`a09b4eae079cfcccd3e1e5ae75e46f2b3b378b2a` (2026-10-03 sync).
The EXL3 integration was source-audited against ExLlamaV3 **1.5.4**,
commit `16a49792a3c93d8432d72e6c4bce800841566577`, and the local 1.5.3-based
fork. The optional dependency accepts `>=1.5.3,<1.6`; the lockfile resolves
1.5.3, the version available on PyPI at sync time.

## Installation and selection

Python 3.10.11 or newer is required. Install this checkout with the `exl3`
extra into a CUDA-enabled environment, or keep your existing editable EXL3
installation and install Heretic without replacing it.

```sh
pip install -e '.[exl3]'
heretic --model /path/to/exl3-model --quantization exl3
```

Upstream's `uv` configuration defaults to **CPU PyTorch wheels** for CI.
For GPU use, explicitly override its `pytorch` index with the CUDA wheel index
appropriate for your installed EXL3 build. Do not run an unqualified `uv sync`
in an existing CUDA environment expecting it to preserve that environment.
The sync itself did not install or upgrade anything in the existing `.venv`
or the neighboring EXL3 checkout.

## Modifier plugins and old configs

EXL3 now participates in upstream's modifier lifecycle: capture activations,
allocate adapters at the plugin's requested rank, reset adapters, and modify.
Both built-in modifiers dispatch to the EXL3 implementation:

- `heretic.modifiers.ara.ARA`: ARA with LoRA; preserves the bounded routed-expert
  capture and the factored objective when row preservation is disabled.
- `heretic.modifiers.abliteration.Abliteration`: directional ablation with
  `row_normalization = "none"`. PRE/FULL directional normalization remains
  unsupported on EXL3 and is rejected before modifier capture.

A new ARA config can use:

```toml
model = "/path/to/exl3-model"
quantization = "exl3"
modifiers = [{ plugin = "heretic.modifiers.ara.ARA" }]

[modifier.ARA]
lora_rank = 128
preserve_row_magnitudes = false
steer_bad_behavior_weight_min = 0.0001
steer_bad_behavior_weight_max = 0.001
invert_target = false
lora_regularization = 0.0

[modifier.ARA.good_prompts]
dataset = "good.txt"

[modifier.ARA.bad_prompts]
dataset = "bad.txt"
```

Legacy config files and CLI flags are accepted and normalized into the new
plugin tables. Explicit plugin settings take precedence. Saved settings use
the canonical plugin schema.

| Legacy setting | Plugin setting |
| --- | --- |
| `use_ara` / `use_ara_lora` | Select ARA; either true selects the LoRA-based 2.x ARA |
| `ara_lora_rank` | `modifier.ARA.lora_rank` |
| `ara_lora_regularization` | `modifier.ARA.lora_regularization` |
| `invert_target` | `modifier.ARA.invert_target` |
| `steer_bad_behavior_weight_min/max` | Same names under `modifier.ARA` |
| ARA `row_normalization` | `preserve_row_magnitudes` (true only for `full`) |
| `[good_prompts]`, `[bad_prompts]` | Corresponding tables under the selected modifier |
| Directional normalization/orthogonalization/winsorization | Same names under `modifier.Abliteration` |

Legacy ARA configs retain this fork's rank-128 and steering-maximum-0.001
fallbacks. New plugin configs use upstream defaults unless explicitly overridden.
`chat_template_kwargs` is preserved and forwarded on both EXL3 and HF paths.
Old optimization checkpoints use a different parameter schema; begin a new study
for 2.x rather than resuming a 1.x checkpoint.

## Loading, generation, and export

Layer splitting, `exl3_gpu_split`, `exl3_reserve_per_device`,
`exl3_load_max_chunk_size`, and sliced quantized-weight reconstruction remain.
Tensor parallelism remains excluded because it bypasses the capture hooks.
LoRA output widths now account for EXL3's trimmed padded projections. Capture
hooks are restored on failure. Resetting adapters invalidates the generator's
prefix cache, and responses come directly from the generator without a second
prompt/completion tokenization pass.

Save either a PEFT adapter or merge into an original HF base model. Set
`exl3_base_model` when the quantized model metadata cannot identify that base.
Adapters retain their active rank and export unpadded tensors in PEFT orientation.
Merging loads the HF base on CPU; it does not rewrite quantized EXL3 weights.
The built-in BenchmarkScore and interactive benchmark use Hugging Face's
lm-eval backend: benchmark a merged HF export instead of an EXL3 model.

## Validation and deferred GPU checks

CPU validation uses an isolated environment with CPU-only PyTorch:

```sh
CUDA_VISIBLE_DEVICES='' UV_PROJECT_ENVIRONMENT=/tmp/heretic-sync-venv uv sync --locked --dev
CUDA_VISIBLE_DEVICES='' /tmp/heretic-sync-venv/bin/python -m unittest discover -s tests -p 'test_*.py'
/tmp/heretic-sync-venv/bin/ruff check --extend-select I .
/tmp/heretic-sync-venv/bin/ruff format --check .
/tmp/heretic-sync-venv/bin/ty check --error-on-warning .
uv build
```

Regression coverage includes config migration, modifier dispatch, CPU ARA
optimization, trimmed padding, adapter export, residual options, response
handling, cache invalidation, and hook cleanup. All repository config files
were validated, CLI help was checked, and wheel/source packages were built.

**No GPU workloads were run.** Real EXL3 loading, CUDA reconstruction kernels,
multi-GPU placement, dense/MoE generation and numerical comparison with saved
adapters still require an explicitly authorized GPU smoke test. CPU tests and
source inspection do not establish GPU numerical correctness. The upstream
model-download/hash integration suite was not run during this sync.
