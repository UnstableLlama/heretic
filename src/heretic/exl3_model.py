# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""ExLlamaV3 (EXL3) backend for Heretic.

This module provides ``Exl3Model``, a drop-in replacement for ``model.Model``
that duck-types the surface ``main.py`` uses. It targets ExLlamaV3 0.0.34.

Design notes (see HANDOFF_EXL3.md):

* Module discovery walks ``model`` via its ``__iter__`` and filters by
  ``.key`` regex. Keys mirror HuggingFace safetensors naming
  (e.g. ``model.layers.0.self_attn.o_proj``,
  ``model.layers.0.mlp.down_proj``,
  ``model.layers.0.block_sparse_moe.experts.3.down_proj``).

* LoRA injection bypasses exllamav3's ``LoRA`` class (which only loads
  from a PEFT directory). Each target ``Linear`` has plain
  ``lora_a_tensors`` / ``lora_b_tensors`` dicts keyed by any hashable; the
  forward path applies ``output += input @ A @ B`` for each pair. We use
  a sentinel object as the key, pre-allocate fp16 A/B tensors of shape
  ``(in_features, 1)`` / ``(1, out_features)``, and mutate them in place
  between trials.

* Weights live as EXL3 trellis blobs. ``LinearEXL3.get_weight_tensor()``
  dequantizes to a fp16 tensor of shape ``(in_features, out_features)``
  — already transposed vs HF's ``(out, in)`` convention. We compute LoRA
  updates on the unpadded slice, then zero-pad to the padded shape
  before copying in.

* Per-layer hidden states (residuals): we wrap every decoder block's
  ``forward`` to append its output to ``params["export_states"]``. The
  first block also captures its input so the list starts with
  post-embedding state, matching HF's ``hidden_states[0]``. We don't
  rely on exllamav3's ``export_state`` attribute because not all block
  classes honor it (Qwen 3.5 hybrid blocks, for example, don't append
  on ``export_state=True``).
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import re
import shutil
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch
import torch.linalg as LA
import torch.nn.functional as F
from torch import Tensor
from torch.optim import LBFGS

from .config import RowNormalization, Settings
from .model import ARAParameters, AbliterationParameters, ModuleIO
from .system import empty_cache
from .utils import Prompt, batchify, mean_distances_to_knn, print


# Match keys like:
#   model.layers.<N>.self_attn.o_proj                     (standard)
#   model.layers.<N>.mlp.down_proj                        (standard)
#   model.layers.<N>.mlp.down_proj.slice.<M>              (sliced MLP)
#   model.layers.<N>.block_sparse_moe.experts.<I>.down_proj  (MoE)
#   model.language_model.layers.<N>.linear_attn.out_proj  (Qwen3.5 hybrid: linear attn + multimodal wrap)
#   model.language_model.layers.<N>.mlp.down_proj         (multimodal wrap)
# Accept any sub-path between the layer block and the leaf suffix. The
# optional ``language_model`` segment covers multimodal models that nest
# the LM under a vision/audio wrapper. The leaf suffix matches both
# ``o_proj`` (standard) and ``out_proj`` (hybrid linear-attention).
_MODULE_KEY_REGEX = re.compile(
    r"^model(?:\.language_model)?\.layers\.(\d+)\..*?\.(o_proj|out_proj|down_proj)(?:\.slice\.\d+)?$"
)

# Block-level keys, used to find decoder layers without depending on
# isinstance(TransformerBlock) (some architectures use custom block
# classes).
_BLOCK_KEY_REGEX = re.compile(
    r"^model(?:\.language_model)?\.layers\.(\d+)$"
)


class _Exl3Tokenizer:
    """Thin wrapper that gives an exllamav3 Tokenizer the small slice of the
    HF tokenizer surface that Heretic's ``main.py`` calls on
    ``model.tokenizer`` (``.encode(text)`` for response-length scoring,
    ``.apply_chat_template`` for prompt rendering, and ``.save_pretrained``
    /``.push_to_hub`` for the save path).

    We expect the user's EXL3 model directory to also contain the original
    HF tokenizer files (``tokenizer.json`` / ``tokenizer_config.json``) —
    every EXL3 conversion script we know of copies them. We load that
    HF tokenizer for everything except token-level ops the exllamav3
    Tokenizer handles natively.
    """

    def __init__(self, exl3_tokenizer: Any, model_path: str):
        self.exl3 = exl3_tokenizer
        self._model_path = model_path
        self._hf = None
        # Lazy: only load the HF tokenizer if something asks for it.

    def _ensure_hf(self) -> Any:
        if self._hf is None:
            from transformers import AutoTokenizer

            self._hf = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=False,
            )
            if self._hf.pad_token is None:
                self._hf.pad_token = self._hf.eos_token
            self._hf.padding_side = "left"
        return self._hf

    @property
    def pad_token_id(self) -> int | None:
        return self._ensure_hf().pad_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._ensure_hf().eos_token_id

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return self._ensure_hf().encode(text, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._ensure_hf()(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self._ensure_hf().apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._ensure_hf().batch_decode(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> Any:
        return self._ensure_hf().save_pretrained(save_directory, **kwargs)

    def push_to_hub(self, *args: Any, **kwargs: Any) -> Any:
        return self._ensure_hf().push_to_hub(*args, **kwargs)


class Exl3Model:
    """Heretic model wrapper backed by ExLlamaV3.

    Duck-types ``model.Model`` for the methods ``main.py`` calls.
    """

    settings: Settings
    needs_reload: bool
    tokenizer: _Exl3Tokenizer

    def __init__(self, settings: Settings, *, inspect_only: bool = False):
        """Load an EXL3 model.

        ``inspect_only=True`` skips weight loading, cache allocation and
        tokenizer instantiation. Use it for module-structure inspection
        when you don't need to run a forward pass — the module tree
        (and ``.key`` strings) is built at ``Model.from_config()`` time,
        so we can discover targets without paying the load cost.
        """
        self.settings = settings
        self.needs_reload = False
        self.revision_kwargs: dict[str, Any] = {}
        self.trusted_models: dict[str, bool | None] = {settings.model: False}
        self._inspect_only = inspect_only

        # Import lazily so the optional dep doesn't crash users on the HF path.
        # We resolve each class from its submodule directly so the integration
        # doesn't depend on which symbols __init__.py happens to re-export
        # (this varies between PyPI releases and the master branch).
        self._exl_api = self._resolve_exllamav3_api()

        print()
        print(
            f"Loading EXL3 model [bold]{settings.model}[/]"
            + (" (inspect-only)" if inspect_only else "")
            + "..."
        )

        model_path = str(Path(settings.model).expanduser())
        self.config = self._exl_api["Config"].from_directory(model_path)
        self.model = self._exl_api["Model"].from_config(self.config)

        if inspect_only:
            # Module tree is already built; no cache, no weight load, no
            # tokenizer. discover_modules walks .key strings — doesn't
            # need weights. Residual hooks aren't installed because no
            # forward pass will run.
            self.cache = None
            self.tokenizer = None  # type: ignore[assignment]
        else:
            # Cache sizing: the user setting is the working bound on
            #   batch_size * seq_len  during any forward pass.
            # For Heretic this is dominated by residual/logprob batches
            # (typically 32 prompts * ~256 tokens = 8k tokens), not the
            # model's declared max_seq_len. We just honour the setting
            # rounded up to the 256-token page size; users who need
            # larger batches or longer sequences can raise it via
            # exl3_max_num_tokens.
            target = max(int(settings.exl3_max_num_tokens), 2048)
            max_num_tokens = ((target + 255) // 256) * 256
            self.cache = self._exl_api["Cache"](
                self.model, max_num_tokens=max_num_tokens
            )
            self.model.load(**self._build_load_kwargs())
            self._report_loaded_devices()

            exl3_tok = self._exl_api["Tokenizer"].from_config(self.config)
            self.tokenizer = _Exl3Tokenizer(exl3_tok, model_path)

        # 2. Discover target modules, group by layer, allocate LoRA slots.
        #    Module-tree discovery only reads .key strings, so it works even
        #    without weights loaded. LoRA slot allocation needs module.device,
        #    which is only valid after load — so we skip it in inspect_only mode.
        self._lora_key = object()  # any hashable; used as the dict key
        self._layer_modules: list[dict[str, list[Any]]] = []
        self._discover_modules(allocate_lora_slots=not inspect_only)

        # 3. Residual hooks and generator only make sense when weights are loaded.
        if not inspect_only:
            self._install_residual_hooks()
            self._generator = None
        else:
            self._blocks = []
            self._num_layers = 0
            self._generator = None
            return

        print(f"* Transformer model with [bold]{len(self._layer_modules)}[/] layers")
        all_components: dict[str, int] = {}
        for per_layer in self._layer_modules:
            for component, modules in per_layer.items():
                all_components[component] = all_components.get(component, 0) + len(modules)
        print("* Abliterable components:")
        for component, count in all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _build_load_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for ``Model.load()`` controlling device
        placement across GPUs.

        Multi-GPU in exllamav3 is controlled by parameters to ``load()``
        (which forwards to ``load_gen()``), not by ``Config`` attributes:

        * ``tensor_p=True``         — tensor parallelism; every GPU is active
                                      on every forward pass.
        * ``use_per_device=[...]``  — explicit GB budget per device (layer split).
        * ``reserve_per_device``    — GB to reserve per device; lets exllamav3
                                      auto-split layers across all visible GPUs.

        We introspect the installed ``load_gen`` signature and only pass
        arguments it actually accepts, so this stays compatible across
        exllamav3 versions (the parameter set has changed over time).
        """
        kwargs: dict[str, Any] = {"progressbar": True}

        try:
            accepted = set(
                inspect.signature(self.model.load_gen).parameters  # ty:ignore[possibly-missing-attribute]
            )
        except (AttributeError, ValueError, TypeError):
            accepted = set()

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if gpu_count > 1:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            print(f"* CUDA devices visible to Heretic: [bold]{gpu_count}[/]")
            print(
                "* CUDA device list: "
                + ", ".join(f"[{i}] {name}" for i, name in enumerate(gpu_names))
            )

        if self.settings.exl3_tensor_parallel and gpu_count > 1:
            if "tensor_p" in accepted:
                kwargs["tensor_p"] = True
                print(
                    f"* EXL3 tensor parallelism enabled across [bold]{gpu_count}[/] GPUs"
                )
            else:
                print(
                    "[yellow]* WARNING:[/] installed exllamav3 has no 'tensor_p' "
                    "parameter; cannot enable tensor parallelism."
                )
        elif self.settings.exl3_gpu_split is not None:
            if "use_per_device" in accepted:
                kwargs["use_per_device"] = self.settings.exl3_gpu_split
                print(
                    "* EXL3 explicit GPU split (GB/device): "
                    f"[bold]{self.settings.exl3_gpu_split}[/]"
                )
            else:
                print(
                    "[yellow]* WARNING:[/] installed exllamav3 has no "
                    "'use_per_device' parameter; ignoring exl3_gpu_split."
                )
        elif gpu_count > 1 and "reserve_per_device" in accepted:
            # Let exllamav3 spread layers across all visible GPUs. Note that a
            # model small enough to fit on one GPU may still land entirely on
            # the first device; use exl3_tensor_parallel or exl3_gpu_split to
            # force both GPUs into active use.
            kwargs["reserve_per_device"] = self.settings.exl3_reserve_per_device
            print(
                f"* EXL3 auto-split across [bold]{gpu_count}[/] GPUs "
                f"(reserving {self.settings.exl3_reserve_per_device} GB/device)"
            )

        # Drop anything the installed version doesn't understand (keep
        # progressbar, which every known version accepts).
        if accepted:
            kwargs = {
                k: v for k, v in kwargs.items() if k == "progressbar" or k in accepted
            }

        return kwargs

    def _report_loaded_devices(self) -> None:
        """Log where exllamav3 actually placed modules after load."""
        device_counts: dict[str, int] = {}
        for module in self.model:
            device = getattr(module, "device", None)
            if device is None:
                continue
            key = str(device)
            device_counts[key] = device_counts.get(key, 0) + 1

        if not device_counts:
            return

        device_summary = ", ".join(
            f"{device}: {count}" for device, count in sorted(device_counts.items())
        )
        print(f"* EXL3 module placement: {device_summary}")

    @staticmethod
    def _import_exllamav3() -> ModuleType:
        try:
            return importlib.import_module("exllamav3")
        except ImportError as error:
            raise ImportError(
                "ExLlamaV3 backend selected but 'exllamav3' is not installed. "
                "Install with: pip install -U 'heretic-llm[exl3]'"
            ) from error

    @classmethod
    def _resolve_exllamav3_api(cls) -> dict[str, type]:
        """Resolve Config / Model / Cache / Tokenizer / Generator across
        exllamav3 versions. The top-level ``__init__.py`` re-exports vary
        between PyPI releases and master; the submodule paths are the
        stable surface.
        """
        cls._import_exllamav3()  # raise a friendly error if missing

        candidates = {
            "Config":    [("exllamav3.model.config", "Config"),
                          ("exllamav3.config",       "Config"),
                          ("exllamav3",              "Config")],
            "Model":     [("exllamav3.model.model",  "Model"),
                          ("exllamav3.model",        "Model"),
                          ("exllamav3",              "Model")],
            "Cache":     [("exllamav3.cache.cache",  "Cache"),
                          ("exllamav3.cache",        "Cache"),
                          ("exllamav3",              "Cache")],
            "Tokenizer": [("exllamav3.tokenizer.tokenizer", "Tokenizer"),
                          ("exllamav3.tokenizer",   "Tokenizer"),
                          ("exllamav3",             "Tokenizer")],
            "Generator": [("exllamav3.generator.generator", "Generator"),
                          ("exllamav3.generator",   "Generator"),
                          ("exllamav3",             "Generator")],
        }

        resolved: dict[str, type] = {}
        misses: dict[str, list[str]] = {}
        for name, attempts in candidates.items():
            for module_name, attr in attempts:
                with suppress(ImportError, AttributeError):
                    mod = importlib.import_module(module_name)
                    obj = getattr(mod, attr)
                    if isinstance(obj, type):
                        resolved[name] = obj
                        break
            if name not in resolved:
                misses[name] = [f"{m}.{a}" for m, a in attempts]

        if misses:
            raise RuntimeError(
                "Could not resolve required exllamav3 API symbols: "
                + ", ".join(f"{k} (tried {' | '.join(v)})" for k, v in misses.items())
                + ". Your exllamav3 install may have rearranged its public surface; "
                "report the package version and __init__.py contents."
            )
        return resolved

    def _discover_modules(self, *, allocate_lora_slots: bool = True) -> None:
        """Walk the loaded model, find o_proj / down_proj Linears, group
        by layer index, allocate a (lora_A, lora_B) pair on each one.

        ``allocate_lora_slots=False`` skips the per-target slot allocation,
        which is required for inspect-only loads where modules are not on
        a device yet.
        """
        # Import the concrete Linear class so we can identity-check.
        linear_mod = importlib.import_module("exllamav3.modules.linear")
        Linear = linear_mod.Linear

        by_layer: dict[int, dict[str, list[Any]]] = {}
        all_keys: list[str] = []

        for module in self.model:
            key = getattr(module, "key", None)
            if not isinstance(key, str):
                continue
            all_keys.append(key)
            m = _MODULE_KEY_REGEX.match(key)
            if m is None:
                continue
            if not isinstance(module, Linear):
                continue
            layer_idx = int(m.group(1))
            leaf = m.group(2)  # "o_proj" | "out_proj" | "down_proj"
            if leaf == "down_proj":
                component = "mlp.down_proj"
            else:
                # Both "o_proj" (standard attention) and "out_proj" (hybrid
                # linear attention, e.g. Qwen3.5 GatedDeltaNet) feed into
                # the same residual stream and should be ablated together.
                component = "attn.o_proj"
            by_layer.setdefault(layer_idx, {}).setdefault(component, []).append(module)

        self._all_module_keys = all_keys

        if not by_layer:
            # Don't crash here — the inspect script needs to dump module
            # keys even for arches we don't yet match. Surface a warning;
            # operations that need the layer list (abliterate, residuals)
            # will raise when called.
            preview = "\n  ".join(all_keys[:20])
            print(
                "[yellow]WARNING:[/] no abliterable modules matched on key regex "
                "^model\\.layers\\.\\d+\\..*(o_proj|down_proj).*$. "
                f"First {min(20, len(all_keys))} discovered keys:\n  {preview}"
            )
            self._layer_modules = []
            return

        # Materialize into a contiguous list indexed by layer number.
        max_layer = max(by_layer.keys())
        self._layer_modules = [by_layer.get(i, {}) for i in range(max_layer + 1)]

        # Pre-allocate LoRA tensors on every target Linear.
        # Shapes: A is (in_features, 1), B is (1, out_features). Both fp16.
        if allocate_lora_slots:
            for per_layer in self._layer_modules:
                for modules in per_layer.values():
                    for module in modules:
                        self._allocate_lora_slot(module)

    def _lora_rank(self) -> int:
        if self.settings.use_ara_lora:
            return self.settings.ara_lora_rank
        return 1

    def _allocate_lora_slot(self, module: Any) -> None:
        device = module.device
        rank = self._lora_rank()
        a = torch.zeros(
            (module.in_features, rank),
            dtype=torch.float16,
            device=device,
        )
        b = torch.zeros(
            (rank, module.out_features),
            dtype=torch.float16,
            device=device,
        )
        module.lora_a_tensors[self._lora_key] = a
        module.lora_b_tensors[self._lora_key] = b

    def _install_residual_hooks(self) -> None:
        """Find decoder block modules by key pattern and wrap each one's
        ``forward`` to append its output into ``params["export_states"]``.
        The first block also captures its input so the captured list is
        ``[embed_out, block0_out, block1_out, ...]``, matching HF's
        ``hidden_states`` shape.

        We identify blocks by ``.key`` matching ``model[.language_model].layers.N``
        rather than ``isinstance(TransformerBlock)`` so this works on
        custom block classes (hybrid, parallel decoder, etc.). We don't
        use exllamav3's ``export_state`` attribute because not all block
        classes honor it — Qwen 3.5 hybrid blocks, for one, don't append
        when it's set. Wrapping ``forward`` is uniform across block
        classes.
        """
        # Find blocks by key pattern. Tuple of (layer_idx, module) so we
        # can sort by layer index reliably.
        block_pairs: list[tuple[int, Any]] = []
        for module in self.model:
            key = getattr(module, "key", None)
            if not isinstance(key, str):
                continue
            m = _BLOCK_KEY_REGEX.match(key)
            if m is None:
                continue
            block_pairs.append((int(m.group(1)), module))

        if not block_pairs:
            raise RuntimeError(
                "No decoder block modules found. Residual capture needs at "
                "least one module whose .key matches "
                "^model(?:\\.language_model)?\\.layers\\.\\d+$."
            )

        block_pairs.sort(key=lambda t: t[0])
        blocks = [m for _, m in block_pairs]

        # Where the upstream attribute exists, ensure it stays off so it
        # doesn't double-capture alongside our wrapper.
        for block in blocks:
            if hasattr(block, "export_state"):
                block.export_state = False

        def _make_wrapper(orig: Any, capture_input: bool) -> Any:
            def wrapped(x, params, out_dtype=None):
                states = params.get("export_states")
                if states is None:
                    states = params["export_states"] = []
                if capture_input:
                    states.append(x.half().clone())
                out = orig(x, params, out_dtype=out_dtype)
                # Some block classes may return (hidden, extras); only
                # capture the hidden tensor.
                captured = out[0] if isinstance(out, tuple) else out
                states.append(captured.half().clone())
                return out
            return wrapped

        for idx, block in enumerate(blocks):
            block.forward = _make_wrapper(block.forward, capture_input=(idx == 0))

        self._blocks = blocks
        self._num_layers = len(blocks)

    # ------------------------------------------------------------------
    # Interface mirroring model.Model
    # ------------------------------------------------------------------

    def get_layers(self) -> list[Any]:
        """Return the per-layer block list (used only for len() in main.py)."""
        return self._blocks

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Any]]:
        return self._layer_modules[layer_index]

    def get_abliterable_components(self) -> list[str]:
        components: set[str] = set()
        for per_layer in self._layer_modules:
            components.update(per_layer.keys())
        return sorted(components)

    # ------------------------------------------------------------------
    # Abliteration: compute LoRA update from refusal direction and write
    # into pre-allocated A/B tensors in place.
    # ------------------------------------------------------------------

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None:
        if self.settings.row_normalization != RowNormalization.NONE:
            # Row normalization paths require either reading and overwriting
            # W (PRE) or a higher-rank SVD decomposition (FULL). The
            # pre-allocated rank-1 slot can't represent FULL, and reading
            # the EXL3 weight for every module on every trial is too
            # expensive to do silently. Surface the limitation explicitly.
            raise NotImplementedError(
                "EXL3 backend currently supports only row_normalization='none'. "
                f"Got '{self.settings.row_normalization.value}'. "
                "PRE/FULL would require dequantizing every target weight per "
                "trial and (for FULL) widening the rank-1 adapter — possible "
                "but not implemented in this pass."
            )

        if direction_index is None:
            refusal_direction = None
        else:
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        for layer_index in range(len(self._layer_modules)):
            for component, modules in self._layer_modules[layer_index].items():
                params = parameters[component]
                distance = cast(float, abs(layer_index - params.max_weight_position))
                if distance > params.min_weight_distance:
                    # Out of kernel support: zero the slot so resetting one
                    # layer's contribution doesn't carry over from a prior
                    # trial that did write to it.
                    for module in modules:
                        module.lora_a_tensors[self._lora_key].zero_()
                        module.lora_b_tensors[self._lora_key].zero_()
                    continue

                kernel_weight = params.max_weight + (
                    distance / params.min_weight_distance
                ) * (params.min_weight - params.max_weight)

                if refusal_direction is None:
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                for module in modules:
                    self._write_lora_for_module(
                        module, layer_refusal_direction, kernel_weight
                    )

    def _write_lora_for_module(
        self, module: Any, v: Tensor, kernel_weight: float
    ) -> None:
        """Compute the rank-1 LoRA update for one Linear and copy it into
        the pre-allocated A/B tensors. Math:

            HF view:     delta_W (out, in) = -k * v v^T W
                         lora_A_hf (1, in) = v^T W
                         lora_B_hf (out, 1) = -k v
            EXL3 view:   W_exl3 (in, out)   = W_hf.T
                         A_exl3 (in, 1)     = W_exl3 @ v == W_hf.T @ v
                         B_exl3 (1, out)    = (-k v).T
            Forward in exllamav3 adds  x @ A @ B  to the projection output.

        Padded vs unpadded: ``v`` has hidden_size = out_features_unpadded;
        anything beyond that is zero-padded.
        """
        in_u = module.in_features_unpadded
        out_u = module.out_features_unpadded
        device = module.device

        # Dequantize the actual functional weight. quant_type == "fp16"
        # falls back to the stored fp16 tensor.
        if module.quant_type == "exl3":
            # LinearEXL3.get_weight_tensor() -> (in_p, out_p) fp16
            W = module.inner.get_weight_tensor()
        elif module.quant_type == "fp16":
            # LinearFP16 stores .weight directly; same (in_p, out_p) layout
            # because exllamav3 keeps everything in input-major form for
            # its kernels.
            W = module.inner.weight
        else:
            raise RuntimeError(
                f"Unknown Linear.quant_type {module.quant_type!r} on {module.key}"
            )

        v_u = v[:out_u].to(device=device, dtype=torch.float32)

        # Compute on the unpadded slice in fp32, then zero-pad and cast.
        W_unpad = W[:in_u, :out_u].to(torch.float32)
        A_unpad = (W_unpad @ v_u).unsqueeze(-1)         # (in_u, 1)
        B_unpad = (-kernel_weight * v_u).unsqueeze(0)   # (1, out_u)

        a_slot = module.lora_a_tensors[self._lora_key]
        b_slot = module.lora_b_tensors[self._lora_key]
        # Pre-zero so any padded rows/cols are clean before we copy unpadded.
        a_slot.zero_()
        b_slot.zero_()
        a_slot[:in_u, :1].copy_(A_unpad.to(torch.float16))
        b_slot[:1, :out_u].copy_(B_unpad.to(torch.float16))

    def reset_model(self) -> None:
        """Zero all LoRA contributions. The current model stays loaded; no
        weight reload is necessary because abliteration happens entirely
        through the additive LoRA path.
        """
        for per_layer in self._layer_modules:
            for modules in per_layer.values():
                for module in modules:
                    module.lora_a_tensors[self._lora_key].zero_()
                    module.lora_b_tensors[self._lora_key].zero_()

    # ------------------------------------------------------------------
    # ARA: module I/O capture + ARA LoRA optimisation
    # ------------------------------------------------------------------

    def _dequantize_weight(self, module: Any) -> Tensor:
        """Dequantize a module's weight to fp32, returning (in_u, out_u)."""
        in_u = module.in_features_unpadded
        out_u = module.out_features_unpadded
        if module.quant_type == "exl3":
            W = module.inner.get_weight_tensor()
        elif module.quant_type == "fp16":
            W = module.inner.weight
        else:
            raise RuntimeError(
                f"Unknown Linear.quant_type {module.quant_type!r} on {module.key}"
            )
        return W[:in_u, :out_u].to(torch.float32)

    def get_module_io(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        module_io: ModuleIO = []

        # Build a mapping from module id to (layer_index, component, module_index)
        # and wrap each target module's forward to capture I/O.
        originals: list[tuple[Any, Any]] = []  # (module, original_forward)

        for layer_index in range(len(self._layer_modules)):
            for component, modules in self._layer_modules[layer_index].items():
                for module_index, module in enumerate(modules):
                    orig = module.forward

                    def _make_wrapper(
                        orig_fn: Any,
                        li: int,
                        comp: str,
                        mi: int,
                    ) -> Any:
                        def wrapped(x, params, out_dtype=None):
                            out = orig_fn(x, params, out_dtype=out_dtype)
                            while len(module_io) <= li:
                                module_io.append({})
                            if comp not in module_io[li]:
                                module_io[li][comp] = {}
                            # x shape: (B, T, in_features). Capture last position.
                            inp = x[:, -1, :].detach().clone().cpu()
                            outp = out[:, -1, :].detach().clone().cpu()
                            module_io[li][comp][mi] = (inp, outp)
                            return out

                        return wrapped

                    module.forward = _make_wrapper(orig, layer_index, component, module_index)
                    originals.append((module, orig))

        # Run a single forward pass to capture I/O.
        input_ids = self._tokenize_chat(prompts)
        with torch.inference_mode():
            self.model.forward(input_ids, params={})

        # Restore original forwards.
        for module, orig in originals:
            module.forward = orig

        return module_io

    def get_module_io_batched(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        module_io_batches: list[ModuleIO] = [
            self.get_module_io(batch)
            for batch in batchify(prompts, self.settings.batch_size)
        ]

        module_io: ModuleIO = []
        for layer_index in range(len(self._layer_modules)):
            module_io.append({})
            for module_io_batch in module_io_batches:
                if layer_index >= len(module_io_batch):
                    continue
                for component, io_map in module_io_batch[layer_index].items():
                    if component not in module_io[layer_index]:
                        module_io[layer_index][component] = {}
                    for module_index in io_map:
                        if module_index not in module_io[layer_index][component]:
                            module_io[layer_index][component][module_index] = (
                                torch.empty(0),
                                torch.empty(0),
                            )

            for component, io_map in module_io[layer_index].items():
                for module_index in io_map:
                    inputs_outputs = [
                        module_io_batch[layer_index][component][module_index]
                        for module_io_batch in module_io_batches
                        if layer_index < len(module_io_batch)
                        and component in module_io_batch[layer_index]
                        and module_index in module_io_batch[layer_index][component]
                    ]
                    inp = torch.cat([io[0] for io in inputs_outputs], dim=0)
                    outp = torch.cat([io[1] for io in inputs_outputs], dim=0)
                    module_io[layer_index][component][module_index] = (inp, outp)

        return module_io

    def ara_lora_abliterate(
        self,
        good_module_io: ModuleIO,
        bad_module_io: ModuleIO,
        parameters: ARAParameters,
    ) -> None:
        """ARA LoRA for EXL3: optimise the pre-allocated LoRA A/B tensors
        using the same objective as standard ARA, but operating in the
        EXL3 weight convention (input-major: (in, out)).
        """
        rank = self._lora_rank()

        for layer_index in range(
            parameters.start_layer_index,
            parameters.end_layer_index,
        ):
            for component, modules in self._layer_modules[layer_index].items():
                for module_index, module in enumerate(modules):
                    in_u = module.in_features_unpadded
                    out_u = module.out_features_unpadded
                    device = module.device

                    W_base = self._dequantize_weight(module).to(device)
                    W_row_norms = LA.vector_norm(W_base, dim=0, keepdim=True).detach()

                    # The optimisable LoRA factors. exllamav3 layout:
                    #   a: (in_features, rank), b: (rank, out_features)
                    #   forward: output += x @ a @ b
                    # We optimise on the unpadded slice.
                    a_slot = module.lora_a_tensors[self._lora_key]
                    b_slot = module.lora_b_tensors[self._lora_key]

                    # Create fp32 parameter copies for the optimizer.
                    a_param = a_slot[:in_u, :rank].float().clone().detach().requires_grad_(True)
                    b_param = b_slot[:rank, :out_u].float().clone().detach().requires_grad_(True)

                    good_input, good_output = good_module_io[layer_index][component][module_index]
                    bad_input, bad_output = bad_module_io[layer_index][component][module_index]

                    good_input = good_input.float().to(device)
                    good_output = good_output.float().to(device)
                    bad_input = bad_input.float().to(device)
                    bad_output = bad_output.float().to(device)

                    def objective(A: Tensor, B: Tensor) -> Tensor:
                        # EXL3 convention: W is (in, out), forward is x @ W.
                        W_eff = W_base + A @ B

                        if self.settings.row_normalization == RowNormalization.FULL:
                            # Column-wise normalisation (dim=0) because W is (in, out).
                            W_eff = F.normalize(W_eff, p=2, dim=0) * W_row_norms

                        # x @ W_eff gives (batch, out_features).
                        new_good_output = good_input @ W_eff
                        new_bad_output = bad_input @ W_eff

                        preserve_good_behavior = (
                            (new_good_output - good_output) ** 2
                        ).mean()

                        steer_bad_behavior = (
                            mean_distances_to_knn(
                                new_bad_output,
                                good_output,
                                parameters.neighbor_count,
                            ).mean()
                            + parameters.overcorrect_relative_weight
                            * -mean_distances_to_knn(
                                new_bad_output,
                                bad_output,
                                parameters.neighbor_count,
                            ).mean()
                        )

                        return (
                            parameters.preserve_good_behavior_weight
                            * preserve_good_behavior
                            + parameters.steer_bad_behavior_weight * steer_bad_behavior
                        )

                    optimizer = LBFGS(
                        [a_param, b_param],
                        lr=1.0,
                        max_iter=20,
                        history_size=10,
                        line_search_fn="strong_wolfe",
                    )

                    def closure():
                        optimizer.zero_grad()
                        loss = objective(a_param, b_param)
                        loss.backward()
                        return loss

                    for step in range(5):
                        optimizer.step(closure)

                    # Write optimised values back into the pre-allocated slots.
                    with torch.no_grad():
                        a_slot.zero_()
                        b_slot.zero_()
                        a_slot[:in_u, :rank].copy_(a_param.to(torch.float16))
                        b_slot[:rank, :out_u].copy_(b_param.to(torch.float16))

    # ------------------------------------------------------------------
    # Forward passes: residuals + logprobs
    # ------------------------------------------------------------------

    def _tokenize_chat(self, prompts: list[Prompt]) -> Tensor:
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats, add_generation_prompt=True, tokenize=False
            ),
        )
        if self.settings.response_prefix:
            chat_prompts = [p + self.settings.response_prefix for p in chat_prompts]

        # Use the HF tokenizer for left-padded batch tokenization. We feed
        # raw input_ids to exllamav3.Model.forward; no attention mask is
        # passed (exllamav3 derives masking from params / cache positions).
        hf = self.tokenizer._ensure_hf()
        enc = hf(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )
        # IMPORTANT: keep input_ids on CPU. exllamav3's Model.forward expects
        # CPU token ids (its own generator feeds them that way) and moves
        # activations onto the compute device internally. The architecture's
        # prepare_inputs helpers (prepare_for_attn, Gemma 4's mm-span builder,
        # etc.) construct index tensors on CPU with no explicit device and then
        # combine them with tensors derived from input_ids; passing input_ids
        # on CUDA triggers "Expected all tensors to be on the same device".
        return enc["input_ids"]

    def _forward(self, input_ids: Tensor, *, last_only: bool = False) -> tuple[Tensor, list[Tensor]]:
        """Run a single forward pass. Returns (logits, export_states)
        where logits is (B, T_out, vocab) and export_states is a list of
        (B, T, H) tensors (one per captured point: embed-out + per-block).

        Note: this is a fresh, stateless forward — we hand the model a
        new params dict each call. The Cache may accumulate, but for
        non-autoregressive forwards over independent batches that
        accumulation doesn't affect correctness, only the cache's
        position counter.
        """
        params: dict[str, Any] = {}
        if last_only:
            params["last_tokens_only"] = 1
        with torch.inference_mode():
            logits = self.model.forward(input_ids, params=params)
        states = params.get("export_states", [])
        return logits, states

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        input_ids = self._tokenize_chat(prompts)
        _, states = self._forward(input_ids, last_only=False)
        if len(states) != self._num_layers + 1:
            raise RuntimeError(
                f"Expected {self._num_layers + 1} captured residuals, got {len(states)}. "
                "Residual hooks may not be installed correctly."
            )
        # Each state is (B, T, H). Take the last position. Stack to (B, L, H).
        residuals = torch.stack(
            [s[:, -1, :].to(torch.float32) for s in states], dim=1
        )

        if 0 <= self.settings.winsorization_quantile < 1:
            abs_residuals = torch.abs(residuals)
            thresholds = torch.quantile(
                abs_residuals,
                self.settings.winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
            empty_cache()
        return residuals

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        out = []
        for batch in batchify(prompts, self.settings.batch_size):
            out.append(self.get_residuals(batch))
        return torch.cat(out, dim=0)

    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")
        running_sum: Tensor | None = None
        total = 0
        for batch in batchify(prompts, self.settings.batch_size):
            r = self.get_residuals(batch)
            s = r.sum(dim=0, dtype=torch.float64).cpu()
            running_sum = s if running_sum is None else running_sum + s
            total += r.shape[0]
        assert running_sum is not None
        return (running_sum / total).to(torch.float32)

    def get_logprobs(self, prompts: list[Prompt]) -> Tensor:
        input_ids = self._tokenize_chat(prompts)
        logits, _ = self._forward(input_ids, last_only=True)
        # logits shape: (B, 1, vocab) when last_tokens_only=1.
        last_logits = logits[:, -1, :]
        logprobs = F.log_softmax(last_logits, dim=-1)
        if self.settings.offload_outputs_to_cpu:
            logprobs = logprobs.cpu()
            empty_cache()
        return logprobs

    def get_logprobs_batched(self, prompts: list[Prompt]) -> Tensor:
        out = []
        for batch in batchify(prompts, self.settings.batch_size):
            out.append(self.get_logprobs(batch))
        return torch.cat(out, dim=0)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _ensure_generator(self) -> Any:
        if self._generator is None:
            Generator = self._exl_api["Generator"]
            Tokenizer = self._exl_api["Tokenizer"]
            self._generator = Generator(
                model=self.model,
                cache=self.cache,
                tokenizer=Tokenizer.from_config(self.config),
            )
        return self._generator

    def _greedy_sampler(self) -> Any:
        """Return a GreedySampler instance. Mirrors HF's do_sample=False
        on the HF backend: pure argmax, no RNG involvement, so repeated
        calls with identical weights produce identical outputs."""
        if getattr(self, "_sampler_cached", None) is None:
            # GreedySampler lives in exllamav3.generator.sampler.presets
            # (re-exported as exllamav3.generator.sampler.*).
            for module_name in (
                "exllamav3.generator.sampler.presets",
                "exllamav3.generator.sampler",
            ):
                with suppress(ImportError, AttributeError):
                    mod = importlib.import_module(module_name)
                    self._sampler_cached = mod.GreedySampler()
                    break
            if getattr(self, "_sampler_cached", None) is None:
                raise RuntimeError(
                    "Could not locate exllamav3.generator.sampler.GreedySampler."
                )
        return self._sampler_cached

    def _render_chat_prompts(self, prompts: list[Prompt]) -> list[str]:
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats, add_generation_prompt=True, tokenize=False
            ),
        )
        if self.settings.response_prefix:
            chat_prompts = [p + self.settings.response_prefix for p in chat_prompts]
        return chat_prompts

    def generate(self, prompts: list[Prompt], **kwargs: Any) -> tuple[Any, Any]:
        """HF-compatible-ish shim. Returns (inputs, outputs) where outputs
        is a tensor of generated token ids (including the prompt) and
        inputs is a dict with input_ids. main.py only reads
        ``inputs["input_ids"].shape[1]`` to slice off the prompt; we mimic
        that by returning the input ids and the concatenated full ids.
        """
        max_new_tokens = int(kwargs.get("max_new_tokens", self.settings.max_response_length))
        chat_prompts = self._render_chat_prompts(prompts)
        generator = self._ensure_generator()

        # Generator.generate(list[str], ...) -> list[str] (completions only,
        # not including the prompt) when completion_only=True (default in
        # examples). We re-tokenize to recover ids.
        # Pin the sampler to GreedySampler so repeated calls under the same
        # weights produce identical outputs (matches HF's do_sample=False).
        completions = generator.generate(
            prompt=chat_prompts,
            max_new_tokens=max_new_tokens,
            completion_only=True,
            add_bos=True,
            sampler=self._greedy_sampler(),
            seed=0,
        )
        if isinstance(completions, str):
            completions = [completions]

        hf = self.tokenizer._ensure_hf()
        prompt_enc = hf(chat_prompts, return_tensors="pt", padding=True)
        full_texts = [p + c for p, c in zip(chat_prompts, completions)]
        full_enc = hf(full_texts, return_tensors="pt", padding=True)

        return prompt_enc, full_enc["input_ids"]

    def get_responses(
        self, prompts: list[Prompt], skip_special_tokens: bool = False
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts, max_new_tokens=self.settings.max_response_length
        )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=skip_special_tokens
        )

    def get_responses_batched(
        self, prompts: list[Prompt], skip_special_tokens: bool = False
    ) -> list[str]:
        out: list[str] = []
        for batch in batchify(prompts, self.settings.batch_size):
            out.extend(self.get_responses(batch, skip_special_tokens=skip_special_tokens))
        return out

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        """Best-effort chat. Not streamed token-by-token (exllamav3's
        streaming API differs from HF's TextStreamer); we just call
        Generator.generate and return the full completion. main.py's
        interactive chat loop accepts the returned string as the final
        response.
        """
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            ),
        )
        generator = self._ensure_generator()
        completion = generator.generate(
            prompt=chat_prompt,
            max_new_tokens=self.settings.max_response_length,
            completion_only=True,
            add_bos=True,
            sampler=self._greedy_sampler(),
            seed=0,
        )
        if isinstance(completion, list):
            completion = completion[0]
        print(completion)
        return completion

    # ------------------------------------------------------------------
    # Save: PEFT-format LoRA adapter sidecar
    # ------------------------------------------------------------------

    def get_merged_model(self) -> Any:
        """Merge the current EXL3 LoRA adapter into the original HF base model.

        EXL3 quantized weights themselves cannot be modified in-place, so this
        path exports the current adapter, loads the original HF base model,
        attaches the adapter with PEFT, and returns ``merge_and_unload()``.
        """
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        hf_base = self.settings.exl3_base_model or self._hf_base_model_name()
        model_class = (
            AutoModelForImageTextToText if self._is_multimodal() else AutoModelForCausalLM
        )

        print("* Loading base model on CPU (this may take a while)...")
        base_model = model_class.from_pretrained(
            hf_base,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=bool(self.settings.trust_remote_code),
        )

        target_module_names: set[str] = set()
        adapter_state: dict[str, Tensor] = {}
        for per_layer in self._layer_modules:
            for modules in per_layer.values():
                for module in modules:
                    in_u = module.in_features_unpadded
                    out_u = module.out_features_unpadded
                    a = module.lora_a_tensors[self._lora_key]
                    b = module.lora_b_tensors[self._lora_key]
                    a_peft = a[:in_u, :].T.contiguous().cpu()
                    b_peft = b[:, :out_u].T.contiguous().cpu()
                    leaf = module.key.rsplit(".", 1)[-1]
                    if leaf.startswith("slice"):
                        leaf = module.key.rsplit(".", 3)[-3]
                    target_module_names.add(leaf)
                    base_key = f"base_model.model.{module.key}"
                    adapter_state[f"{base_key}.lora_A.weight"] = a_peft
                    adapter_state[f"{base_key}.lora_B.weight"] = b_peft

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=1,
            lora_alpha=1,
            lora_dropout=0.0,
            bias="none",
            target_modules=sorted(target_module_names),
            inference_mode=True,
        )

        print("* Applying LoRA adapters...")
        peft_model = get_peft_model(base_model, peft_config)
        for name, param in peft_model.named_parameters():
            if name in adapter_state:
                param.data = adapter_state[name].to(param.device)

        print("* Merging LoRA adapters into base model...")
        return peft_model.merge_and_unload()

    def _is_multimodal(self) -> bool:
        """Check whether the model uses a multimodal wrapper (i.e. module
        keys contain a ``language_model`` segment). Multimodal models must
        be loaded with ``AutoModelForImageTextToText`` when merging the
        adapter back into HF weights.
        """
        return any(
            ".language_model." in key for key in self._all_module_keys
        )

    def get_base_model_hint(self) -> str:
        """Return best-effort HF base model hint for EXL3 merge prompts."""
        return self.settings.exl3_base_model or self._hf_base_model_name()

    def _hf_base_model_name(self) -> str:
        """Best-effort lookup for the original HF model name. Falls back
        to the EXL3 quant path if nothing better is available.
        """
        model_dir = Path(self.settings.model).expanduser()
        config_path = model_dir / "config.json"
        if config_path.exists():
            with suppress(Exception):
                cfg = json.loads(config_path.read_text())
                name = cfg.get("_name_or_path", "")
                if name:
                    return name
        return self.settings.model

    def save_adapter(self, save_directory: str) -> None:
        """Write a PEFT-format LoRA adapter dir compatible with both the
        peft library and exllamav3's own LoRA.from_directory.

        Output:
            <save_directory>/adapter_config.json
            <save_directory>/adapter_model.safetensors
            <save_directory>/merge.py  (ready-to-use merge script)

        Tensor keys are
            base_model.model.<full_key>.lora_A.weight  shape (rank, in_unpad)
            base_model.model.<full_key>.lora_B.weight  shape (out_unpad, rank)
        in fp16, matching what PEFT writes.
        """
        try:
            from safetensors.torch import save_file as st_save_file
        except ImportError as error:
            raise ImportError(
                "Saving the EXL3 adapter requires safetensors. "
                "Install with: pip install -U 'heretic-llm[exl3]'"
            ) from error

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        tensors: dict[str, Tensor] = {}
        target_module_names: set[str] = set()

        for per_layer in self._layer_modules:
            for modules in per_layer.values():
                for module in modules:
                    in_u = module.in_features_unpadded
                    out_u = module.out_features_unpadded
                    a = module.lora_a_tensors[self._lora_key]  # (in_p, 1) fp16
                    b = module.lora_b_tensors[self._lora_key]  # (1, out_p) fp16

                    # Slice off the padding, then transpose back to PEFT layout:
                    #   peft.lora_A.weight: (rank, in_features)
                    #   peft.lora_B.weight: (out_features, rank)
                    a_peft = a[:in_u, :].T.contiguous().cpu()
                    b_peft = b[:, :out_u].T.contiguous().cpu()

                    base_key = f"base_model.model.{module.key}"
                    tensors[f"{base_key}.lora_A.weight"] = a_peft
                    tensors[f"{base_key}.lora_B.weight"] = b_peft

                    # Leaf name for adapter_config.target_modules.
                    leaf = module.key.rsplit(".", 1)[-1]
                    if leaf.startswith("slice"):
                        # mlp.down_proj.slice.N -> down_proj
                        leaf = module.key.rsplit(".", 3)[-3]
                    target_module_names.add(leaf)

        multimodal = self._is_multimodal()
        hf_base = self._hf_base_model_name()

        rank = 1  # We only allocate rank-1 slots in this backend.
        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": hf_base,
            "r": rank,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": sorted(target_module_names),
            "fan_in_fan_out": False,
            "inference_mode": True,
            "heretic_multimodal": multimodal,
        }
        (out_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=2) + "\n"
        )
        st_save_file(tensors, str(out_dir / "adapter_model.safetensors"))

        # Also copy the original tokenizer files alongside the adapter so the
        # output dir is self-sufficient as a PEFT adapter against the base.
        src = Path(self.settings.model).expanduser()
        for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            sp = src / fname
            if sp.exists():
                with suppress(Exception):
                    shutil.copy2(sp, out_dir / fname)

        # Write a ready-to-use merge script so users don't have to guess
        # which AutoModel class to use (this is the #1 pitfall for EXL3
        # adapters on multimodal models like Qwen 3.5).
        model_class = (
            "AutoModelForImageTextToText" if multimodal
            else "AutoModelForCausalLM"
        )
        merge_script = (
            "#!/usr/bin/env python3\n"
            '"""Merge this Heretic EXL3 adapter into a full-precision HF model.\n'
            "\n"
            "Usage:\n"
            "    python merge.py --base <HF_MODEL> --output <OUTPUT_DIR>\n"
            "\n"
            "The --base argument should point to the original (unquantized)\n"
            "HuggingFace model that the EXL3 quant was derived from.\n"
            '"""\n'
            "\n"
            "import argparse\n"
            "from pathlib import Path\n"
            "\n"
            "import torch\n"
            f"from transformers import {model_class}, AutoTokenizer\n"
            "from peft import PeftModel\n"
            "\n"
            "\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            f'    parser.add_argument("--base", default={hf_base!r})\n'
            '    parser.add_argument("--output", required=True)\n'
            '    parser.add_argument("--dtype", default="bfloat16")\n'
            "    args = parser.parse_args()\n"
            "\n"
            "    dtype = getattr(torch, args.dtype)\n"
            "    adapter_dir = str(Path(__file__).resolve().parent)\n"
            "\n"
            f"    model = {model_class}.from_pretrained(\n"
            '        args.base, torch_dtype=dtype, device_map="auto",\n'
            "    )\n"
            "    model = PeftModel.from_pretrained(model, adapter_dir)\n"
            "    merged = model.merge_and_unload()\n"
            "\n"
            "    merged.save_pretrained(args.output, safe_serialization=True)\n"
            "    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.output)\n"
            '    print(f"Saved merged model to {args.output}")\n'
            "\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
        (out_dir / "merge.py").write_text(merge_script)

        print(f"* Wrote PEFT adapter to [bold]{out_dir}[/]")
        if multimodal:
            print(
                f"* [yellow]Note:[/] This model uses a multimodal architecture. "
                f"When merging, load the base model with "
                f"[bold]{model_class}[/], not AutoModelForCausalLM."
            )
        print(f"* A ready-to-use merge script was written to [bold]{out_dir}/merge.py[/]")
