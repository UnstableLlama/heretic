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

* Per-layer hidden states (residuals): we flip
  ``block.export_state = True`` on every ``TransformerBlock``, then
  inject a pre-block-0 capture so the first entry mirrors HF's
  ``hidden_states[0]`` (embedding output).
"""

from __future__ import annotations

import importlib
import json
import math
import re
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import RowNormalization, Settings
from .system import empty_cache
from .utils import Prompt, batchify, print


# Match keys like:
#   model.layers.<N>.self_attn.o_proj
#   model.layers.<N>.mlp.down_proj
#   model.layers.<N>.mlp.down_proj.slice.<M>
#   model.layers.<N>.block_sparse_moe.experts.<I>.down_proj
#   model.layers.<N>.<any>.down_proj
# We accept any sub-path between the layer index and the leaf suffix so
# we cover MoE expert paths, hybrid linear-attention paths, and other
# architecture variants that put o_proj / down_proj at deeper nesting.
_MODULE_KEY_REGEX = re.compile(
    r"^model\.layers\.(\d+)\.(.*?)(o_proj|down_proj)(?:\.slice\.\d+)?$"
)


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


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

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False
        self.revision_kwargs: dict[str, Any] = {}
        self.trusted_models: dict[str, bool | None] = {settings.model: False}

        # Import lazily so the optional dep doesn't crash users on the HF path.
        self._exl = self._import_exllamav3()

        print()
        print(f"Loading EXL3 model [bold]{settings.model}[/]...")

        model_path = str(Path(settings.model).expanduser())
        # 1. Config -> Model -> Cache -> Tokenizer
        self.config = self._exl.Config.from_directory(model_path)
        self.model = self._exl.Model.from_config(self.config)
        self.cache = self._exl.Cache(
            self.model,
            max_num_tokens=int(settings.exl3_max_num_tokens),
        )
        self.model.load(progressbar=True)

        # Tokenizer
        exl3_tok = self._exl.Tokenizer.from_config(self.config)
        self.tokenizer = _Exl3Tokenizer(exl3_tok, model_path)

        # 2. Discover target modules, group by layer, allocate LoRA slots
        self._lora_key = object()  # any hashable; used as the dict key
        self._layer_modules: list[dict[str, list[Any]]] = []
        self._discover_modules()

        # 3. Enable residual capture on every TransformerBlock + inject
        #    a pre-block-0 capture so we get an "embedding output" entry
        #    that mirrors HF's hidden_states[0].
        self._install_residual_hooks()

        # 4. Generator (lazy; only built on first generate call so we
        #    don't pay the cost for runs that only need residuals).
        self._generator = None

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

    @staticmethod
    def _import_exllamav3() -> ModuleType:
        try:
            return importlib.import_module("exllamav3")
        except ImportError as error:
            raise ImportError(
                "ExLlamaV3 backend selected but 'exllamav3' is not installed. "
                "Install with: pip install -U 'heretic-llm[exl3]'"
            ) from error

    def _discover_modules(self) -> None:
        """Walk the loaded model, find o_proj / down_proj Linears, group
        by layer index, allocate a (lora_A, lora_B) pair on each one."""
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
            leaf = m.group(3)  # "o_proj" or "down_proj"
            component = "attn.o_proj" if leaf == "o_proj" else "mlp.down_proj"
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
        for per_layer in self._layer_modules:
            for modules in per_layer.values():
                for module in modules:
                    self._allocate_lora_slot(module)

    def _allocate_lora_slot(self, module: Any) -> None:
        device = module.device
        a = torch.zeros(
            (module.in_features, 1),
            dtype=torch.float16,
            device=device,
        )
        b = torch.zeros(
            (1, module.out_features),
            dtype=torch.float16,
            device=device,
        )
        module.lora_a_tensors[self._lora_key] = a
        module.lora_b_tensors[self._lora_key] = b

    def _install_residual_hooks(self) -> None:
        """Set ``export_state = True`` on each TransformerBlock and wrap
        the first one's ``forward`` to also capture x at entry — so the
        captured list is ``[embed_out, block0_out, block1_out, ...]``,
        matching HF's ``hidden_states`` shape.
        """
        transformer_mod = importlib.import_module("exllamav3.modules.transformer")
        TransformerBlock = transformer_mod.TransformerBlock

        blocks: list[Any] = []
        for module in self.model:
            if isinstance(module, TransformerBlock):
                module.export_state = True
                blocks.append(module)

        if not blocks:
            raise RuntimeError(
                "No TransformerBlock instances found in model. "
                "Residual capture requires the standard transformer block class; "
                "this runtime may use ParallelDecoderBlock or a custom block."
            )

        # Sort by layer_idx so blocks[0] really is layer 0.
        blocks.sort(key=lambda b: (b.layer_idx if b.layer_idx is not None else -1))
        first_block = blocks[0]
        original_forward = first_block.forward

        def capturing_forward(x, params, out_dtype=None, _orig=original_forward):
            states = params.get("export_states")
            if states is None:
                states = params["export_states"] = []
            # Capture pre-block-0 state == post-embedding state.
            states.append(x.half().clone())
            return _orig(x, params, out_dtype=out_dtype)

        first_block.forward = capturing_forward
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
        input_ids = enc["input_ids"].to(self._first_device())
        return input_ids

    def _first_device(self) -> torch.device:
        # Pick the device of any loaded module (model may be sharded).
        for per_layer in self._layer_modules:
            for modules in per_layer.values():
                for module in modules:
                    return module.device
        return torch.device("cuda:0")

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
            Generator = self._exl.Generator
            self._generator = Generator(
                model=self.model,
                cache=self.cache,
                tokenizer=self._exl.Tokenizer.from_config(self.config),
            )
        return self._generator

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
        completions = generator.generate(
            prompt=chat_prompts,
            max_new_tokens=max_new_tokens,
            completion_only=True,
            add_bos=True,
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
        )
        if isinstance(completion, list):
            completion = completion[0]
        print(completion)
        return completion

    # ------------------------------------------------------------------
    # Save: PEFT-format LoRA adapter sidecar
    # ------------------------------------------------------------------

    def get_merged_model(self) -> Any:
        """EXL3 doesn't support baking LoRA back into the quantized
        storage. The save UI in main.py should branch on backend and
        offer adapter-only saving for EXL3.
        """
        raise NotImplementedError(
            "EXL3 weights cannot be merged in-place. Save the LoRA adapter "
            "sidecar (heretic offers this path for the EXL3 backend), or "
            "re-quantize the merged model manually."
        )

    def save_adapter(self, save_directory: str) -> None:
        """Write a PEFT-format LoRA adapter dir compatible with both the
        peft library and exllamav3's own LoRA.from_directory.

        Output:
            <save_directory>/adapter_config.json
            <save_directory>/adapter_model.safetensors

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

        rank = 1  # We only allocate rank-1 slots in this backend.
        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": self.settings.model,
            "r": rank,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": sorted(target_module_names),
            "fan_in_fan_out": False,
            "inference_mode": True,
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

        print(f"* Wrote PEFT adapter to [bold]{out_dir}[/]")
