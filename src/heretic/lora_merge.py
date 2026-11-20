# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Merge PEFT-format LoRA tensors directly into a Hugging Face model.

PEFT can only attach LoRA adapters to modules (``nn.Linear`` and friends), and
it locates them by exact module name. Adapters produced by the EXL3 backend
break both assumptions:

* ExLlamaV3 *unfuses* MoE experts during conversion: every expert becomes its
  own Linear (``...experts.<I>.down_proj``), whereas Transformers stores the
  experts of most MoE architectures as a single 3D parameter
  (``...experts.down_proj`` with shape ``(experts, out, in)``). Such adapters
  have no module to attach to and PEFT silently drops them.
* ExLlamaV3 slices very wide MLP projections (``...down_proj.slice.<M>``).
* The quantized model's tensor names can carry a different prefix than the
  Hugging Face base model (e.g. a Gemma backbone nested inside a custom
  wrapper model).

This module therefore applies ``W += B @ A`` itself: it re-fuses per-expert
adapters into the fused parameter, reassembles slices, and falls back to
matching on the ``layers.<N>.…`` tail of the name when the prefix differs.

IMPORTANT: This file must stay self-contained (standard library, torch and
lazily imported transformers only). ``Exl3Model.save_adapter`` embeds its
source verbatim into the standalone ``merge.py`` shipped with adapters.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

_ADAPTER_KEY_REGEX = re.compile(
    r"^(?:base_model\.model\.)?(.+)\.lora_([AB])(?:\.[^.]+)?\.weight$"
)
_SLICE_REGEX = re.compile(r"^(.+)\.slice\.(\d+)$")
_UNFUSED_EXPERT_REGEX = re.compile(r"^(.+)\.(\d+)\.([^.]+)$")
_LAYER_TAIL_REGEX = re.compile(r"(?:^|\.)(layers\.\d+\..+)$")


@dataclass
class MergeReport:
    # Adapters added to an ordinary 2D weight (including slices).
    merged_linear: int = 0
    # Per-expert adapters re-fused into a 3D expert parameter.
    merged_fused: int = 0
    # Adapter module names that could not be applied, with the reason.
    unmatched: dict[str, str] = field(default_factory=dict)

    @property
    def merged(self) -> int:
        return self.merged_linear + self.merged_fused


def select_model_class(model: str) -> Any:
    """Choose the Auto class that can load ``model``.

    Multimodal models need AutoModelForImageTextToText, text-only models
    AutoModelForCausalLM. Models that ship custom code and register it only
    for AutoModel (e.g. a wrapper around a Gemma backbone) need AutoModel.
    """
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        PretrainedConfig,
    )

    config = PretrainedConfig.get_config_dict(model)[0]

    if "vision_config" in config:
        name, model_class = "AutoModelForImageTextToText", AutoModelForImageTextToText
    else:
        name, model_class = "AutoModelForCausalLM", AutoModelForCausalLM

    auto_map = config.get("auto_map") or {}
    if "AutoModel" in auto_map and name not in auto_map:
        return AutoModel

    return model_class


def _collect_adapters(state: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    adapters: dict[str, dict[str, Tensor]] = {}
    for key, tensor in state.items():
        match = _ADAPTER_KEY_REGEX.match(key)
        if match is not None:
            adapters.setdefault(match.group(1), {})[match.group(2)] = tensor
    return adapters


def _common_suffix_length(a: str, b: str) -> int:
    a_parts, b_parts = a.split("."), b.split(".")
    length = 0
    while (
        length < min(len(a_parts), len(b_parts))
        and a_parts[-1 - length] == b_parts[-1 - length]
    ):
        length += 1
    return length


def _find(name: str, weights: dict[str, Tensor], accept: Any) -> str | None:
    """Find the weight called ``name``, tolerating a different name prefix."""
    if name in weights:
        return name if accept(weights[name]) else None

    match = _LAYER_TAIL_REGEX.search(name)
    if match is None:
        return None
    tail = match.group(1)

    candidates = [
        candidate
        for candidate, weight in weights.items()
        if (candidate == tail or candidate.endswith("." + tail)) and accept(weight)
    ]
    if len(candidates) > 1:
        # Prefer the candidate that shares the most trailing name segments,
        # e.g. "...language_model.layers.0..." over "...vision_tower.layers.0...".
        best = max(_common_suffix_length(name, candidate) for candidate in candidates)
        candidates = [
            candidate
            for candidate in candidates
            if _common_suffix_length(name, candidate) == best
        ]
    return candidates[0] if len(candidates) == 1 else None


def _add_delta(target: Tensor, delta: Tensor) -> None:
    # Add in float32 and round once, rather than rounding the (small) delta
    # to the weight's dtype before adding it.
    target.copy_((target.float() + delta.to(target.device)).to(target.dtype))


@torch.no_grad()
def merge_lora_state_dict(
    model: Any,
    state: dict[str, Tensor],
    scale: float = 1.0,
) -> MergeReport:
    """Add ``scale * B @ A`` to the weights of ``model``, in place.

    ``state`` uses PEFT's layout: ``[base_model.model.]<module>.lora_A.weight``
    with shape ``(rank, in_features)`` and ``<module>.lora_B.weight`` with
    shape ``(out_features, rank)``. ``scale`` is ``lora_alpha / r``.
    """
    report = MergeReport()

    weights: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.dim() == 2 and name.endswith(".weight"):
            weights[name[: -len(".weight")]] = parameter
        elif parameter.dim() == 3:
            weights[name] = parameter

    # Column offsets of sliced projections: slices are consecutive ranges of
    # the input features, in slice order.
    slice_offsets: dict[str, int] = {}
    slice_widths: dict[str, list[tuple[int, int]]] = {}
    adapters = _collect_adapters(state)
    for module_name, pair in adapters.items():
        match = _SLICE_REGEX.match(module_name)
        if match is not None and "A" in pair:
            slice_widths.setdefault(match.group(1), []).append(
                (int(match.group(2)), pair["A"].shape[1])
            )
    for base_name, widths in slice_widths.items():
        offset = 0
        for index, width in sorted(widths):
            slice_offsets[f"{base_name}.slice.{index}"] = offset
            offset += width

    for module_name, pair in adapters.items():
        if "A" not in pair or "B" not in pair:
            report.unmatched[module_name] = "incomplete lora_A/lora_B pair"
            continue

        lora_a, lora_b = pair["A"].float(), pair["B"].float()
        if not lora_a.any() or not lora_b.any():
            # An all-zero adapter (e.g. a never-routed expert) changes nothing.
            continue

        delta = (lora_b @ lora_a) * scale
        out_features, in_features = delta.shape

        target_name = module_name
        offset = 0
        slice_match = _SLICE_REGEX.match(module_name)
        if slice_match is not None:
            target_name = slice_match.group(1)
            offset = slice_offsets[module_name]

        # 1. An ordinary linear weight.
        def accept_linear(weight: Tensor) -> bool:
            return (
                weight.dim() == 2
                and weight.shape[0] == out_features
                and (
                    weight.shape[1] >= offset + in_features
                    if slice_match is not None
                    else weight.shape[1] == in_features
                )
            )

        name = _find(target_name, weights, accept_linear)
        if name is not None:
            _add_delta(weights[name][:, offset : offset + in_features], delta)
            report.merged_linear += 1
            continue

        # 2. One expert of a fused expert parameter.
        expert_match = _UNFUSED_EXPERT_REGEX.match(target_name)
        if expert_match is not None and slice_match is None:
            fused_name = f"{expert_match.group(1)}.{expert_match.group(3)}"
            expert_index = int(expert_match.group(2))

            def accept_fused(weight: Tensor) -> bool:
                return (
                    weight.dim() == 3
                    and expert_index < weight.shape[0]
                    and tuple(weight.shape[1:])
                    in ((out_features, in_features), (in_features, out_features))
                )

            name = _find(fused_name, weights, accept_fused)
            if name is not None:
                fused = weights[name]
                if out_features != in_features:
                    transposed = tuple(fused.shape[1:]) == (in_features, out_features)
                else:
                    # Square experts: ask the owning module (Transformers'
                    # experts interface records the layout there).
                    owner = model.get_submodule(name.rpartition(".")[0])
                    transposed = bool(getattr(owner, "is_transposed", False))
                _add_delta(fused[expert_index], delta.T if transposed else delta)
                report.merged_fused += 1
                continue

        report.unmatched[module_name] = (
            f"no weight of shape ({out_features}, {in_features}) with a matching name"
        )

    return report
