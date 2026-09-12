# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from typing import Any

import torch
from isolated_settings import IsolatedSettings
from transformers import Qwen2Config, Qwen2ForCausalLM

from heretic.model import Model


def make_model(target_components: list[str] | None = None) -> Model:
    """
    Build a Model around a tiny randomly initialized Qwen2 without going
    through Model.__init__ (which loads from disk), so the module discovery
    and capture code can be exercised on CPU in a fraction of a second.
    """
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    tiny = Qwen2ForCausalLM(config).eval()

    overrides: dict[str, Any] = {"model": "tiny"}
    if target_components is not None:
        overrides["target_components"] = target_components

    model = Model.__new__(Model)
    model.settings = IsolatedSettings(**overrides)
    model.model = tiny
    return model


def hide_mlp(model: Model, layer_index: int) -> None:
    """
    Make one layer's MLP undiscoverable (no `.down_proj` attribute) while
    keeping it functional, mimicking a hybrid architecture whose layers do
    not all carry the same components.
    """
    layer = model.get_layers()[layer_index]
    layer.mlp = torch.nn.Sequential(layer.mlp)


class TargetComponentsDiscoveryTests(unittest.TestCase):
    def test_discovers_both_components_by_default(self) -> None:
        model = make_model()

        self.assertEqual(
            model.get_abliterable_components(), ["attn.o_proj", "mlp.down_proj"]
        )
        self.assertEqual(
            set(model.get_layer_modules(0)), {"attn.o_proj", "mlp.down_proj"}
        )

    def test_excluded_component_is_not_discovered(self) -> None:
        model = make_model(["attn.o_proj"])

        self.assertEqual(model.get_abliterable_components(), ["attn.o_proj"])
        for layer_index in range(len(model.get_layers())):
            modules = model.get_layer_modules(layer_index)
            self.assertEqual(set(modules), {"attn.o_proj"})
            self.assertEqual(len(modules["attn.o_proj"]), 1)

    def test_layer_without_target_component_is_empty_not_fatal(self) -> None:
        model = make_model(["mlp.down_proj"])
        hide_mlp(model, 1)

        self.assertEqual(set(model.get_layer_modules(0)), {"mlp.down_proj"})
        self.assertEqual(model.get_layer_modules(1), {})
        self.assertEqual(model.get_abliterable_components(), ["mlp.down_proj"])


class ModuleIOCaptureTests(unittest.TestCase):
    def test_capture_keeps_one_entry_per_layer_across_gaps(self) -> None:
        model = make_model(["mlp.down_proj"])
        hide_mlp(model, 1)

        # get_module_io only needs generate() to run a forward pass that fires
        # the hooks; stand in for the tokenizer-driven version with raw ids.
        input_ids = torch.randint(0, 64, (3, 4))
        model.generate = lambda prompts, max_new_tokens=1: model.model(  # ty:ignore[invalid-assignment]
            input_ids=input_ids
        )

        module_io = model.get_module_io([])

        self.assertEqual(len(module_io), 2)
        self.assertEqual(module_io[1], {})
        (captured_input, captured_output) = module_io[0]["mlp.down_proj"][0]
        self.assertEqual(captured_input.shape, (3, 32))
        self.assertEqual(captured_output.shape, (3, 16))
        self.assertEqual(captured_input.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
