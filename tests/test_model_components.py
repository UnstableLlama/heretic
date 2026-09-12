# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from typing import Any

import torch
from isolated_settings import IsolatedSettings
from transformers import Qwen2Config, Qwen2ForCausalLM

from heretic.model import Model

EOS_TOKEN_ID = 63


def make_model(target_components: list[str] | None = None, **settings: Any) -> Model:
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
    tiny.generation_config.eos_token_id = EOS_TOKEN_ID

    overrides: dict[str, Any] = {"model": "tiny", **settings}
    if target_components is not None:
        overrides["target_components"] = target_components

    model = Model.__new__(Model)
    model.settings = IsolatedSettings(**overrides)
    model.model = tiny
    return model


class FakeLinearAttention(torch.nn.Module):
    """Stand-in for a GatedDeltaNet block: exposes `out_proj` only."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)


def make_hybrid_layer(model: Model, layer_index: int) -> None:
    """
    Turn one layer into a hybrid linear-attention layer the way Qwen3.5/3.8
    lay them out: `linear_attn.out_proj` instead of `self_attn.o_proj`.
    Discovery only looks at attributes, so the layer need not stay runnable.
    """
    layer = model.get_layers()[layer_index]
    hidden_size = layer.self_attn.o_proj.out_features
    del layer.self_attn
    layer.linear_attn = FakeLinearAttention(hidden_size)


class FakeInputs:
    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids


def install_fake_generate(model: Model, generated: torch.Tensor) -> list[str | None]:
    """
    Replace generate() with a stand-in that runs the tiny model once per
    generated token (a prefill pass, then one decode pass per further token),
    so the capture hooks fire exactly as they would during real generation.
    `generated` holds the token ids the fake run "produces". The response
    prefix each call receives is recorded in the returned list.
    """
    prompt_ids = torch.randint(0, EOS_TOKEN_ID, (generated.shape[0], 4))
    received: list[str | None] = []

    def fake_generate(prompts, response_prefix=None, max_new_tokens=1):
        received.append(response_prefix)
        steps = min(max_new_tokens, generated.shape[1])
        with torch.no_grad():
            out = model.model(input_ids=prompt_ids, use_cache=True)
            for step in range(1, steps):
                out = model.model(
                    input_ids=generated[:, step - 1 : step],
                    past_key_values=out.past_key_values,
                    use_cache=True,
                )
        sequences = torch.cat([prompt_ids, generated[:, :steps]], dim=1)
        return FakeInputs(prompt_ids), sequences

    model.generate = fake_generate  # ty:ignore[invalid-assignment]
    return received


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

    def test_linear_attention_is_its_own_component(self) -> None:
        model = make_model()
        make_hybrid_layer(model, 1)

        self.assertEqual(
            model.get_abliterable_components(),
            ["attn.o_proj", "attn.out_proj", "mlp.down_proj"],
        )
        self.assertEqual(
            set(model.get_layer_modules(0)), {"attn.o_proj", "mlp.down_proj"}
        )
        self.assertEqual(
            set(model.get_layer_modules(1)), {"attn.out_proj", "mlp.down_proj"}
        )

    def test_linear_attention_can_be_targeted_alone(self) -> None:
        model = make_model(["attn.out_proj"])
        make_hybrid_layer(model, 1)

        self.assertEqual(model.get_abliterable_components(), ["attn.out_proj"])
        self.assertEqual(model.get_layer_modules(0), {})
        self.assertEqual(set(model.get_layer_modules(1)), {"attn.out_proj"})

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
        install_fake_generate(model, torch.randint(0, EOS_TOKEN_ID, (3, 1)))

        module_io = model.get_module_io([])

        self.assertEqual(len(module_io), 2)
        self.assertEqual(module_io[1], {})
        (captured_input, captured_output) = module_io[0]["mlp.down_proj"][0]
        self.assertEqual(captured_input.shape, (3, 32))
        self.assertEqual(captured_output.shape, (3, 16))
        self.assertEqual(captured_input.device.type, "cpu")

    def test_capture_tokens_add_one_sample_per_generated_position(self) -> None:
        model = make_model(["attn.o_proj"], ara_capture_tokens=4)
        install_fake_generate(model, torch.randint(0, EOS_TOKEN_ID, (3, 4)))

        module_io = model.get_module_io([])

        captured_input, captured_output = module_io[0]["attn.o_proj"][0]
        # 3 prompts x 4 positions (prompt end + 3 generated tokens).
        self.assertEqual(captured_input.shape, (12, 16))
        self.assertEqual(captured_output.shape, (12, 16))

    def test_capture_drops_positions_after_end_of_sequence(self) -> None:
        model = make_model(["attn.o_proj"], ara_capture_tokens=4)
        generated = torch.randint(0, EOS_TOKEN_ID, (3, 4))
        generated[0, 1] = EOS_TOKEN_ID  # prompt 0 ends at its second token
        install_fake_generate(model, generated)

        module_io = model.get_module_io([])

        captured_input, _ = module_io[0]["attn.o_proj"][0]
        # Prompt 0 keeps the positions up to and including EOS; the others all 4.
        self.assertEqual(captured_input.shape[0], 2 + 4 + 4)

    def test_capture_prefix_overrides_response_prefix(self) -> None:
        model = make_model(
            ["attn.o_proj"], response_prefix="", ara_capture_prefix="\n</think>\n\n"
        )
        received = install_fake_generate(model, torch.randint(0, EOS_TOKEN_ID, (3, 1)))

        model.get_module_io([])

        self.assertEqual(received, ["\n</think>\n\n"])

    def test_capture_uses_response_prefix_by_default(self) -> None:
        model = make_model(["attn.o_proj"], response_prefix="<think></think>\n")
        received = install_fake_generate(model, torch.randint(0, EOS_TOKEN_ID, (3, 1)))

        model.get_module_io([])

        self.assertEqual(received, ["<think></think>\n"])


if __name__ == "__main__":
    unittest.main()
