"""CPU-only tests for merging EXL3-style adapters into Hugging Face models."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    Inkling2ForCausalLM,
    Inkling2TextConfig,
)

from heretic.exl3_model import Exl3Model
from heretic.lora_merge import merge_lora_state_dict, select_model_class
from heretic.model import Model, _collapse_looped_residual_streams

HIDDEN, EXPERT_INTERMEDIATE, EXPERTS = 8, 12, 4


def gemma4_moe() -> Any:
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "vocab_size": 64,
        "hidden_size": HIDDEN,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "global_head_dim": 4,
        "num_global_key_value_heads": 1,
        "enable_moe_block": True,
        "num_experts": EXPERTS,
        "top_k_experts": 2,
        "moe_intermediate_size": EXPERT_INTERMEDIATE,
        "layer_types": ["sliding_attention", "full_attention"],
        "vocab_size_per_layer_input": 64,
        "hidden_size_per_layer_input": 0,
    }
    config = Gemma4TextConfig(**kwargs)
    return Gemma4ForCausalLM(config).eval()


def inkling2() -> Any:
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "swa_head_dim": 8,
        "swa_num_attention_heads": 2,
        "swa_num_key_value_heads": 1,
        "n_routed_experts": 4,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "d_rel": 4,
        "rel_extent": 8,
        "sliding_window_size": 8,
        "eos_token_id": 1,
    }
    config = Inkling2TextConfig(**kwargs)
    return Inkling2ForCausalLM(config).eval()


def adapter(key: str, out_features: int, in_features: int, rank: int = 2):
    """An adapter for one module, with the key layout the EXL3 backend exports."""
    lora_a = torch.randn(rank, in_features)
    lora_b = torch.randn(out_features, rank)
    return {
        f"base_model.model.{key}.lora_A.weight": lora_a,
        f"base_model.model.{key}.lora_B.weight": lora_b,
    }, lora_b @ lora_a


class MergeTests(unittest.TestCase):
    def test_unfused_expert_adapter_is_refused_into_expert_tensor(self):
        model = gemma4_moe()
        experts = model.model.layers[1].experts
        before = experts.down_proj.detach().clone()
        gate_up = experts.gate_up_proj.detach().clone()

        # ExLlamaV3 keys each expert separately; Transformers has no such module.
        state, delta = adapter(
            "model.layers.1.experts.2.down_proj", HIDDEN, EXPERT_INTERMEDIATE
        )
        report = merge_lora_state_dict(model, state)

        self.assertEqual((report.merged_fused, report.merged_linear), (1, 0))
        self.assertEqual(report.unmatched, {})
        after = experts.down_proj.detach()
        torch.testing.assert_close(after[2], before[2] + delta)
        for untouched in (0, 1, 3):
            torch.testing.assert_close(after[untouched], before[untouched])
        torch.testing.assert_close(experts.gate_up_proj.detach(), gate_up)

    def test_refused_expert_changes_forward_like_a_lora_on_that_expert(self):
        model = gemma4_moe()
        experts = model.model.layers[0].experts
        hidden_states = torch.randn(5, HIDDEN)
        # Route every token to expert 1 only, with weight 1.
        top_k_index = torch.ones(5, 1, dtype=torch.long)
        top_k_weights = torch.ones(5, 1)

        with torch.no_grad():
            before = experts(hidden_states, top_k_index, top_k_weights)
            gate, up = (hidden_states @ experts.gate_up_proj[1].T).chunk(2, dim=-1)
            activation = experts.act_fn(gate) * up

        state, delta = adapter(
            "model.layers.0.experts.1.down_proj", HIDDEN, EXPERT_INTERMEDIATE
        )
        merge_lora_state_dict(model, state)

        with torch.no_grad():
            after = experts(hidden_states, top_k_index, top_k_weights)
        # A LoRA on the expert's down-projection adds activation @ (B A)^T.
        torch.testing.assert_close(
            after - before, activation @ delta.T, atol=1e-5, rtol=1e-4
        )

    def test_linear_adapters_match_peft_merge(self):
        state: dict[str, torch.Tensor] = {}
        for layer in range(2):
            for key, in_features in (("self_attn.o_proj", 8), ("mlp.down_proj", 16)):
                state |= adapter(f"model.layers.{layer}.{key}", HIDDEN, in_features)[0]

        ours = gemma4_moe()
        report = merge_lora_state_dict(ours, state)
        self.assertEqual((report.merged_linear, report.unmatched), (4, {}))

        peft_model = get_peft_model(
            gemma4_moe(),
            LoraConfig(r=2, lora_alpha=2, target_modules=["o_proj", "down_proj"]),
        )
        result = set_peft_model_state_dict(peft_model, state)
        self.assertEqual(result.unexpected_keys, [])
        reference = peft_model.merge_and_unload()

        for (name, ours_param), (_, reference_param) in zip(
            ours.named_parameters(), reference.named_parameters()
        ):
            torch.testing.assert_close(ours_param, reference_param, msg=name)

    def test_peft_drops_unfused_expert_adapters(self):
        # Documents why the EXL3 merge cannot use PEFT: this is the bug that
        # silently produced merged models without any expert modification.
        state, _ = adapter(
            "model.layers.0.experts.1.down_proj", HIDDEN, EXPERT_INTERMEDIATE
        )
        peft_model = get_peft_model(
            gemma4_moe(),
            LoraConfig(r=2, lora_alpha=2, target_modules=["o_proj", "down_proj"]),
        )
        result = set_peft_model_state_dict(peft_model, state)
        self.assertEqual(len(result.unexpected_keys), 2)

    def test_matches_base_model_with_a_different_name_prefix(self):
        # A Gemma backbone nested in a custom wrapper (as in Apriel Voice), while
        # the EXL3 quantization uses plain Gemma tensor names.
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_tower = nn.ModuleDict(
                    {"layers": nn.ModuleList([nn.Linear(3, 3)])}
                )
                self.language_model = gemma4_moe()

        wrapper: Any = Wrapper()
        backbone = wrapper.language_model.model.layers[1]
        o_proj_before = backbone.self_attn.o_proj.weight.detach().clone()
        expert_before = backbone.experts.down_proj.detach().clone()

        state, o_proj_delta = adapter(
            "model.language_model.layers.1.self_attn.o_proj", HIDDEN, 8
        )
        expert_state, expert_delta = adapter(
            "model.language_model.layers.1.experts.3.down_proj",
            HIDDEN,
            EXPERT_INTERMEDIATE,
        )
        report = merge_lora_state_dict(wrapper, state | expert_state)

        self.assertEqual((report.merged_linear, report.merged_fused), (1, 1))
        torch.testing.assert_close(
            backbone.self_attn.o_proj.weight.detach(), o_proj_before + o_proj_delta
        )
        torch.testing.assert_close(
            backbone.experts.down_proj.detach()[3], expert_before[3] + expert_delta
        )

    def test_sliced_projection_slices_are_reassembled_in_order(self):
        model: Any = nn.ModuleDict(
            {
                "layers": nn.ModuleList(
                    [nn.ModuleDict({"down_proj": nn.Linear(10, 4, bias=False)})]
                )
            }
        )
        before = model["layers"][0]["down_proj"].weight.detach().clone()

        # Deliberately out of order and of unequal width (4, 4, 2).
        second, second_delta = adapter("layers.0.down_proj.slice.1", 4, 4)
        third, third_delta = adapter("layers.0.down_proj.slice.2", 4, 2)
        first, first_delta = adapter("layers.0.down_proj.slice.0", 4, 4)
        report = merge_lora_state_dict(model, second | third | first)

        self.assertEqual(report.merged_linear, 3)
        torch.testing.assert_close(
            model["layers"][0]["down_proj"].weight.detach(),
            before + torch.cat([first_delta, second_delta, third_delta], dim=1),
        )

    def test_transposed_expert_layout_and_scale(self):
        class Experts(nn.Module):
            def __init__(self):
                super().__init__()
                # (experts, in_features, out_features), as in gpt-oss.
                self.down_proj = nn.Parameter(torch.randn(2, 6, 3))

        model: Any = nn.ModuleDict(
            {"layers": nn.ModuleList([nn.ModuleDict({"experts": Experts()})])}
        )
        before = model["layers"][0]["experts"].down_proj.detach().clone()
        state, delta = adapter("layers.0.experts.1.down_proj", 3, 6)
        report = merge_lora_state_dict(model, state, scale=0.5)

        self.assertEqual(report.merged_fused, 1)
        torch.testing.assert_close(
            model["layers"][0]["experts"].down_proj.detach()[1],
            before[1] + 0.5 * delta.T,
        )

    def test_unmatched_adapters_are_reported_not_ignored(self):
        model = gemma4_moe()
        wrong_shape, _ = adapter("model.layers.0.self_attn.o_proj", HIDDEN, 5)
        missing, _ = adapter("model.layers.7.self_attn.o_proj", HIDDEN, 8)
        report = merge_lora_state_dict(model, wrong_shape | missing)

        self.assertEqual(report.merged, 0)
        self.assertEqual(
            set(report.unmatched),
            {"model.layers.0.self_attn.o_proj", "model.layers.7.self_attn.o_proj"},
        )


def exl3_module(key: str, in_features: int, out_features: int) -> SimpleNamespace:
    return SimpleNamespace(
        key=key,
        device=torch.device("cpu"),
        in_features=in_features,
        in_features_unpadded=in_features,
        out_features=out_features,
        out_features_unpadded=out_features,
        trim_padded_out=False,
        lora_a_tensors={},
        lora_b_tensors={},
    )


def exl3_backend() -> tuple[Any, list[SimpleNamespace]]:
    model: Any = Exl3Model.__new__(Exl3Model)
    model.settings = SimpleNamespace(exl3_base_model="base", model="quant")
    model.processor = None
    model._lora_key = object()
    model._adapter_rank = 2
    model._generator = None
    model._all_module_keys = []
    modules = [
        exl3_module("model.layers.0.self_attn.o_proj", 8, HIDDEN),
        exl3_module("model.layers.0.experts.0.down_proj", EXPERT_INTERMEDIATE, HIDDEN),
        exl3_module("model.layers.0.experts.3.down_proj", EXPERT_INTERMEDIATE, HIDDEN),
    ]
    model._layer_modules = [{"attn.o_proj": modules[:1], "mlp.down_proj": modules[1:]}]
    for module in modules:
        model._allocate_lora_slot(module)
        # ExLlamaV3 layout: A is (in, rank), B is (rank, out); x @ A @ B.
        module.lora_a_tensors[model._lora_key].copy_(torch.randn(module.in_features, 2))
        module.lora_b_tensors[model._lora_key].copy_(
            torch.randn(2, module.out_features)
        )
    return model, modules


class Exl3MergeTests(unittest.TestCase):
    def test_merged_model_contains_expert_modifications(self):
        backend, modules = exl3_backend()
        backend._is_multimodal = Mock(return_value=False)
        base = gemma4_moe()
        before = {name: p.detach().clone() for name, p in base.named_parameters()}
        loader = SimpleNamespace(from_pretrained=Mock(return_value=base))

        with patch("heretic.exl3_model.select_model_class", return_value=loader):
            merged = backend.get_merged_model()

        def delta(module: SimpleNamespace) -> torch.Tensor:
            lora_a = module.lora_a_tensors[backend._lora_key].float()
            lora_b = module.lora_b_tensors[backend._lora_key].float()
            return (lora_a @ lora_b).T

        after = dict(merged.named_parameters())
        torch.testing.assert_close(
            after["model.layers.0.self_attn.o_proj.weight"],
            before["model.layers.0.self_attn.o_proj.weight"] + delta(modules[0]),
        )
        fused = "model.layers.0.experts.down_proj"
        torch.testing.assert_close(
            after[fused][0], before[fused][0] + delta(modules[1])
        )
        torch.testing.assert_close(
            after[fused][3], before[fused][3] + delta(modules[2])
        )
        torch.testing.assert_close(after[fused][1], before[fused][1])

    def test_merge_refuses_to_return_an_unmodified_base(self):
        backend, _ = exl3_backend()
        backend._is_multimodal = Mock(return_value=False)
        loader = SimpleNamespace(from_pretrained=Mock(return_value=nn.Linear(2, 2)))
        with (
            patch("heretic.exl3_model.select_model_class", return_value=loader),
            self.assertRaisesRegex(RuntimeError, "No adapter could be merged"),
        ):
            backend.get_merged_model()

    def test_generated_merge_script_refuses_experts(self):
        backend, _ = exl3_backend()
        backend._is_multimodal = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory:
            base_dir = Path(directory, "base")
            gemma4_moe().save_pretrained(base_dir)
            adapter_dir = Path(directory, "adapter")
            backend.save_adapter(str(adapter_dir))
            script = (adapter_dir / "merge.py").read_text()
            compile(script, "merge.py", "exec")
            self.assertIn("def merge_lora_state_dict", script)
            self.assertNotIn("heretic", script.split('"""', 2)[2])

            # Run only the embedded merge, not the tokenizer export.
            driver = (
                "import runpy, sys, torch\n"
                "from safetensors.torch import load_file\n"
                "from transformers import AutoModelForCausalLM\n"
                "ns = runpy.run_path(sys.argv[1])\n"
                "model = AutoModelForCausalLM.from_pretrained(sys.argv[2])\n"
                "state = load_file(sys.argv[1].replace('merge.py', "
                "'adapter_model.safetensors'))\n"
                "report = ns['merge_lora_state_dict'](model, state)\n"
                "print(report.merged_linear, report.merged_fused, "
                "len(report.unmatched))\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", driver, str(adapter_dir / "merge.py"), base_dir],
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(result.stdout.split()[-3:], ["1", "2", "0"])


class ModelClassTests(unittest.TestCase):
    def select(self, config: dict):
        with patch(
            "transformers.PretrainedConfig.get_config_dict", return_value=(config, {})
        ):
            return select_model_class("model")

    def test_text_and_multimodal_models(self):
        self.assertIs(self.select({}), AutoModelForCausalLM)
        self.assertIs(self.select({"vision_config": {}}), AutoModelForImageTextToText)

    def test_custom_code_registered_for_auto_model_only(self):
        # Apriel Voice: vision config nested under text_config, AutoModel only.
        config = {
            "auto_map": {"AutoConfig": "c.Config", "AutoModel": "m.Model"},
            "text_config": {"vision_config": {}},
        }
        self.assertIs(self.select(config), AutoModel)

    def test_custom_code_registered_for_the_expected_class(self):
        config = {"auto_map": {"AutoModel": "m.A", "AutoModelForCausalLM": "m.B"}}
        self.assertIs(self.select(config), AutoModelForCausalLM)


class HfBackendArchitectureTests(unittest.TestCase):
    def backend(self, model: nn.Module) -> Any:
        backend: Any = Model.__new__(Model)
        backend.model = model
        return backend

    def test_inkling2_layers_and_modules(self):
        backend = self.backend(inkling2())
        self.assertEqual(len(backend.get_layers()), 6)
        self.assertEqual(backend.get_modifiable_components(), ["attn.o_proj"])
        (o_proj,) = backend.get_layer_modules(0)["attn.o_proj"]
        self.assertIsInstance(o_proj, nn.Linear)

    def test_gated_output_projection_is_unwrapped(self):
        # Inkling 2 Mini: self_attn.o_proj is a gate module holding the Linear.
        linear = nn.Linear(4, 4)
        gated = nn.ModuleDict({"gate_proj": nn.Linear(4, 4), "linear": linear})
        layer = nn.ModuleDict({"self_attn": nn.ModuleDict({"o_proj": gated})})
        backend = self.backend(
            nn.ModuleDict({"model": nn.ModuleDict({"layers": nn.ModuleList([layer])})})
        )
        self.assertEqual(backend.get_layer_modules(0), {"attn.o_proj": [linear]})

    def test_layers_of_wrapped_multimodal_backbone(self):
        layers = nn.ModuleList([nn.Identity()])
        backbone = nn.ModuleDict(
            {
                "model": nn.ModuleDict(
                    {"language_model": nn.ModuleDict({"layers": layers})}
                )
            }
        )
        backend = self.backend(nn.ModuleDict({"language_model": backbone}))
        self.assertIs(backend.get_layers(), layers)


class LoopedResidualTests(unittest.TestCase):
    def test_keeps_final_pass_of_looped_layers(self):
        # Inkling 2 Mini: 24 layers, layers 6..17 run three times.
        config = SimpleNamespace(
            num_hidden_layers=24,
            loop_layer_index_start=6,
            loop_layer_index_end=18,
            loop_num_passes=3,
        )
        names = (
            ["embeddings"]
            + [f"layer{i}" for i in range(6)]
            + [f"pass{p}.layer{i}" for p in range(3) for i in range(6, 18)]
            + [f"layer{i}" for i in range(18, 24)]
        )
        self.assertEqual(len(names), 49)
        collapsed = _collapse_looped_residual_streams(cast(Any, tuple(names)), config)
        self.assertEqual(
            list(collapsed),
            ["embeddings"]
            + [f"layer{i}" for i in range(6)]
            + [f"pass2.layer{i}" for i in range(6, 18)]
            + [f"layer{i}" for i in range(18, 24)],
        )

    def test_leaves_other_models_unchanged(self):
        states = cast(Any, tuple(range(25)))
        plain = SimpleNamespace(num_hidden_layers=24)
        self.assertIs(_collapse_looped_residual_streams(states, plain), states)
        single_pass = SimpleNamespace(
            num_hidden_layers=24,
            loop_layer_index_start=6,
            loop_layer_index_end=18,
            loop_num_passes=1,
        )
        self.assertIs(_collapse_looped_residual_streams(states, single_pass), states)


if __name__ == "__main__":
    unittest.main()
