"""CPU-only EXL3 integration tests; never import EXL3's CUDA extension."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from pydantic import ValidationError
from safetensors.torch import load_file

from heretic.config import Settings, SingleDatasetSpecification
from heretic.exl3_model import Exl3Model
from heretic.modifier import load_and_init_modifiers
from heretic.modifiers.abliteration import (
    Abliteration,
    RowNormalization,
    WeightDistribution,
)
from heretic.modifiers.abliteration import Parameters as AbliterationParameters
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.modifiers.ara import ARA, Parameters, mean_distances_to_knn
from heretic.modifiers.ara import Settings as ARASettings
from heretic.plugin import Context
from heretic.utils import Prompt


class IsolatedSettings(Settings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (init_settings,)


def settings(**kwargs):
    return IsolatedSettings(model="unused", quantization="exl3", **kwargs)


def backend(trim=True):
    model = Exl3Model.__new__(Exl3Model)
    model.settings = settings(batch_size=1, seed=42)
    model._lora_key = object()
    model._adapter_rank = 1
    model._generator = None
    module = SimpleNamespace(
        key="model.layers.0.self_attn.o_proj",
        device=torch.device("cpu"),
        in_features=4,
        in_features_unpadded=3,
        out_features=4,
        out_features_unpadded=2,
        trim_padded_out=trim,
        quant_type="fp16",
        inner=SimpleNamespace(weight=torch.arange(16).reshape(4, 4).half() / 16),
        lora_a_tensors={},
        lora_b_tensors={},
    )
    model._layer_modules = [{"attn.o_proj": [module]}]
    model._allocate_lora_slot(module)
    return model, module


class LegacyConfigTests(unittest.TestCase):
    def test_legacy_ara_maps_to_plugin_and_preserves_defaults(self):
        config = settings(
            use_ara=False,
            use_ara_lora=True,
            row_normalization="none",
            invert_target=True,
            good_prompts={"dataset": "good.txt"},
            bad_prompts={"dataset": "bad.txt"},
        )
        self.assertEqual(config.modifiers[0].plugin, "heretic.modifiers.ara.ARA")
        raw = config.model_extra["modifier"]["ARA"]
        plugin_settings = ARASettings.model_validate(raw)
        self.assertEqual(plugin_settings.lora_rank, 128)
        self.assertEqual(plugin_settings.steer_bad_behavior_weight_max, 0.001)
        self.assertTrue(plugin_settings.invert_target)
        self.assertFalse(plugin_settings.preserve_row_magnitudes)
        assert isinstance(plugin_settings.good_prompts, SingleDatasetSpecification)
        self.assertEqual(plugin_settings.good_prompts.dataset, "good.txt")
        self.assertNotIn("use_ara", config.model_dump())
        self.assertNotIn("good_prompts", config.model_dump())

    def test_explicit_plugin_settings_win(self):
        config = settings(
            use_ara_lora=True,
            ara_lora_rank=128,
            modifiers=[
                {"plugin": "heretic.modifiers.ara.ARA", "instance_name": "custom"}
            ],
            modifier={
                "ARA_custom": {"lora_rank": 8, "steer_bad_behavior_weight_max": 0.5}
            },
        )
        self.assertEqual(config.model_extra["modifier"]["ARA_custom"]["lora_rank"], 8)
        self.assertEqual(
            config.model_extra["modifier"]["ARA_custom"][
                "steer_bad_behavior_weight_max"
            ],
            0.5,
        )

    def test_legacy_false_strings_select_directional(self):
        config = settings(
            use_ara="false", use_ara_lora="false", row_normalization="none"
        )
        self.assertEqual(
            config.modifiers[0].plugin, "heretic.modifiers.abliteration.Abliteration"
        )
        self.assertEqual(
            config.model_extra["modifier"]["Abliteration"]["row_normalization"], "none"
        )

    def test_new_config_uses_upstream_defaults(self):
        config = settings()
        self.assertEqual(config.modifiers[0].plugin, "heretic.modifiers.ara.ARA")
        self.assertNotIn("modifier", config.model_extra)

    def test_steering_range_is_validated(self):
        with self.assertRaises(ValidationError):
            ARASettings(
                steer_bad_behavior_weight_min=1, steer_bad_behavior_weight_max=0.1
            )

    def test_serialized_migration_roundtrip(self):
        config = settings(use_ara_lora=True, ara_lora_rank=16)
        restored = IsolatedSettings(**config.model_dump())
        self.assertEqual(restored.model_extra, config.model_extra)


class Exl3Tests(unittest.TestCase):
    def test_adapter_shape_tracks_trimmed_output_and_rank(self):
        for trim, width in [(True, 2), (False, 4)]:
            model, module = backend(trim)
            model.apply_lora(3)
            self.assertEqual(module.lora_a_tensors[model._lora_key].shape, (4, 3))
            self.assertEqual(module.lora_b_tensors[model._lora_key].shape, (3, width))

    def test_directional_update_matches_dense_reference(self):
        model, module = backend()
        v = torch.tensor([0.6, 0.8])
        model._write_lora_for_module(module, v, 0.7)
        a, b = (
            module.lora_a_tensors[model._lora_key],
            module.lora_b_tensors[model._lora_key],
        )
        expected = (
            -0.7 * (module.inner.weight[:3, :2].float() @ v)[:, None] @ v[None, :]
        )
        torch.testing.assert_close(
            (a @ b)[:3].float(), expected, atol=0.001, rtol=0.001
        )
        self.assertTrue((a[3] == 0).all())

    def test_sliced_reconstruction_uses_contiguous_final_slice(self):
        model, _ = backend()
        model.settings.exl3_reconstruct_slice_n = 256
        weight = torch.ones(128, 384, dtype=torch.float16) / 128
        offsets = []

        def had(source, target, pre, post, scale):
            target.copy_(source)
            if pre is not None:
                target.mul_(pre)
            if post is not None:
                target.mul_(post)

        def reconstruct(target, trellis, k, mcg, mul1, offset):
            self.assertTrue(target.is_contiguous())
            offsets.append(offset)
            target.copy_(weight[:, offset : offset + target.shape[1]])

        model._exl3_ext = Mock(
            return_value=SimpleNamespace(had_r_128=had, reconstruct_slice=reconstruct)
        )
        inner = SimpleNamespace(
            trellis=torch.empty(0),
            in_features=128,
            out_features=384,
            suh=torch.ones(128),
            svh=torch.ones(384),
            K=0,
            mcg=0,
            mul1=0,
        )
        result = model._ablation_a_exl3(inner, torch.ones(380), 120)
        torch.testing.assert_close(result, torch.full((120,), 380 / 128))
        self.assertEqual(offsets, [0, 256])

    def test_reset_clears_adapters_and_prefix_cache(self):
        model, module = backend()
        generator = Mock()
        model._generator = generator
        module.lora_a_tensors[model._lora_key].fill_(1)
        self.assertTrue(model.reset_model())
        self.assertTrue((module.lora_a_tensors[model._lora_key] == 0).all())
        generator.clear_queue.assert_called_once()
        generator.filter_pool.shutdown.assert_called_once_with(wait=True)
        self.assertIsNone(model._generator)

    def test_export_uses_active_rank_and_unpadded_shapes(self):
        model, _ = backend()
        model.apply_lora(3)
        model._is_multimodal = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory:
            model.save_adapter(directory)
            config = json.loads(Path(directory, "adapter_config.json").read_text())
            tensors = load_file(str(Path(directory, "adapter_model.safetensors")))
        self.assertEqual(config["r"], 3)
        self.assertEqual(config["lora_alpha"], 3)
        self.assertEqual({tuple(t.shape) for t in tensors.values()}, {(3, 3), (2, 3)})

    def test_capture_restores_hooks_on_failure(self):
        model, module = backend()
        original = Mock()
        module.forward = original
        model._tokenize_chat = Mock(side_effect=RuntimeError("tokenize failed"))
        with (
            patch("heretic.exl3_model._check_ram_guard"),
            self.assertRaisesRegex(RuntimeError, "tokenize failed"),
        ):
            model.get_module_io_batched([Prompt(system="", user="x")])
        self.assertIs(module.forward, original)

    def test_residual_quantile_argument_is_forwarded(self):
        model, _ = backend()
        model.get_residuals = Mock(return_value=torch.ones(1, 2, 3))
        prompts = [Prompt(system="", user="x"), Prompt(system="", user="y")]
        torch.testing.assert_close(
            model.get_residuals_mean(prompts, 0.9), torch.ones(2, 3)
        )
        self.assertEqual(model.get_residuals.call_count, 2)
        self.assertEqual(model.get_residuals.call_args.args[1], 0.9)

    def test_responses_are_not_retokenized_or_truncated(self):
        model, _ = backend()
        generator = Mock()
        generator.generate.return_value = ["short", "a longer completion"]
        model._ensure_generator = Mock(return_value=generator)
        model._render_chat_prompts = Mock(return_value=["prompt A", "prompt B"])
        model._greedy_sampler = Mock(return_value="greedy")
        self.assertEqual(
            model.get_responses([], True), ["short", "a longer completion"]
        )
        self.assertFalse(generator.generate.call_args.kwargs["add_bos"])
        self.assertTrue(generator.generate.call_args.kwargs["encode_special_tokens"])
        self.assertFalse(generator.generate.call_args.kwargs["decode_special_tokens"])

    def test_ara_runs_on_cpu_with_trimmed_padding_and_no_outer_grad(self):
        model, module = backend()
        model.apply_lora(2)
        good_x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        bad_x = torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
        weight = module.inner.weight.float()
        good = [{"attn.o_proj": {0: (good_x, good_x @ weight)}}]
        bad = [{"attn.o_proj": {0: (bad_x, bad_x @ weight)}}]
        with torch.no_grad():
            model.ara_lora_abliterate(
                good,
                bad,
                Parameters(0, 1, 1, 0.01, 0, 15),
                ARASettings(
                    lora_rank=2,
                    preserve_row_magnitudes=False,
                    n_optimization_steps=1,
                    max_iter=2,
                ),
            )
        a, b = (
            module.lora_a_tensors[model._lora_key],
            module.lora_b_tensors[model._lora_key],
        )
        self.assertTrue(torch.isfinite(a @ b).all())
        self.assertGreater((a @ b).abs().sum().item(), 0)
        self.assertTrue((a[3] == 0).all())

    def test_knn_caps_neighbors_for_sparse_experts(self):
        torch.testing.assert_close(
            mean_distances_to_knn(
                torch.tensor([[0.0]]), torch.tensor([[1.0], [3.0]]), 15
            ),
            torch.tensor([2.0]),
        )

    def test_plugin_dispatch_uses_backend_ara(self):
        model, _ = backend()
        model.get_module_io_batched = Mock(return_value=[{}])
        model.ara_lora_abliterate = Mock()
        config = settings(modifier={"ARA": {"lora_rank": 4}})
        with patch.object(Context, "load_prompts", return_value=[]):
            entry = load_and_init_modifiers(config, model)[0]
        self.assertIsInstance(entry.modifier, ARA)
        self.assertEqual(model._lora_rank(), 4)
        parameters = Parameters(0, 1, 1, 0.001, 0, 1)
        entry.modifier.modify_model(Context(config, model), parameters)
        self.assertEqual(model.ara_lora_abliterate.call_args.args[2], parameters)
        self.assertEqual(entry.modifier.parameters_class, Parameters)

    def test_directional_dispatch_and_early_normalization_validation(self):
        model, _ = backend()
        plugin = Abliteration(
            heretic_settings=model.settings, settings=AbliterationSettings()
        )
        with self.assertRaisesRegex(ValueError, "row_normalization"):
            plugin.init(Context(model.settings, model))
        plugin.settings = AbliterationSettings(row_normalization="none")
        plugin.residual_directions = torch.ones(2, 2)
        model.abliterate = Mock()
        params = AbliterationParameters(
            None, {"attn.o_proj": WeightDistribution(1, 0, 1, 1)}
        )
        plugin.modify_model(Context(model.settings, model), params)
        self.assertEqual(model.abliterate.call_args.args[3], RowNormalization.NONE)


if __name__ == "__main__":
    unittest.main()
