"""GPU integration test for the EXL3 backend's runtime LoRA layout.

Loads a tiny synthetic 4-bit EXL3 Ricursive checkpoint (built with the
exllamav3 fork's ``tests/testlib``) through ``Exl3Model``, writes a rank-1
adapter the way directional abliteration does and a rank-8 adapter the way
ARA-LoRA does, and checks the forward output against a dense fp16 reference
model whose weights carry the LoRA delta folded in. The exported PEFT adapter
is checked numerically and by reloading it through exllamav3's own LoRA
loader.

exllamav3 stores LoRA A as ``(rank, in)`` (PEFT's orientation) and B as
``(rank, out)`` and applies ``x @ A.T @ B``; the dense fold is therefore
``W (in, out) += A.T @ B``.

Skipped unless exllamav3 (with its CUDA extension), a CUDA device and the
fork's ``tests/testlib`` are available. Set ``HERETIC_EXL3_TEST_DIR`` to
choose where the checkpoints are written (default: a fresh temp dir under
``TMPDIR``); ``HERETIC_EXL3_FORK_TESTS`` overrides the fork tests dir.
"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from heretic.config import Settings
from heretic.exl3_model import Exl3Model
from heretic.modifiers.abliteration import RowNormalization, WeightDistribution
from heretic.modifiers.ara import Parameters as ARAParameters
from heretic.modifiers.ara import Settings as ARASettings

# Resolved dynamically (like exl3_model.py does) so this module imports and
# type-checks cleanly where exllamav3 is not installed.
try:
    EXL3: Any = importlib.import_module("exllamav3")
except Exception:  # pragma: no cover - import side effects vary
    EXL3 = None

VOCAB = 512


def fork_tests_dir() -> Path | None:
    override = os.environ.get("HERETIC_EXL3_FORK_TESTS")
    if override:
        candidates = [Path(override)]
    elif EXL3 is not None:
        candidates = [Path(EXL3.__file__).resolve().parents[1] / "tests"]
    else:
        candidates = []
    for candidate in candidates:
        if (candidate / "testlib" / "tiny_models.py").is_file():
            return candidate
    return None


FORK_TESTS = fork_tests_dir()
SKIP_REASON = (
    None
    if EXL3 is not None and torch.cuda.is_available() and FORK_TESTS is not None
    else "needs exllamav3 with CUDA and the fork's tests/testlib"
)


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


def write_tiny_tokenizer(model_dir: Path) -> None:
    """A WordLevel tokenizer over the tiny vocab (words are ``t<id>``; ids 0..3
    are special tokens), enough for ``Tokenizer.from_config`` to load."""
    tokenizers = importlib.import_module("tokenizers")
    HFTokenizer = tokenizers.Tokenizer
    models, pre_tokenizers, decoders = (
        tokenizers.models,
        tokenizers.pre_tokenizers,
        tokenizers.decoders,
    )

    specials = ["<pad>", "<bos>", "<eos>", "<eot>"]
    vocab = {s: i for i, s in enumerate(specials)}
    vocab["▁"] = 4
    vocab["▁t"] = 5
    vocab.update({f"▁t{i}": i for i in range(6, VOCAB)})
    tok = HFTokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.add_special_tokens(specials)
    tok.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "added_tokens_decoder": {
                    str(vocab[s]): {"content": s, "special": True} for s in specials
                },
            }
        )
    )


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class Exl3LoraLayoutGpuTests(unittest.TestCase):
    """Rank-1 (abliteration) and rank-8 (ARA-LoRA) adapters on a real EXL3
    model must equal a dense reference with ``A.T @ B`` folded in."""

    # Logit agreement between the EXL3 kernels and an fp16 dense matmul on
    # the dequantized weights; the adapter deltas are forced well above it.
    ATOL = 3e-2
    RTOL = 2e-2

    @classmethod
    def setUpClass(cls):
        if str(FORK_TESTS) not in sys.path:
            sys.path.insert(0, str(FORK_TESTS))
        make_ricursive_checkpoint = importlib.import_module(
            "testlib.tiny_models"
        ).make_ricursive_checkpoint

        base = os.environ.get("HERETIC_EXL3_TEST_DIR")
        if base:
            cls.work_dir = Path(base)
            cls.work_dir.mkdir(parents=True, exist_ok=True)
            cls._cleanup = None
        else:
            cls._cleanup = tempfile.TemporaryDirectory(prefix="heretic-lorafix-")
            cls.work_dir = Path(cls._cleanup.name)

        cls.model_dir = Path(
            make_ricursive_checkpoint(str(cls.work_dir / "ricursive_exl3"), exl3_bits=4)
        )
        write_tiny_tokenizer(cls.model_dir)

        cls.settings = IsolatedSettings(
            model=str(cls.model_dir),
            quantization="exl3",
            batch_size=1,
            seed=0,
            exl3_max_num_tokens=2048,
        )
        cls.backend = Exl3Model(cls.settings)
        cls.targets = [
            module
            for per_layer in cls.backend._layer_modules
            for modules in per_layer.values()
            for module in modules
        ]
        cls.ids = [
            torch.randint(
                4, VOCAB, (2, 40), generator=torch.Generator().manual_seed(1)
            ),
            torch.randint(4, VOCAB, (1, 1), generator=torch.Generator().manual_seed(2)),
        ]

    @classmethod
    def tearDownClass(cls):
        model = getattr(cls.backend, "model", None)
        if model is not None:
            model.unload()
        if cls._cleanup is not None:
            cls._cleanup.cleanup()

    # -- helpers ----------------------------------------------------------

    def forward(self, model=None) -> list[torch.Tensor]:
        model = self.backend.model if model is None else model
        out = []
        with torch.inference_mode():
            for ids in self.ids:
                out.append(model.forward(ids, params={}).float().cpu())
        return out

    def dense_reference(self, name: str, deltas: dict[str, torch.Tensor]):
        """Write a dense fp16 copy of the loaded EXL3 checkpoint with
        ``deltas[key]`` (in_u, out_u) added to each named Linear, load it with
        exllamav3 and return (model, unload)."""
        Config = importlib.import_module("exllamav3.model.config").Config
        Model = importlib.import_module("exllamav3.model.model").Model
        Linear = importlib.import_module("exllamav3.modules.linear").Linear

        tensors = load_file(str(self.model_dir / "model.safetensors"))
        for module in self.backend.model:
            if not isinstance(module, Linear) or module.quant_type != "exl3":
                continue
            for key in [k for k in tensors if k.startswith(module.key + ".")]:
                del tensors[key]
            w = module.inner.get_weight_tensor().float()  # (in_p, out_p)
            delta = deltas.get(module.key)
            if delta is not None:
                w[: delta.shape[0], : delta.shape[1]] += delta.to(w.device)
            tensors[module.key + ".weight"] = w.T.contiguous().half().cpu()

        out_dir = self.work_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_dir / "model.safetensors"))
        shutil.copy(self.model_dir / "config.json", out_dir / "config.json")

        cfg = Config.from_directory(str(out_dir))
        cfg.override_dynamic_seq_len(256)
        model = Model.from_config(cfg)
        model.load(torch.device(self.targets[0].device))
        return model

    def slot_deltas(self) -> dict[str, torch.Tensor]:
        """The (in_u, out_u) delta each slot encodes: A.T @ B on the unpadded slice."""
        deltas = {}
        for module in self.targets:
            a = module.lora_a_tensors[self.backend._lora_key]
            b = module.lora_b_tensors[self.backend._lora_key]
            in_u, out_u = module.in_features_unpadded, module.out_features_unpadded
            deltas[module.key] = a[:, :in_u].float().T @ b[:, :out_u].float()
        return deltas

    def assert_matches_dense(self, name: str, base: list[torch.Tensor]):
        deltas = self.slot_deltas()
        got = self.forward()
        ref_model = self.dense_reference(name, deltas)
        try:
            ref = self.forward(ref_model)
        finally:
            ref_model.unload()
        for g, r, b in zip(got, ref, base):
            self.assertGreater(
                (r - b).abs().max().item(),
                10 * self.ATOL,
                "the adapter must change the output well beyond the tolerance",
            )
            torch.testing.assert_close(g, r, atol=self.ATOL, rtol=self.RTOL)

    # -- tests ------------------------------------------------------------

    def test_01_targets_and_slots(self):
        self.assertEqual(len(self.backend._layer_modules), 3)
        self.assertEqual(len(self.targets), 6)
        for module in self.targets:
            self.assertEqual(module.quant_type, "exl3")
            a = module.lora_a_tensors[self.backend._lora_key]
            b = module.lora_b_tensors[self.backend._lora_key]
            self.assertEqual(tuple(a.shape), (1, module.in_features))
            self.assertEqual(tuple(b.shape), (1, module.out_features_unpadded))
            self.assertEqual(a.dtype, torch.float16)

    def test_02_no_adapter_matches_dense(self):
        """Noise floor: zero slots vs the dequantized dense model."""
        self.backend.apply_lora(1)
        self.assertTrue(self.backend.reset_model())
        got = self.forward()
        ref_model = self.dense_reference("dense_base", {})
        try:
            ref = self.forward(ref_model)
        finally:
            ref_model.unload()
        for g, r in zip(got, ref):
            self.assertTrue(torch.isfinite(g).all())
            torch.testing.assert_close(g, r, atol=self.ATOL, rtol=self.RTOL)

    def test_03_rank1_abliteration_matches_dense(self):
        self.backend.apply_lora(1)
        self.assertTrue(self.backend.reset_model())
        base = self.forward()

        hidden = self.backend.config.hidden_size
        gen = torch.Generator().manual_seed(3)
        directions = torch.nn.functional.normalize(
            torch.randn(len(self.backend._layer_modules) + 1, hidden, generator=gen),
            dim=-1,
        )
        # Strong weights so a random direction on a random model moves the
        # logits well past the kernel-vs-dense noise floor.
        dist = WeightDistribution(
            max_weight=4.0,
            max_weight_position=1.0,
            min_weight=2.0,
            min_weight_distance=2.0,
        )
        weights = dict.fromkeys(self.backend.get_modifiable_components(), dist)
        self.backend.abliterate(directions, None, weights, RowNormalization.NONE)

        # Each slot is the rank-1 update (W v)(-k v)^T on the unpadded slice.
        for layer, per_layer in enumerate(self.backend._layer_modules):
            for modules in per_layer.values():
                for module in modules:
                    a = module.lora_a_tensors[self.backend._lora_key]
                    self.assertEqual(tuple(a.shape), (1, module.in_features))
                    in_u, out_u = (
                        module.in_features_unpadded,
                        module.out_features_unpadded,
                    )
                    w = self.backend._dequantize_weight(module)
                    v = directions[layer + 1].to(w.device)[:out_u]
                    distance = abs(layer - dist.max_weight_position)
                    k = dist.max_weight + distance / dist.min_weight_distance * (
                        dist.min_weight - dist.max_weight
                    )
                    expected = torch.outer(w @ v, -k * v)
                    torch.testing.assert_close(
                        self.slot_deltas()[module.key].to(w.device),
                        expected,
                        atol=5e-3,
                        rtol=1e-2,
                    )
                    self.assertTrue((a[:, in_u:] == 0).all())

        self.assert_matches_dense("dense_rank1", base)

    def test_04_rank8_ara_lora_matches_dense(self):
        self.backend.apply_lora(8)
        self.assertTrue(self.backend.reset_model())
        base = self.forward()

        gen = torch.Generator().manual_seed(4)
        good, bad = [], []
        for per_layer in self.backend._layer_modules:
            good_layer, bad_layer = {}, {}
            for component, modules in per_layer.items():
                good_layer[component], bad_layer[component] = {}, {}
                for index, module in enumerate(modules):
                    w = self.backend._dequantize_weight(module).cpu()
                    good_x = torch.randn(64, module.in_features_unpadded, generator=gen)
                    bad_x = (
                        torch.randn(64, module.in_features_unpadded, generator=gen)
                        + 1.0
                    )
                    good_layer[component][index] = (good_x, good_x @ w)
                    bad_layer[component][index] = (bad_x, bad_x @ w)
            good.append(good_layer)
            bad.append(bad_layer)

        with torch.no_grad():
            self.backend.ara_lora_abliterate(
                good,
                bad,
                ARAParameters(
                    start_layer_index=0,
                    end_layer_index=len(self.backend._layer_modules),
                    preserve_good_behavior_weight=1.0,
                    steer_bad_behavior_weight=1.0,
                    overcorrect_relative_weight=0.0,
                    neighbor_count=15,
                ),
                ARASettings(
                    lora_rank=8,
                    preserve_row_magnitudes=False,
                    n_optimization_steps=2,
                    max_iter=10,
                ),
            )

        for module in self.targets:
            a = module.lora_a_tensors[self.backend._lora_key]
            b = module.lora_b_tensors[self.backend._lora_key]
            self.assertEqual(tuple(a.shape), (8, module.in_features))
            self.assertEqual(tuple(b.shape), (8, module.out_features_unpadded))
            self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
            self.assertGreater(b.abs().max().item(), 0.0, module.key)

        self.assert_matches_dense("dense_rank8", base)

    def test_05_export_roundtrips_through_exllamav3_lora(self):
        """save_adapter writes PEFT (rank, in) / (out, rank) tensors whose
        delta equals the slot's, and exllamav3's loader reproduces the
        forward from the exported files alone."""
        LoRA = importlib.import_module("exllamav3.model.lora").LoRA

        self.backend.apply_lora(8)
        self.assertTrue(self.backend.reset_model())
        gen = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for module in self.targets:
                a = module.lora_a_tensors[self.backend._lora_key]
                b = module.lora_b_tensors[self.backend._lora_key]
                in_u, out_u = module.in_features_unpadded, module.out_features_unpadded
                a[:, :in_u] = (torch.randn(8, in_u, generator=gen) * 0.05).half()
                b[:, :out_u] = (torch.randn(8, out_u, generator=gen) * 0.05).half()
        deltas = self.slot_deltas()
        with_slots = self.forward()

        adapter_dir = self.work_dir / "adapter_rank8"
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        self.backend.save_adapter(str(adapter_dir))

        config = json.loads((adapter_dir / "adapter_config.json").read_text())
        self.assertEqual(config["r"], 8)
        self.assertEqual(config["lora_alpha"], 8)
        tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
        for module in self.targets:
            in_u, out_u = module.in_features_unpadded, module.out_features_unpadded
            a_peft = tensors[f"base_model.model.{module.key}.lora_A.weight"]
            b_peft = tensors[f"base_model.model.{module.key}.lora_B.weight"]
            self.assertEqual(tuple(a_peft.shape), (8, in_u))
            self.assertEqual(tuple(b_peft.shape), (out_u, 8))
            # PEFT's delta on the HF (out, in) weight is B @ A, i.e. the
            # transpose of the (in, out) slot delta.
            torch.testing.assert_close(
                (b_peft.float() @ a_peft.float()),
                deltas[module.key].T.cpu(),
                atol=1e-3,
                rtol=1e-3,
            )

        # Reload through exllamav3's PEFT loader with heretic's slots zeroed.
        self.assertTrue(self.backend.reset_model())
        lora = LoRA.from_directory(self.backend.model, str(adapter_dir))
        try:
            self.assertEqual(len(lora.target_modules), len(self.targets))
            reloaded = self.forward()
        finally:
            lora.unload()
        for g, r in zip(reloaded, with_slots):
            torch.testing.assert_close(g, r, atol=self.ATOL, rtol=self.RTOL)
        # And detaching restores the base output.
        detached = self.forward()
        base = self.forward()
        for d, b in zip(detached, base):
            torch.testing.assert_close(d, b, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
