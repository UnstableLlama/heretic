# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""CPU-only tests for the self-calibrated KL scorer and its backend helpers."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn.functional as F
from optuna import TrialPruned

from heretic.config import Settings
from heretic.scorers.self_calibrated_kl import (
    SelfCalibratedKL,
    find_answer_start,
    log1mexp,
    position_kl_divergence,
    summarize_distribution,
)
from heretic.scorers.self_calibrated_kl import Settings as ScorerSettings
from heretic.utils import (
    Prompt,
    build_teacher_forced_batch,
    slice_response_logits,
    trim_response_token_ids,
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


def exact_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits_p.float(), dim=-1)
    logq = F.log_softmax(logits_q.float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


class HelperTests(unittest.TestCase):
    def test_trim_cuts_after_first_eos_and_keeps_it(self):
        self.assertEqual(trim_response_token_ids([5, 6, 2, 7, 2], {2}, 0), [5, 6, 2])

    def test_trim_strips_trailing_padding_without_eos(self):
        self.assertEqual(trim_response_token_ids([5, 6, 0, 0], {2}, 0), [5, 6])
        self.assertEqual(trim_response_token_ids([0, 0], {2}, 0), [])

    def test_teacher_forced_batch_is_left_padded_with_response_last(self):
        input_ids, mask = build_teacher_forced_batch(
            [[1, 2], [3]], [[10, 11, 12], [20]], 99
        )
        self.assertEqual(input_ids.tolist(), [[1, 2, 10, 11, 12], [99, 99, 99, 3, 20]])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1, 1], [0, 0, 0, 1, 1]])

    def test_slice_returns_positions_predicting_each_response_token(self):
        # Encode the absolute position into the logits so the slice can be checked.
        length, vocab = 7, 3
        full = torch.arange(length).view(1, length, 1).expand(2, length, vocab).float()
        slices = slice_response_logits(full, [3, 1])
        # Row 0: response at positions 4, 5, 6 -> predicted by positions 3, 4, 5.
        self.assertEqual(slices[0][:, 0].tolist(), [3.0, 4.0, 5.0])
        # Row 1: response at position 6 -> predicted by position 5.
        self.assertEqual(slices[1][:, 0].tolist(), [5.0])

    def test_slice_works_on_kept_tail_only(self):
        kept = torch.arange(4).view(1, 4, 1).expand(1, 4, 2).float() + 10
        (sliced,) = slice_response_logits(kept, [3])
        self.assertEqual(sliced[:, 0].tolist(), [10.0, 11.0, 12.0])
        with self.assertRaises(ValueError):
            slice_response_logits(kept, [4])

    def test_log1mexp_matches_naive_formula_away_from_zero(self):
        x = torch.tensor([-0.01, -0.5, -1.0, -5.0, -40.0])
        torch.testing.assert_close(log1mexp(x), torch.log(1 - torch.exp(x)))
        self.assertTrue(torch.isneginf(log1mexp(torch.tensor([0.0]))).all())

    def test_find_answer_start_locates_marker(self):
        tokens = {1: "think ", 2: "more ", 3: "</think>", 4: " answer"}

        def decode(ids):
            return "".join(tokens[i] for i in ids)

        self.assertEqual(find_answer_start([1, 2, 3, 4, 4], decode, ["</think>"]), 3)
        self.assertEqual(find_answer_start([4, 4], decode, ["</think>"]), 0)
        self.assertEqual(find_answer_start([1, 2, 3], decode, []), 0)


class KLMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vocab = 50
        self.p_logits = torch.randn(6, self.vocab) * 3
        self.q_logits = self.p_logits + torch.randn(6, self.vocab) * 0.5

    def test_exact_mode_matches_reference_kl(self):
        logprobs, ids, tail = summarize_distribution(self.p_logits, top_k=0)
        self.assertIsNone(ids)
        self.assertIsNone(tail)
        kl = position_kl_divergence(logprobs, ids, tail, self.q_logits)
        torch.testing.assert_close(kl, exact_kl(self.p_logits, self.q_logits))

    def test_top_k_covering_vocabulary_is_exact(self):
        logprobs, ids, tail = summarize_distribution(self.p_logits, top_k=self.vocab)
        self.assertIsNone(ids)
        kl = position_kl_divergence(logprobs, ids, tail, self.q_logits)
        torch.testing.assert_close(kl, exact_kl(self.p_logits, self.q_logits))

    def test_coarsened_kl_is_lower_bound_and_tight_for_large_k(self):
        exact = exact_kl(self.p_logits, self.q_logits)
        for k in (2, 8, 32, 49):
            logprobs, ids, tail = summarize_distribution(self.p_logits, top_k=k)
            kl = position_kl_divergence(logprobs, ids, tail, self.q_logits)
            self.assertTrue((kl <= exact + 1e-5).all(), f"k={k}")
            self.assertTrue((kl >= 0).all(), f"k={k}")
        logprobs, ids, tail = summarize_distribution(self.p_logits, top_k=49)
        kl = position_kl_divergence(logprobs, ids, tail, self.q_logits)
        torch.testing.assert_close(kl, exact, atol=5e-3, rtol=5e-2)

    def test_identical_distributions_give_zero(self):
        logprobs, ids, tail = summarize_distribution(self.p_logits, top_k=8)
        kl = position_kl_divergence(logprobs, ids, tail, self.p_logits)
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-5, rtol=0)

    def test_mass_moved_outside_top_k_is_detected(self):
        logprobs, ids, tail = summarize_distribution(self.p_logits, top_k=4)
        # Build a modified distribution that dumps most of its mass on tokens
        # the original model considered unlikely.
        outside = torch.ones(self.vocab, dtype=torch.bool)
        outside[ids[0]] = False
        q = self.p_logits[0].clone()
        q[outside] += 8.0
        kl = position_kl_divergence(logprobs[:1], ids[:1], tail[:1], q.unsqueeze(0))
        self.assertGreater(kl.item(), 1.0)


class FakeContext:
    """Minimal plugin context backed by a deterministic toy 'model'."""

    def __init__(self, vocab=12, heretic_settings=None, perturbation=0.0, nan=False):
        self.vocab = vocab
        self.perturbation = perturbation
        self.nan = nan
        self._settings = heretic_settings
        self._model = SimpleNamespace(
            tokenizer=SimpleNamespace(
                batch_decode=lambda batches, skip_special_tokens=False: [
                    "".join(f"<{i}>" if i != 3 else "</think>" for i in ids)
                    for ids in batches
                ]
            )
        )
        self.logit_calls = 0
        self.generation_calls = 0

    def load_prompts(self, specification):
        return [Prompt(system="s", user=f"user {i}") for i in range(5)]

    def get_model(self):
        return self._model

    def get_response_token_ids(self, prompts, max_new_tokens):
        self.generation_calls += 1
        out = []
        for prompt in prompts:
            index = int(prompt.user.split()[-1])
            if index == 4:
                out.append([])  # empty response: must be dropped
            else:
                out.append([1, 2, 3, 4, 5][: index + 2])
        return out

    def _logits_for(self, prompt, token_ids):
        generator = torch.Generator().manual_seed(hash(prompt.user) % 1000)
        logits = torch.randn(len(token_ids), self.vocab, generator=generator) * 2
        logits = logits + self.perturbation
        if self.nan:
            logits[0, 0] = float("nan")
        return logits

    def get_response_logits(self, prompts, response_token_ids):
        self.logit_calls += 1
        return [self._logits_for(p, ids) for p, ids in zip(prompts, response_token_ids)]


def make_scorer(**overrides):
    overrides.setdefault("cache_dir", "")
    heretic_settings = IsolatedSettings(
        model="unused", batch_size=2, max_response_length=8
    )
    scorer = SelfCalibratedKL(
        heretic_settings=heretic_settings,
        settings=ScorerSettings(**overrides),
    )
    return scorer, heretic_settings


class ScorerTests(unittest.TestCase):
    def test_baseline_is_zero_and_unchanged_model_scores_zero(self):
        scorer, settings = make_scorer(top_k=5)
        ctx = FakeContext(heretic_settings=settings)
        scorer.init(ctx)
        self.assertEqual(len(scorer._references), 4)
        self.assertEqual(scorer.get_baseline_score(ctx).value, 0)
        score = scorer.get_score(ctx)
        self.assertAlmostEqual(score.value, 0.0, places=5)
        self.assertIn("thinking", score.rich_display)

    def test_perturbed_model_scores_positive_and_prunes_on_nan(self):
        scorer, settings = make_scorer(top_k=0)
        ctx = FakeContext(heretic_settings=settings)
        scorer.init(ctx)
        perturbed = FakeContext(heretic_settings=settings)
        # Perturb only the first token's logit so the distributions differ.
        original = perturbed._logits_for

        def shifted(prompt, token_ids):
            logits = original(prompt, token_ids)
            logits[:, 0] += 3.0
            return logits

        perturbed._logits_for = shifted
        self.assertGreater(scorer.get_score(perturbed).value, 0.01)

        broken = FakeContext(heretic_settings=settings, nan=True)
        with self.assertRaises(TrialPruned):
            scorer.get_score(broken)

    def test_batch_size_is_derived_from_budget_and_capped(self):
        scorer, settings = make_scorer(top_k=0)
        ctx = FakeContext(heretic_settings=settings)
        scorer.init(ctx)
        # Toy vocabulary is tiny, so the budget allows a huge batch; it must
        # be capped by the global batch size (2).
        self.assertEqual(scorer._batch_size, 2)

    def test_explicit_batch_size_is_respected(self):
        scorer, settings = make_scorer(top_k=0, batch_size=1)
        ctx = FakeContext(heretic_settings=settings)
        scorer.init(ctx)
        self.assertEqual(scorer._batch_size, 1)
        self.assertEqual(ctx.logit_calls, 4)

    def test_reference_responses_are_cached_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer, settings = make_scorer(top_k=4, cache_dir=directory)
            ctx = FakeContext(heretic_settings=settings)
            scorer.init(ctx)
            self.assertEqual(ctx.generation_calls, 3)
            first = [r.response_token_ids for r in scorer._references]
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)

            again, _ = make_scorer(top_k=4, cache_dir=directory)
            ctx2 = FakeContext(heretic_settings=settings)
            again.init(ctx2)
            self.assertEqual(ctx2.generation_calls, 0)
            self.assertEqual([r.response_token_ids for r in again._references], first)

            # A different generation length is a different corpus.
            other, _ = make_scorer(top_k=4, cache_dir=directory, max_response_tokens=3)
            ctx3 = FakeContext(heretic_settings=settings)
            other.init(ctx3)
            self.assertEqual(ctx3.generation_calls, 3)

    def test_scorer_contract(self):
        SelfCalibratedKL.validate_contract()
        self.assertIs(SelfCalibratedKL.get_settings_model(), ScorerSettings)


class HFBackendTests(unittest.TestCase):
    def test_get_response_logits_passes_positions_mask_and_kept_logits(self):
        from heretic.model import Model

        model = Model.__new__(Model)
        model.settings = IsolatedSettings(model="unused")
        vocab = 7
        pad = 0

        class Tokenizer:
            pad_token_id = pad
            eos_token_id = 2

            def apply_chat_template(self, chats, **kwargs):
                return [f"<{c[1]['content']}>" for c in chats]

            def __call__(self, texts, **kwargs):
                # First prompt is 3 tokens, second is 1, left-padded.
                return {
                    "input_ids": torch.tensor([[11, 12, 13], [pad, pad, 21]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [0, 0, 1]]),
                }

        model.tokenizer = Tokenizer()
        captured = {}

        def forward(input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
            captured.update(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                logits_to_keep=logits_to_keep,
            )
            batch, length = input_ids.shape
            logits = (
                torch.arange(length)
                .view(1, length, 1)
                .expand(batch, length, vocab)
                .float()
            )
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

        fake = Mock(side_effect=forward)
        fake.forward = forward
        fake.device = torch.device("cpu")
        model.model = fake

        slices = model.get_response_logits(
            [Prompt(system="", user="a"), Prompt(system="", user="b")],
            [[31, 32], [41, 42, 43]],
        )

        # Rows: [11 12 13 31 32] and [pad 21 41 42 43] (left padded to 5).
        self.assertEqual(
            captured["input_ids"].tolist(),
            [[11, 12, 13, 31, 32], [pad, 21, 41, 42, 43]],
        )
        self.assertEqual(
            captured["attention_mask"].tolist(), [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]
        )
        self.assertEqual(
            captured["position_ids"].tolist(), [[0, 1, 2, 3, 4], [0, 0, 1, 2, 3]]
        )
        self.assertEqual(captured["logits_to_keep"], 4)
        # Response tokens at positions 3,4 (row 0) and 2,3,4 (row 1) are
        # predicted by positions 2,3 and 1,2,3 respectively.
        self.assertEqual(slices[0][:, 0].tolist(), [2.0, 3.0])
        self.assertEqual(slices[1][:, 0].tolist(), [1.0, 2.0, 3.0])

    def test_get_response_token_ids_trims_at_eos(self):
        from heretic.model import Model

        model = Model.__new__(Model)
        model.settings = IsolatedSettings(model="unused")
        model.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        model.model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[2, 9])
        )
        model.generate = Mock(
            return_value=(
                {"input_ids": torch.zeros(2, 3, dtype=torch.long)},
                torch.tensor([[5, 5, 5, 7, 9, 0, 0], [5, 5, 5, 7, 8, 0, 0]]),
            )
        )
        self.assertEqual(
            model.get_response_token_ids([Prompt(system="", user="x")] * 2, 4),
            [[7, 9], [7, 8]],
        )


class Exl3BackendTests(unittest.TestCase):
    def test_get_response_logits_requests_tail_logits_and_crops_vocab(self):
        from heretic.exl3_model import Exl3Model

        model = Exl3Model.__new__(Exl3Model)
        model.settings = IsolatedSettings(model="unused", quantization="exl3")
        model.config = SimpleNamespace(vocab_size=5)

        class HF:
            pad_token_id = 0

            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]]),
                    "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
                }

        model.tokenizer = SimpleNamespace(_ensure_hf=lambda: HF())
        model._render_chat_prompts = Mock(return_value=["a", "b"])
        captured = {}

        def forward(
            input_ids, *, last_only=False, last_tokens=None, capture_residuals=False
        ):
            captured.update(input_ids=input_ids, last_tokens=last_tokens)
            batch, length = input_ids.shape
            # Padded vocabulary of 8 columns, model vocab is 5.
            logits = (
                torch.arange(length).view(1, length, 1).expand(batch, length, 8).float()
            )
            return logits[:, -last_tokens:], []

        model._forward = forward
        slices = model.get_response_logits(
            [Prompt(system="", user="a"), Prompt(system="", user="b")],
            [[31], [41, 42]],
        )
        self.assertEqual(
            captured["input_ids"].tolist(), [[0, 0, 11, 12, 31], [21, 22, 23, 41, 42]]
        )
        self.assertEqual(captured["last_tokens"], 3)
        self.assertEqual(slices[0].shape, (1, 5))
        self.assertEqual(slices[1].shape, (2, 5))
        self.assertEqual(slices[0][:, 0].tolist(), [3.0])
        self.assertEqual(slices[1][:, 0].tolist(), [2.0, 3.0])

    def test_get_response_token_ids_appends_eos_on_stop_token(self):
        from heretic.exl3_model import Exl3Model

        model = Exl3Model.__new__(Exl3Model)
        model.settings = IsolatedSettings(model="unused", quantization="exl3")
        generator = Mock()
        generator.generate.return_value = (
            ["hello", "cut off"],
            [{"eos_reason": "stop_token"}, {"eos_reason": "max_new_tokens"}],
        )
        model._ensure_generator = Mock(return_value=generator)
        model._render_chat_prompts = Mock(return_value=["p1", "p2"])
        model._greedy_sampler = Mock(return_value="greedy")
        hf = SimpleNamespace(
            eos_token_id=2,
            encode=lambda text, add_special_tokens=False: [len(text)],
        )
        model.tokenizer = SimpleNamespace(_ensure_hf=lambda: hf)
        self.assertEqual(
            model.get_response_token_ids([Prompt(system="", user="x")] * 2, 16),
            [[5, 2], [7]],
        )
        kwargs = generator.generate.call_args.kwargs
        self.assertTrue(kwargs["return_last_results"])
        # Without explicit stop conditions exllamav3 never stops on EOS.
        self.assertEqual(kwargs["stop_conditions"], [2])

    def test_reset_model_reloads_when_model_path_changes(self):
        from heretic.exl3_model import Exl3Model

        model = Exl3Model.__new__(Exl3Model)
        model.settings = IsolatedSettings(model="/other/model", quantization="exl3")
        model._loaded_model_path = "/loaded/model"
        model._reload_model = Mock()
        self.assertFalse(model.reset_model())
        model._reload_model.assert_called_once_with()

    def test_get_response_token_ids_appends_triggering_stop_token(self):
        from heretic.exl3_model import Exl3Model

        model = Exl3Model.__new__(Exl3Model)
        model.settings = IsolatedSettings(model="unused", quantization="exl3")
        model.config = SimpleNamespace(eos_token_id=2, eos_token_id_list=[2, 9])
        generator = Mock()
        generator.generate.return_value = (
            ["hello"],
            [{"eos_reason": "stop_token", "eos_triggering_token_id": 9}],
        )
        model._ensure_generator = Mock(return_value=generator)
        model._render_chat_prompts = Mock(return_value=["p1"])
        model._greedy_sampler = Mock(return_value="greedy")
        hf = SimpleNamespace(
            eos_token_id=2,
            encode=lambda text, add_special_tokens=False: [len(text)],
        )
        model.tokenizer = SimpleNamespace(_ensure_hf=lambda: hf, eos_token_id=2)
        self.assertEqual(
            model.get_response_token_ids([Prompt(system="", user="x")], 16),
            [[5, 9]],
        )
        kwargs = generator.generate.call_args.kwargs
        self.assertEqual(kwargs["stop_conditions"], [2, 9])
        self.assertTrue(kwargs["decode_special_tokens"])


if __name__ == "__main__":
    unittest.main()
