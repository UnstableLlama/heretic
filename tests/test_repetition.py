# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from isolated_settings import IsolatedSettings
from pydantic import ValidationError

from heretic.scorers.repetition import Repetition, Settings, max_consecutive_repeats
from heretic.utils import Prompt


class MaxConsecutiveRepeatsTests(unittest.TestCase):
    def test_text_without_repetition_scores_one(self) -> None:
        self.assertEqual(max_consecutive_repeats("the quick brown fox jumps"), 1)

    def test_empty_text_scores_zero(self) -> None:
        self.assertEqual(max_consecutive_repeats("   \n  "), 0)

    def test_counts_a_repeated_word(self) -> None:
        self.assertEqual(max_consecutive_repeats("wait " * 12), 12)

    def test_counts_a_repeated_sentence(self) -> None:
        text = "I need to reconsider this approach. " * 30
        self.assertEqual(max_consecutive_repeats(text), 30)

    def test_finds_a_run_inside_normal_text(self) -> None:
        text = "Okay, so " + "let me think " * 7 + "about what the user wants."
        self.assertEqual(max_consecutive_repeats(text), 7)

    def test_respects_the_ngram_bounds(self) -> None:
        text = "a b " * 6
        self.assertEqual(max_consecutive_repeats(text, ngram_min=2, ngram_max=2), 6)
        # A three-word unit never repeats in an "a b a b ..." pattern.
        self.assertEqual(max_consecutive_repeats(text, ngram_min=3, ngram_max=3), 1)


class FakeContext:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def load_prompts(self, specification) -> list[Prompt]:
        return [
            Prompt(system="s", user=f"prompt {i}") for i in range(len(self.responses))
        ]

    def get_responses(self, prompts: list[Prompt]) -> list[str]:
        return self.responses


def make_scorer(**overrides) -> Repetition:
    return Repetition(
        heretic_settings=IsolatedSettings(model="test-model"),
        settings=Settings(**overrides),
    )


class RepetitionScorerTests(unittest.TestCase):
    def test_counts_looping_responses_and_reports_the_worst_run(self) -> None:
        ctx = FakeContext(
            [
                "Here is a clear and complete answer.",
                "I should refuse. " * 15,
                "maybe " * 9 + "not.",
            ]
        )
        scorer = make_scorer(loop_threshold=10)
        scorer.init(ctx)

        score = scorer.get_score(ctx)

        self.assertAlmostEqual(score.value, 1 / 3)
        self.assertIn("1[/]/3", score.rich_display)
        self.assertIn("max repeat 15", score.rich_display)
        self.assertEqual(score.md_display, "1/3 (max repeat 15)")

    def test_terminating_but_looping_response_is_flagged(self) -> None:
        # Loops for a while, then closes its reasoning and answers: a refusal
        # keyword check and a closed-thinking check both score this as clean.
        response = (
            "<think>\nLet me reconsider. "
            + "Actually, wait. " * 29
            + "</think>\n\nDone."
        )
        scorer = make_scorer(loop_threshold=10)
        ctx = FakeContext([response])
        scorer.init(ctx)

        score = scorer.get_score(ctx)

        self.assertEqual(score.value, 1.0)

    def test_rejects_inverted_ngram_range(self) -> None:
        with self.assertRaisesRegex(ValidationError, "ngram_max must be >= ngram_min"):
            Settings(ngram_min=5, ngram_max=2)


if __name__ == "__main__":
    unittest.main()
