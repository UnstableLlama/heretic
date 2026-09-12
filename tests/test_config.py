# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from isolated_settings import IsolatedSettings
from pydantic import ValidationError

from heretic.config import ScorerConfig


class ScorerConfigTests(unittest.TestCase):
    def test_accepts_slug_like_instance_name(self) -> None:
        config = ScorerConfig(
            plugin="heretic.scorers.keyword_rate.KeywordRate",
            optimization="minimize",
            instance_name="small-1",
        )

        self.assertEqual(config.instance_name, "small-1")

    def test_rejects_empty_instance_name(self) -> None:
        with self.assertRaises(ValidationError):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name=" \t",
            )

    def test_rejects_whitespace_in_instance_name(self) -> None:
        for instance_name in ["small name", "small\tname", "small\nname"]:
            with self.subTest(instance_name=instance_name):
                with self.assertRaisesRegex(
                    ValidationError, "whitespace is not allowed"
                ):
                    ScorerConfig(
                        plugin="heretic.scorers.keyword_rate.KeywordRate",
                        optimization="minimize",
                        instance_name=instance_name,
                    )

    def test_rejects_dot_in_instance_name(self) -> None:
        with self.assertRaisesRegex(ValidationError, "'\\.' is not allowed"):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name="small.name",
            )


class TargetComponentsTests(unittest.TestCase):
    def test_defaults_to_all_components(self) -> None:
        settings = IsolatedSettings(**{"model": "test-model"})

        self.assertEqual(settings.target_components, ["attn.o_proj", "mlp.down_proj"])

    def test_accepts_subset(self) -> None:
        settings = IsolatedSettings(
            **{"model": "test-model", "target_components": ["attn.o_proj"]}
        )

        self.assertEqual(settings.target_components, ["attn.o_proj"])

    def test_drops_duplicates(self) -> None:
        settings = IsolatedSettings(
            **{
                "model": "test-model",
                "target_components": ["mlp.down_proj", "attn.o_proj", "mlp.down_proj"],
            }
        )

        self.assertEqual(settings.target_components, ["mlp.down_proj", "attn.o_proj"])

    def test_rejects_unknown_component(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unknown component"):
            IsolatedSettings(
                **{"model": "test-model", "target_components": ["attn.q_proj"]}
            )

    def test_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValidationError, "at least one component"):
            IsolatedSettings(**{"model": "test-model", "target_components": []})


if __name__ == "__main__":
    unittest.main()
