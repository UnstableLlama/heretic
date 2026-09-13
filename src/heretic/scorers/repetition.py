# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from pydantic import BaseModel, Field, PositiveInt, model_validator

from heretic.config import DatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import print


def max_consecutive_repeats(text: str, ngram_min: int = 1, ngram_max: int = 20) -> int:
    """
    Largest number of times any run of `ngram_min` to `ngram_max` words occurs
    back to back in `text`. 1 means nothing repeats consecutively; 0 means the
    text contains no words at all.
    """
    words = text.split()
    length = len(words)
    if length == 0:
        return 0

    best = 1
    for n in range(ngram_min, min(ngram_max, length // 2) + 1):
        i = 0
        while i + 2 * n <= length:
            gram = words[i : i + n]
            if words[i + n : i + 2 * n] != gram:
                i += 1
                continue

            count = 2
            j = i + 2 * n
            while j + n <= length and words[j : j + n] == gram:
                count += 1
                j += n
            best = max(best, count)

            # Every start inside this run yields the same run or a shorter one,
            # so resume after its last complete unit.
            i = j - n + 1

    return best


class Settings(BaseModel):
    score_name: str = Field(
        default="Looping responses",
        description="Name that describes what the configured repetition check measures.",
    )

    prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts to evaluate repetition on.",
    )

    ngram_min: PositiveInt = Field(
        default=1,
        description="Shortest run of words that counts as a repeated unit.",
    )

    ngram_max: PositiveInt = Field(
        default=20,
        description="Longest run of words that counts as a repeated unit.",
    )

    loop_threshold: PositiveInt = Field(
        default=10,
        description=(
            "A response whose most repeated unit occurs at least this many times "
            "in a row counts as looping."
        ),
    )

    print_responses: bool = Field(
        default=False,
        description="Whether to print the looping responses with their repeat counts.",
    )

    @model_validator(mode="after")
    def _validate_ranges(self) -> "Settings":
        if self.ngram_max < self.ngram_min:
            raise ValueError("ngram_max must be >= ngram_min.")
        if self.loop_threshold < 2:
            raise ValueError("loop_threshold must be at least 2.")
        return self


class Repetition(Scorer):
    """
    Counts responses that degenerate into repetition: the same run of words
    repeated back to back at least `loop_threshold` times.

    This catches the failure that response-keyword checks cannot see: a model
    that loops for a while, then still closes its reasoning and stops, scores
    as a clean response on a refusal check and on a closed-thinking check.
    Give this scorer the same prompts as those checks so all of them share one
    generation per trial.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading {self.settings.score_name} evaluation prompts from [bold]{self.settings.prompts.dataset}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

    def get_score(self, ctx: Context) -> Score:
        responses = ctx.get_responses(self.prompts)
        repeats = [
            max_consecutive_repeats(
                response, self.settings.ngram_min, self.settings.ngram_max
            )
            for response in responses
        ]
        looping = sum(1 for count in repeats if count >= self.settings.loop_threshold)
        worst = max(repeats, default=0)

        if self.settings.print_responses:
            for prompt, response, count in zip(self.prompts, responses, repeats):
                if count < self.settings.loop_threshold:
                    continue
                print()
                print(f"[bold]Prompt:[/] {prompt.user}")
                print(f"[bold]Max repeat:[/] [red]{count}[/]")
                print(f"[bold]Response:[/] {response}")
            print()

        return Score(
            value=float(looping / len(self.prompts)),
            rich_display=(
                f"[bold]{looping}[/]/{len(self.prompts)} [italic](max repeat {worst})[/]"
            ),
            md_display=f"{looping}/{len(self.prompts)} (max repeat {worst})",
        )
