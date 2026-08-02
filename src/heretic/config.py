# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from enum import Enum
from typing import Dict, Literal

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# !!!IMPORTANT!!!
#
# Any settings added to the classes defined in this module
# must be evaluated for privacy implications and have
# exclude=True set in their field definitions if appropriate.


class QuantizationMethod(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"
    EXL3 = "exl3"


class RowNormalization(str, Enum):
    NONE = "none"
    PRE = "pre"
    # POST = "post"  # Theoretically possible, but provides no advantage.
    FULL = "full"


class ExportStrategy(str, Enum):
    MERGE = "merge"
    ADAPTER = "adapter"


class DatasetSpecification(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset ID, or path to dataset on disk."
    )

    commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the dataset.",
    )

    split: str | None = Field(
        default=None,
        description="Portion of the dataset to use. Required for datasets, optional for plain text files.",
    )

    column: str | None = Field(
        default=None,
        description="Column in the dataset that contains the prompts. Required for datasets, ignored for plain text files.",
    )

    prefix: str = Field(
        default="",
        description="Text to prepend to each prompt.",
    )

    suffix: str = Field(
        default="",
        description="Text to append to each prompt.",
    )

    system_prompt: str | None = Field(
        default=None,
        description="System prompt to use with the prompts (overrides global system prompt if set).",
    )

    residual_plot_label: str | None = Field(
        default=None,
        description="Label to use for the dataset in plots of residual vectors.",
        exclude=True,
    )

    residual_plot_color: str | None = Field(
        default=None,
        description="Matplotlib color to use for the dataset in plots of residual vectors.",
        exclude=True,
    )


class ScorerConfig(BaseModel):
    """
    Configuration for a scorer plugin.

    TOML format:
    - { plugin = "<plugin>", optimization = "<optimization>", instance_name = "<optional>" }
    """

    plugin: str = Field(
        description=(
            "Plugin to load. Either a file path with class name "
            "(`path/to/plugin.py:ClassName`) or a fully-qualified import path "
            "(`module.submodule.ClassName`)."
        ),
    )

    optimization: Literal["minimize", "maximize", "none"] = Field(
        description=(
            "Optimization direction for this scorer. "
            '"minimize" / "maximize" to include the scorer as an objective, '
            '"none" to compute the score without optimizing for it.'
        ),
    )

    instance_name: str | None = Field(
        default=None,
        description=(
            "Optional name to distinguish multiple instances of the same plugin class. "
            "Instance-specific settings live under `[scorer.<ClassName>_<instance_name>]`."
        ),
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if not value.strip():
            raise ValueError("cannot be empty or whitespace")

        if "." in value:
            raise ValueError("'.' is not allowed")

        if any(char.isspace() for char in value):
            raise ValueError("whitespace is not allowed")

        return value


class BenchmarkSpecification(BaseModel):
    task: str = Field(
        description="Task ID of the benchmark in the Language Model Evaluation Harness."
    )

    name: str = Field(description="Name of the benchmark for presentation purposes.")

    description: str = Field(
        description="Description of the benchmark for presentation purposes."
    )


class Settings(BaseSettings):
    model: str = Field(description="Hugging Face model ID, or path to model on disk.")


    exl3_max_num_tokens: int = Field(
        default=8192,
        description=(
            "EXL3 backend only: max_num_tokens for the KV cache. Must be a multiple of 256. "
            "Bound on batch_size * seq_len during forward."
        ),
    )

    exl3_load_max_chunk_size: int = Field(
        default=512,
        description=(
            "EXL3 backend only: max_chunk_size passed to exllamav3's Model.load(). "
            "This sizes the single-sequence measuring forward the autosplit runs "
            "through each module to estimate activation memory; it does NOT limit "
            "runtime forwards (those are bounded by exl3_max_num_tokens). Large "
            "MoE models (e.g. Laguna, 256 experts) OOM during load with the "
            "exllamav3 default of 2048 even though weights and real forwards fit, "
            "so Heretic probes with a smaller chunk. Raise it if a model needs a "
            "larger activation reservation to place layers correctly."
        ),
    )

    exl3_base_model: str | None = Field(
        default=None,
        description=(
            "EXL3 merge only: explicit Hugging Face base model ID/path to use when "
            "merging LoRA into a full model. If not set, Heretic will try to infer "
            "it from the EXL3 model directory metadata."
        ),
    )

    exl3_gpu_split: list[float] | None = Field(
        default=None,
        description=(
            "EXL3 backend only: explicit per-device memory budget in GB used to "
            'split the model across GPUs (e.g. [20.0, 20.0]). If not set, '
            "exllamav3 auto-splits across all visible devices."
        ),
    )

    exl3_reserve_per_device: float = Field(
        default=0.5,
        description=(
            "EXL3 backend only: gigabytes of VRAM to reserve per device when "
            "auto-splitting the model across multiple GPUs."
        ),
    )

    exl3_reconstruct_slice_n: int = Field(
        default=4096,
        description=(
            "EXL3 backend only: column-slice width (in output features) used when "
            "computing the abliteration update. Instead of materializing the full "
            "effective weight matrix (which transiently upcasts to fp32 and can OOM "
            "a busy GPU), the weight is reconstructed and consumed in column slices "
            "of this width. Lower values reduce the peak VRAM of the abliteration "
            "pass at the cost of more reconstruction calls; higher values do the "
            "reverse. Rounded up to a multiple of 128."
        ),
    )

    model_commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the model.",
    )

    evaluate_model: str | None = Field(
        default=None,
        description=(
            "If this model ID or path is set, then instead of abliterating the main model, "
            "evaluate this model relative to the main model."
        ),
        exclude=True,
    )

    collect_reproducibles: str | None = Field(
        default=None,
        description=(
            "If this directory path is set, then instead of abliterating a model, "
            "download all reproduce.json files from public Heretic model repositories "
            "on Hugging Face, and store them in that directory for archival purposes."
        ),
        exclude=True,
    )

    reproduce: str | None = Field(
        default=None,
        description=(
            "If this path or URL to a reproduce.json file is set, load reproduction information "
            "from that file, and attempt to reproduce the abliterated model it originated from."
        ),
        exclude=True,
    )

    dtypes: list[str] = Field(
        default=[
            # In practice, "auto" almost always means bfloat16.
            "auto",
            # If that doesn't work (e.g. on pre-Ampere hardware), fall back to float16.
            "float16",
            # If "auto" resolves to float32, and that fails because it is too large,
            # and float16 fails due to range issues, try bfloat16.
            "bfloat16",
            # If neither of those work, fall back to float32 (which will of course fail
            # if that was the dtype "auto" resolved to).
            "float32",
        ],
        description=(
            "List of PyTorch dtypes to try when loading model tensors. "
            "If loading with a dtype fails, the next dtype in the list will be tried."
        ),
    )

    quantization: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description=(
            "Quantization method to use when loading the model. Options: "
            '"none" (no quantization), '
            '"bnb_4bit" (4-bit quantization using bitsandbytes), '
            '"exl3" (ExLlamaV3 for EXL3-quantized models; requires "pip install heretic-llm[exl3]" and a path to an EXL3 model directory).'
        ),
    )

    device_map: str | Dict[str, int | str] = Field(
        default="auto",
        description="Device map to pass to Accelerate when loading the model.",
    )

    max_memory: Dict[str, str] | None = Field(
        default=None,
        description='Maximum memory to allocate per device (e.g., { "0" = "20GB", "cpu" = "64GB" }).',
    )

    offload_outputs_to_cpu: bool = Field(
        default=True,
        description=(
            "Whether to move intermediate analysis tensors (such as residuals and logprobs) "
            "to CPU memory as soon as possible to reduce peak VRAM usage. "
            "This lowers peak VRAM usage during residual analysis and evaluation, "
            "but may slightly reduce performance due to host/device transfers."
        ),
    )

    batch_size: NonNegativeInt = Field(
        default=0,  # auto
        description="Number of input sequences to process in parallel (0 = auto).",
    )

    max_batch_size: PositiveInt = Field(
        default=128,
        description="Maximum batch size to try when automatically determining the optimal batch size.",
        # When storing a settings object, the batch size is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    max_response_length: PositiveInt = Field(
        default=100,
        description="Maximum number of tokens to generate for each response.",
    )

    response_prefix: str | None = Field(
        default=None,
        description=(
            "Common prefix to assume for all responses, so that evaluation happens "
            "at the point where responses start to differ for different prompts. "
            "If not set, the prefix is determined automatically by comparing multiple responses."
        ),
    )

    chain_of_thought_skips: list[tuple[str, str]] = Field(
        default=[
            # Most thinking models.
            (
                "<think>",
                "<think></think>",
            ),
            # gpt-oss.
            (
                "<|channel|>analysis<|message|>",
                "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>",
            ),
            # Unknown, suggested by user.
            (
                "<thought>",
                "<thought></thought>",
            ),
            # Unknown, suggested by user.
            (
                "[THINK]",
                "[THINK][/THINK]",
            ),
        ],
        description=(
            "List of pairs of the form (cot_initializer, closed_cot_block) used to skip "
            "the Chain-of-Thought block in responses, so that evaluation happens "
            "at the start of the actual response."
        ),
        # When storing a settings object, the response prefix is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    print_debug_information: bool = Field(
        default=False,
        description="Whether to print additional information that can help with debugging.",
        exclude=True,
    )

    print_residual_geometry: bool = Field(
        default=False,
        description="Whether to print detailed information about residuals and residual directions.",
        exclude=True,
    )

    plot_residuals: bool = Field(
        default=False,
        description="Whether to generate plots showing PaCMAP projections of residual vectors.",
        exclude=True,
    )

    residual_plot_path: str = Field(
        default="plots",
        description="Base path to save plots of residual vectors to.",
        exclude=True,
    )

    residual_plot_title: str = Field(
        default='PaCMAP Projection of Residual Vectors for "Harmless" and "Harmful" Prompts',
        description="Title placed above plots of residual vectors.",
        exclude=True,
    )

    residual_plot_style: str = Field(
        default="dark_background",
        description="Matplotlib style sheet to use for plots of residual vectors.",
        exclude=True,
    )

    scorers: list[ScorerConfig] = Field(
        default_factory=lambda: [
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
            ),
            ScorerConfig(
                plugin="heretic.scorers.kl_divergence.KLDivergence",
                optimization="minimize",
            ),
        ],
        description=(
            "List of scorer plugin configs. Each entry is an object"
            " { plugin = <plugin>, optimization = <optimization>, instance_name = <optional> }."
            " <optimization> is one of 'minimize', 'maximize', 'none' (do not optimize)."
        ),
    )

    use_ara: bool = Field(
        default=False,
        description=(
            "Whether to use Arbitrary-Rank Ablation (ARA), an abliteration method based on matrix optimization, "
            "instead of traditional directional ablation."
        ),
    )

    use_ara_lora: bool = Field(
        default=False,
        description=(
            "Use LoRA in ARA instead of full-weight editing. "
            "Makes ARA compatible with quantization and removes model reloads between trials. "
            "Based on work by kabachuha (https://github.com/p-e-w/heretic/pull/332)."
        ),
    )

    ara_lora_rank: int = Field(
        default=128,
        description=(
            "If LoRA is used in ARA, this sets its rank. "
            "Keep it high enough to simulate the 'arbitrary' effect."
        ),
    )

    ara_lora_regularization: float = Field(
        default=0.0,
        description=(
            "L2 regularization strength on the ARA LoRA factors A and B "
            "(adds value * (mean(A^2) + mean(B^2)) to the optimization loss). "
            "0 (the default) disables it. The ARA overcorrection objective is "
            "unbounded below and LoRA has a scale degeneracy, so the factors can "
            "blow up and overflow the fp16 forward (nan KL), especially on "
            "low-bit quants. A small positive value (e.g. 1e-3) bounds the loss "
            "and keeps the factors well-scaled; too large weakens abliteration."
        ),
    )

    steer_bad_behavior_weight_min: float = Field(
        default=0.0001,
        description=(
            "Lower bound of the search range for the ARA inner-objective "
            "'steer_bad_behavior_weight', which the outer optimizer samples on a "
            "log scale. Applies to both ARA and ARA-LoRA. Must be positive and "
            "no greater than steer_bad_behavior_weight_max."
        ),
    )

    steer_bad_behavior_weight_max: float = Field(
        default=0.001,
        description=(
            "Upper bound of the search range for the ARA inner-objective "
            "'steer_bad_behavior_weight', which the outer optimizer samples on a "
            "log scale. Applies to both ARA and ARA-LoRA. Defaults to 0.001: the "
            "optimizer reliably prefers the low end, and large values are the "
            "dominant source of non-finite-KL ('blown up') trials on low-bpw EXL3 "
            "ARA-LoRA. Raise it (e.g. to 1.0) to explore a wider phase space."
        ),
    )

    @model_validator(mode="after")
    def _validate_steer_bad_behavior_weight_range(self) -> "Settings":
        if self.steer_bad_behavior_weight_min <= 0:
            raise ValueError("steer_bad_behavior_weight_min must be positive.")
        if self.steer_bad_behavior_weight_max < self.steer_bad_behavior_weight_min:
            raise ValueError(
                "steer_bad_behavior_weight_max must be >= "
                "steer_bad_behavior_weight_min."
            )
        return self

    invert_target: bool = Field(
        default=False,
        description=(
            "Invert the steering target: instead of pushing 'bad' outputs toward "
            "'good' outputs (suppressing the targeted behavior), push them further "
            "from 'good' and deeper into the 'bad' cluster (amplifying the "
            "targeted behavior). To make the outer optimizer maximize the "
            "behavior's expression, also set optimization = \"maximize\" on the "
            "KeywordRate scorer in [[scorers]]. Useful for "
            "general behavioral steering when 'good'='neutral' and 'bad'='target "
            "behavior'. Applies to both ARA and ARA-LoRA inner objectives. "
            "Like the default direction, the inner ARA objective is geometrically "
            "unbounded; consider setting ara_lora_regularization to a small "
            "positive value if A@B overflows fp16."
        ),
    )

    @model_validator(mode="after")
    def _ara_lora_implies_ara(self) -> "Settings":
        # ARA LoRA is a sub-mode of ARA: the module-I/O collection and ARA
        # parameter suggestion are gated on use_ara, but the abliteration
        # dispatch enters the LoRA branch on use_ara_lora alone. Passing
        # --use-ara-lora without --use-ara would otherwise crash with a
        # NameError on good_module_io. Treat use_ara_lora as implying use_ara.
        if self.use_ara_lora:
            self.use_ara = True
        return self

    @model_validator(mode="after")
    def _exl3_ara_requires_lora(self) -> "Settings":
        # Plain ARA edits each weight matrix in place. EXL3 weights are stored
        # as quantized trellis blobs that can't be modified in place, so the
        # EXL3 backend only implements the LoRA variant (ara_lora_abliterate).
        # Without this guard, --quantization exl3 --use-ara would load the
        # model and capture module I/O before crashing on the first trial with
        # an AttributeError. Fail fast instead.
        if (
            self.quantization == QuantizationMethod.EXL3
            and self.use_ara
            and not self.use_ara_lora
        ):
            raise ValueError(
                "The EXL3 backend cannot run plain ARA (it can't edit quantized "
                "weights in place). Add --use-ara-lora to use the LoRA variant."
            )
        return self

    orthogonalize_direction: bool = Field(
        default=True,
        description=(
            "Whether to adjust the residual directions so that only the component that is "
            "orthogonal to the good direction is subtracted during abliteration."
        ),
    )

    row_normalization: RowNormalization = Field(
        default=RowNormalization.FULL,
        description=(
            "How to apply row normalization of the weights. Options: "
            '"none" (no normalization), '
            '"pre" (compute LoRA adapter relative to row-normalized weights), '
            '"full" (like "pre", but renormalizes to preserve original row magnitudes).'
        ),
    )

    full_normalization_lora_rank: PositiveInt = Field(
        default=3,
        description=(
            'The rank of the LoRA adapter to use when "full" row normalization is used. '
            "Row magnitude preservation is approximate due to non-linear effects, "
            "and this determines the rank of that approximation. Higher ranks produce "
            "larger output files and may slow down evaluation."
        ),
    )

    winsorization_quantile: float = Field(
        default=1.0,
        description=(
            "The symmetric winsorization to apply to the per-prompt, per-layer residual vectors, "
            "expressed as the quantile to clamp to (between 0 and 1). Disabled by default. "
            'This can tame so-called "massive activations" that occur in some models. '
            "Example: winsorization_quantile = 0.95 computes the 0.95-quantile of the absolute values "
            "of the components, then clamps the magnitudes of all components to that quantile."
        ),
    )

    n_trials: PositiveInt = Field(
        default=200,
        description="Number of abliteration trials to run during optimization.",
    )

    n_startup_trials: NonNegativeInt = Field(
        default=60,
        description="Number of trials that use random sampling for the purpose of exploration.",
    )

    seed: int | None = Field(
        default=None,
        description=(
            "Random seed for reproducible optimization. "
            "Applies to Python's random module, NumPy, PyTorch, and Optuna."
        ),
    )

    study_checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory to save and load study progress to/from.",
        exclude=True,
    )

    benchmarks: list[BenchmarkSpecification] = Field(
        default=[
            BenchmarkSpecification(
                task="agieval",
                name="AGIEval",
                description="A Human-Centric Benchmark for Evaluating Foundation Models",
            ),
            BenchmarkSpecification(
                task="bbh",
                name="BIG-Bench Hard (BBH)",
                description="Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
            ),
            BenchmarkSpecification(
                task="commonsense_qa",
                name="CommonsenseQA",
                description="A Question Answering Challenge Targeting Commonsense Knowledge",
            ),
            BenchmarkSpecification(
                task="eq_bench",
                name="EQ-Bench",
                description="An Emotional Intelligence Benchmark for Large Language Models",
            ),
            BenchmarkSpecification(
                task="gsm8k",
                name="GSM8K",
                description="Training Verifiers to Solve Math Word Problems",
            ),
            BenchmarkSpecification(
                task="hellaswag",
                name="HellaSwag",
                description="Can a Machine Really Finish Your Sentence?",
            ),
            BenchmarkSpecification(
                task="ifeval",
                name="IFEval",
                description="Instruction-Following Evaluation for Large Language Models",
            ),
            BenchmarkSpecification(
                task="mmlu",
                name="MMLU",
                description="Measuring Massive Multitask Language Understanding",
            ),
            BenchmarkSpecification(
                task="mmlu_pro",
                name="MMLU-Pro",
                description="A More Robust and Challenging Multi-Task Language Understanding Benchmark",
            ),
            BenchmarkSpecification(
                task="piqa",
                name="PIQA",
                description="Reasoning about Physical Commonsense in Natural Language",
            ),
            BenchmarkSpecification(
                task="winogrande",
                name="WinoGrande",
                description="An Adversarial Winograd Schema Challenge at Scale",
            ),
        ],
        description="Benchmarks to offer to the user for evaluating abliterated models.",
        exclude=True,
    )

    max_shard_size: PositiveInt | str = Field(
        default="5GB",
        description="Maximum size for individual safetensors files generated when exporting a model.",
    )

    export_strategy: ExportStrategy | None = Field(
        default=None,
        description='How to export the model: "merge", "adapter", or unset to prompt the user.',
    )

    checkpoint_action: str | None = Field(
        default=None,
        description='Action to take in case a checkpoint exists: "continue", "restart", or unset to prompt the user.',
    )

    trial_index: NonNegativeInt | None = Field(
        default=None,
        description="Index (in the sorted Pareto front) of the trial to use, or unset to prompt the user.",
    )

    n_additional_trials: PositiveInt | None = Field(
        default=None,
        description="Number of additional trials to run, or unset to prompt the user.",
    )

    model_action: str | None = Field(
        default=None,
        description='Action to take with the decensored model: "save", "upload", or unset to prompt the user.',
    )

    save_directory: str | None = Field(
        default=None,
        description="Directory to save the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_id: str | None = Field(
        default=None,
        description="Name of the Hugging Face repository to upload the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_private: bool | None = Field(
        default=None,
        description="Whether the Hugging Face repository to upload the model to should be private, or unset to prompt the user.",
    )

    upload_reproducibility_information: str | None = Field(
        default=None,
        description='Which reproducibility information to add to the Hugging Face repository: "full", "basic", "none", or unset to prompt the user.',
    )

    ignore_mismatches: bool | None = Field(
        default=None,
        description="Whether to attempt to reproduce the model even if there are environment mismatches, or unset to prompt the user.",
    )

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to use when prompting the model.",
    )

    good_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmless" prompts',
            residual_plot_color="royalblue",
        ),
        description="Dataset of prompts that tend to not result in refusals (used for calculating refusal directions).",
    )

    bad_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmful" prompts',
            residual_plot_color="darkorange",
        ),
        description="Dataset of prompts that tend to result in refusals (used for calculating refusal directions).",
    )

    # We intentionally allow extra keys so users can provide plugin-specific
    # configuration in TOML tables like `[scorer.KeywordRate]` which are later
    # consumed via `settings.model_extra` (see `Evaluator._get_plugin_namespace`).
    model_config = SettingsConfigDict(extra="allow")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,  # Used during resume - should override *all* other sources.
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_implicit_flags=True,
                cli_kebab_case=True,
            ),
            EnvSettingsSource(settings_cls, env_prefix="HERETIC_"),
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls, toml_file="config.toml"),
        )
