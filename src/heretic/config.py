# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from copy import deepcopy
from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
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


class ExportStrategy(str, Enum):
    MERGE = "merge"
    ADAPTER = "adapter"


class SingleDatasetSpecification(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset ID, or path to dataset on disk."
    )

    commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the dataset.",
    )

    config: str | None = Field(
        default=None,
        description=(
            "Dataset config/subset name. Each config can have its own split. "
            "Used to load a specific config of a dataset that has multiple configurations."
        ),
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


DatasetSpecification: TypeAlias = (
    SingleDatasetSpecification | list[SingleDatasetSpecification]
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


class ModifierConfig(BaseModel):
    """
    Configuration for a modifier plugin.

    TOML format:
    - { plugin = "<plugin>", instance_name = "<optional>" }
    """

    plugin: str = Field(
        description=(
            "Plugin to load. Either a file path with class name "
            "(`path/to/plugin.py:ClassName`) or a fully-qualified import path "
            "(`module.submodule.ClassName`)."
        ),
    )

    instance_name: str | None = Field(
        default=None,
        description=(
            "Optional name to distinguish multiple instances of the same plugin class. "
            "Instance-specific settings live under `[modifier.<ClassName>_<instance_name>]`."
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
            "split the model across GPUs (e.g. [20.0, 20.0]). If not set, "
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

    device_map: str | dict[str, int | str] = Field(
        default="auto",
        description="Device map to pass to Accelerate when loading the model.",
    )

    max_memory: dict[str, str] | None = Field(
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

    batch_size_test_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:256]",
            column="text",
        ),
        description="Dataset of prompts to use for automatically determining the optimal batch size.",
        # When storing a settings object, the batch size is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    max_response_length: PositiveInt = Field(
        default=100,
        description="Maximum number of tokens to generate for each response.",
    )

    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments forwarded to the tokenizer's "
            "apply_chat_template when rendering prompts (capture, evaluation, and "
            "chat). E.g. { enable_thinking = false } to suppress a reasoning "
            "model's <think> block so evaluation sees the actual response."
        ),
    )

    response_prefix: str | None = Field(
        default=None,
        description=(
            "Common prefix to assume for all responses, so that evaluation happens "
            "at the point where responses start to differ for different prompts. "
            "If not set, the prefix is determined automatically by comparing multiple responses."
        ),
    )

    response_prefix_test_prompts: DatasetSpecification = Field(
        default=[
            SingleDatasetSpecification(
                dataset="mlabonne/harmless_alpaca",
                split="train[:100]",
                column="text",
            ),
            SingleDatasetSpecification(
                dataset="mlabonne/harmful_behaviors",
                split="train[:100]",
                column="text",
            ),
        ],
        description="Dataset of prompts to use for automatically determining the response prefix.",
        # When storing a settings object, the response prefix is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
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

    scorers: list[ScorerConfig] = Field(
        default=[
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
            "List of scorer plugin configs. Each entry is an object "
            "{ plugin = <plugin>, optimization = <optimization>, instance_name = <optional> }. "
            '<optimization> is one of "minimize", "maximize", or "none" (do not optimize).'
        ),
    )

    modifiers: list[ModifierConfig] = Field(
        default=[
            ModifierConfig(
                plugin="heretic.modifiers.ara.ARA",
            ),
        ],
        description=(
            "List of modifier plugin configs. Each entry is an object "
            "{ plugin = <plugin>, instance_name = <optional> }. "
            "Note that only a single modifier can currently be applied, "
            "and this list must contain exactly one entry."
        ),
    )

    # Legacy CLI flags remain accepted; normalize them into the 2.x plugin schema.
    use_ara: bool | None = Field(default=None, exclude=True)
    use_ara_lora: bool | None = Field(default=None, exclude=True)
    ara_lora_rank: PositiveInt | None = Field(default=None, exclude=True)
    ara_lora_regularization: float | None = Field(default=None, ge=0, exclude=True)
    steer_bad_behavior_weight_min: float | None = Field(
        default=None, gt=0, exclude=True
    )
    steer_bad_behavior_weight_max: float | None = Field(
        default=None, gt=0, exclude=True
    )
    invert_target: bool | None = Field(default=None, exclude=True)
    row_normalization: Literal["none", "pre", "full"] | None = Field(
        default=None, exclude=True
    )
    orthogonalize_direction: bool | None = Field(default=None, exclude=True)
    full_normalization_lora_rank: PositiveInt | None = Field(default=None, exclude=True)
    winsorization_quantile: float | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_modifier_settings(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = deepcopy(value)
        for key in ("use_ara", "use_ara_lora"):
            if value.get(key) is not None:
                value[key] = TypeAdapter(bool).validate_python(value[key])
        legacy_keys = {
            "use_ara",
            "use_ara_lora",
            "ara_lora_rank",
            "ara_lora_regularization",
            "steer_bad_behavior_weight_min",
            "steer_bad_behavior_weight_max",
            "invert_target",
            "row_normalization",
            "orthogonalize_direction",
            "full_normalization_lora_rank",
            "winsorization_quantile",
            "good_prompts",
            "bad_prompts",
        }
        if not any(value.get(key) is not None for key in legacy_keys):
            return value
        if "modifiers" not in value:
            ara = value.get("use_ara") or value.get("use_ara_lora")
            plugin = (
                "heretic.modifiers.ara.ARA"
                if ara
                else "heretic.modifiers.abliteration.Abliteration"
            )
            value["modifiers"] = [{"plugin": plugin}]
        for entry in value["modifiers"]:
            entry = entry.model_dump() if isinstance(entry, ModifierConfig) else entry
            plugin = entry["plugin"]
            if plugin not in (
                "heretic.modifiers.ara.ARA",
                "heretic.modifiers.abliteration.Abliteration",
            ):
                continue
            ara = plugin.endswith(".ARA")
            name = "ARA" if ara else "Abliteration"
            if entry.get("instance_name"):
                name += "_" + entry["instance_name"]
            target = value.setdefault("modifier", {}).setdefault(name, {})
            mapping = {"good_prompts": "good_prompts", "bad_prompts": "bad_prompts"}
            if ara:
                mapping.update(
                    {
                        "ara_lora_rank": "lora_rank",
                        "ara_lora_regularization": "lora_regularization",
                        "invert_target": "invert_target",
                        "steer_bad_behavior_weight_min": "steer_bad_behavior_weight_min",
                        "steer_bad_behavior_weight_max": "steer_bad_behavior_weight_max",
                    }
                )
                # Preserve this fork's defaults for legacy configurations.
                target.setdefault("lora_rank", value.get("ara_lora_rank") or 128)
                target.setdefault(
                    "steer_bad_behavior_weight_max",
                    value.get("steer_bad_behavior_weight_max") or 0.001,
                )
                if value.get("row_normalization") is not None:
                    target.setdefault(
                        "preserve_row_magnitudes", value["row_normalization"] == "full"
                    )
            else:
                for key in (
                    "orthogonalize_direction",
                    "row_normalization",
                    "full_normalization_lora_rank",
                    "winsorization_quantile",
                ):
                    mapping[key] = key
            for old, new in mapping.items():
                if value.get(old) is not None:
                    target.setdefault(new, value[old])
        # Dataset tables now belong to the modifier, including in saved run metadata.
        value.pop("good_prompts", None)
        value.pop("bad_prompts", None)
        return value

    n_trials: PositiveInt = Field(
        default=100,
        description="Number of abliteration trials to run during optimization.",
    )

    n_startup_trials: NonNegativeInt = Field(
        default=30,
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

    # We intentionally allow extra keys so users can provide plugin-specific
    # configuration in TOML tables like `[scorer.KeywordRate]` which are later
    # consumed via `settings.model_extra` (see `plugin.get_plugin_namespace`).
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
