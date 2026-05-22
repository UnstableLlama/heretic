# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib
from contextlib import suppress
from types import ModuleType
from typing import Any

from torch import LongTensor, Tensor
from transformers import BatchEncoding
from transformers.generation import GenerateDecoderOnlyOutput

from heretic.backends.base import ModelBackend
from heretic.utils import Prompt


class Exl3Backend(ModelBackend):
    """Prototype backend for ExLlamaV3 + EXL3 integration."""

    def __init__(self):
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.generator: Any | None = None
        self.config: Any | None = None
        self.cache: Any | None = None
        self._exllamav3: ModuleType | None = None

    def _load_exllamav3_module(self) -> ModuleType:
        if self._exllamav3 is None:
            self._exllamav3 = importlib.import_module("exllamav3")
        return self._exllamav3

    def _require_exllamav3(self) -> ModuleType:
        return self._load_exllamav3_module()

    def _resolve_exl3_types(self, exllamav3: ModuleType) -> tuple[type[Any], type[Any], type[Any], type[Any], type[Any]]:
        """Resolve documented exllamav3 public API types."""
        modules: list[ModuleType] = [exllamav3]
        for module_name in ("exllamav3.model", "exllamav3.generator"):
            with suppress(ModuleNotFoundError):
                modules.append(importlib.import_module(module_name))

        def resolve(candidates: tuple[str, ...]) -> type[Any] | None:
            for module in modules:
                for candidate in candidates:
                    resolved = getattr(module, candidate, None)
                    if isinstance(resolved, type):
                        return resolved
            return None

        config_type = resolve(("Config", "ExLlamaV3Config"))
        model_type = resolve(("Model", "ExLlamaV3"))
        cache_type = resolve(("Cache", "ExLlamaV3Cache"))
        tokenizer_type = resolve(("Tokenizer", "ExLlamaV3Tokenizer"))
        generator_type = resolve(("Generator", "ExLlamaV3DynamicGenerator"))

        if any(t is None for t in (config_type, model_type, cache_type, tokenizer_type, generator_type)):
            discovered = sorted(
                {
                    name
                    for module in modules
                    for name in dir(module)
                    if name in {
                        "Config",
                        "Model",
                        "Cache",
                        "Tokenizer",
                        "Generator",
                        "ExLlamaV3Config",
                        "ExLlamaV3",
                        "ExLlamaV3Cache",
                        "ExLlamaV3Tokenizer",
                        "ExLlamaV3DynamicGenerator",
                    }
                }
            )
            raise RuntimeError(
                "Unsupported exllamav3 API surface: expected Config/Model/Cache/Tokenizer/Generator "
                "(or legacy ExLlamaV3* aliases). "
                f"Discovered relevant symbols: {discovered}"
            )

        return config_type, model_type, cache_type, tokenizer_type, generator_type

    def load_model(self, model_path: str, **kwargs: Any) -> Any:
        exllamav3 = self._require_exllamav3()
        config_type, model_type, cache_type, tokenizer_type, generator_type = self._resolve_exl3_types(exllamav3)

        self.config = config_type.from_directory(model_path)
        self.model = model_type.from_config(self.config)
        self.cache = cache_type(self.model, max_num_tokens=int(kwargs.get("max_num_tokens", 8192)))
        self.model.load(progressbar=bool(kwargs.get("progressbar", False)))
        self.tokenizer = tokenizer_type.from_config(self.config)
        self.generator = generator_type(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )
        return self.model

    def load_tokenizer(self, tokenizer_path: str | None = None, **kwargs: Any) -> Any:
        self._require_exllamav3()
        if self.tokenizer is None:
            raise RuntimeError("load_model must be called before load_tokenizer.")
        return self.tokenizer

    def generate_raw_text(self, prompt: str, **kwargs: Any) -> str:
        self._require_exllamav3()
        if self.generator is None:
            raise RuntimeError("Model is not loaded; call load_model first.")

        max_new_tokens = int(kwargs.get("max_new_tokens", 64))
        output = self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            add_bos=bool(kwargs.get("add_bos", True)),
            completion_only=bool(kwargs.get("completion_only", True)),
        )
        if isinstance(output, list):
            return str(output[0])
        return str(output)

    def generate(self, prompts: list[Prompt], **kwargs: Any) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        raise NotImplementedError(
            "Exl3Backend.generate (batched HF-compatible return format) is not "
            "implemented in the prototype. Use generate_raw_text in smoke scripts."
        )

    def forward_logits(self, input_ids: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError("Exl3Backend.forward_logits is not implemented yet.")

    def list_modules(self) -> list[str]:
        self._require_exllamav3()
        if self.model is None:
            raise RuntimeError("Model is not loaded; call load_model first.")

        if hasattr(self.model, "named_modules"):
            return [name for name, _ in self.model.named_modules()]

        if hasattr(self.model, "modules") and isinstance(self.model.modules, list):
            return [getattr(module, "name", f"module_{index}") for index, module in enumerate(self.model.modules)]

        return []

    def list_target_modules(self) -> list[str]:
        modules = self.list_modules()
        return [m for m in modules if m.endswith(".o_proj") or m.endswith(".down_proj")]

    def get_effective_weight(self, module_name: str) -> Tensor:
        raise NotImplementedError("Exl3Backend.get_effective_weight requires EXL3 runtime investigation.")

    def apply_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        self._require_exllamav3()
        if self.model is None:
            raise RuntimeError("Model is not loaded; call load_model first.")
        self.model.load_lora(adapter_path, adapter_name)

    def unload_adapter(self, adapter_name: str = "default") -> None:
        self._require_exllamav3()
        if self.model is None:
            raise RuntimeError("Model is not loaded; call load_model first.")
        self.model.unload_lora(adapter_name)

    def reset_adapters(self) -> None:
        self._require_exllamav3()
        if self.model is None:
            raise RuntimeError("Model is not loaded; call load_model first.")
        self.model.reset_loras()
