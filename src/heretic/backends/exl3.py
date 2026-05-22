# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from torch import LongTensor, Tensor
from transformers import BatchEncoding
from transformers.generation import GenerateDecoderOnlyOutput

from heretic.backends.base import ModelBackend
from heretic.utils import Prompt


class Exl3Backend(ModelBackend):
    """Prototype backend for ExLlamaV3 + EXL3 integration.

    This backend intentionally implements only the minimum runtime surface needed
    for the EXL3 smoke-test scripts. Methods required by the generic backend
    interface but not yet supported by the prototype continue to raise explicit
    ``NotImplementedError``.
    """

    def __init__(self):
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.generator: Any | None = None
        self._exllamav3: ModuleType | None = None

    def _load_exllamav3_module(self) -> ModuleType:
        if self._exllamav3 is None:
            self._exllamav3 = importlib.import_module("exllamav3")
        return self._exllamav3

    def _require_exllamav3(self) -> ModuleType:
        return self._load_exllamav3_module()

    def _resolve_exl3_types(self, exllamav3: ModuleType) -> tuple[type[Any], type[Any], type[Any]]:
        config_type = getattr(exllamav3, "ExLlamaV3Config", None)
        model_type = getattr(exllamav3, "ExLlamaV3", None)
        generator_type = getattr(exllamav3, "ExLlamaV3DynamicGenerator", None)

        if config_type is not None and model_type is not None and generator_type is not None:
            return config_type, model_type, generator_type

        module_model = importlib.import_module("exllamav3.model")
        module_generator = importlib.import_module("exllamav3.generator")

        config_type = config_type or getattr(module_model, "ExLlamaV3Config", None)
        model_type = model_type or getattr(module_model, "ExLlamaV3", None)
        generator_type = generator_type or getattr(
            module_generator,
            "ExLlamaV3DynamicGenerator",
            None,
        )

        if config_type is None or model_type is None or generator_type is None:
            raise RuntimeError(
                "Unsupported exllamav3 API surface: missing one or more required "
                "types (ExLlamaV3Config, ExLlamaV3, ExLlamaV3DynamicGenerator)."
            )

        return config_type, model_type, generator_type

    def load_model(self, model_path: str, **kwargs: Any) -> Any:
        exllamav3 = self._require_exllamav3()
        config_type, model_type, generator_type = self._resolve_exl3_types(exllamav3)

        config = config_type(model_path)
        self.model = model_type(config)
        self.model.load()
        self.generator = generator_type(self.model)
        return self.model

    def load_tokenizer(self, tokenizer_path: str | None = None, **kwargs: Any) -> Any:
        self._require_exllamav3()
        if self.model is None:
            raise RuntimeError("load_model must be called before load_tokenizer.")

        self.tokenizer = self.model.get_tokenizer()
        return self.tokenizer

    def generate_raw_text(self, prompt: str, **kwargs: Any) -> str:
        self._require_exllamav3()
        if self.generator is None:
            raise RuntimeError("Model is not loaded; call load_model first.")

        max_new_tokens = int(kwargs.get("max_new_tokens", 64))
        output = self.generator.generate(prompt=prompt, max_new_tokens=max_new_tokens)
        if isinstance(output, list):
            return str(output[0])
        return str(output)

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
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

        modules = []
        for name, _ in self.model.named_modules():
            modules.append(name)
        return modules

    def list_target_modules(self) -> list[str]:
        modules = self.list_modules()
        return [m for m in modules if m.endswith(".o_proj") or m.endswith(".down_proj")]

    def get_effective_weight(self, module_name: str) -> Tensor:
        raise NotImplementedError(
            "Exl3Backend.get_effective_weight requires EXL3 runtime investigation."
        )

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
