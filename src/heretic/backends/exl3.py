# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import Any

from torch import LongTensor, Tensor
from transformers import BatchEncoding
from transformers.generation import GenerateDecoderOnlyOutput

from heretic.backends.base import ModelBackend
from heretic.utils import Prompt


class Exl3Backend(ModelBackend):
    """Skeleton backend for ExLlamaV3 + EXL3 integration."""

    def __init__(self):
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def load_model(self, model_path: str, **kwargs: Any) -> Any:
        raise NotImplementedError("Exl3Backend.load_model is not implemented yet.")

    def load_tokenizer(self, tokenizer_path: str | None = None, **kwargs: Any) -> Any:
        raise NotImplementedError("Exl3Backend.load_tokenizer is not implemented yet.")

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        raise NotImplementedError("Exl3Backend.generate is not implemented yet.")

    def forward_logits(self, input_ids: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError("Exl3Backend.forward_logits is not implemented yet.")

    def list_modules(self) -> list[str]:
        raise NotImplementedError("Exl3Backend.list_modules is not implemented yet.")

    def list_target_modules(self) -> list[str]:
        raise NotImplementedError("Exl3Backend.list_target_modules is not implemented yet.")

    def get_effective_weight(self, module_name: str) -> Tensor:
        raise NotImplementedError(
            "Exl3Backend.get_effective_weight requires EXL3 runtime investigation."
        )

    def apply_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        raise NotImplementedError("Exl3Backend.apply_adapter is not implemented yet.")

    def unload_adapter(self, adapter_name: str = "default") -> None:
        raise NotImplementedError("Exl3Backend.unload_adapter is not implemented yet.")

    def reset_adapters(self) -> None:
        raise NotImplementedError("Exl3Backend.reset_adapters is not implemented yet.")
