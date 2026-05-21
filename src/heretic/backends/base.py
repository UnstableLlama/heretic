# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import LongTensor, Tensor
from transformers import BatchEncoding
from transformers.generation import GenerateDecoderOnlyOutput

from heretic.utils import Prompt


class ModelBackend(ABC):
    """Backend interface for model runtime operations used by Heretic."""

    @abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def load_tokenizer(self, tokenizer_path: str | None = None, **kwargs: Any) -> Any: ...

    @abstractmethod
    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]: ...

    @abstractmethod
    def forward_logits(self, input_ids: Tensor, **kwargs: Any) -> Tensor: ...

    @abstractmethod
    def list_modules(self) -> list[str]: ...

    @abstractmethod
    def list_target_modules(self) -> list[str]: ...

    @abstractmethod
    def get_effective_weight(self, module_name: str) -> Tensor: ...

    @abstractmethod
    def apply_adapter(self, adapter_path: str, adapter_name: str = "default") -> None: ...

    @abstractmethod
    def unload_adapter(self, adapter_name: str = "default") -> None: ...

    @abstractmethod
    def reset_adapters(self) -> None: ...
