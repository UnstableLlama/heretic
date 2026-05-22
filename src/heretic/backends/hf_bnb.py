# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import Any

from torch import LongTensor, Tensor
from transformers import BatchEncoding
from transformers.generation import GenerateDecoderOnlyOutput

from heretic.backends.base import ModelBackend
from heretic.utils import Prompt


class HfBnbBackend(ModelBackend):
    """HF/Transformers + optional bitsandbytes backend facade.

    This class is intentionally lightweight for now and delegates to the existing
    model wrapper. It provides a migration bridge while Heretic runtime logic is
    moved toward a backend-agnostic architecture.
    """

    def __init__(self, model_wrapper: Any):
        self.model_wrapper = model_wrapper

    def load_model(self, model_path: str, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "HfBnbBackend.load_model is not yet independently implemented. "
            "Use Model(settings) construction for now."
        )

    def load_tokenizer(self, tokenizer_path: str | None = None, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "HfBnbBackend.load_tokenizer is not yet independently implemented."
        )

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        return self.model_wrapper.generate(prompts, **kwargs)

    def forward_logits(self, input_ids: Tensor, **kwargs: Any) -> Tensor:
        outputs = self.model_wrapper.model(input_ids=input_ids, **kwargs)
        return outputs.logits

    def list_modules(self) -> list[str]:
        return [name for name, _ in self.model_wrapper.model.named_modules()]

    def list_target_modules(self) -> list[str]:
        return self.model_wrapper.get_abliterable_components()

    def get_effective_weight(self, module_name: str) -> Tensor:
        raise NotImplementedError(
            "HfBnbBackend.get_effective_weight is pending extraction from Model.abliterate()."
        )

    def apply_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        raise NotImplementedError(
            "HfBnbBackend.apply_adapter is pending adapter lifecycle refactor."
        )

    def unload_adapter(self, adapter_name: str = "default") -> None:
        raise NotImplementedError(
            "HfBnbBackend.unload_adapter is pending adapter lifecycle refactor."
        )

    def reset_adapters(self) -> None:
        self.model_wrapper.reset_model()
