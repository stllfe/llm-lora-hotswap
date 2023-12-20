"""Base LLM adapter interface for inheritance."""

import abc

from typing import Iterable, Iterator

import torch

from transformers import PreTrainedModel


class LLMAdapterBase(abc.ABC):

    def __init__(self, llm: PreTrainedModel, model_id: str, adapter_name: str | None) -> None:
        super().__init__()
        self._llm = llm
        self._llm.load_adapter(model_id, adapter_name=adapter_name)
        self._name = adapter_name

    @property
    def name(self) -> str:
        return self._name

    @torch.inference_mode()
    def generate(self, message: str, history: list[tuple[str, str]] | None = None) -> Iterable[str]:
        self._llm.eval()
        self._llm.set_adapter(self.name)

        prompt = self.preformat(message, history=history or [])
        for token in self.generator(self._llm, prompt):
            yield token

    @abc.abstractmethod
    def preformat(self, message: str, history: list[tuple[str, str]]) -> str:
        """Creates a model prompt from the current chat message and message history."""

    @abc.abstractmethod
    def generator(self, llm: PreTrainedModel, prompt: str) -> Iterator[str]:
        """Generates the model outputs with the given model."""
