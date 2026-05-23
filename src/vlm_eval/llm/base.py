from __future__ import annotations

from abc import ABC, abstractmethod


class LLM(ABC):
    def __init__(self, model: str):
        self.model = model
        self.cum_prompt_tokens = 0
        self.cum_completion_tokens = 0

    @abstractmethod
    def generate(self, prompt: str, sys_prompt: str | None = None) -> tuple[str, int, int]:
        raise NotImplementedError

    @abstractmethod
    def change_model(self, model: str) -> None:
        raise NotImplementedError
