from abc import ABC, abstractmethod
import random
import time
import openai
from typing import Any


class BaseModel(ABC):
    def __init__(self, model_name: str, **kwargs):
        self._name = model_name
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.reasoning_effort = kwargs.get("reasoning_effort", None)
        self.n = kwargs.get("n", 1)
        self.input_tokens = 0
        self.output_tokens = 0

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs):
        pass

    @property
    def name(self) -> str:
        return self._name
