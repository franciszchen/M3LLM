# model_wrappers/vlm_base.py
from abc import ABC, abstractmethod

class VLMBase(ABC):
    @abstractmethod
    def generate_output(self, messages: dict) -> str:
        pass