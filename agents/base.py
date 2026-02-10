from abc import ABC, abstractmethod
from core.llm import get_llm

class BaseAgent(ABC):
    def __init__(self):
        self.llm = get_llm()

    @abstractmethod
    def run(self, *args, **kwargs):
        pass