
"""
BaseAgent: defines interface and common utilities for agents.
"""
from abc import ABC, abstractmethod
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.memory = []

    @abstractmethod
    def plan(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def act(self, *args, **kwargs):
        raise NotImplementedError

    def reflect(self, note: str):
        logger.info(f"{self.name} reflecting: {note}")
        self.memory.append(note)
