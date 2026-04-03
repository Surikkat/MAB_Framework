from abc import ABC, abstractmethod
import numpy as np

class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_context(self) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[float, float]:
        pass