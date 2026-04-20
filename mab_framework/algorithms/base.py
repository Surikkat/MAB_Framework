from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Dict, Any
from mab_framework.models.base import BaseModel

class BaseAlgorithm(ABC):
    def __init__(self, n_arms: int, model: Union[BaseModel, List[BaseModel]]):
        self.n_arms = n_arms
        self.model = model

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        pass

    @abstractmethod
    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        pass