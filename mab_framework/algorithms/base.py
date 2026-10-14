from abc import ABC, abstractmethod
import copy
import numpy as np
from typing import Union, List, Dict, Any
from mab_framework.models.base import BaseModel


class BaseAlgorithm(ABC):
    def __init__(self, n_arms: int, model: Union[BaseModel, List[BaseModel]]):
        self.n_arms = n_arms
        if isinstance(model, list):
            if len(model) != n_arms:
                raise ValueError(
                    f"Expected {n_arms} models (one per arm), got {len(model)}"
                )
            self.model = model
        else:
            # Deep-copy so each arm has its own independent model
            self.model = [copy.deepcopy(model) for _ in range(n_arms)]

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        pass

    @abstractmethod
    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        pass