import numpy as np
from typing import Union, List, Dict, Any
from .base import BaseAlgorithm
from models.base import BaseModel

class ThompsonSampling(BaseAlgorithm):
    def __init__(self, n_arms: int, model: Union[BaseModel, List[BaseModel]]):
        super().__init__(n_arms, model)

    def select_arm(self, context: np.ndarray) -> int:
        sampled_values = []
        for a in range(self.n_arms):
            x_a = context[a] if context.ndim > 1 else context
            sample = self.model[a].sample(x_a)
            sampled_values.append(sample)
        return int(np.argmax(sampled_values))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            x_a = context[action] if context.ndim > 1 else context
            self.model[action].fit(x_a, reward)

