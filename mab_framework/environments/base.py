from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional

class DelayedFeedbackBuffer:
    def __init__(self):
        self.queue = {}
        self.current_time = 0

    def add(self, action: int, reward: float, delay: int, context: np.ndarray = None) -> None:
        target_time = self.current_time + delay
        if target_time not in self.queue:
            self.queue[target_time] = []
        self.queue[target_time].append({
            "action": action,
            "reward": reward,
            "context": context
        })

    def step(self) -> List[Dict[str, Any]]:
        matured = self.queue.pop(self.current_time, [])
        self.current_time += 1
        return matured


class BaseEnvironment(ABC):
    def __init__(self, delay_config: Optional[Dict] = None, **kwargs):
        self.delay_config = delay_config or {"type": "fixed", "value": 0}
        self.delay_buffer = DelayedFeedbackBuffer()
        super().__init__(**kwargs)

    def _sample_delay(self) -> int:
        d_type = self.delay_config.get("type", "fixed")
        if d_type == "fixed":
            return int(self.delay_config.get("value", 0))
        elif d_type == "geometric":
            p = float(self.delay_config.get("p", 1.0))
            return int(np.random.geometric(p) - 1)
        return 0

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_context(self) -> np.ndarray:
        pass

    @abstractmethod
    def _step_raw(self, action: int) -> tuple[float, float]:
        """
        Execute action and return (reward, optimal_reward) before any delay is applied.
        This must be implemented by subclasses.
        """
        pass
        
    def step(self, action: int) -> Dict[str, Any]:
        """
        Takes an action, potentially holds the reward in a buffer,
        and returns available rewards for this step.
        """
        context = self.get_context()
        reward, optimal_reward = self._step_raw(action)
        delay = self._sample_delay()
        
        self.delay_buffer.add(action=action, reward=reward, delay=delay, context=context)
        available_rewards = self.delay_buffer.step()
        
        return {
            "available_rewards": available_rewards,
            "instant_reward": float(reward),
            "optimal_reward": float(optimal_reward)
        }