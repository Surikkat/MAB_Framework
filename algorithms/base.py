from abc import ABC, abstractmethod
import numpy as np

class BaseAlgorithm(ABC):
    """Базовый абстрактный класс для алгоритмов многоруких бандитов."""

    def __init__(self, n_arms: int):
        self.n_arms = n_arms

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """
        Метод должен возвращать индекс выбранной ручки (от 0 до n_arms - 1).
        """
        pass

    @abstractmethod
    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """
        Метод для обновления весов/знаний алгоритма после получения награды.
        """
        pass