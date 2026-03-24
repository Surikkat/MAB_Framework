from abc import ABC, abstractmethod
import numpy as np

class BaseEnvironment(ABC):
    """Базовый абстрактный класс для сред и генераторов данных."""
    
    @abstractmethod
    def reset(self) -> None:
        """
        Сбрасывает среду в начальное состояние (например, для начала нового эксперимента
        на тех же данных без пересоздания объекта среды).
        """
        pass

    @abstractmethod
    def get_context(self) -> np.ndarray:
        """
        Возвращает текущий контекст (вектор признаков ситуации/пользователя).
        """
        pass

    @abstractmethod
    def step(self, action: int) -> float:
        """
        Принимает действие (номер ручки) и возвращает полученную награду (reward).
        """
        pass