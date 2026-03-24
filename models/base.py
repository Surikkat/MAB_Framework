from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: float):
        """Обновить веса модели на основе новых данных."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[float, float]:
        """Возвращает (ожидаемая_награда, неопределенность/стандартное_отклонение)."""
        pass

