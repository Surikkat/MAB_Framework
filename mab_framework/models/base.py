from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y: float) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def sample(self, x: np.ndarray) -> np.ndarray:
        pass

    def get_grad_features(self, x: np.ndarray) -> np.ndarray:
        """Return the gradient features (e.g. for Fisher matrix) if supported, else None."""
        return None

    def finalize_update(self) -> None:
        """Called by algorithms at the end of an update batch to sync/fit underlying heavy processes."""
        pass
