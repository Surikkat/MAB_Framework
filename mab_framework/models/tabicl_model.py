import numpy as np
import torch
from typing import Any
from tabicl import TabICLRegressor
from tabicl._model.quantile_dist import QuantileDistribution
from mab_framework.models.base import BaseModel

class TabICLRegressorPPD(TabICLRegressor, BaseModel):
    """
    Extends TabICLRegressor with CDF evaluation and posterior sampling,
    needed for the predictive CLT and rollout algorithms.
    """

    @staticmethod
    def _check_shapes(
        x_new: np.ndarray | None = None,
        x_prev: np.ndarray | None = None,
        y_prev: np.ndarray | None = None,
    ) -> None:
        """Assert that input arrays have the shapes documented in the public API."""
        if x_prev is not None:
            assert x_prev.ndim == 2, f"x_prev must be 2D, got shape {x_prev.shape}"
        if y_prev is not None:
            assert y_prev.ndim == 1, f"y_prev must be 1D, got shape {y_prev.shape}"
        if x_prev is not None and y_prev is not None:
            assert x_prev.shape[0] == y_prev.shape[0], (
                f"x_prev and y_prev must have the same n, "
                f"got {x_prev.shape[0]} and {y_prev.shape[0]}"
            )
        if x_new is not None:
            assert x_new.ndim == 2, f"x_new must be 2D, got shape {x_new.shape}"
        if x_new is not None and x_prev is not None:
            assert x_new.shape[1] == x_prev.shape[1], (
                f"x_new and x_prev must have the same d, "
                f"got {x_new.shape[1]} and {x_prev.shape[1]}"
            )

    def condition(self, x_prev: np.ndarray, y_prev: np.ndarray) -> None:
        """
        Fit the model on (x_prev, y_prev).
        """
        x_prev = np.asarray(x_prev)
        y_prev = np.asarray(y_prev)
        self._check_shapes(x_prev=x_prev, y_prev=y_prev)
        self.fit(x_prev, y_prev)

    def _predict(self, x_new: np.ndarray, output_type: str = "raw_quantiles") -> Any:
        """Call predict() forwarding output_type. Default is ``"raw_quantiles"`` for CDF/sampling."""
        return self.predict(x_new, output_type=output_type)

    def _condition_and_predict(
        self,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        output_type: str = "raw_quantiles",
    ) -> Any:
        """Condition on (x_prev, y_prev) and return predictive output for x_new."""
        self.condition(x_prev, y_prev)
        return self._predict(x_new, output_type=output_type)

    @staticmethod
    def _cdf_from_output(pred_output: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute CDF values from a ``"raw_quantiles"`` predict output. Returns (p, m)."""
        dist = QuantileDistribution(torch.from_numpy(pred_output).float())
        m = pred_output.shape[0]
        t = np.atleast_1d(t).astype(np.float32)
        p = len(t)
        z = torch.from_numpy(t).unsqueeze(0).expand(m, p)  # (m, p)
        return dist.cdf(z).detach().numpy().T  # (p, m)

    def cdf(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        x_new = np.asarray(x_new)
        x_prev = np.asarray(x_prev)
        y_prev = np.asarray(y_prev)
        self._check_shapes(x_new=x_new, x_prev=x_prev, y_prev=y_prev)
        return self._cdf_from_output(
            self._condition_and_predict(x_new, x_prev, y_prev), t
        )

    def cdf_cached(self, t: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new)
        self._check_shapes(x_new=x_new)
        return self._cdf_from_output(self._predict(x_new), t)

    def sample(
        self,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        rng: np.random.Generator = None,
        size: int = 1,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        x_new = np.asarray(x_new)
        x_prev = np.asarray(x_prev)
        y_prev = np.asarray(y_prev)
        self._check_shapes(x_new=x_new, x_prev=x_prev, y_prev=y_prev)
        raw_q = self._condition_and_predict(x_new, x_prev, y_prev)
        dist = QuantileDistribution(torch.from_numpy(raw_q).float())
        m = x_new.shape[0]
        u = rng.uniform(1e-5, 1 - 1e-5, size=(m, size)).astype(np.float32)
        samples = dist.icdf(torch.from_numpy(u))  # (m, size)
        return samples.detach().numpy().T  # (size, m)

    def mean(
        self, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
    ) -> np.ndarray:
        x_new = np.asarray(x_new)
        x_prev = np.asarray(x_prev)
        y_prev = np.asarray(y_prev)
        self._check_shapes(x_new=x_new, x_prev=x_prev, y_prev=y_prev)
        return self._condition_and_predict(x_new, x_prev, y_prev, output_type="mean")

    def mean_cached(self, x_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new)
        self._check_shapes(x_new=x_new)
        return self._predict(x_new, output_type="mean")

    def sample_cached(
        self,
        x_new: np.ndarray,
        rng: np.random.Generator = None,
        size: int = 1,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        x_new = np.asarray(x_new)
        self._check_shapes(x_new=x_new)
        raw_q = self._predict(x_new)
        dist = QuantileDistribution(torch.from_numpy(raw_q).float())
        m = x_new.shape[0]
        u = rng.uniform(1e-5, 1 - 1e-5, size=(m, size)).astype(np.float32)
        samples = dist.icdf(torch.from_numpy(u))  # (m, size)
        return samples.detach().numpy().T  # (size, m)
