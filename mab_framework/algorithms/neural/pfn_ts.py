import copy
from bisect import bisect_right
from typing import Optional, List, Dict, Any
import numpy as np

from mab_framework.algorithms.base import BaseAlgorithm
from mab_framework.models.base import BaseModel

class PFNTSAlgorithm(BaseAlgorithm):
    """
    PFN Thompson Sampling via the Universal Subsampling CLT.
    Supports fixed encoding ("disjoint", "one_hot", "block") or adaptive 
    encoding selection ("adaptive", "auto").
    """
    def __init__(
        self,
        n_arms: int,
        n_features: int,
        model: BaseModel,
        encoding: str = "disjoint",
        alpha: float = 1.0,
        max_context: Optional[int] = None,
        initial_random_selections: int = 5,
        n_quantiles: int = 200,
        subsample_base: float = 2.0,
        random_state: Optional[int] = None,
        joint_threshold: int = 5,
        switch_times: Optional[list] = None,
        n_crps_thresholds: int = 50,
    ) -> None:
        super().__init__(n_arms, model)
        
        self.adaptive_mode = encoding in ("auto", "adaptive")
        if self.adaptive_mode:
            self.encoding = "one_hot" if n_arms >= joint_threshold else "disjoint"
        else:
            if encoding not in ("disjoint", "one_hot", "block"):
                raise ValueError(f"Unknown encoding '{encoding}'")
            self.encoding = encoding
            
        self.n_features = n_features
        self.max_context = max_context
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.t = 1
        
        self.alpha = alpha
        self.n_quantiles = n_quantiles
        self.subsample_base = subsample_base

        self._warmstart_arms: list[int] = []
        for _ in range(initial_random_selections):
            self._warmstart_arms.extend(self.rng.permutation(n_arms).tolist())

        if self.encoding == "disjoint":
            self._X: list[list[np.ndarray]] = [[] for _ in range(n_arms)]
            self._y: list[list[float]] = [[] for _ in range(n_arms)]
        else:
            self._X_joint: list[np.ndarray] = []
            self._y_joint: list[float] = []

        self.clf = copy.deepcopy(model)
        
        self._arm_grid_clfs: dict[int, list] = {}
        self._arm_eval_times: dict[int, list[int]] = {}
        self._arm_next_grid_point: dict[int, int] = {}
        
        if self.adaptive_mode:
            self.joint_threshold = joint_threshold
            self.challenger_encoding: str = (
                "disjoint" if self.encoding in ("one_hot", "block") else "one_hot"
            )
            self.n_crps_thresholds = n_crps_thresholds
            if switch_times is None:
                switch_times = [2**k for k in range(5, 12)]
            self._switch_times: list[int] = sorted(switch_times)
            self._next_switch_idx: int = 0
            self._dual_caching: bool = True

            self.challenger_clf = copy.deepcopy(model)

            self._raw_obs: list[tuple[np.ndarray, int, float]] = []

            if self.challenger_encoding == "disjoint":
                self._chal_X: list[list] = [[] for _ in range(n_arms)]
                self._chal_y: list[list] = [[] for _ in range(n_arms)]
            else:
                self._chal_X_joint: list = []
                self._chal_y_joint: list = []

            self._chal_arm_grid_clfs: dict[int, list] = {}
            self._chal_arm_eval_times: dict[int, list[int]] = {}
            self._chal_arm_next_grid_point: dict[int, int] = {}

            self._cumul_crps_active: float = 0.0
            self._cumul_crps_chal: float = 0.0

            self.switch_log: list[dict] = []

    def _encode_with(self, x: np.ndarray, arm: int, encoding: str) -> np.ndarray:
        if encoding == "disjoint":
            return x
        if encoding == "one_hot":
            return np.append(x, int(arm))
        expanded = np.zeros(self.n_features * self.n_arms, dtype=x.dtype)
        expanded[arm * self.n_features : (arm + 1) * self.n_features] = x
        return expanded

    def _encode(self, x: np.ndarray, arm: int) -> np.ndarray:
        return self._encode_with(x, arm, self.encoding)

    def _get_dataset(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        if self.encoding == "disjoint":
            rows = self._X[arm]
            d = self.n_features
            y_list = self._y[arm]
        else:
            rows = self._X_joint
            d = self.n_features * self.n_arms if self.encoding == "block" else self.n_features + 1
            y_list = self._y_joint

        X = np.array(rows) if rows else np.empty((0, d))
        y = np.array(y_list)
        return X, y

    def _get_query(self, x: np.ndarray, arm: int) -> np.ndarray:
        return self._encode(x, arm).reshape(1, -1)

    def _in_warmup(self) -> bool:
        return (self.t - 1) < len(self._warmstart_arms)

    def _warmup_arm(self) -> int:
        return int(self._warmstart_arms[self.t - 1])

    def _store(self, arm: int, x: np.ndarray, y: float) -> None:
        if self.encoding == "disjoint":
            self._X[arm].append(x.copy())
            self._y[arm].append(y)
            if self.max_context is not None and len(self._y[arm]) > self.max_context:
                self._X[arm] = self._X[arm][-self.max_context :]
                self._y[arm] = self._y[arm][-self.max_context :]
        else:
            self._X_joint.append(self._encode(x, arm).copy())
            self._y_joint.append(y)
            if self.max_context is not None and len(self._y_joint) > self.max_context:
                self._X_joint = self._X_joint[-self.max_context :]
                self._y_joint = self._y_joint[-self.max_context :]

    def _next_grid_point(self, s: int) -> int:
        return max(s + 1, int(np.floor(s * self.subsample_base)))

    def _build_eval_grid(self, n: int) -> list[int]:
        assert n >= 2, f"Expected at least 2 observations to build eval grid, got n={n}"
        grid: list[int] = []
        s = 2
        while s < n:
            grid.append(s)
            s = self._next_grid_point(s)
        grid.append(n)
        return grid

    def _init_clf_cache(self, x_prev: np.ndarray, y_prev: np.ndarray, arm: int) -> None:
        n = int(len(y_prev))
        eval_times = self._build_eval_grid(n)
        grid_clfs: list = []
        for s in eval_times:
            self.clf.condition(x_prev[:s], y_prev[:s])
            grid_clfs.append(copy.deepcopy(self.clf))
        self._arm_grid_clfs[arm] = grid_clfs
        self._arm_eval_times[arm] = eval_times
        self._arm_next_grid_point[arm] = self._next_grid_point(n)

    def _extend_clf_cache(self, x_prev: np.ndarray, y_prev: np.ndarray, arm: int) -> None:
        n = int(len(y_prev))
        self.clf.condition(x_prev, y_prev)
        self._arm_grid_clfs[arm].append(copy.deepcopy(self.clf))
        self._arm_eval_times[arm].append(n)
        self._arm_next_grid_point[arm] = self._next_grid_point(n)

    def _ensure_arm_cache(self, arm: int, x_prev: np.ndarray, y_prev: np.ndarray) -> None:
        n = len(y_prev)
        if arm not in self._arm_grid_clfs:
            self._init_clf_cache(x_prev, y_prev, arm=arm)
        elif n >= self._arm_next_grid_point[arm]:
            self._extend_clf_cache(x_prev, y_prev, arm=arm)

    def _thompson_sample(self, arm: int, x: np.ndarray) -> float:
        x_prev, y_prev = self._get_dataset(arm)
        n = len(y_prev)

        if n < 2:
            raise ValueError(f"Unexpected dataset with n={n} < 2 for arm {arm} at Thompson sampling")

        y_min, y_max = float(y_prev.min()), float(y_prev.max())
        if y_min == y_max:
            return y_min

        x_query = self._get_query(x, arm)
        self._ensure_arm_cache(arm, x_prev, y_prev)

        grid_clfs = self._arm_grid_clfs[arm]
        eval_times = self._arm_eval_times[arm]
        means = [float(clf_snap.mean_cached(x_query)[0]) for clf_snap in grid_clfs]

        mu = means[-1]
        sigma2 = self._sigma2_from_means(means, eval_times)
        return float(self.rng.normal(mu, np.sqrt(self.alpha * sigma2 + 1e-12)))

    @staticmethod
    def _sigma2_from_means(means: list[float], eval_times: list[int]) -> float:
        k_blocks = len(eval_times) - 1
        n = eval_times[-1]
        if k_blocks == 0:
            return 0.0

        v_hat = 0.0
        for k in range(1, k_blocks + 1):
            s_k = eval_times[k]
            s_prev = eval_times[k - 1]
            d_k = float(means[k] - means[k - 1])
            harmonic_weight = (s_k * s_prev) / (s_k - s_prev)
            v_hat += harmonic_weight * (d_k * d_k)

        v_hat /= k_blocks
        return max(0.0, float(v_hat / n))

    def select_arm(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=float).reshape(-1)
        if self._in_warmup():
            return self._warmup_arm()
        samples = np.array([self._thompson_sample(k, x) for k in range(self.n_arms)])
        return int(np.argmax(samples))

    # Adaptive methods
    def _chal_dataset(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        if self.challenger_encoding == "disjoint":
            rows = self._chal_X[arm]
            d = self.n_features
            y_list = self._chal_y[arm]
        else:
            rows = self._chal_X_joint
            d = self.n_features + 1 if self.challenger_encoding == "one_hot" else self.n_features * self.n_arms
            y_list = self._chal_y_joint
        X = np.array(rows) if rows else np.empty((0, d))
        y = np.array(y_list)
        return X, y

    def _maybe_extend_chal_cache(self, arm: int) -> None:
        cache_arm = arm if self.challenger_encoding == "disjoint" else 0
        x_prev, y_prev = self._chal_dataset(arm)
        n = len(y_prev)
        if n < 2:
            return

        if cache_arm not in self._chal_arm_grid_clfs:
            eval_times = self._build_eval_grid(n)
            grid_clfs: list = []
            for s in eval_times:
                self.challenger_clf.condition(x_prev[:s], y_prev[:s])
                grid_clfs.append(copy.deepcopy(self.challenger_clf))
            self._chal_arm_grid_clfs[cache_arm] = grid_clfs
            self._chal_arm_eval_times[cache_arm] = eval_times
            self._chal_arm_next_grid_point[cache_arm] = self._next_grid_point(n)
        elif n >= self._chal_arm_next_grid_point[cache_arm]:
            self.challenger_clf.condition(x_prev, y_prev)
            self._chal_arm_grid_clfs[cache_arm].append(copy.deepcopy(self.challenger_clf))
            self._chal_arm_eval_times[cache_arm].append(n)
            self._chal_arm_next_grid_point[cache_arm] = self._next_grid_point(n)

    @staticmethod
    def _crps_one(snap, x_query: np.ndarray, y_obs: float, thresholds: np.ndarray) -> float:
        cdf_vals = snap.cdf_cached(thresholds, x_query).ravel()
        indicators = (thresholds >= y_obs).astype(float)
        return float(np.trapezoid((cdf_vals - indicators) ** 2, thresholds))

    def _lookup_crps(self, j: int, per_arm_pos: int, x: np.ndarray, arm: int, y: float, grid_clfs: dict, eval_times_dict: dict, encoding: str, thresholds: np.ndarray) -> Optional[float]:
        cache_arm = arm if encoding == "disjoint" else 0
        pos = per_arm_pos if encoding == "disjoint" else j

        times = eval_times_dict.get(cache_arm)
        if not times:
            return None

        idx = bisect_right(times, pos) - 1
        if idx < 0:
            return None

        x_query = self._encode_with(x, arm, encoding).reshape(1, -1)
        return self._crps_one(grid_clfs[cache_arm][idx], x_query, y, thresholds)

    def _score_interval(self, t_prev: int, t_curr: int) -> tuple[float, float]:
        interval_obs = self._raw_obs[t_prev:t_curr]
        if not interval_obs:
            return 0.0, 0.0

        y_arr = np.array([y for _, _, y in interval_obs])
        y_std = max(float(y_arr.std()), 1e-8)
        thresholds = np.linspace(float(y_arr.min()) - y_std, float(y_arr.max()) + y_std, self.n_crps_thresholds)

        arm_counts = [0] * self.n_arms
        for _, a, _ in self._raw_obs[:t_prev]:
            arm_counts[a] += 1

        delta_a = 0.0
        delta_c = 0.0

        for j_off, (x, arm, y) in enumerate(interval_obs):
            j = t_prev + j_off
            per_arm_pos = arm_counts[arm]

            crps_a = self._lookup_crps(j, per_arm_pos, x, arm, y, self._arm_grid_clfs, self._arm_eval_times, self.encoding, thresholds)
            crps_c = self._lookup_crps(j, per_arm_pos, x, arm, y, self._chal_arm_grid_clfs, self._chal_arm_eval_times, self.challenger_encoding, thresholds)

            arm_counts[arm] += 1

            if crps_a is None or crps_c is None:
                continue

            delta_a += crps_a
            delta_c += crps_c

        return delta_a, delta_c

    def _rebuild_data_buffers(self) -> None:
        if self.encoding == "disjoint":
            self._X = [[] for _ in range(self.n_arms)]
            self._y = [[] for _ in range(self.n_arms)]
            for x, arm, y in self._raw_obs:
                self._X[arm].append(x.copy())
                self._y[arm].append(y)
            if self.max_context is not None:
                for arm in range(self.n_arms):
                    self._X[arm] = self._X[arm][-self.max_context :]
                    self._y[arm] = self._y[arm][-self.max_context :]
        else:
            self._X_joint = [self._encode(x, a).copy() for x, a, _ in self._raw_obs]
            self._y_joint = [y for _, _, y in self._raw_obs]
            if self.max_context is not None:
                self._X_joint = self._X_joint[-self.max_context :]
                self._y_joint = self._y_joint[-self.max_context :]

    def _rebuild_chal_buffers(self) -> None:
        if self.challenger_encoding == "disjoint":
            self._chal_X = [[] for _ in range(self.n_arms)]
            self._chal_y = [[] for _ in range(self.n_arms)]
            for x, arm, y in self._raw_obs:
                self._chal_X[arm].append(x.copy())
                self._chal_y[arm].append(y)
        else:
            self._chal_X_joint = [self._encode_with(x, a, self.challenger_encoding).copy() for x, a, _ in self._raw_obs]
            self._chal_y_joint = [y for _, _, y in self._raw_obs]

    def _run_switch_check(self) -> None:
        t_prev = self._switch_times[self._next_switch_idx - 1] if self._next_switch_idx > 0 else 0
        t_curr = len(self._raw_obs)

        delta_a, delta_c = self._score_interval(t_prev, t_curr)
        self._cumul_crps_active += delta_a
        self._cumul_crps_chal += delta_c

        do_switch = self._cumul_crps_chal < self._cumul_crps_active
        is_last = self._next_switch_idx == len(self._switch_times) - 1
        terminate = is_last

        self.switch_log.append({
            "n": t_curr,
            "active_encoding": self.encoding,
            "active_cumul_crps": self._cumul_crps_active,
            "challenger_encoding": self.challenger_encoding,
            "chal_cumul_crps": self._cumul_crps_chal,
            "switched": do_switch,
            "terminated": terminate,
        })

        if do_switch:
            self.encoding, self.challenger_encoding = self.challenger_encoding, self.encoding
            self.clf, self.challenger_clf = self.challenger_clf, self.clf
            self._arm_grid_clfs, self._chal_arm_grid_clfs = self._chal_arm_grid_clfs, self._arm_grid_clfs
            self._arm_eval_times, self._chal_arm_eval_times = self._chal_arm_eval_times, self._arm_eval_times
            self._arm_next_grid_point, self._chal_arm_next_grid_point = self._chal_arm_next_grid_point, self._arm_next_grid_point
            self._rebuild_data_buffers()
            self._rebuild_chal_buffers()
            self._cumul_crps_active, self._cumul_crps_chal = self._cumul_crps_chal, self._cumul_crps_active

        if terminate:
            self._dual_caching = False
            self.challenger_clf = None
            self._chal_arm_grid_clfs = {}
            self._chal_arm_eval_times = {}
            self._chal_arm_next_grid_point = {}
            if hasattr(self, "_chal_X"):
                self._chal_X = [[] for _ in range(self.n_arms)]
                self._chal_y = [[] for _ in range(self.n_arms)]
            if hasattr(self, "_chal_X_joint"):
                self._chal_X_joint = []
                self._chal_y_joint = []

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for feedback in feedbacks:
            arm = feedback.get("arm", feedback.get("action"))
            context = np.asarray(feedback["context"], dtype=float).reshape(-1)
            reward = float(feedback["reward"])

            if self.adaptive_mode:
                self._raw_obs.append((context.copy(), int(arm), reward))
                if self._dual_caching:
                    if self.challenger_encoding == "disjoint":
                        self._chal_X[arm].append(context.copy())
                        self._chal_y[arm].append(reward)
                    else:
                        self._chal_X_joint.append(self._encode_with(context, arm, self.challenger_encoding))
                        self._chal_y_joint.append(reward)

            self._store(arm, context, reward)
            self.t += 1

            if self.adaptive_mode and self._dual_caching:
                self._maybe_extend_chal_cache(arm)

                n = len(self._raw_obs)
                if self._next_switch_idx < len(self._switch_times) and n == self._switch_times[self._next_switch_idx]:
                    self._run_switch_check()
                    self._next_switch_idx += 1
