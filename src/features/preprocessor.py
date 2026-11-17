"""Shared feature preprocessor (log1p + RobustScaler) fitted strictly on X_train."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.features.extractor import FEATURE_NAMES, NUM_FEATURES


class FlowPreprocessor:
    """Clips, log1p-transforms, and Robust-scales flow feature matrices using pure NumPy."""

    def __init__(
        self,
        clip_max: float = 1_000_000_000.0,
        quantile_range: tuple[float, float] = (10.0, 90.0),
    ) -> None:
        self.clip_max = float(clip_max)
        self.quantile_range = quantile_range
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.is_fitted = False
        self.feature_names = list(FEATURE_NAMES)
        self.active_mask: np.ndarray | None = None

    def _apply_mask(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if self.active_mask is not None:
            arr = arr * self.active_mask
        return arr

    def _log_transform(self, x: np.ndarray) -> np.ndarray:
        arr = self._apply_mask(x)
        arr = np.nan_to_num(arr, nan=0.0, posinf=self.clip_max, neginf=0.0)
        arr = np.clip(arr, 0.0, self.clip_max)
        return np.log1p(arr)

    def fit(self, x_train: np.ndarray) -> "FlowPreprocessor":
        """Fit robust median/IQR scaler strictly on shuffled X_train."""
        if x_train.ndim != 2 or x_train.shape[1] != NUM_FEATURES:
            raise ValueError(f"Expected shape (N, {NUM_FEATURES}), got {x_train.shape}")
        self.active_mask = np.any(x_train != 0.0, axis=0).astype(np.float32)
        x_log = self._log_transform(x_train)

        q_low, q_high = self.quantile_range
        q_vals = np.percentile(x_log, [q_low, 50.0, q_high], axis=0)
        center = q_vals[1].astype(np.float64)
        scale = (q_vals[2] - q_vals[0]).astype(np.float64)
        scale[scale == 0.0] = 1.0

        # Prevent unscaled continuous log1p features when inter-decile range is 0
        for i in range(NUM_FEATURES):
            if self.active_mask[i] > 0 and scale[i] == 1.0 and np.max(x_log[:, i]) > 1.0:
                nz = x_log[x_log[:, i] > 0, i]
                if len(nz) > 0:
                    scale[i] = float(max(np.mean(nz), 1.0))

        self.center_ = center
        self.scale_ = scale
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform feature matrix to float32 normalized representation."""
        if not self.is_fitted or self.center_ is None or self.scale_ is None:
            raise RuntimeError("FlowPreprocessor must be fitted before transform().")
        x_log = self._log_transform(x)
        scaled = (x_log - self.center_) / self.scale_
        scaled = self._apply_mask(scaled)
        scaled = np.clip(scaled, -20.0, 20.0)
        return np.nan_to_num(scaled, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        """Fit on X_train and return transformed X_train."""
        return self.fit(x_train).transform(x_train)

    def save(self, path: str | Path) -> None:
        """Persist preprocessor artifact with joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "FlowPreprocessor":
        """Load persisted preprocessor artifact."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected FlowPreprocessor, got {type(obj).__name__}")
        return obj
