"""Tests for FlowPreprocessor fit/transform and persistence."""

from pathlib import Path

import numpy as np

from src.features.extractor import NUM_FEATURES
from src.features.preprocessor import FlowPreprocessor


def test_preprocessor_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    x_train = np.abs(rng.normal(loc=100.0, scale=50.0, size=(200, NUM_FEATURES))).astype(np.float32)
    x_test = np.abs(rng.normal(loc=100.0, scale=50.0, size=(20, NUM_FEATURES))).astype(np.float32)
    x_test[0, 0] = np.nan
    x_test[1, 1] = 1e15

    prep = FlowPreprocessor(clip_max=1e9)
    x_train_s = prep.fit_transform(x_train)
    assert x_train_s.shape == x_train.shape
    assert np.all(np.isfinite(x_train_s))

    save_path = tmp_path / "prep.joblib"
    prep.save(save_path)
    loaded = FlowPreprocessor.load(save_path)

    x_test_1 = prep.transform(x_test)
    x_test_2 = loaded.transform(x_test)
    assert np.all(np.isfinite(x_test_1))
    assert np.allclose(x_test_1, x_test_2)
