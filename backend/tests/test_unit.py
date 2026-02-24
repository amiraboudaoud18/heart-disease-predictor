import sys
from pathlib import Path

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Test 1: Preprocessing output shapes are correct ──────────────────────────
def test_preprocessing_output_shapes():
    """Preprocessing must return correct shapes for train/test splits."""
    from app.preprocess import load_and_preprocess

    data_path = str(Path(__file__).parent.parent.parent / "data/heart.csv")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        data_path
    )

    assert X_train.shape[1] == 13, "Must have 13 features"
    assert X_train.shape[0] == len(y_train), "X_train and y_train must match"
    assert X_test.shape[0] == len(y_test), "X_test and y_test must match"
    assert len(feature_names) == 13, "Must return 13 feature names"


# ── Test 2: Target is binary (0 or 1 only) ───────────────────────────────────
def test_preprocessing_binary_target():
    """Target variable must be binary after preprocessing."""
    from app.preprocess import load_and_preprocess

    data_path = str(Path(__file__).parent.parent.parent / "data/heart.csv")
    _, _, y_train, y_test, _, _ = load_and_preprocess(data_path)

    all_values = set(y_train.unique()) | set(y_test.unique())
    assert all_values.issubset({0, 1}), f"Target must be binary, got {all_values}"


# ── Test 3: Scaler normalizes data correctly ──────────────────────────────────
def test_scaler_normalization():
    """StandardScaler must produce zero mean and unit variance."""
    from app.preprocess import load_and_preprocess

    data_path = str(Path(__file__).parent.parent.parent / "data/heart.csv")
    X_train, _, _, _, _, _ = load_and_preprocess(data_path)

    means = np.abs(X_train.mean(axis=0))
    stds = X_train.std(axis=0)

    assert np.all(means < 0.1), "Scaled features must have near-zero mean"
    assert np.all(np.abs(stds - 1.0) < 0.1), "Scaled features must have unit variance"
