
"""Automated tests for the occupancy prediction project."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import prepare_occupancy_data
from src.models import (
    train_linear_model,
    train_decision_tree,
    predict_occupancy
)
from src.evaluation import (
    evaluate_regression,
    significant_underprediction_rate
)


def test_preprocessing_output():
    """Verify processed occupancy values and feature columns."""
    index = pd.date_range(
        "2024-08-05",
        periods=180,
        freq="1min",
        tz="Australia/Melbourne"
    )

    sensor = pd.DataFrame(
        {"occupied": [1] * 90 + [0] * 90},
        index=index
    )

    result = prepare_occupancy_data(sensor, fill_limit=5)

    assert not result.empty
    assert result["occupancy_fraction"].between(0, 1).all()
    assert result["target"].between(0, 1).all()
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique


def test_invalid_fill_limit():
    """Reject negative carry-forward limits."""
    index = pd.date_range("2024-08-05", periods=30, freq="1min")
    sensor = pd.DataFrame({"occupied": [1] * 30}, index=index)

    with pytest.raises(ValueError):
        prepare_occupancy_data(sensor, fill_limit=-1)


def test_missing_column():
    """Reject sensor data without an occupied column."""
    index = pd.date_range("2024-08-05", periods=30, freq="1min")
    sensor = pd.DataFrame({"headcount": [1] * 30}, index=index)

    with pytest.raises(ValueError):
        prepare_occupancy_data(sensor)


def test_linear_regression():
    """Verify Linear Regression predictions are bounded."""
    X = pd.DataFrame({"hour": [8, 9, 10, 11, 12]})
    y = pd.Series([0.1, 0.3, 0.6, 0.9, 1.0])

    model = train_linear_model(X, y)
    predictions = predict_occupancy(model, X)

    assert len(predictions) == len(y)
    assert np.all((predictions >= 0) & (predictions <= 1))


def test_decision_tree_reproducibility():
    """Verify deterministic Decision Tree predictions."""
    X = pd.DataFrame({"hour": [8, 9, 10, 11, 12]})
    y = pd.Series([0.1, 0.3, 0.6, 0.9, 1.0])

    model1 = train_decision_tree(X, y, max_depth=3)
    model2 = train_decision_tree(X, y, max_depth=3)

    np.testing.assert_allclose(
        predict_occupancy(model1, X),
        predict_occupancy(model2, X)
    )


def test_invalid_tree_depth():
    """Reject a tree with non-positive maximum depth."""
    X = pd.DataFrame({"hour": [8, 9, 10]})
    y = pd.Series([0.1, 0.5, 0.9])

    with pytest.raises(ValueError):
        train_decision_tree(X, y, max_depth=0)


def test_evaluation_metrics():
    """Verify MAE and RMSE against manually calculated values."""
    actual = np.array([0.0, 1.0])
    predicted = np.array([0.2, 0.8])

    result = evaluate_regression(actual, predicted)

    assert np.isclose(result["MAE"], 0.2)
    assert np.isclose(result["RMSE"], 0.2)


def test_underprediction_rate():
    """Verify significant underprediction rate."""
    actual = np.array([0.9, 0.9, 0.1])
    predicted = np.array([0.4, 0.8, 0.1])

    rate = significant_underprediction_rate(actual, predicted)

    assert np.isclose(rate, 50.0)
