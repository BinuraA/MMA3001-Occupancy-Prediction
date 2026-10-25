
"""Performance evaluation functions for occupancy prediction."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_regression(y_true, y_pred):
    """
    Calculate occupancy prediction error metrics.

    Parameters
    ----------
    y_true : array-like
        Actual occupancy fractions.
    y_pred : array-like
        Predicted occupancy fractions.

    Returns
    -------
    dict
        Dictionary containing MAE and RMSE.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }


def significant_underprediction_rate(
    y_true,
    y_pred,
    high_threshold=0.8,
    error_threshold=0.3
):
    """
    Calculate significant underprediction during high occupancy.

    Parameters
    ----------
    y_true : array-like
        Actual occupancy fractions.
    y_pred : array-like
        Predicted occupancy fractions.
    high_threshold : float
        Minimum actual fraction considered high occupancy.
    error_threshold : float
        Minimum underprediction magnitude considered significant.

    Returns
    -------
    float
        Percentage of high-occupancy observations with
        significant underprediction. Returns NaN if there
        are no high-occupancy observations.
    """
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    high_occupancy = actual >= high_threshold

    if high_occupancy.sum() == 0:
        return np.nan

    significant_errors = (
        high_occupancy &
        ((actual - predicted) >= error_threshold)
    )

    return 100 * significant_errors.sum() / high_occupancy.sum()
