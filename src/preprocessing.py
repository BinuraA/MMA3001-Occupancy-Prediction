
"""Functions for preparing building occupancy sensor data."""

import numpy as np
import pandas as pd


def prepare_occupancy_data(clean_data, fill_limit=5):
    """
    Convert irregular occupancy readings into model-ready data.

    Parameters
    ----------
    clean_data : pandas.DataFrame
        DataFrame with a DatetimeIndex and an 'occupied' column
        containing binary occupancy values (0 or 1).
    fill_limit : int, optional
        Maximum number of missing one-minute intervals to fill.
        Default is 5 minutes.

    Returns
    -------
    pandas.DataFrame
        Occupancy fractions, time-based features and prediction
        targets for the following 15-minute interval.

    Raises
    ------
    ValueError
        If fill_limit is invalid or required data is missing.
    """
    if not isinstance(fill_limit, int) or fill_limit < 0:
        raise ValueError("fill_limit must be a non-negative integer")

    if 'occupied' not in clean_data.columns:
        raise ValueError("Missing required column: occupied")

    if not isinstance(clean_data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex")

    # Convert readings into one-minute intervals
    minute_series = (
        clean_data['occupied']
        .sort_index()
        .resample('1min')
        .last()
    )

    # Carry previous occupancy state forward
    if fill_limit > 0:
        minute_series = minute_series.ffill(limit=fill_limit)

    # Aggregate into 15-minute occupancy fractions
    intervals = minute_series.resample('15min').agg(
        ['mean', 'count']
    )

    intervals.columns = [
        'occupancy_fraction',
        'valid_minutes'
    ]

    # Require at least 12 valid minutes per interval
    intervals.loc[
        intervals['valid_minutes'] < 12,
        'occupancy_fraction'
    ] = np.nan

    result = intervals[['occupancy_fraction']].copy()

    # Create model input features
    result['current_occupancy'] = result['occupancy_fraction']
    result['previous_occupancy'] = result['occupancy_fraction'].shift(1)

    # Next interval is the prediction target
    result['target'] = result['occupancy_fraction'].shift(-1)

    next_time = result.index + pd.Timedelta(minutes=15)

    result['hour'] = next_time.hour
    result['day_of_week'] = next_time.dayofweek
    result['weekend'] = (result['day_of_week'] >= 5).astype(int)

    return result.dropna().copy()
