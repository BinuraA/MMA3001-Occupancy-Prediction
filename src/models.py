
"""Machine learning models for predicting building occupancy."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def train_linear_model(X_train, y_train):
    """
    Train a Linear Regression model.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training input features.
    y_train : pandas.Series
        Actual occupancy fractions between 0 and 1.

    Returns
    -------
    sklearn.linear_model.LinearRegression
        Trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def train_decision_tree(X_train, y_train, max_depth=5):
    """
    Train a Decision Tree Regression model.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training input features.
    y_train : pandas.Series
        Actual occupancy fractions.
    max_depth : int
        Maximum depth of the decision tree.

    Returns
    -------
    sklearn.tree.DecisionTreeRegressor
        Trained Decision Tree model.
    """
    if not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def predict_occupancy(model, X):
    """
    Predict occupancy fractions using a trained model.

    Parameters
    ----------
    model : fitted regression model
        A trained model supporting predict().
    X : pandas.DataFrame
        Input features to make predictions on.

    Returns
    -------
    numpy.ndarray
        Predicted occupancy fractions clipped between 0 and 1.
    """
    predictions = model.predict(X)

    return np.clip(predictions, 0, 1)
