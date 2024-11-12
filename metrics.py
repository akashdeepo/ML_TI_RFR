# metrics.py
# Code for computing various metrics and plots.
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""
  I'd like to add a sell/hold/buy confusion matrix in here, but it will take some thought.
  It would be nice to know, e.g., how often we got a "buy" signal but should have sold,
  and so on.
"""

############################################################################
def calculate_metrics(SD, Model):
    """
    Calculate performance metrics for prediction model.
    
    Parameters:
    -----------
    SD : StockData object
        Contains training and test data
    Model : Predictor object
        The trained prediction model
        
    Returns:
    --------
    dict
        Dictionary containing calculated metrics
    """
    trainX, trainY = SD.get_train_set()
    testX, testY = SD.get_test_set()
    trainY_pred = Model.apply(trainX)
    testY_pred = Model.apply(testX)

    metrics = {
        'Model': Model.get_name(),
        'train start': str(trainX.index[0]),
        'train end': str(trainX.index[-1]),
        'test start': str(testX.index[0]),
        'test end': str(testX.index[-1]),
        'Train RMSE': np.sqrt(mean_squared_error(trainY, trainY_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(testY, testY_pred)),
        'Train MAE': mean_absolute_error(trainY, trainY_pred),
        'Test MAE': mean_absolute_error(testY, testY_pred),
        'Train R²': r2_score(trainY, trainY_pred),
        'Test R²': r2_score(testY, testY_pred),
        'Train trend accuracy (%)': calculate_trend_accuracy(trainY, trainY_pred),
        'Test trend accuracy (%)': calculate_trend_accuracy(testY, testY_pred),
        'Train dir. accuracy (%)': calculate_direction_accuracy(trainY, trainY_pred),
        'Test dir. accuracy (%)': calculate_direction_accuracy(testY, testY_pred),
        'Train correlation coefficient': np.corrcoef(trainY, trainY_pred)[0, 1],
        'Test correlation coefficient': np.corrcoef(testY, testY_pred)[0, 1]
    }
    
    # Handle feature importances with better error checking
    try:
        feature_importances = Model.feature_importances()
        if feature_importances is not None:
            metrics['features'] = SD.feature_column_names()
            metrics['feature_importances'] = feature_importances
    except (AttributeError, TypeError):
        pass

    return metrics

############################################################################
def save_predictions_to_csv(y_test, y_test_pred, model_name):
    """
    Save actual and predicted values to CSV.
    
    Parameters:
    -----------
    y_test : array-like
        Actual values
    y_test_pred : array-like
        Predicted values
    model_name : str
        Name of the model for the filename
    """
    predictions_df = pd.DataFrame({
        'y_test': y_test,
        'predictions': y_test_pred
    })
    predictions_df.to_csv(f'predictions_{model_name}.csv', index=False)

############################################################################
def calculate_trend_accuracy(y_true, y_pred):
    """
    Calculate accuracy of trend prediction (up/down movement).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        Percentage of correctly predicted trends
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
        
    actual_trend = np.sign(np.diff(y_true))
    predicted_trend = np.sign(np.diff(y_pred))
    trend_accuracy = np.mean(actual_trend == predicted_trend)
    return trend_accuracy * 100

############################################################################
def calculate_direction_accuracy(y_true, y_pred):
    """
    Calculate accuracy of direction prediction (positive/negative).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        Percentage of correctly predicted directions
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
        
    return np.mean(np.sign(y_true) == np.sign(y_pred)) * 100