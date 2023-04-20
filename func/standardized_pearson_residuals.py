import pandas as pd
import numpy as np

from rocky3.func.hat_matrix import calculate_hat_matrix_poisson, calculate_hat_matrix_gamma

def standardized_pearson_residuals_poisson(df, model, y_col='y', y_pred_col='y_pred', x_col=None):
    """
    Calculate standardized Pearson residuals for the given DataFrame using a Poisson GLM.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the specified columns
    model : sklearn.linear_model.PoissonRegressor
        The fitted PoissonRegressor model
    y_col : str, optional
        The name of the column containing the observed values (default: 'y')
    y_pred_col : str, optional
        The name of the column containing the expected values (default: 'y_pred')
    x_col : list of str, optional
        The list of column names used as predictors in the model
    """
    
    hat_matrix = calculate_hat_matrix_poisson(df, model, y_pred_col=y_pred_col, x_col=x_col)
    
    y = df[y_col].values
    y_pred = df[y_pred_col].values
    pearson_res = (y - y_pred) / np.sqrt(y_pred)

    h = np.diag(hat_matrix)
    standardized_res = pearson_res / np.where(np.logical_not(np.equal(h, 1)), np.sqrt(1 - h), np.nan)
    return standardized_res