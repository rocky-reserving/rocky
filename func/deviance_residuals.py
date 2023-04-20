import numpy as np
import pandas as pd

def poisson_deviance_residuals(df : pd.DataFrame
                               , y_col : str = 'y'
                               , yhat_col : str = 'y_pred'
                               ) -> pd.Series:
    """
    Calculate the deviance residuals for a Poisson GLM.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing 'y' and 'yhat' columns
    y_col : str, optional
        The name of the column containing the observed response variable (default: 'y')
    yhat_col : str, optional
        The name of the column containing the predicted response variable (default: 'yhat')
        
    Returns
    -------
    pd.Series
        A pandas Series containing the deviance residuals
    """
    y = df[y_col]
    yhat = df[yhat_col]
    
    # deviance residuals for a Poisson GLM are defined as:
    # 2 * (y * log(y / yhat) - (y - yhat))
    residuals = 2 * (y * np.log(y / yhat) - (y - yhat))
    return pd.Series(np.where(y == 0, -2 * yhat, residuals), name='Deviance Residuals')

def gamma_deviance_residuals(df: pd.DataFrame
                             , y_col: str = 'y'
                             , yhat_col: str = 'y_pred'
                             ) -> pd.Series:
    """
    Calculate the deviance residuals for a Gamma GLM.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing 'y' and 'yhat' columns
    y_col : str, optional
        The name of the column containing the observed response variable (default: 'y')
    yhat_col : str, optional
        The name of the column containing the predicted response variable (default: 'yhat')
        
    Returns
    -------
    pd.Series
        A pandas Series containing the deviance residuals
    """
    y = df[y_col]
    yhat = df[yhat_col]
    
    # deviance residuals for a Gamma GLM are defined as:
    # deviance_residual_i = sign(y_i - y_hat_i) * 2 * (y_i * log(y_i / y_hat_i) - (y_i - y_hat_i))
    # Where y_i is the observed value, y_hat_i is the predicted value, and i is the index of the observation.
    residuals = 2 * (y * np.log(y / yhat) - (y - yhat))
    residuals = np.where(y > yhat, residuals, -residuals)
    return pd.Series(residuals, name='Deviance Residuals')

def deviance_residuals(df: pd.DataFrame
                       , y_col: str = 'y'
                       , yhat_col: str = 'y_pred'
                       , family: str = 'poisson'
                       ) -> pd.Series:
    """
    Shortcut function to calculate the deviance residuals for a GLM. The family argument
    will be stored as part of the rocky model metadata, so this is a convenience function
    to avoid having to import the correct function for the family.
    """
    if family == 'poisson':
        return poisson_deviance_residuals(df, y_col, yhat_col)
    elif family == 'gamma':
        return gamma_deviance_residuals(df, y_col, yhat_col)
    else:
        raise ValueError(f'Invalid family: {family}')