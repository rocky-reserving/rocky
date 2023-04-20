import numpy as np
import pandas as pd

from sklearn.linear_model import PoissonRegressor, GammaRegressor

from typing import Union

def calculate_hat_matrix_poisson(df : pd.DataFrame
                                 , model : PoissonRegressor
                                 , y_pred_col : str = 'y_pred'
                                 , x_col : list = None
                                 ) -> np.ndarray:
    """
    Calculate the hat matrix for the given DataFrame using a Poisson GLM.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the specified columns
    model : sklearn.linear_model.PoissonRegressor
        The fitted PoissonRegressor model
    y_pred_col : str, optional
        The name of the column containing the expected values (default: 'y_pred')
    x_col : list of str, optional
        The list of column names used as predictors in the model
    """

    y_pred = df[y_pred_col].values

    if x_col is None:
        x_col = ['const'] + list(df.columns)

    X = df[x_col].values
    W = np.diag(y_pred)

    # Compute the condition number of the matrix (X.T @ W @ X)
    matrix_to_invert = X.T @ W @ X
    cond_num = np.linalg.cond(matrix_to_invert)

    # If the condition number is large, use the pseudoinverse
    if cond_num > 1 / np.sqrt(np.finfo(matrix_to_invert.dtype).eps):
        inverted_matrix = np.linalg.pinv(matrix_to_invert)
    else:
        inverted_matrix = np.linalg.inv(matrix_to_invert)

    # Compute hat matrix for Poisson GLM
    hat_matrix = np.sqrt(W) @ X @ inverted_matrix @ X.T @ np.sqrt(W)
    return 

def calculate_hat_matrix_gamma(df : pd.DataFrame
                                , model : GammaRegressor
                                , y_pred_col : str = 'y_pred'
                                , x_col : list = None
                                ) -> np.ndarray:
     """
     Calculate the hat matrix for the given DataFrame using a Gamma GLM.
    
     Parameters
     ----------
     df : pd.DataFrame
          The input DataFrame containing the specified columns
     model : sklearn.linear_model.GammaRegressor
          The fitted GammaRegressor model
     y_pred_col : str, optional
          The name of the column containing the expected values (default: 'y_pred')
     x_col : list of str, optional
          The list of column names used as predictors in the model
     """
    
     y_pred = df[y_pred_col].values
    
     if x_col is None:
          x_col = ['const'] + list(df.columns)
    
     X = df[x_col].values
     W = np.diag(y_pred)
    
     # Compute the condition number of the matrix (X.T @ W @ X)
     matrix_to_invert = X.T @ W @ X
     cond_num = np.linalg.cond(matrix_to_invert)
    
     # If the condition number is large, use the pseudoinverse
     if cond_num > 1 / np.sqrt(np.finfo(matrix_to_invert.dtype).eps):
          inverted_matrix = np.linalg.pinv(matrix_to_invert)
     else:
          inverted_matrix = np.linalg.inv(matrix_to_invert)
    
     # Compute hat matrix for Gamma GLM
     hat_matrix = np.sqrt(W) @ X @ inverted_matrix @ X.T @ np.sqrt(W)
     return hat_matrix

def hat_matrix(df : pd.DataFrame
                            , model : Union[PoissonRegressor, GammaRegressor]
                            , y_pred_col : str = 'y_pred'
                            , x_col : list = None
                            ) -> np.ndarray:
        """
        Calculate the hat matrix for the given DataFrame using the specified GLM.
    
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the specified columns
        model : Union[sklearn.linear_model.PoissonRegressor, sklearn.linear_model.GammaRegressor]
            The fitted PoissonRegressor or GammaRegressor model
        y_pred_col : str, optional
            The name of the column containing the expected values (default: 'y_pred')
        x_col : list of str, optional
            The list of column names used as predictors in the model
        """
        if isinstance(model, PoissonRegressor):
            return calculate_hat_matrix_poisson(df, model, y_pred_col, x_col)
        elif isinstance(model, GammaRegressor):
            return calculate_hat_matrix_gamma(df, model, y_pred_col, x_col)
        else:
            raise ValueError('The model must be a PoissonRegressor or GammaRegressor.')