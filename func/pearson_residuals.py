import pandas as pd
import numpy as np

def pearson_residuals(df : pd.DataFrame
                      , observed_col : str = 'y'
                      , expected_col : str = 'y_pred'
                      ) -> pd.Series:
    """
    Calculate Pearson residuals for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the specified columns
    observed_col : str, optional
        The name of the column containing the observed values (default: 'y')
    expected_col : str, optional
        The name of the column containing the expected values (default: 'y_pred')

    Returns
    -------
    pd.Series
        A pandas Series containing the Pearson residuals
    """
    observed = df[observed_col]
    expected = df[expected_col]

    pearson_res = (observed - expected) / np.sqrt(expected)
    return pearson_res