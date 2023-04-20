"""
This module contains the GLM class, which is the base class for the Poisson and Gamma GLM models.

Classes:
--------
GLM: Generalized Linear Model (GLM) class.
Poisson: Poisson GLM model. Inherits from GLM.
Gamma: Gamma GLM model. Inherits from GLM.

References:
-----------
[1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
[2] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html
"""
import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from base import BaseEstimator

from dataclasses import dataclass

@dataclass
class GLM(BaseEstimator):
    """
    Generalized Linear Model (GLM) class.

    Parameters:
    -----------
    alpha: float, default=1.0
        The regularization strength (L2 penalty) for the underlying regressor.
    max_iter: int, default=1000
        The maximum number of iterations for the underlying regressor.
    tol: float, default=1e-4
        The tolerance for the stopping criterion for the underlying regressor.
    """
    alpha: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-4

    def __repr__(self) -> str:
        return f"GLM(id={self.id})"

@dataclass
class Poisson(GLM):
    """
    Poisson GLM model for property-casualty reserve modeling.

    Parameters:
    -----------
    alpha: float, default=1.0
        The regularization strength (L2 penalty) for the underlying regressor.
    max_iter: int, default=1000
        The maximum number of iterations for the underlying regressor.
    tol: float, default=1e-4
        The tolerance for the stopping criterion for the underlying regressor.
    """
    def __repr__(self) -> str:
        return f"Poisson(id={self.id})"
    

@dataclass
class Gamma(GLM):
    """
    Gamma GLM model for property-casualty reserve modeling.

    Parameters:
    -----------
    alpha: float, default=1.0
        The regularization strength (L2 penalty) for the underlying regressor.
    max_iter: int, default=1000
        The maximum number of iterations for the underlying regressor.
    tol: float, default=1e-4
        The tolerance for the stopping criterion for the underlying regressor.
    """
    def __repr__(self) -> str:
        return f"Poisson(id={self.id})"
