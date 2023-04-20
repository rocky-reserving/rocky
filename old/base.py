"""
This is the base estimator class for all estimators in the rocky3 package. The basic idea comes from
scikit-learn's BaseEstimator class. The main difference is that the rocky3 BaseEstimator class
is specifically set up for the reserve modelling process, and doesn't include all the functionality
of the scikit-learn BaseEstimator class, which is more general and can be used for a wider range
of applications. For this specific use case, we don't need all the functionality of the scikit-learn
BaseEstimator class, so we've simplified it.

Methods:
--------
get_params: Get parameters for this estimator.
set_params: Set the parameters of this estimator.
_get_param_names: Get parameter names for the estimator.

Attributes:
-----------
alpha: float, default=1.0
    The regularization strength (L1 or L2 penalty) for the underlying regressor.
max_iter: int, default=1000
    The maximum number of iterations for the underlying regressor.
tol: float, default=1e-4
    The tolerance for the stopping criterion for the underlying regressor.

References:
-----------
https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
"""

import inspect
import numpy as np
# from triangle import Triangle
from dataclasses import dataclass

from dataclasses import dataclass

@dataclass
class BaseEstimator:
    """
    Base class for all property-casualty reserving models.
    """
    model: object = None

    def fit(self, X, y):
        """
        Fits the model and returns the fitted model object.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        # code to fit the model and assign it to self.model
        return self

    def predict(self, X):
        """
        Predicts the response vector `y` from the fitted model object, and a given design matrix `X`.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        # code to predict y from X and self.model
        return y_pred

    def summary(self):
        """
        Prints a summary of the fitted model object.
        """
        # code to print the model summary
        pass

    def score(self, X, y):
        """
        Returns an appropriate score for the fitted model object.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        score : float
            The score of the model.
        """
        # code to calculate the model score from X, y, and self.model
        return score

    def plot(self):
        """
        Handles plotting, including plots of pp plot, qq plot, plots of normalized residuals by fitted values, accident period, development period, etc., boxplots of normalized residuals by accident period, development period, etc., deviance residuals by accident period, development period, etc., and deviance residual histograms, density plots, etc.
        """
        # code to generate the plots for the fitted model object
        pass

    
    def get_params(self,
                   deep : bool = True) -> dict:
        """
        Get parameters for this estimator. This is a simplified version of the scikit-learn
        BaseEstimator.get_params() method. The main difference is that this method doesn't
        
        Parameters:
        -----------
        deep: bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns:
        --------
        params: dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters:
        -----------
        **params: dict
            Estimator parameters.
            
        Returns:
        --------
        self: object
            Estimator instance.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split_key = key.split('__', 1)
            if len(split_key) > 1:
                name, sub_name = split_key
                if name not in valid_params:
                    raise ValueError(f'Invalid parameter {name} for estimator {self}.')
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                if key not in valid_params:
                    raise ValueError(f'Invalid parameter {key} for estimator {self}.')
                setattr(self, key, value)
        return self