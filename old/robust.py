"""
This module contains the robust linear regression models that are implemented in the rocky3 package.

Classes:
--------
Robust: Robust linear regression model base class.
Huber: Huber regressor model.
RANSAC: RANSAC regressor model.
"""

from sklearn.linear_model import HuberRegressor, RANSACRegressor
from base import BaseEstimator

class Robust(BaseEstimator):
    def __init__(self):
        """
        Robust linear regression model base class.
        """
        super().__init__()

    def fit(self, X, y):
        raise NotImplementedError("This method should be implemented in the child class.")

    def predict(self, X):
        raise NotImplementedError("This method should be implemented in the child class.")


class Huber(Robust):
    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001, tol=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol
        self.model = HuberRegressor(epsilon=self.epsilon, max_iter=self.max_iter, alpha=self.alpha, tol=self.tol)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class RANSAC(Robust):
    def __init__(self, base_estimator=None, min_samples=None, residual_threshold=None, max_trials=100, random_state=None):
        super().__init__()
        self.base_estimator = base_estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.random_state = random_state
        self.model = RANSACRegressor(base_estimator=self.base_estimator, min_samples=self.min_samples, 
                                      residual_threshold=self.residual_threshold, max_trials=self.max_trials, 
                                      random_state=self.random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
