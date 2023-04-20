"""
This module contains the regularized linear regression model base class and the Ridge, Lasso, and
ElasticNet models that are implemented in the rocky3 package.

Classes:
--------
Regularized: Regularized linear regression model base class.
Ridge: Ridge regressor model.
Lasso: Lasso regressor model.
ElasticNet: ElasticNet regressor model.

References:
-----------
[1] https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
[2] https://scikit-learn.org/stable/modules/linear_model.html#lasso
[3] https://scikit-learn.org/stable/modules/linear_model.html#elastic-net
"""

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from base import BaseEstimator

class Regularized(BaseEstimator):
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        """
        Regularized linear regression model base class.

        Parameters:
        -----------
        alpha: float, default=1.0
            The regularization strength (L1 or L2 penalty) for the underlying regressor.
        max_iter: int, default=1000
            The maximum number of iterations for the underlying regressor.
        tol: float, default=1e-4
            The tolerance for the stopping criterion for the underlying regressor.
        """
        super().__init__()

    def fit(self, X, y):
        raise NotImplementedError("This method should be implemented in the child class.")

    def predict(self, X):
        raise NotImplementedError("This method should be implemented in the child class.")


class Ridge(Regularized):
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        super().__init__(alpha=alpha, max_iter=max_iter, tol=tol)
        self.model = Ridge(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class Lasso(Regularized):
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        super().__init__(alpha=alpha, max_iter=max_iter, tol=tol)
        self.model = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self,
                X: np.ndarray
                ) -> np.ndarray:
        return self.model.predict(X)


class ElasticNet(Regularized):
    def __init__(self,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 max_iter: int = 1000,
                 tol : float = 1e-4
                 ) -> None:
        super().__init__(alpha=alpha, max_iter=max_iter, tol=tol)
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter, tol=self.tol)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray
            ) -> None:
        self.model.fit(X, y)
        return self

    def predict(self,
                X: np.ndarray
                )-> np.ndarray:
        return self.model.predict(X)
