try:
    from .triangle import Triangle
except:
    from triangle import Triangle

try:
    from .TriangleTimeSeriesSplit import TriangleTimeSeriesSplit
except:
    from TriangleTimeSeriesSplit import TriangleTimeSeriesSplit

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import GridSearchCV


@dataclass
class glm:
    """
    Base GLM class. All GLM models inherit from this class, and are Tweedie
    GLMs. The GLM models are fit using the scikit-learn TweedieRegressor class.
    """

    id: str = None
    model_class: str = None
    model: object = None
    tri: Triangle = None
    intercept: float = None
    coef: np.ndarray = None
    is_fitted: bool = False
    cal: bool = False
    n_validation: int = 0
    validate: bool = None
    saturated_model = None
    weights = None
    distribution_family = None

    def __post_init__(self):
        if self.n_validation > 0:
            self.validate = True
        else:
            self.validate = False

        if self.weights is None:
            self.weights = np.ones(self.tri.tri.shape[0])

    def GetX(self, validate=False):
        df = self.tri.get_X_base("train")

        # if using validation, remove the last n_validation calendar periods
        # from the training data
        if self.n_validation > 0 and validate:
            cur_cal = self.tri.getCurCalendarIndex()
            cal_tri = self.tri.getCalendarIndex()
            val_periods = [cur_cal - i for i in range(self.n_validation + 1)]

            # return the training data without the validation periods
            return df[~cal_tri.isin(val_periods) & (cal_tri.le(cur_cal))]
        else:
            return df

    def GetY(self, validate=False):
        df = self.tri.get_y_base("train")

        # if using validation, remove the last n_validation calendar periods
        # from the training data
        if self.n_validation > 0 and validate:
            cur_cal = self.tri.getCurCalendarIndex()
            cal_tri = self.tri.getCalendarIndex()
            val_periods = [cur_cal - i for i in range(self.n_validation + 1)]

            # return the training data without the validation periods
            return df[~cal_tri.isin(val_periods) & (cal_tri.le(cur_cal))]
        else:
            return df

    def GetXVal(self):
        df = self.tri.get_X_base("train")

        # if using validation, remove the last n_validation calendar periods
        # from the training data
        if self.n_validation > 0:
            cur_cal = self.tri.getCurCalendarIndex()
            cal_tri = self.tri.getCalendarIndex()
            val_periods = [cur_cal - i for i in range(self.n_validation + 1)]

            # return the training data with only the validation periods
            return df[cal_tri.isin(val_periods)]
        else:
            return None

    def GetYVal(self):
        df = self.tri.get_y_base("train")

        # if using validation, remove the last n_validation calendar periods
        # from the training data
        if self.n_validation > 0:
            cur_cal = self.tri.getCurCalendarIndex()
            cal_tri = self.tri.getCalendarIndex()
            val_periods = [cur_cal - i for i in range(self.n_validation + 1)]

            # return the training data with only the validation periods
            return df[cal_tri.isin(val_periods)]
        else:
            return None

    def GetXForecast(self):
        return self.tri.get_X_base("forecast")

    def GetN(self, validate=False):
        return self.GetX(validate=validate).shape[0]

    def GetP(self, validate=False):
        return self.GetX(validate=validate).shape[1]

    def GetDegreesOfFreedom(self, validate=False):
        return self.GetN(validate=validate) - self.GetP(validate=validate)

    def VarY(self, validate=False):
        return self.Predict(self.GetX(validate=validate)).values * self.ScaleParameter()

    @classmethod
    def FitGLM(cls, tri, model_class, id=None):
        if id is None:
            id = model_class
        mod = cls(id=id, model_class=model_class, tri=tri)
        mod.Fit()
        return mod

    def TuneHyperparameters(self, n_splits=5):
        grid = {
            "alpha": np.linspace(0, 2, 10),
            "power": np.linspace(1, 3, 5),
            "max_iter": [1000],
        }

        # Since it is a Tweedie GLM, we use TweedieRegressor
        model = TweedieRegressor()

        # Define a scoring metric (e.g., mean squared error)
        scoring = "neg_mean_squared_error"

        # Define cross-validation strategy
        cv = TriangleTimeSeriesSplit(self.tri, n_splits=5)

        # Grid search
        search = GridSearchCV(model, grid, scoring=scoring, cv=cv.GetSplit(), verbose=1)

        # Fit data to GridSearchCV
        X = self.GetX()
        y = self.GetY()
        search.fit(X, y)

        # Get the best parameters
        best_params = search.best_params_

        # Use the best parameters for your model
        self.model = TweedieRegressor(**best_params)

    def Fit(self, X=None, y=None, validate=False):
        if X is None:
            X = self.GetX(validate=validate)
        if y is None:
            y = self.GetY(validate=validate)

        # if self.distribution_family == 'poisson':

    def Predict(self, X):
        raise NotImplementedError

    def Ultimate(self):
        current = self.tri.diag()
        forecast = self.Predict(self.GetXForecast())

    def PredictTriangle(self):
        raise NotImplementedError

    def GetSaturatedModel(self, validate=False):
        y = self.GetY(validate=validate)
        X = y.copy().reset_index()
        X["variable"] = X["index"].astype(str).str.pad(5, fillchar="0")

        X_dm = pd.get_dummies(X, columns=["variable"], drop_first=False)
        X_dm = X_dm.drop(columns=["index"])

        self.saturated_model = self.Fit(X=X_dm, y=y, validate=False)

    def LogPi(self, validate=False):
        cls_name = self.__class__.__name__
        print("LogPi not implemented for this model")
        print(f"They must be specifically implemented in the {cls_name} class")
        raise NotImplementedError

    def LogLikelihood(self, model=None, validate=False):
        return np.sum(self.LogPi(validate=validate))

    def Deviance(self, validate=False):
        """
        D(Y, Y_hat) = 2 * sum(Y * log(Y / Y_hat) - (Y - Y_hat))
                    = 2 * sum[loglik(saturated model) - loglik(fitted model)]
        """
        raise NotImplementedError

    def RawResiduals(self, validate=False):
        return self.GetY(validate=validate) - self.Predict(self.GetX(validate=validate))

    def PearsonResiduals(self, validate=False):
        return self.RawResiduals(validate=validate) / np.sqrt(
            self.Predict(self.GetX(validate=validate))
        )

    def DevianceResiduals(self, validate=False):
        """
        Deviance residuals are defined as:
        R_i^D = sign(Y_i - Y_hat_i) *
                sqrt(2 * [Y_i * log(Y_i / Y_hat_i) - (Y_i - Y_hat_i)])
        """
        y = self.GetY(validate=validate)
        y_hat = self.Predict(self.GetX(validate=validate))
        raw_resid = self.RawResiduals(validate=validate)
        resid = np.multply(
            np.sign(raw_resid),
            np.sqrt(
                np.multiply(
                    2,
                    np.subtract(np.multiply(y, np.log(np.divide(y, y_hat))), raw_resid),
                )
            ),
        )
        return resid

    def ScaleParameter(self, validate=False):
        dev = self.Deviance(validate=validate)
        dof = self.GetDegreesOfFreedom(validate=validate)
        return dev / dof

    def _p(self, distribution="poisson"):
        if distribution == "normal":
            return 0
        elif distribution == "poisson":
            return 1
        elif distribution == "gamma":
            return 2
        elif distribution == "inv_gaussian":
            return 3

    def _b(
        self,
        distribution="poisson",
    ):
        if distribution == "normal":
            return 1
        elif distribution == "poisson":
            return 1
        elif distribution == "gamma":
            return 1
        elif distribution == "inv_gaussian":
            return 1
