"""
This module contains the BaseEstimator class, which is the base class for all
models and estimators in the rocky package.
"""
# import Triangle
try:
    from ..triangle import Triangle
except:
    from triangle import Triangle

# import TriangleTimeSeriesSplit
try:
    from ..TriangleTimeSeriesSplit import TriangleTimeSeriesSplit
except:
    from TriangleTimeSeriesSplit import TriangleTimeSeriesSplit

from dataclasses import dataclass
import numpy as np
import pandas as pd


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

    def VarY(self, validate=False, scale_parameter=1):
        return self.Predict(self.GetX(validate=validate)).values * scale_parameter

    @classmethod
    def FitGLM(cls, tri, model_class, id=None):
        if id is None:
            id = model_class
        mod = cls(id=id, model_class=model_class, tri=tri)
        mod.Fit()
        return mod

    def TuneHyperparameters(self, n_splits=5):
        raise NotImplementedError

    def Fit(self, X=None, y=None, validate=False, model_obj=None):
        raise NotImplementedError

    def Predict(self, X):
        raise NotImplementedError

    def Ultimate(self):
        raise NotImplementedError

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

    def LogLikelihood(self, validate=False):
        return np.sum(self.LogPi(validate=validate))

    def RawResiduals(self, validate=False):
        return self.GetY(validate=validate) - self.Predict(self.GetX(validate=validate))

    def PearsonResiduals(self, validate=False):
        return self.RawResiduals(validate=validate) / np.sqrt(
            self.Predict(self.GetX(validate=validate))
        )
