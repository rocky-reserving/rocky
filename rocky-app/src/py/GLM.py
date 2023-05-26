try:
    from .triangle import Triangle
except ImportError:
    from triangle import Triangle

try:
    from .TriangleTimeSeriesSplit import TriangleTimeSeriesSplit
except ImportError:
    from TriangleTimeSeriesSplit import TriangleTimeSeriesSplit

try:
    from .ModelPlot import Plot
except ImportError:
    from ModelPlot import Plot

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import TweedieRegressor

import plotly.express as px

import warnings


@dataclass
class glm:
    """
    Base GLM class. All GLM models inherit from this class, and are Tweedie
    GLMs. The GLM models are fit using the scikit-learn TweedieRegressor class.
    """

    id: str
    model_class: str = None
    model: object = None
    tri: Triangle = None
    intercept: float = None
    coef: np.ndarray[float] = None
    is_fitted: bool = False
    n_validation: int = 0
    saturated_model = None
    weights: pd.Series = None
    distribution_family: str = None
    alpha: float = None
    power: float = None
    max_iter: int = 100000
    link: str = "log"
    cv: TriangleTimeSeriesSplit = None
    X_train: pd.DataFrame = None
    X_forecast: pd.DataFrame = None
    y_train: pd.Series = None
    plot: Plot = None
    acc: pd.Series = None
    dev: pd.Series = None
    cal: pd.Series = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones(self.tri.tri.shape[0])

        # initialize X_train, X_forecast, and y_train
        self.X_train = self.GetXBase("train")
        self.X_forecast = self.GetXBase("forecast")
        self.y_train = self.GetYBase("train")

        # initialize the plotting object
        self.plot = Plot()

        # initialize the acc, dev, cal attributes
        self.acc = self.tri.get_X_id("train")["accident_period"]
        self.dev = self.tri.get_X_id("train")["development_period"]
        self.cal = self.tri.get_X_id("train")["cal"]

    def _update_attributes(self, after="fit", **kwargs):
        """
        Update the model's attributes after fitting.
        """
        if after.lower() == "fit":
            self.intercept = self.model.intercept_
            self.coef = self.model.coef_
            self.is_fitted = True
            # self.plot.X_train = self.GetX("train")
            # self.plot.X_id = self.tri.get_X_id("train")
            # self.plot.X_forecast = self.GetX("forecast")
            # self.plot.y_train = self.GetY("train")
            # self.plot.yhat = self.GetYhat()
            # self.plot.acc = self.acc
            # self.plot.dev = self.dev
            # self.plot.cal = self.cal
        elif after.lower() == "x":
            self.X_train = kwargs.get("X_train", None)
            self.X_forecast = kwargs.get("X_forecast", None)

    def _update_plot_attributes(self, **kwargs):
        """
        Update the model's attributes for plotting.
        """
        if self.plot is not None:
            for arg in kwargs:
                if hasattr(self.plot, arg):
                    setattr(self.plot, arg, kwargs[arg])
                else:
                    warnings.warn(f"Attribute {arg} not found in plot attribute.")
        else:
            raise AttributeError(f"{self.id} model object has no plot attribute.")

    def GetXBase(self, kind="train"):
        """
        This is a wrapper function for the Triangle class's get_X_base method.

        Returns the base triangle data for the model.
        """
        return self.tri.get_X_base(kind, cal=self.cal)

    def GetYBase(self, kind="train"):
        """
        This is a wrapper function for the Triangle class's get_y_base method.

        Returns the base triangle data for the model.
        """
        return self.tri.get_y_base(kind)

    def GetX(self, kind="train"):
        """
        Getter for the model's X data. If there is no X data, take the base design
        matrix directly from the triangle. When parameters are combined, X is
        created as the design matrix of the combined parameters.
        """
        if kind.lower() in ["train", "forecast"]:
            if kind.lower() == "train":
                return self.X_train
            elif kind.lower() == "forecast":
                return self.X_forecast
        else:
            raise ValueError("kind must be 'train' or 'forecast'")

    def GetY(self, kind="train"):
        """
        Getter for the model's y data. If there is no y data, take the y vector
        directly from the triangle.
        """
        if kind.lower() in ["train", "forecast"]:
            if kind.lower() == "train":
                return self.y_train
            elif kind.lower() == "forecast":
                raise ValueError("y_forecast is what we are trying to predict!")
        else:
            raise ValueError("kind must be 'train' for `y`")

    def GetN(self):
        return self.GetX("train").shape[0]

    def GetP(self):
        return self.GetX("train").shape[1]

    def GetDegreesOfFreedom(self):
        return self.GetN() - self.GetP()

    def VarY(self, kind="train"):
        return self.Predict(self.GetX(kind=kind)).values * self.ScaleParameter()

    @classmethod
    def FitGLM(cls, tri, model_class, id=None):
        if id is None:
            id = model_class
        mod = cls(id=id, model_class=model_class, tri=tri)
        mod.Fit()
        return mod

    def SetHyperparameters(self, alpha, power, max_iter=100000, link="log"):
        self.alpha = alpha
        self.power = power
        self.max_iter = max_iter
        self.link = link

    def TuneFitHyperparameters(
        self,
        n_splits=5,
        param_grid=None,
        measures=None,
        tie_criterion="ave_mse_test",
        **kwargs,
    ):
        # set the parameter grid to default if none is provided
        if param_grid is None:
            param_grid = {
                "alpha": np.arange(0, 3.1, 0.1),
                "power": np.array([0]) + np.arange(1, 3.1, 0.1),
                "max_iter": 100000,
            }

        # if kwargs for alpha, p and max_iter are provided, use those
        if "alpha" in kwargs:
            param_grid["alpha"] = kwargs["alpha"]
        if "power" in kwargs:
            param_grid["power"] = kwargs["power"]
        if "max_iter" in kwargs:
            param_grid["max_iter"] = kwargs["max_iter"]

        # set the cross-validation object
        cv = TriangleTimeSeriesSplit(
            self.tri, n_splits=n_splits, tweedie_grid=param_grid
        )

        # set the parameter search grid
        cv.GridTweedie(
            alpha=param_grid["alpha"],
            power=param_grid["power"],
            max_iter=param_grid["max_iter"],
        )

        # grid search & return the optimal model
        opt_tweedie = cv.OptimalTweedie(measures=measures, tie_criterion=tie_criterion)

        # set the optimal hyperparameters
        self.alpha = opt_tweedie.alpha
        self.power = opt_tweedie.power
        self.link = opt_tweedie.link

        # save cv object
        self.cv = cv

    def Fit(
        self,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        alpha: float = None,
        power: float = None,
        link: str = None,
        max_iter: int = None,
        **kwargs,
    ) -> None:
        """
        Fit the model to the Triangle data.

        Parameters
        ----------
        X : pd.DataFrame, optional
            The design matrix, by default None, which will use the design matrix
            from the Triangle object.
        y : pd.Series, optional
            The response vector, by default None, which will use the response
            vector from the Triangle object.
        alpha : float, optional
            The alpha hyperparameter, by default None, which will use the
            alpha from the glm object. If there is no alpha hyperparameter
            set, then a Pareto-optimal set of hyperparameters will be found
            using `TuneFitHyperparameters()`.
        power : float, optional
            The p hyperparameter, by default None, which will use the
            p from the glm object. If there is no p hyperparameter
            set, then a Pareto-optimal set of hyperparameters will be found
            using `TuneFitHyperparameters()`.
        link : str, optional
            The link function, by default None, which will use the
            link function from the glm object. If there is no link function
            set, it will default to "log".
        max_iter : int, optional
            The maximum number of iterations, by default None, which will use
            the maximum number of iterations from the glm object. If there is
            no maximum number of iterations set, it will default to 100000.
        **kwargs
            Additional keyword arguments to pass to the glm object. See
            `sklearn.linear_model.TweedieRegressor` for more details.

        Returns
        -------
        None
            The model is fit in place.

        Notes
        -----
        If the hyperparameters are not set, then a Pareto-optimal set of
        hyperparameters will be found using `TuneFitHyperparameters()`.

        Examples
        --------
        >>> from rockycore import ROCKY
        >>> # create a ROCKY object
        >>> rky = ROCKY()
        >>> # load a triangle from the clipboard
        >>> rky.FromClipboard()
        >>> # add a GLM to `rky`
        >>> rky.AddModel()

        """
        # get X, y if not provided
        if X is None:
            X = self.GetX("train")

        if y is None:
            y = self.GetY("train")

        # if alpha or p are not provided, calculate the optimal values
        if alpha is None or power is None:
            if self.alpha is None or self.power is None:
                if self.alpha is None:
                    message = "`alpha`"
                    if self.power is None:
                        message += " and `power`"
                else:
                    message = "`power`"
                    message += " is/are not set. Running `TuneFitHyperparameters()`..."

                warnings.warn(message)
                self.TuneFitHyperparameters()

        # now either alpha and p are set or they were provided to begin with
        if alpha is None:
            alpha = self.alpha

        if power is None:
            power = self.power

        if link is None:
            if self.link is None:
                link = "log"
            else:
                link = self.link

        if max_iter is None:
            if self.max_iter is None:
                max_iter = 100000
            else:
                max_iter = self.max_iter

        # tweedie regressor object
        self.model = TweedieRegressor(
            alpha=alpha, power=power, link=link, max_iter=max_iter, verbose=0
        )

        # fit the model
        self.model.fit(X, y)

        # update attributes
        self._update_attributes("fit")

        # add a plot object to the glm object now that it has been fit
        self.plot = Plot()
        self._update_plot_attributes(
            X_train=self.GetX("train"),
            y_train=self.GetY("train"),
            X_forecast=self.GetX("forecast"),
            X_id=self.tri.get_X_id("train"),
            yhat=self.GetYhat(),
            acc=self.acc,
            dev=self.dev,
            cal=self.cal,
        )

    def Predict(self, kind: str = "train", X: pd.DataFrame = None) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been fit")

        if X is None:
            X = self.GetX(kind=kind)

        yhat = self.model.predict(X)
        return yhat

    def GetYhat(self):
        return self.Predict()

    def PredictTriangle(self):
        raise NotImplementedError

    def GetSaturatedModel(self):
        y = self.GetY("train")
        X = y.copy().reset_index()
        X["variable"] = X["index"].astype(str).str.pad(5, fillchar="0")

        X_dm = pd.get_dummies(X, columns=["variable"], drop_first=False)
        X_dm = X_dm.drop(columns=["index"])

        self.saturated_model = self.Fit(X=X_dm, y=y)

    def LogPi(self):
        cls_name = self.__class__.__name__
        print("LogPi not implemented for this model")
        print(f"They must be specifically implemented in the {cls_name} class")
        raise NotImplementedError

    def LogLikelihood(self):
        return np.sum(self.LogPi())

    def Deviance(self):
        """
        D(Y, Y_hat) = 2 * sum(Y * log(Y / Y_hat) - (Y - Y_hat))
                    = 2 * sum[loglik(saturated model) - loglik(fitted model)]
        """
        raise NotImplementedError

    def RawResiduals(self):
        return self.GetY() - self.GetYhat()

    def PearsonResiduals(self, show_plot=False, by=None, **kwargs):
        res = np.divide(
            self.RawResiduals(),
        )

        if show_plot:
            df = pd.DataFrame(dict(resid=res))
            if by is None:
                df["y"] = self.GetY()
            else:
                try:
                    df[by] = getattr(self.tri, by)
                except AttributeError:
                    raise ValueError(
                        f"""by={by} must be a valid attribute of the triangle object.
                        Try `ay` or `dev` instead."""
                    )
            self.plot.residual(df, plot_by=by, **kwargs)
        else:
            return res

    def PearsonResidualPlot(self, by=None):
        if by is None:
            fig = px.scatter(
                x=self.Predict(), y=self.PearsonResiduals(), trendline="ols"
            )
            fig.show()
        else:
            if by not in ["accident_period", "development_period", "cal"]:
                raise ValueError(
                    'by must be one of "accident_period", "development_period", "cal"'
                )

            df = self.tri.get_X_id("train")
            df["Pearson Residuals"] = self.PearsonResiduals()

            pad_map = {"accident_period": 4, "development_period": 3, "cal": 2}
            df[by] = df[by].astype(int).astype(str).str.pad(pad_map[by], fillchar="0")
            fig = px.scatter(df, x=df[by], y=self.PearsonResiduals(), trendline="ols")
            fig.show()

    def DevianceResiduals(self):
        """
        Deviance residuals are defined as:
        R_i^D = sign(Y_i - Y_hat_i) *
                sqrt(2 * [Y_i * log(Y_i / Y_hat_i) - (Y_i - Y_hat_i)])
        """
        y = self.GetY()
        y_hat = self.Predict()
        raw_resid = self.RawResiduals()
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

    def ScaleParameter(self):
        dev = self.Deviance()
        dof = self.GetDegreesOfFreedom()
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
