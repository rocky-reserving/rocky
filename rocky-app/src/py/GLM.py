# try:
#     from .triangle import Triangle
# except ImportError:
#     from triangle import Triangle
from triangle import Triangle

# try:
#     from .TriangleTimeSeriesSplit import TriangleTimeSeriesSplit
# except ImportError:
#     from TriangleTimeSeriesSplit import TriangleTimeSeriesSplit
from TriangleTimeSeriesSplit import TriangleTimeSeriesSplit

# try:
#     from .ModelPlot import Plot
# except ImportError:
#     from ModelPlot import Plot
from ModelPlot import Plot

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import TweedieRegressor

import plotly.express as px

import warnings

from typing import List, Optional, Any

from _util.BaseEstimator import BaseEstimator


@dataclass
class glm(BaseEstimator):
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
    use_cal: bool = False
    plot: Plot = None
    acc: pd.Series = None
    dev: pd.Series = None
    cal: pd.Series = None
    must_be_positive: bool = True
    model_name: str = "GLM"

    def __post_init__(self):
        super().__post_init__()

    def __repr__(self):
        if self.alpha is None:
            a = ""
        else:
            a = f"alpha={self.alpha}"

        if self.power is None:
            p = ""
        else:
            if self.alpha is None:
                p = f"power={self.power}"
            else:
                p = f", power={self.power}"
        return f"tweedieGLM({a}{p})"

    def _update_attributes(self, after="fit", **kwargs):
        """
        Update the model's attributes after fitting.
        """
        if after.lower() == "fit":
            self.intercept = self.model.intercept_
            self.coef = self.model.coef_
            self.is_fitted = True
        elif after.lower() == "x":
            self.X_train = kwargs.get("X_train", None)
            self.X_forecast = kwargs.get("X_forecast", None)

    def _update_plot_attributes(self, **kwargs):
        """
        Update the model's attributes for plotting.
        """
        if self.plot is not None:
            for arg in kwargs:
                setattr(self.plot, arg, kwargs[arg])
        else:
            raise AttributeError(f"{self.id} model object has no plot attribute.")

    # def GetXBase(self, kind="train"):
    #     """
    #     This is a wrapper function for the Triangle class's get_X_base method.

    #     Returns the base triangle data for the model.
    #     """
    #     return self.tri.get_X_base(kind, cal=self.use_cal)

    # def GetYBase(self, kind="train"):
    #     """
    #     This is a wrapper function for the Triangle class's get_y_base method.

    #     Returns the base triangle data for the model.

    #     Parameters
    #     ----------
    #     """
    #     return self.tri.get_y_base(kind)

    # def GetX(self, kind=None):
    #     """
    #     Getter for the model's X data. If there is no X data, take the base design
    #     matrix directly from the triangle. When parameters are combined, X is
    #     created as the design matrix of the combined parameters.
    #     """
    #     if kind is None:
    #         return pd.concat(
    #             [
    #                 self.tri.get_X_base("train", cal=self.use_cal),
    #                 self.tri.get_X_base("forecast", cal=self.use_cal),
    #             ]
    #         )
    #     else:
    #         if kind.lower() in ["train", "forecast"]:
    #             if kind.lower() == "train":
    #                 if self.is_fitted:
    #                     return self.X_train
    #                 else:
    #                     df = self.tri.get_X_base("train", cal=self.use_cal)
    #                     return df
    #             elif kind.lower() == "forecast":
    #                 if self.is_fitted:
    #                     return self.X_forecast
    #                 else:
    #                     df = self.tri.get_X_base("forecast", cal=self.use_cal)
    #                     return df
    #         else:
    #             raise ValueError("kind must be 'train' or 'forecast'")

    def GetParameterNames(self, column=None):
        """
        Getter for the model's parameter names.
        """
        if column is None:
            return self.GetX().columns.tolist()
        else:
            return self.GetX().columns.to_series().str.startswith(column)

    # def GetY(self, kind="train"):
    #     """
    #     Getter for the model's y data. If there is no y data, take the y vector
    #     directly from the triangle.
    #     """
    #     if kind.lower() in ["train", "forecast"]:
    #         if kind.lower() == "train":
    #             return self.y_train
    #         elif kind.lower() == "forecast":
    #             raise ValueError("y_forecast is what we are trying to predict!")
    #     else:
    #         raise ValueError("kind must be 'train' for `y`")

    # def GetN(self):
    #     return self.GetX("train").shape[0]

    # def GetP(self):
    #     return self.GetX("train").shape[1]

    # def GetDegreesOfFreedom(self):
    #     return self.GetN() - self.GetP()

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

        # make sure X does not have the `is_observed` column
        if "is_observed" in X.columns.tolist():
            X = X.drop(columns=["is_observed"])

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
            yhat=self.GetYhat("train"),
            acc=self.acc,
            dev=self.dev,
            cal=self.cal,
            fitted_model=self,
        )

    def ManualFit(self, **kwargs):
        # TODO: docstring

        # parameters
        params = self.GetParameters()

        # loop through the kwargs and set the coefficients of the model
        for key, value in kwargs.items():
            # get index of the key from the design matrix
            idx = params[params["parameter"] == key].index[0]

            # set the coefficient
            self.model.coef_[idx] = value

        # update attributes
        self._update_attributes("fit")
        self._update_plot_attributes(
            X_id=self.tri.get_X_id("train"),
            yhat=self.GetYhat("train"),
            acc=self.acc,
            dev=self.dev,
            cal=self.cal,
            fitted_model=self,
        )

    def Predict(self, kind: str = None, X: pd.DataFrame = None) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been fit")

        if X is None:
            X = self.GetX(kind=kind)

        # drop the `is_observed` column if it exists
        if "is_observed" in X.columns.tolist():
            X = X.drop(columns=["is_observed"])

        yhat = self.model.predict(X)
        return pd.Series(yhat)

    # def GetYhat(self, kind: str = None) -> pd.Series:
    #     return self.Predict(kind=kind)

    def GetParameters(self, column: str = None) -> pd.DataFrame:
        """
        Get the parameters of the model.

        Parameters
        ----------
        column : str, optional
            The type of parameter to return, by default None, which will return
            all parameters. The options are:
                - "acc" : the accident period parameters
                - "dev" : the development period parameters
                - "cal" : the calendar period parameters
                - "all" : all parameters
                - None  : all parameters

                - 'intercept' : the intercept

        Returns
        -------
        pd.DataFrame
            The parameters of the model.
        """
        if self.model is None:
            raise ValueError("Model has not been fit")

        params = pd.DataFrame(
            dict(
                parameter=self.GetParameterNames(),
                value=self.model.coef_,
            )
        )

        # group parameter by variable
        params["var_gp"] = params["parameter"].str.split("_").str[0]

        # cumulative sum of parameters by variable
        params["cumsum"] = params.groupby("var_gp")["value"].cumsum()

        # get the parameters for the intercept
        params["intercept"] = params["cumsum"].shift(1).fillna(0)

        if column is None or column == "all":
            return params
        elif column == "acc":
            return params.query("var_gp == 'accident'")
        elif column == "dev":
            return params.query("var_gp == 'development'")
        elif column == "cal":
            return params.query("var_gp == 'calendar'")
        elif column == "intercept":
            return params.query("parameter == 'intercept'")

    def PlotParameter(
        self, column: str = "acc", value: str = "cumsum", **kwargs
    ) -> None:
        """
        Plot the parameters of the model.

        Parameters
        ----------
        column : str, optional
            The column to plot, by default "acc".
        value : str, optional
            The value to plot, by default "cumsum".
        **kwargs
            Additional keyword arguments to pass to `Plot.PlotParameter()`.

        Returns
        -------
        None
            The plot is displayed in the console.

        """
        if self.model is None:
            raise ValueError("Model has not been fit")

        params = self.GetParameters()
        if column is None:
            column = "acc"
        params = self.GetParameters(column=column)
        hover_data = ["parameter", "value", "cumsum"]
        fig = px.line(
            params, x="parameter", y="cumsum", hover_data=hover_data, **kwargs
        )
        fig.update_layout(
            title=f"Cumulative {column} parameters",
            xaxis_title="Parameter",
            yaxis_title="Cumulative Parameter Value",
        )
        fig.show()

    # def PredictTriangle(self):
    #     yhat = pd.DataFrame(dict(yhat=self.Predict()))
    #     _ids = self.tri.get_X_id()
    #     return pd.concat([_ids, yhat], axis=1)

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

    # def LogLikelihood(self):
    #     return np.sum(self.LogPi())

    def Deviance(self):
        """
        D(Y, Y_hat) = 2 * sum(Y * log(Y / Y_hat) - (Y - Y_hat))
                    = 2 * sum[loglik(saturated model) - loglik(fitted model)]
        """
        raise NotImplementedError

    # def RawResiduals(self):
    #     return self.GetY() - self.GetYhat()

    # def PearsonResiduals(self, show_plot=False, by=None, **kwargs):
    #     res = np.divide(
    #         self.RawResiduals(),
    #     )

    #     if show_plot:
    #         df = pd.DataFrame(dict(resid=res))
    #         if by is None:
    #             df["y"] = self.GetY()
    #         else:
    #             try:
    #                 df[by] = getattr(self.tri, by)
    #             except AttributeError:
    #                 raise ValueError(
    #                     f"""by={by} must be a valid attribute of the triangle object.
    #                     Try `ay` or `dev` instead."""
    #                 )
    #         self.plot.residual(df, plot_by=by, **kwargs)
    #     else:
    #         return res

    # def PearsonResidualPlot(self, by=None):
    #     if by is None:
    #         fig = px.scatter(
    #             x=self.Predict(), y=self.PearsonResiduals(), trendline="ols"
    #         )
    #         fig.show()
    #     else:
    #         if by not in ["accident_period", "development_period", "cal"]:
    #             raise ValueError(
    #                 'by must be one of "accident_period", "development_period", "cal"'
    #             )

    #         df = self.tri.get_X_id("train")
    #         df["Pearson Residuals"] = self.PearsonResiduals()

    #         pad_map = {"accident_period": 4, "development_period": 3, "cal": 2}
    #         df[by] = df[by].astype(int).astype(str).str.pad(pad_map[by], fillchar="0")
    #         fig = px.scatter(df, x=df[by], y=self.PearsonResiduals(), trendline="ols")
    #         fig.show()

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

    # def ScaleParameter(self):
    #     dev = self.Deviance()
    #     dof = self.GetDegreesOfFreedom()
    #     return dev / dof

    # def GetYearTypeDict(self):
    #     d = {
    #         "accident": "acc",
    #         "acc": "acc",
    #         "ay": "acc",
    #         "accident_year": "acc",
    #         "development": "dev",
    #         "dy": "dev",
    #         "dev": "dev",
    #         "development_year": "dev",
    #         "development_period": "dev",
    #         "calendar": "cal",
    #         "cal": "cal",
    #         "cy": "cal",
    #         "calendar_year": "cal",
    #     }
    #     return d

    # def Combine(
    #     self, year_type: str, year_list: list, combined_name: str = None
    # ) -> pd.DataFrame:
    #     """
    #     This function combines parameters of the design matrix to have the same value.

    #     THIS SHOULD JUST BE A MAPPING BETWEEN ORIGINAL "BASE" PARAMETERS AND COMBINED
    #     PARAMETERS SO THE BASE SET CAN BE USED FOR ORDERING, ETC, AND KEEPING AY
    #     PARAMETERS TOGETHER, DEV PARAM TOGETHER, ETC
    #     """
    #     # run through recode year-type dictionary
    #     year_type = self.GetYearTypeDict()[year_type.lower()]

    #     # check that acceptable year_type was passed
    #     if year_type not in ["acc", "dev", "cal"]:
    #         raise ValueError("Year type must be 'acc', 'dev', or 'cal'")

    #     # make a copy of the current design matrix
    #     X_train = self.GetX("train").copy()
    #     X_forecast = self.GetX("forecast").copy()
    #     X_new_train = X_train.copy()
    #     X_new_forecast = X_forecast.copy()

    #     # recode combined name
    #     if combined_name is None:
    #         combined_name = f"{year_type}_combined"
    #     combined_param_name = combined_name.lower().replace(" ", "_").replace(".", "_")

    #     if year_type == "acc":
    #         # generate column names
    #         cols_to_combine = ["accident_period_" + str(year) for year in year_list]

    #         # check if columns exist in the DataFrame
    #         if not set(cols_to_combine).issubset(X_train.columns):
    #             raise ValueError(
    #                 "One or more columns specified do not exist in the DataFrame"
    #             )

    #         # add new column that is the sum of the specified accident year columns
    #         X_new_train[combined_param_name] = X_new_train[cols_to_combine].sum(axis=1)
    #         X_new_forecast[combined_param_name] = X_new_forecast[cols_to_combine].sum(
    #             axis=1
    #         )

    #         # drop the original columns
    #         X_new_train = X_new_train.drop(columns=cols_to_combine)
    #         X_new_forecast = X_new_forecast.drop(columns=cols_to_combine)

    #     elif year_type == "dev":
    #         # generate column names
    #         cols_to_combine = ["development_year_" + str(year) for year in year_list]

    #         # check if columns exist in the DataFrame
    #         if not set(cols_to_combine).issubset(X_train.columns):
    #             raise ValueError(
    #                 "One or more columns specified do not exist in the DataFrame"
    #             )

    #         # add new column that is the sum of the specified development year columns
    #         X_new_train[combined_param_name] = X_new_train[cols_to_combine].sum(axis=1)
    #         X_new_forecast[combined_param_name] = X_new_forecast[cols_to_combine].sum(
    #             axis=1
    #         )

    #         # drop the original columns
    #         X_new_train = X_new_train.drop(columns=cols_to_combine)
    #         X_new_forecast = X_new_forecast.drop(columns=cols_to_combine)

    #     # reset X_train, X_forecast
    #     self.X_train = X_new_train
    #     self.X_forecast = X_new_forecast

    #     # refit the model
    #     self.Fit()
