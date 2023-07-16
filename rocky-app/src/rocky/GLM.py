# rocky code
from rocky.triangle import Triangle
from rocky.TriangleTimeSeriesSplit import TriangleTimeSeriesSplit
from rocky.ModelPlot import Plot
from rocky._util.BaseEstimator import BaseEstimator

# for class attributes/definitions
from dataclasses import dataclass

# for working with data
import numpy as np
import pandas as pd

# for fitting the model
from sklearn.linear_model import TweedieRegressor

# for plotting
import plotly.express as px

# for warnings
import warnings

@dataclass
class glm(BaseEstimator):
    """
    Base GLM class. All GLM models are based on, and are Tweedie
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
    model_name: str = "tweedieGLM"
    

    def __post_init__(self):
        super().__post_init__()
        self.model_name_params = {
            'alpha': self.alpha,
            'power': self.power,
        }

    # def __repr__(self):
    #     super().__repr__()

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

    def GetParameterNames(self, column=None):
        """
        Getter for the model's parameter names.
        """
        if column is None:
            return self.GetX().columns.tolist()
        else:
            return self.GetX().columns.to_series().str.startswith(column)

    def GetVarY(self, kind="train"):
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
            params = params
        elif column == "acc":
            params = params.query("var_gp == 'accident'")
        elif column == "dev":
            params = params.query("var_gp == 'development'")
        elif column == "cal":
            params = params.query("var_gp == 'calendar'")
        elif column == "intercept":
            params = params.query("parameter == 'intercept'")
        
        # params = params.drop(columns=['var_gp intercept'.split()])
        return params
        

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

        # params = self.GetParameters()
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

    def Deviance(self):
        """
        D(Y, Y_hat) = 2 * sum(Y * log(Y / Y_hat) - (Y - Y_hat))
                    = 2 * sum[loglik(saturated model) - loglik(fitted model)]
        """
        print("Deviance not implemented for this model")
        raise NotImplementedError

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
