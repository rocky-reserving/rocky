"""
This module contains the BaseEstimator class, which is the base class for all
models and estimators in the rocky package.
"""
from rocky.triangle import Triangle
from rocky.TriangleTimeSeriesSplit import TriangleTimeSeriesSplit
from rocky.ModelPlot import Plot

from dataclasses import dataclass
import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"

@dataclass
class BaseEstimator:
    """
    Base model class.

    The methods in this class are used by all models and estimators in the
    rocky package. Those that are not implemented in this class must be
    implemented in the child class.

    All added models should inherit from this class, and should implement methods
    as needed.

    Parameters
    ----------
    id : str
        Model ID. This is used to identify the model in the rocky object.
    model_class : str, optional
        Model class. This is used to identify the model in the rocky object, and
        is used in the model's repr.
    model : object, optional
        Model object. This is the model object that is fit to the data. For example,
        the GLM rocky model object is based on the sklearn.linear_model.TweedieRegressor
        object. This is used in the model's repr.
    tri : Triangle
        rocky Triangle object, or a dictionary of Triangles. This is used to
        identify the model in the rocky object.
    exposure : pd.Series, optional
        Exposure vector. Before fitting, the triangle data are divided by the
        exposure vector. If None, the exposure vector is set to a vector of ones.
    coef : pd.Series, optional
        Fitted model coefficients. Starts as None, and is set to the model
        coefficients after the model is fit.
    is_fitted : bool, optional
        Whether the model has been fit. Starts as False, and is set to True
        after the model is fit.
    n_validation : int, optional
        Number of calendar periods to use for validation. If 0, no validation
        is performed. If n_validation is not greater than or equal to 0, an
        error is raised.
    weights : pd.Series, optional
        Weights vector. This is used to weight the data when fitting the model.
        If None, the weights vector is set to a vector of ones.
    cv : TriangleTimeSeriesSplit, optional
        Cross-validation object. This is used to split the data into training
        and testing sets for cross-validation. If None, the data are not split
        into training and testing sets.
    X_train : pd.DataFrame, optional
        Training data. This is the data used to fit the model. If None, X_train
        is calculated from the triangle data.
    X_forecast : pd.DataFrame, optional
        Forecasting data. This is the data used to forecast the model. If None,
        X_forecast is calculated from the triangle data.
    y_train : pd.Series, optional
        Training response. This is the unadjusted response variable. If None,
        y_train is calculated from the triangle data.
        Note that this is not the same as the response used to fit the model.
        Before fitting, the triangle data are divided by the exposure vector
        and the weights vector. The response used to fit the model is the
        adjusted response.
    use_cal : bool, optional
        Whether to use calendar periods. If True, the calendar periods are
        included in X_train and X_forecast. If False, the calendar periods
        are not included in X_train and X_forecast.
    plot : Plot, optional
        Plot object. This is used to plot the model results. If None, a new
        Plot object is created.
    acc : pd.Series, optional
        Accident period labels. This is used to identify the accident periods
        in the triangle data. If None, the accident periods are identified
        from the triangle data.
    dev : pd.Series, optional
        Development period labels. This is used to identify the development
        periods in the triangle data. If None, the development periods are
        identified from the triangle data.
    cal : pd.Series, optional
        Calendar period labels. This is used to identify the calendar periods
        in the triangle data. If None, the calendar periods are identified
        from the triangle data.
    acc_gp : pd.Series, optional
        Accident period variable group. This is used to group the accident
        periods in the triangle data. If None, the accident periods are not
        grouped.
    dev_gp : pd.Series, optional
        Development period variable group. This is used to group the development
        periods in the triangle data. If None, the development periods are not
        grouped.
    cal_gp : pd.Series, optional
        Calendar period variable group. This is used to group the calendar
        periods in the triangle data. If None, the calendar periods are not
        grouped.
    acc_gp_filter : pd.Series, optional
        Accident period variable group filter. This is used to blend the accident
        periods in the triangle data. If None, the accident periods are not
        blended.
    hetero_gp : pd.Series, optional
        Heteroskedasticity variable group. This is used to identify the
        heteroskedasticity groups in the triangle data. If None, the
        heteroskedasticity groups are not identified.
    has_combined_params : bool, optional
        Whether the model has combined parameters. If True, the model has
        combined parameters. If False, the model does not have combined
        parameters.
    acc_forecast : pd.Series, optional
        Forecast accident period labels. This is used to identify the accident
        periods in the forecast data. If None, the accident periods are
        identified from the forecast data.
    dev_forecast : pd.Series, optional
        Forecast development period labels. This is used to identify the
        development periods in the forecast data. If None, the development
        periods are identified from the forecast data.
    cal_forecast : pd.Series, optional
        Forecast calendar period labels. This is used to identify the calendar
        periods in the forecast data. If None, the calendar periods are
        identified from the forecast data.
    must_be_positive : bool, optional
        Whether the model must be positive. If True, the model must be positive.
        If False, the model does not have to be positive.

    Public Methods
    --------------
    GetIdx
        Returns the index for the model. This is used to get the index for the
        train, forecast, or all data, and is used to filter the X and y data when
        GetX and GetY are called.
    GetX
        Returns the X data for the model. This is used to get the design matrix
        for fitting a rocky model with a linear predictor. The X data are filtered
        by the index returned by GetIdx.

    """

    id: str
    model_class: str = None
    model: object = None
    tri: Triangle = None
    exposure: pd.Series = None
    coef: pd.Series = None
    is_fitted: bool = False
    n_validation: int = 0
    weights: pd.Series = None
    cv: TriangleTimeSeriesSplit = None
    X_train: pd.DataFrame = None
    X_forecast: pd.DataFrame = None
    y_train: pd.Series = None
    use_cal: bool = False
    plot: Plot = None
    acc: pd.Series = None
    dev: pd.Series = None
    cal: pd.Series = None
    acc_gp: pd.Series = None
    dev_gp: pd.Series = None
    cal_gp: pd.Series = None
    acc_gp_filter: pd.Series = None
    hetero_gp: pd.Series = None
    has_combined_params: bool = False
    acc_forecast: pd.Series = None
    dev_forecast: pd.Series = None
    cal_forecast: pd.Series = None
    must_be_positive: bool = False

    def __post_init__(self):
        # print(f"must be positive: {self.must_be_positive}")
        if self.weights is None:
            self.weights = np.ones(self.tri.tri.shape[0])

        # build idx
        self._build_idx()

        # initialize matrices
        self._initialize_matrices()

        # initialize the plotting object
        self.plot = Plot()

        # order the columns of X_train and X_forecast
        self.column_order = self.GetX("train").columns.tolist()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"

    def _build_idx1(self):
        # index where y is positive and observed
        is_positive = self.tri.positive_y
        is_observed = self.tri.X_base.loc[
            self.tri.X_base["is_observed"] == 1
        ].index.values
        total = self.tri.X_base.index.values
        is_forecast = np.setdiff1d(total, is_observed)
        return is_positive, is_observed, is_forecast

    def _build_idx(self):
        is_positive, is_observed, is_forecast = self._build_idx1()

        # depending on the model, we may want to drop non-positive y values
        # if so, set must_be_positive to True in the child class
        if self.must_be_positive:
            self.initial_train_idx = np.intersect1d(is_positive, is_observed)
        else:
            self.initial_train_idx = is_observed
        self.initial_forecast_idx = is_forecast

        self.train_index = self.initial_train_idx
        self.forecast_index = self.initial_forecast_idx

    def _initialize_matrices(self):
        # initialize X_train, X_forecast, and y_train
        self.X_train = self.GetXBase("train")
        self.X_forecast = self.GetXBase("forecast")
        self.y_train = self.GetYBase("train")

        # initialize the acc, dev, cal attributes (train set)
        self.acc = self.tri.get_X_id("train")["accident_period"]
        self.dev = self.tri.get_X_id("train")["development_period"]
        self.cal = self.tri.get_X_id("train")["calendar_period"]
        exp = pd.DataFrame(
            {"acc": self.tri.acc.dt.year.values, "exposure": self.tri.exposure.values}
        )
        exp2 = pd.DataFrame(
            {"acc": self.tri.get_X_id("train")["accident_period"]}
        ).merge(exp, on="acc", how="left")
        self.exposure = exp2["exposure"]

        # initialize forecasting attributes
        self.acc_forecast = self.tri.get_X_id("forecast")["accident_period"]
        self.dev_forecast = self.tri.get_X_id("forecast")["development_period"]
        self.cal_forecast = self.tri.get_X_id("forecast")["calendar_period"]
        exp_f = pd.DataFrame(
            {"acc": self.tri.acc.dt.year.values, "exposure": self.tri.exposure.values}
        )
        exp2_f = pd.DataFrame(
            {"acc": self.tri.get_X_id("forecast")["accident_period"]}
        ).merge(exp_f, on="acc", how="left")
        self.exposure_forecast = exp2_f["exposure"]

    def combine_indices(self, *args):
        """
        Combine indices into a single index.

        Parameters
        ----------
        args : np.ndarray
            Indices to combine. If multiple indices are passed, they must each
            be an array. The indicies will be concatenated along the first axis
            to form a single index.

        Returns
        -------
        np.ndarray
            Combined index.
        """
        return (
            pd.Series(np.concatenate(args, axis=0))
            .drop_duplicates()
            .sort_values()
            .values
        )

    def __repr__(self):
        raise NotImplementedError

    def _update_attributes(self, after="fit", **kwargs):
        """
        Update the model's attributes after fitting.
        """
        raise NotImplementedError

    def _update_plot_attributes(self, **kwargs):
        """
        Update the model's attributes for plotting.
        """
        raise NotImplementedError

    def GetIdx(self, kind: str = "train") -> pd.Series:
        """
        Get the index for the model.

        This is used to get the index for the train, forecast, or all data,
        and is used to filter the X and y data when GetX and GetY are called.

        Parameters
        ----------
        kind : str, optional
            The type of index to return. Must be one of "train", "forecast",
            or "all". If "all", the index for both the train and forecast
            data will be returned. The default is "train".

        Returns
        -------
        pd.Series
            The index for the model.
        """
        if kind == "train":
            idx = self.train_index
        elif kind == "forecast":
            idx = self.forecast_index
        elif kind == "all" or kind is None:
            idx = self.train_index.tolist() + self.forecast_index.tolist()
            idx = pd.Series(idx).sort_values().drop_duplicates().values
        else:
            raise ValueError("kind must be 'train', 'forecast', or 'all'")
        return idx

    def GetXBase(self, kind: str = "train") -> pd.DataFrame:
        """
        This is a wrapper function for the Triangle class's get_X_base method.

        Returns the base triangle data for the model. Uses GetIdx to filter
        the data.

        Parameters
        ----------
        kind : str, optional
            The type of data to return. If None, return the train data.
            The default is "train".

        Returns
        -------
        pd.DataFrame
            The base triangle data for the model.
        """
        # get index
        idx = self.GetIdx(kind)

        # get X
        X = self.tri.get_X_base(kind, cal=self.use_cal)

        # filter X and return
        X = X.loc[idx, :]
        return X

    def GetYBase(self, kind="train"):
        """
        This is a wrapper function for the Triangle class's get_y_base method.

        Returns the base triangle data for the model.

        Parameters
        ----------
        """
        idx = self.GetIdx(kind)

        y = self.tri.get_y_base(kind)
        y = y[idx]
        return y

    def GetAcc(self, kind: str = None) -> pd.Series:
        """
        Getter for the model's accident period vector, filtered for the
        current model context and kind.

        Parameters
        ----------
        kind : str, optional
            The type of data to return. If None, return all data. If "train",
            return only the training data. If "forecast", return only the
            forecast data. If "all", return both the training and forecast
            data. The default is None.

        Returns
        -------
        pd.Series
            The accident period vector.
        """
        # get current context index
        idx = self.GetIdx(kind)

        # get accident period vector
        if kind is None or kind == "all":
            acc = pd.concat([self.acc, self.acc_forecast])
        elif kind == "train":
            acc = self.acc
        elif kind == "forecast":
            acc = self.acc_forecast
        else:
            raise ValueError("kind must be 'train', 'forecast', 'all'")

        # filter and return
        return acc[idx]

    def GetDev(self, kind: str = None) -> pd.Series:
        """
        Getter for the model's development period vector, filtered for the
        current model context and kind.

        Parameters
        ----------
        kind : str, optional
            The type of data to return. If None, return all data. If "train",
            return only the training data. If "forecast", return only the
            forecast data. If "all", return both the training and forecast
            data. The default is None.

        Returns
        -------
        pd.Series
            The development period vector.
        """
        # get current context index
        idx = self.GetIdx(kind)

        # get development period vector
        if kind is None or kind == "all":
            dev = pd.concat([self.dev, self.dev_forecast])
        elif kind == "train":
            dev = self.dev
        elif kind == "forecast":
            dev = self.dev_forecast
        else:
            raise ValueError("kind must be 'train', 'forecast', 'all'")

        # filter and return
        return dev[idx]

    def GetCal(self, kind: str = None) -> pd.Series:
        """
        Getter for the model's calendar period vector, filtered for the
        current model context and kind.

        Parameters
        ----------
        kind : str, optional
            The type of data to return. If None, return all data. If "train",
            return only the training data. If "forecast", return only the
            forecast data. If "all", return both the training and forecast
            data. The default is None.

        Returns
        -------
        pd.Series
            The calendar period vector.
        """
        # get current context index
        idx = self.GetIdx(kind)

        # get calendar period vector
        if kind is None or kind == "all":
            cal = pd.concat([self.cal, self.cal_forecast])
        elif kind == "train":
            cal = self.cal
        elif kind == "forecast":
            cal = self.cal_forecast
        else:
            raise ValueError("kind must be 'train', 'forecast', 'all'")

        # filter and return
        return cal[idx]

    def GetAccGp(self) -> pd.Series:
        """
        Getter for the model's accident period group vector, filtered for the
        current model context and kind.

        Parameters
        ----------
        None

        Returns
        -------
        pd.Series
            The accident period group vector.

        Notes
        -----
        This method will always return the full accident period group vector, since
        there are no accident period groups applicable to the forecast data.
        """
        # get current context index (always train)
        idx = self.GetIdx("train")

        # return NotImplementedError if no accident period group
        if self.acc_gp is None:
            raise NotImplementedError("acc_gp not implemented")

        # filter acciden and return
        return self.acc_gp[idx]

    def GetDevGp(self) -> pd.Series:
        """
        Getter for the model's development period group vector, filtered for the
        current model context and kind.

        Parameters
        ----------
        None

        Returns
        -------
        pd.Series
            The development period group vector.

        Notes
        -----
        This method will always return the full development period group vector, since
        there are no development period groups applicable to the forecast data.
        """
        # get current context index (always train)
        idx = self.GetIdx("train")

        # return NotImplementedError if no development period group
        if self.dev_gp is None:
            raise NotImplementedError("dev_gp not implemented")

        # filter development and return
        return self.dev_gp[idx]

    def GetCalGp(self) -> pd.Series:
        """
        Getter for the model's calendar period group vector, filtered for the
        current model context and kind.

        Parameters
        ----------
        None

        Returns
        -------
        pd.Series
            The calendar period group vector.

        Notes
        -----
        This method will always return the full calendar period group vector, since
        there are no calendar period groups applicable to the forecast data.
        """
        # get current context index (always train)
        idx = self.GetIdx("train")

        # return NotImplementedError if no calendar period group
        if self.cal_gp is None:
            raise NotImplementedError("cal_gp not implemented")

        # filter calendar and return
        return self.cal_gp[idx]

    def GetAccGpMap(self) -> pd.Series:
        """
        Getter for the model's accident period group map vector, filtered for the
        current model context and kind.

        Parameters
        ----------
        None

        Returns
        -------
        pd.Series
            The accident period group map vector.

        Notes
        -----
        This method will always return the full accident period group map vector, since
        there are no accident period groups applicable to the forecast data.
        """
        # get current context index (always train)
        idx = self.GetIdx("train")

        # return NotImplementedError if no accident period group map
        if self.acc_gp_map is None:
            raise NotImplementedError("acc_gp_map not implemented")

        # filter accident and return
        return self.acc_gp_map[idx]

    def GetDevGpMap(self) -> pd.Series:
        """
        Getter for the model's development period group map vector, filtered for the
        current model context and kind.

        Parameters
        ----------
        None

        Returns
        -------
        pd.Series
            The development period group map vector.

        Notes
        -----
        This method will always return the full development period group map vector, since
        there are no development period groups applicable to the forecast data.
        """
        # get current context index (always train)
        idx = self.GetIdx("train")

        # return NotImplementedError if no development period group map
        if self.dev_gp_map is None:
            raise NotImplementedError("dev_gp_map not implemented")

        # filter development and return
        return self.dev_gp_map[idx]

    def SetDevGpMap(self, dev_gp_map: pd.Series) -> None:
        """
        Setter for the model's development period group map vector.

        Parameters
        ----------
        dev_gp_map : pd.Series
            The development period group map vector.
        """
        # set development period group map vector
        self.dev_gp_map = dev_gp_map

    def GetCalGpMap(self) -> pd.Series:
        """
        Getter for the model's calendar period group map vector, filtered for the
        current model context and kind.

        Parameters
        ----------
        None

        Returns
        -------
        pd.Series
            The calendar period group map vector.

        Notes
        -----
        This method will always return the full calendar period group map vector, since
        there are no calendar period groups applicable to the forecast data.
        """
        # get current context index (always train)
        idx = self.GetIdx("train")

        # return NotImplementedError if no calendar period group map
        if self.cal_gp_map is None:
            raise NotImplementedError("cal_gp_map not implemented")

        # filter calendar and return
        return self.cal_gp_map[idx]

    def SetCalGpMap(self, cal_gp_map: pd.Series) -> None:
        """
        Setter for the model's calendar period group map vector.

        Parameters
        ----------
        cal_gp_map : pd.Series
            The calendar period group map vector.

        Returns
        -------
        None
        """
        # set calendar period group map
        self.cal_gp_map = cal_gp_map

    def GetHeteroGp(self) -> pd.Series:
        """
        Getter for the model's heteroskedasticity group vector,
        filtered for the current model context and kind.

        Parameters
        ----------
        None

        Returns
        -------
        pd.Series
            The heterogeneity group vector.

        Notes
        -----
        This method will always return the full hetero group vector, since
        there are no hetero groups applicable to the forecast data.
        """
        # get current context index (always train)
        idx = self.GetIdx("train")

        # return NotImplementedError if no heterogeneity group
        if self.hetero_gp is None:
            raise NotImplementedError("hetero_gp not implemented")

        # filter heterogeneity and return
        return self.hetero_gp[idx]

    def SetHeteroGp(self, hetero_gp: pd.Series) -> None:
        """
        Setter for the model's heteroskedasticity group vector.

        Parameters
        ----------
        hetero_gp : pd.Series
            The heteroskedasticity group vector.

        Returns
        -------
        None
        """
        # set hetero_gp
        self.hetero_gp = hetero_gp

    def GetExposure(self, kind: str = None) -> pd.Series:
        """
        Getter for the model's exposure vector, filtered for the current model
        context and kind.

        Parameters
        ----------
        kind : str, optional
            The type of data to return. If None, return all data. If "train",
            return only the training data. If "forecast", return only the
            forecast data. If "all", return both the training and forecast
            data. The default is None.

        Returns
        -------
        pd.Series
            The exposure vector.
        """
        # get current context index
        idx = self.GetIdx(kind)

        # get exposure vector
        if kind is None or kind == "all":
            exposure = pd.concat([self.exposure, self.exposure_forecast])
        elif kind == "train":
            exposure = self.exposure
        elif kind == "forecast":
            exposure = self.exposure_forecast
        else:
            raise ValueError("kind must be 'train', 'forecast', 'all'")

        # filter and return
        return exposure[idx]

    def GetX(self, kind=None):
        """
        Getter for the model's X data. If there is no X data, take the base design
        matrix directly from the triangle. When parameters are combined, X is
        created as the design matrix of the combined parameters.
        """
        idx = self.GetIdx(kind)

        if kind is None or kind == "all":
            df = pd.concat([self.X_train, self.X_forecast])
        elif kind == "train":
            df = self.X_train
        elif kind == "forecast":
            df = self.X_forecast
        else:
            raise ValueError("kind must be 'train', 'forecast', 'all'")

        def cond(x):
            return x != "intercept" and x != "is_observed"

        condition = ["intercept"]
        for c in df.columns:
            if cond(c):
                condition.append(c)

        df = df[condition]

        return df.loc[idx, :]

    def SetX(
        self, kind: str = "train", X: pd.DataFrame = None, vars: list = None
    ) -> None:
        """
        Setter for the model's X data. If there is no X data, exit.
        If `vars` is included, only set the X data for those variables.
        Raise an error if the variables are not in the model's X data.
        """
        # get current context index
        idx = self.GetIdx("train")

        # get the correct version of X depending on the kind
        if self.X_train is None:
            raise ValueError("X_train is not defined!")
        else:
            if vars is None:
                if X is None:
                    quit
                else:
                    self.X_train.loc[idx, :] = X
            else:
                if set(vars).issubset(set(self.X_train.columns)):
                    self.X_train.loc[idx, vars] = X
                else:
                    raise ValueError("vars must be in X_train!")

    def GetParameterNames(self, column=None):
        """
        Getter for the model's parameter names.
        """
        print("GetParameterNames is not implemented for this model.")
        raise NotImplementedError

    def GetY(self, kind: str = "train") -> pd.Series:
        """
        Getter for the model's y data. If there is no y data, take the y vector
        directly from the triangle.
        """
        # get index to filter y
        idx = self.GetIdx(kind)

        # get the correct version of y depending on the kind
        if kind.lower() in ["train", "forecast"]:
            if kind.lower() == "train":
                return self.y_train[idx]
            elif kind.lower() == "forecast":
                raise ValueError("y_forecast is what we are trying to predict!")
        else:
            raise ValueError("kind must be 'train' for `y`")

    def SetY(self, y: pd.Series) -> None:
        """
        Setter for the model's y data.
        """
        # if the index of y is not the same as the index of the model,
        # raise an error since we cannot change the index of the model
        # without breaking the index of the triangle
        if not y.index.equals(self.GetIdx("train")):
            raise ValueError("y must have the same index as the model")

        # set y
        self.y_train = y

    def GetWeights(self, kind: str = "train") -> pd.Series:
        """
        Getter for the model's weights. If there are no weights, return None.
        """
        idx = self.GetIdx(kind)
        if kind.lower() == "train":
            return self.weights[idx]
        else:
            raise ValueError("kind must be 'train' for `weights`")

    def SetWeights(self, weights: pd.Series) -> None:
        """
        Setter for the model's weights.
        """
        idx = self.GetIdx("train")
        self.weights[idx] = weights

    def GetN(self) -> int:
        """
        Getter for the model's number of observations.

        This is equal to the number of rows in the "train"
        design matrix.

        Returns
        -------
        int
            Number of observations.
        """
        return self.GetX("train").shape[0]

    def GetP(self) -> int:
        """
        Getter for the model's number of parameters.

        This is equal to the number of columns in the "train"
        design matrix, excluding those that have no nonzero
        values in the design matrix.

        Must do this because if there are calendar year parameters,
        about half of them will be unobserved and thus have no
        nonzero values in the design matrix. They are not parameters
        in the normal sense, so we exclude them from the count.

        Returns
        -------
        int
            Number of parameters.
        """
        # p, unadjusted for columns that are completely 0
        unadj_p = self.GetX("train").shape[1]

        # p, adjusted for columns that are completely 0
        adj_p = unadj_p - self.GetX("train").isna().all().sum()

        return adj_p

    def GetDegreesOfFreedom(self) -> int:
        """
        Getter for the model's degrees of freedom.

        Degrees of freedom is equal to the number of observations
        minus the number of parameters.

        Returns
        -------
        int
            Degrees of freedom.
        """
        return self.GetN() - self.GetP()

    def GetVarY(self, kind="train"):
        """ """
        print("VarY is not implemented for this model.")
        raise NotImplementedError

    def SetVarY(self, var_y, kind="train"):
        """ """
        print("SetVarY is not implemented for this model.")
        raise NotImplementedError

    def Fit(
        self,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        **kwargs,
    ) -> None:
        print("Fit is not implemented for this model.")
        raise NotImplementedError

    def ManualFit(self, **kwargs):
        print("ManualFit is not implemented for this model.")
        raise NotImplementedError

    def Predict(self, kind: str = None, X: pd.DataFrame = None) -> pd.Series:
        print("Predict is not implemented for this model.")
        raise NotImplementedError

    def Ultimate(self, tail=None) -> pd.Series:
        X = self.GetX(
            kind="forecast",
        )
        df = pd.DataFrame(
            {
                "Accident Period": self.tri.get_X_id("all").accident_period,
                f"{self.model_name} Ultimate": self.GetY(kind="train").tolist()
                + self.Predict("forecast", X).tolist(),
            }
        )

        df = df.groupby("Accident Period").sum()[f"{self.model_name} Ultimate"].round(0)

        if tail is None:
            tail = 1
        df = df * tail

        df.index = self.tri.tri.index
        return df

    def GetYhat(self, kind: str = None) -> pd.Series:
        return self.Predict(kind=kind)

    def GetParameters(self) -> pd.DataFrame:
        """
        Get the parameters of the model.

        Returns
        -------
        pd.DataFrame
            The parameters of the model.
        """
        print("GetParameters is not implemented for this model.")
        raise NotImplementedError

    def PredictTriangle(self):
        yhat = pd.DataFrame(dict(yhat=self.Predict()))
        _ids = self.tri.get_X_id()
        return pd.concat([_ids, yhat], axis=1)

    def GetSaturatedModel(self):
        raise NotImplementedError

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
        print("Deviance not implemented for this model")
        raise NotImplementedError

    def RawResiduals(self, kind="train"):
        return self.GetY(kind=kind) - self.GetYhat(kind=kind)

    def PearsonResiduals(self, kind="train"):
        res = np.divide(self.RawResiduals(kind=kind), np.sqrt(self.GetVarY(kind=kind)))
        return res

    def DevianceResiduals(self):
        """
        Deviance residuals are defined as:
        R_i^D = sign(Y_i - Y_hat_i) *
                sqrt(2 * [Y_i * log(Y_i / Y_hat_i) - (Y_i - Y_hat_i)])
        """
        print("DevianceResiduals not implemented for this model")
        raise NotImplementedError

    def ScaleParameter(self):
        print("ScaleParameter not implemented for this model")
        raise NotImplementedError

    def GetYearTypeDict(self):
        d = {
            "accident": "acc",
            "acc": "acc",
            "ay": "acc",
            "accident_year": "acc",
            "development": "dev",
            "dy": "dev",
            "dev": "dev",
            "development_year": "dev",
            "development_period": "dev",
            "calendar": "cal",
            "cal": "cal",
            "cy": "cal",
            "calendar_year": "cal",
        }
        return d

    def Combine(
        self, year_type: str, year_list: list, combined_name: str = None
    ) -> pd.DataFrame:
        """
        This function combines parameters of the design matrix to have the same value.

        THIS SHOULD JUST BE A MAPPING BETWEEN ORIGINAL "BASE" PARAMETERS AND COMBINED
        PARAMETERS SO THE BASE SET CAN BE USED FOR ORDERING, ETC, AND KEEPING AY
        PARAMETERS TOGETHER, DEV PARAM TOGETHER, ETC
        """
        # run through recode year-type dictionary
        year_type = self.GetYearTypeDict()[year_type.lower()]

        # check that acceptable year_type was passed
        if year_type not in ["acc", "dev", "cal"]:
            raise ValueError("Year type must be 'acc', 'dev', or 'cal'")

        # make a copy of the current design matrix
        X_train = self.GetX("train").copy()
        X_forecast = self.GetX("forecast").copy()
        X_new_train = X_train.copy()
        X_new_forecast = X_forecast.copy()

        # recode combined name
        if combined_name is None:
            combined_name = f"{year_type}_combined"
        combined_param_name = combined_name.lower().replace(" ", "_").replace(".", "_")

        if year_type == "acc":
            # generate column names
            cols_to_combine = ["accident_period_" + str(year) for year in year_list]

            # check if columns exist in the DataFrame
            if not set(cols_to_combine).issubset(X_train.columns):
                raise ValueError(
                    "One or more columns specified do not exist in the DataFrame"
                )

            # add new column that is the sum of the specified accident year columns
            X_new_train[combined_param_name] = X_new_train[cols_to_combine].sum(axis=1)
            X_new_forecast[combined_param_name] = X_new_forecast[cols_to_combine].sum(
                axis=1
            )

            # drop the original columns
            X_new_train = X_new_train.drop(columns=cols_to_combine)
            X_new_forecast = X_new_forecast.drop(columns=cols_to_combine)

        elif year_type == "dev":
            # generate column names
            cols_to_combine = [
                "development_period_" + pd.Series([year]).astype(str).str.zfill(4)[0]
                for year in year_list
            ]

            # check if columns exist in the DataFrame
            # if not set(cols_to_combine).issubset(X_train.columns.tolist()):
            for col in cols_to_combine:
                print(col)
                assert col in X_train.columns.tolist(), f"{col} not in X_train.columns"

            # add new column that is the sum of the specified development year columns
            X_new_train[combined_param_name] = X_new_train[cols_to_combine].sum(axis=1)
            X_new_forecast[combined_param_name] = X_new_forecast[cols_to_combine].sum(
                axis=1
            )

            # drop the original columns
            X_new_train = X_new_train.drop(columns=cols_to_combine)
            X_new_forecast = X_new_forecast.drop(columns=cols_to_combine)

        # reset X_train, X_forecast
        self.X_train = X_new_train
        self.X_forecast = X_new_forecast

        self.has_combined_params = True

        # refit the model
        self.Fit()
