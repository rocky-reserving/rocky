# try:
#     from .triangle import Triangle
# except ModuleNotFoundError:
#     from triangle import Triangle
# except ImportError:
#     from triangle import Triangle
from triangle import Triangle

import pandas as pd
import numpy as np

import sklearn
import xgboost
from sklearn.linear_model import TweedieRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, d2_tweedie_score
from sklearn.exceptions import ConvergenceWarning
import warnings

import itertools
from tqdm import tqdm


class TriangleTimeSeriesSplit:
    """
    Class for splitting a triangle into training and validation sets.

    Methods
    -------
    - GetSplit()
    - GridTweedie()
    - TuneTweedie()
    - GetBestModel()
    - TweedieParetoFront()
    - OptimalTweedie()
    """

    def __init__(
        self,
        triangle: Triangle = None,
        n_splits: int = 5,
        tie_criterion: str = "ave_mse_test",
        model_type: str = "tweedie",
        log_transform: bool = False,
        tweedie_grid: dict = None,
        randomforest_grid: dict = None,
        xgboost_grid: dict = None,
        model=None,
        **kwargs,
    ):
        self.tri = triangle
        self.n_splits_ = n_splits
        self.split = []
        self.has_tuning_results = False
        self.is_tuned = False
        self.model_type = model_type
        self.model = model
        self.log_transform = log_transform

        # if no grid is provided, use the default grid
        if tweedie_grid is None:
            if model_type == "tweedie":
                self.tweedie_grid = {
                    "alpha": np.arange(0, 3.1, 0.1),
                    "power": np.array([0]) + np.arange(1, 3.1, 0.1),
                    "max_iter": 100000,
                }
            elif model_type == "loglinear":
                self.tweedie_grid = {
                    "alpha": np.arange(0, 3.1, 0.1),
                    "l1_ratio": np.arange(0, 1.05, 0.05),
                    "max_iter": 100000,
                }
        else:
            self.tweedie_grid = tweedie_grid
        if randomforest_grid is None:
            self.randomforest_grid = {
                "n_estimators": [10, 25, 50, 100],
                "max_depth": [None, 10, 25, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            }
        else:
            self.randomforest_grid = randomforest_grid
        if xgboost_grid is None:
            self.xgboost_grid = {
                "n_estimators": [100, 200, 500, 1000],
                "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                "max_depth": [3, 4, 5, 6, 7, 8],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.2, 0.3],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "alpha": [0, 0.5, 1],
                "lambda": [1, 1.5, 2],
            }
        else:
            self.xgboost_grid = xgboost_grid

        # if **kwargs are provided, use them to update the grid
        if "alpha" in kwargs:
            self.tweedie_grid["alpha"] = kwargs["alpha"]
            self.xgboost_grid["alpha"] = kwargs["alpha"]
        if "power" in kwargs:
            self.tweedie_grid["power"] = kwargs["power"]
        if "l1_ratio" in kwargs:
            self.tweedie_grid["l1_ratio"] = kwargs["l1_ratio"]
        if "max_iter" in kwargs:
            self.tweedie_grid["max_iter"] = kwargs["max_iter"]
        if "n_estimators" in kwargs:
            self.randomforest_grid["n_estimators"] = kwargs["n_estimators"]
            self.xgboost_grid["n_estimators"] = kwargs["n_estimators"]
        if "max_depth" in kwargs:
            self.randomforest_grid["max_depth"] = kwargs["max_depth"]
            self.xgboost_grid["max_depth"] = kwargs["max_depth"]
        if "min_samples_split" in kwargs:
            self.randomforest_grid["min_samples_split"] = kwargs["min_samples_split"]
        if "min_samples_leaf" in kwargs:
            self.randomforest_grid["min_samples_leaf"] = kwargs["min_samples_leaf"]
        if "bootstrap" in kwargs:
            self.randomforest_grid["bootstrap"] = kwargs["bootstrap"]
        if "learning_rate" in kwargs:
            self.xgboost_grid["learning_rate"] = kwargs["learning_rate"]
        if "min_child_weight" in kwargs:
            self.xgboost_grid["min_child_weight"] = kwargs["min_child_weight"]
        if "gamma" in kwargs:
            self.xgboost_grid["gamma"] = kwargs["gamma"]
        if "subsample" in kwargs:
            self.xgboost_grid["subsample"] = kwargs["subsample"]
        if "colsample_bytree" in kwargs:
            self.xgboost_grid["colsample_bytree"] = kwargs["colsample_bytree"]
        if "lambda" in kwargs:
            self.xgboost_grid["lambda"] = kwargs["lambda"]

        # set tie criterion if there is more than one
        # pareto optimal model
        self.tie_criterion = tie_criterion

    def __repr__(self):
        return f"TriangleTimeSeriesTuner(n_splits={self.n_splits_}, \
model_type={self.model_type})"

    def GetSplit(self):
        """Yields the indices for the training and validation sets."""
        X_id = self.tri.get_X_id().reset_index(drop=True)

        # current calendar period
        current_cal = self.tri.getCurCalendarIndex()

        for i in range(1, self.n_splits_ + 1):
            # get the calendar period for the current split
            split_cal = current_cal - i

            # get the indices for training and validation set
            train_indices = X_id.calendar_period[
                X_id.calendar_period.lt(split_cal)
            ].index.to_numpy()
            test_indices = X_id.calendar_period[
                X_id.calendar_period.ge(split_cal)
                & X_id.calendar_period.le(current_cal)
            ].index.to_numpy()

            yield train_indices, test_indices

    def GridTweedie(
        self, alpha=None, power=None, l1_ratio=None, max_iter=None, model_type="tweedie"
    ):
        """
        Sets the grid for the hyperparameters of the Tweedie models.

        Default values for the seach are:
        alpha = [0, 0.1, 0.2, ..., 3]
        power = [0, 1, 1.1, 1.2, ..., 3] (does not include (0, 1) interval)
        max_iter = 100000

        Parameters
        ----------
        alpha : array-like, default=None
        power : array-like, default=None
        l1_ratio : array-like, default=None
        max_iter : int, default=None
        model_type : str, default='tweedie'
            The model type to use. Currently can be 'tweedie' or 'loglinear'.
        """
        if model_type in ["tweedie", "loglinear"]:
            if alpha is not None:
                self.tweedie_grid["alpha"] = alpha

        if model_type in ["loglinear"]:
            if l1_ratio is not None:
                self.tweedie_grid["l1_ratio"] = l1_ratio

        if model_type in ["tweedie"]:
            if power is not None:
                self.tweedie_grid["power"] = power

        if model_type in ["tweedie", "loglinear"]:
            if max_iter is not None:
                self.tweedie_grid["max_iter"] = max_iter

    def TuneTweedie(self):
        """
        Trains a set of Tweedie models with different hyperparameters,
        and stores the result.

        This method will fit a Tweedie model for each combination of hyperparameters
        (alpha, power), for each split of the data. The models are then evaluated
        on the validation set, and the results are stored in the self.split list.

        The results for each are stored in a dictionary with the following
        structure:
        {
            "alpha_0.0_power_1.0": {
                "train": {
                    "mse": 0.0,
                    "mae": 0.0,
                    "d2": 0.0
                },
                "test": {
                    "mse": 0.0,
                    "mae": 0.0,
                    "d2": 0.0
                }
            },
            "alpha_0.0_power_1.1": {
                "train": {
                    "mse": 0.0,
                    "mae": 0.0,
                    "d2": 0.0
                },
                "test": {
                    "mse": 0.0,
                    "mae": 0.0,
                    "d2": 0.0
                }
            },
            ...
        }

        The results are stored in the self.split list, which is a list of dictionaries
        """
        # get the minimum and maximum values of ay
        first_ay = self.tri.ay.min()

        result_dict = {}

        model_params = {
            "tweedie": ["alpha", "power"],
            "loglinear": ["alpha", "l1_ratio"],
        }

        if self.model_type == "tweedie":
            param_map = {
                "tweedie": itertools.product(
                    self.tweedie_grid["alpha"], self.tweedie_grid["power"]
                )
            }

        elif self.model_type == "loglinear":
            param_map = {
                "loglinear": itertools.product(
                    self.tweedie_grid["alpha"], self.tweedie_grid["l1_ratio"]
                )
            }

        parameters = list(param_map[self.model_type])
        n_parameters = len(parameters)

        n_failed_to_converge = 0

        cur_params = {}
        for param in model_params[self.model_type]:
            cur_params[param] = None

        ############################ 6/14 right here ##########################################
        # for **cur_params in parameters:
        if self.model_type == "tweedie":
            for alpha, power in parameters:
                result_dict[
                    f"alpha_{np.round(alpha, 2)}_power_{np.round(power, 2)}"
                ] = {}
        elif self.model_type == "loglinear":
            for alpha, l1_ratio in parameters:
                result_dict[
                    f"alpha_{np.round(alpha, 2)}_l1_ratio_{np.round(l1_ratio, 2)}"
                ] = {}

        for train, val in self.GetSplit():
            X_train = self.tri.get_X_base().iloc[train]
            X_test = self.tri.get_X_base().iloc[val]

            if self.log_transform:
                y_train = np.log(self.tri.get_y_base()[train])
                y_test = np.log(self.tri.get_y_base()[val])
            else:
                y_train = self.tri.get_y_base()[train]
                y_test = self.tri.get_y_base()[val]

            X_id_test = self.tri.get_X_id().iloc[val]

            excl_cal = X_id_test.calendar_period.min() + first_ay - 1

            for alpha, power in tqdm(
                parameters,
                total=n_parameters,
                desc=f"Training models with data through {excl_cal}",
            ):
                try:
                    # catch warnings related to convergence
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=ConvergenceWarning)
                        warnings.filterwarnings("error", category=RuntimeWarning)
                        warnings.filterwarnings("ignore", category=UserWarning)
                        if self.model_type == "tweedie":
                            model = TweedieRegressor(
                                power=np.round(power, 2),
                                alpha=np.round(alpha, 2),
                                max_iter=self.tweedie_grid["max_iter"],
                            ).fit(X_train, y_train)
                        elif self.model_type == "loglinear":
                            model = ElasticNet(
                                alpha=np.round(alpha, 2),
                                l1_ratio=np.round(power, 2),
                                max_iter=self.tweedie_grid["max_iter"],
                            ).fit(X_train, y_train)
                except (ConvergenceWarning, RuntimeWarning):
                    # if n_failed_to_converge == 0:
                    #     tqdm.write("Failed to converge: ", end="")
                    n_failed_to_converge += 1
                    # tqdm.write(f"{n_failed_to_converge} ", end="")
                    continue

                if self.model_type == "tweedie":
                    p = {"alpha": np.round(alpha, 2), "power": np.round(power, 2)}
                elif self.model_type == "loglinear":
                    p = {"alpha": np.round(alpha, 2), "l1_ratio": np.round(l1_ratio, 2)}

                # add mean squared error to result dictionary
                if self.model_type == "tweedie":
                    key = f"alpha_{np.round(alpha, 2)}_power_{np.round(power, 2)}"
                elif self.model_type == "loglinear":
                    key = f"alpha_{np.round(alpha, 2)}_l1_ratio_{np.round(l1_ratio, 2)}"

                result_dict[key][f"cy_{excl_cal}_mse_train"] = mean_squared_error(
                    y_train, model.predict(X_train)
                )
                result_dict[key][f"cy_{excl_cal}_mse_test"] = mean_squared_error(
                    y_test, model.predict(X_test)
                )

                # add d2 score to result dictionary
                if self.model_type == "tweedie":
                    result_dict[key][f"cy_{excl_cal}_d2_train"] = d2_tweedie_score(
                        y_train, model.predict(X_train), power=np.round(power, 2)
                    )
                    result_dict[key][f"cy_{excl_cal}_d2_test"] = d2_tweedie_score(
                        y_test, model.predict(X_test), power=np.round(power, 2)
                    )

                # add MAE to result dictionary
                result_dict[key][f"cy_{excl_cal}_mae_train"] = mean_absolute_error(
                    y_train, model.predict(X_train)
                )
                result_dict[key][f"cy_{excl_cal}_mae_test"] = mean_absolute_error(
                    y_test, model.predict(X_test)
                )

            print(f"{n_failed_to_converge} / {n_parameters} models failed to converge.")

            tweedie_result = pd.DataFrame(result_dict).T

            for m in "mse_train mse_test mae_train mae_test".split():
                tweedie_result[f"ave_{m}"] = tweedie_result.filter(like=m).mean(axis=1)
                tweedie_result[f"sd_{m}"] = tweedie_result.filter(like=m).std(axis=1)
                tweedie_result[f"cv_{m}"] = (
                    tweedie_result[f"sd_{m}"] / tweedie_result[f"ave_{m}"]
                )

            if self.model_type == "tweedie":
                for m in "d2_train d2_test".split():
                    tweedie_result[f"ave_{m}"] = tweedie_result.filter(like=m).mean(
                        axis=1
                    )
                    tweedie_result[f"sd_{m}"] = tweedie_result.filter(like=m).std(
                        axis=1
                    )
                    tweedie_result[f"cv_{m}"] = (
                        tweedie_result[f"sd_{m}"] / tweedie_result[f"ave_{m}"]
                    )

            cols_to_drop = [c for c in tweedie_result.columns if "cy_" in c]

            tweedie_result = tweedie_result.drop(columns=cols_to_drop)

            tweedie_result = tweedie_result.T

            for c in tweedie_result.columns.tolist():
                if self.model_type == "tweedie":
                    _, a, _, power = c.split("_")
                    tweedie_result.loc["alpha", c] = np.round(float(a), 2)
                    tweedie_result.loc["power", c] = np.round(float(power), 2)
                elif self.model_type == "loglinear":
                    print(c)
                    _, a, _, l1 = c.split("_")
                    tweedie_result.loc["alpha", c] = np.round(float(a), 2)
                    tweedie_result.loc["l1_ratio", c] = np.round(float(l1), 2)

            self.tweedie_result = tweedie_result.T.sort_values("ave_mse_test")

        self.has_tuning_results = True

    def TweedieParetoFront(self, measures=None):
        """
        Get the Pareto front from the results of TuneTweedie.
        In multi-objective optimization, the Pareto front consists of the solutions
        that are optimal in the sense that no other solution is superior to them
        when considering all objectives. Here, we consider models with the lowest mean
        squared error and the lowest standard deviation (for stability)
        as the optimal solutions.

        Parameters:
        ----------
        measures: list, default=None
            A dictionary where keys are the names of measures to consider when computing
            the Pareto front, and the values are 'min' if we want to minimize that
            measure or 'max' if we want to maximize that measure. For example, if we
            want to minimize the mean squared error and maximize the d2 score, we would
            pass measures={'mse': 'min', 'd2': 'max'}. If None, the default is to
            minimize the mean squared error and minimize the standard deviation of the
            mean squared error. This is the recommended setting, and is meant to balance
            model performance and stability.

        Returns:
        -------
        A pandas DataFrame containing the Pareto front.
        """

        # Ensure TuneTweedie has been called
        if not hasattr(self, "tweedie_result"):
            self.TuneTweedie()

        # If no measures are passed, use the default
        if measures is None:
            measures = {"ave_mse_test": "min", "sd_mse_test": "min"}

        # Create a copy of the results dataframe
        results = self.tweedie_result.copy()

        # Initialize a boolean Series to keep track of whether each model
        # is Pareto optimal
        is_pareto_optimal = pd.Series([True] * len(results), index=results.index)

        for i, model in results.iterrows():
            # Compare the current model to all other models
            for measure, direction in measures.items():
                if direction == "min":
                    is_pareto_optimal[i] = not any(
                        (results[measure] <= model[measure])
                        & (
                            results[list(measures.keys())]
                            != model[list(measures.keys())]
                        ).any(axis=1)
                    )
                elif direction == "max":
                    is_pareto_optimal[i] = not any(
                        (results[measure] >= model[measure])
                        & (
                            results[list(measures.keys())]
                            != model[list(measures.keys())]
                        ).any(axis=1)
                    )

        # Return only the Pareto optimal models
        self.tweedie_optimal = results[is_pareto_optimal]
        return self.tweedie_optimal

    def OptimalTweedie(self, measures=None, tie_criterion=None):
        """
        Returns the optimal Tweedie model, which is defined to be the Pareto-optimal
        model, optimized on selected optimization criteria. If more than one model is
        Pareto-optimal, the model with the lowest mean squared error is returned.

        If the TweedieParetoFront method has not been called, we will run it
        using the measures passed to this method.

        This method is intended to be the only method that the user needs to call
        to get the optimal Tweedie model.
        """
        # Ensure TweedieParetoFront has been called
        if not hasattr(self, "tweedie_optimal"):
            self.TweedieParetoFront(measures=measures)

        if tie_criterion is None:
            tie_criterion = self.tie_criterion

        # if there is more than one optimal model, return the one with the lowest MSE
        if self.tweedie_optimal.shape[0] > 1:
            optimal_model = self.tweedie_optimal.sort_values(tie_criterion).iloc[0]
        else:
            optimal_model = self.tweedie_optimal.iloc[0]

        # get the optimal set of hyperparameters
        if self.model_type == "tweedie":
            alpha = optimal_model["alpha"]
            power = optimal_model["power"]
        elif self.model_type == "loglinear":
            alpha = optimal_model["alpha"]
            l1_ratio = optimal_model["l1_ratio"]

        # Re-fit a model with the optimal hyperparameters, and return it
        if self.model_type == "tweedie":
            best_model = TweedieRegressor(alpha=alpha, power=power, link="log")
        elif self.model_type == "loglinear":
            best_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        best_model.fit(self.tri.get_X_base("train"), self.tri.get_y_base("train"))
        return best_model

    def TuneModel(
        self, model, grid=None, n_jobs=-1, verbose=0, use_cal=False, **kwargs
    ):
        """
        Tune a model's hyperparameters using time series cross validation.

        Parameters:
        ----------
        model: model object
            A model object that has a fit and predict method.
        grid: dictionary, default=None
            A dictionary of hyperparameter values to try. The keys should be the
            hyperparameter names and the values should be lists of values to try.
            For example, if we want to try 3 values of alpha and 2 values of power,
            we would pass grid={'alpha': [0.5, 1, 2], 'power': [1, 2]}. If None,
            the default is to try 3 values of alpha and 2 values of power.

            This method has default parameter grids for the following models:
                1. TweedieRegressor
                2. RandomForestRegressor
                3. XGBRegressor

            If you pass a model that is not one of these three, you must pass a grid,
            otherwise an error will be raised.
        n_jobs: int, default=-1
            The number of jobs to run in parallel. -1 means use all processors.
        verbose: int, default=0
            The verbosity level.
        **kwargs: keyword arguments
            Additional keyword arguments to pass to the model's fit method. For example,
            if you want to pass a validation set, you can pass X_valid and y_valid here.
        """
        first_ay = self.tri.acc.min()

        result_dict = {}

        # apply default grid if none is passed (if model is one of the three models
        # that have default grids)
        if grid is None:
            if isinstance(model, sklearn.linear_model.TweedieRegressor):
                model_type = "tweedie"
                grid = self.tweedie_grid
            elif isinstance(model, sklearn.ensemble.RandomForestRegressor):
                model_type = "randomforest"
                grid = self.randomforest_grid
            elif isinstance(model, xgboost.XGBRegressor):
                model_type = "xgboost"
                grid = self.xgboost_grid
            else:
                raise ValueError(
                    "You must pass a grid for this model or use a model that\
                        has a default grid."
                )

        # anonymous function to get all combinations of hyperparameters
        def get_grid(grid):
            return list(itertools.product(*[grid[k] for k in grid.keys()]))

        # anonymous function to get the number of combinations of hyperparameters
        def get_n_parameters(grid):
            return len(get_grid(grid))

        # anonymous function to get a key for the result dictionary
        def get_key(grid, param_set):
            keys = grid.keys()
            key = "-".join(
                f"{k}_{np.round(v, 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}"
                for k, v in zip(keys, param_set)
            )
            return key

        # get all combinations of hyperparameters
        parameters = get_grid(grid)
        n_parameters = get_n_parameters(grid)

        # count number of models that fail to converge
        n_failed_to_converge = 0

        # anonymous function to train the passed model on the passed data
        def train_model(model, X_train, y_train, X_test, hyperparameters, **kwargs):
            model.set_params(**hyperparameters)
            try:
                model.fit(X_train, y_train, **kwargs)
            except ValueError:
                nonlocal n_failed_to_converge
                n_failed_to_converge += 1
                return None

            return model.predict(X_test)

        # anonymous function to add the mean squared error to the result_dict
        def add_mse(result_dict, grid, parameters, para, y_train, y_test, y_pred):
            result_dict[get_key(grid, parameters[parameters.index(para)])][
                f"cy_{excl_cal}_mse_train"
            ] = mean_squared_error(y_train, model.predict(X_train))
            result_dict[get_key(grid, parameters[parameters.index(para)])][
                f"cy_{excl_cal}_mse_test"
            ] = mean_squared_error(y_test, y_pred)
            return result_dict

        # add d2 score to result dictionary
        def add_d2(result_dict, grid, parameters, para, y_train, y_test, power):
            result_dict[get_key(grid, parameters[parameters.index(para)])][
                f"cy_{excl_cal}_d2_train"
            ] = d2_tweedie_score(
                y_train, model.predict(X_train), power=np.round(power, 2)
            )
            result_dict[get_key(grid, parameters[parameters.index(para)])][
                f"cy_{excl_cal}_d2_test"
            ] = d2_tweedie_score(
                y_test, model.predict(X_test), power=np.round(power, 2)
            )
            return result_dict

        def add_mae(result_dict, grid, parameters, para, y_train, y_test, y_pred):
            result_dict[get_key(grid, parameters[parameters.index(para)])][
                f"cy_{excl_cal}_mae_train"
            ] = mean_absolute_error(y_train, model.predict(X_train))
            result_dict[get_key(grid, parameters[parameters.index(para)])][
                f"cy_{excl_cal}_mae_test"
            ] = mean_absolute_error(y_test, y_pred)
            return result_dict

        # loop through all combinations of hyperparameters and initialize result_dict
        for param_set in parameters:
            result_dict[get_key(grid, param_set)] = {}

        # loop through all combinations of train/val splits
        for train, val in self.GetSplit():
            # training data
            X_train = self.tri.get_X_base(cal=use_cal).iloc[train]
            y_train = self.tri.get_y_base()[train]

            # validation data
            X_test = self.tri.get_X_base(cal=use_cal).iloc[val]
            y_test = self.tri.get_y_base()[val]

            # labels for the training and validation data
            X_id_test = self.tri.get_X_id().iloc[val]

            # get the breakpoint year for the current train/val split
            excl_cal = X_id_test.calendar_period.min() + first_ay.year - 1

            # loop through all combinations of hyperparameters in the current
            # train/val split
            for para in tqdm(
                parameters,
                total=n_parameters,
                desc=f"Training models with data through {excl_cal}",
            ):
                # train model and get predictions
                y_pred = train_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    dict(zip(grid.keys(), para)),
                    **kwargs,
                )

                # add mean squared error to result dictionary
                result_dict = add_mse(
                    result_dict, grid, parameters, para, y_train, y_test, y_pred
                )

                # add d2 score to result dictionary if tweedie
                if model.__class__.__name__ == "TweedieRegressor":
                    result_dict = add_d2(
                        result_dict, grid, parameters, para, y_train, y_test, y_pred
                    )

                # add MAE to result dictionary
                result_dict = add_mae(
                    result_dict, grid, parameters, para, y_train, y_test, y_pred
                )

        print(f"{n_failed_to_converge} / {n_parameters} models failed to converge.")

        result = pd.DataFrame(result_dict).T

        # list of methods to loop through
        methods = "mse_train mse_test mae_train mae_test".split()
        if model.__class__.__name__ == "TweedieRegressor":
            methods += "d2_train d2_test".split()

        for m in methods:
            result[f"ave_{m}"] = result.filter(like=m).mean(axis=1)
            result[f"sd_{m}"] = result.filter(like=m).std(axis=1)
            result[f"cv_{m}"] = result[f"sd_{m}"] / result[f"ave_{m}"]

        cols_to_drop = [c for c in result.columns if "cy_" in c]

        result = result.drop(columns=cols_to_drop)

        result = result.T

        for c in result.columns.tolist():
            result[c] = result[c].astype(float)

        setattr(self, f"{model_type}_result", result.T.sort_values("ave_mse_test"))

        return result.T.sort_values("ave_mse_test")

    def ModelParetoFront(self, model_type, measures=None):
        """
        Get the Pareto front from the results of a model tuning process.
        In multi-objective optimization, the Pareto front consists of the solutions
        that are optimal in the sense that no other solution is superior to them
        when considering all objectives. Here, we consider models with the lowest mean
        squared error and the lowest standard deviation (for stability)
        as the optimal solutions.

        Parameters:
        ----------
        model_type: str
            The type of the model. It should be one of the following: 'Tweedie',
            'RandomForest', 'XGBoost'.

        measures: dict, default=None
            A dictionary where keys are the names of measures to consider when computing
            the Pareto front, and the values are 'min' if we want to minimize that
            measure or 'max' if we want to maximize that measure. For example, if we
            want to minimize the mean squared error and maximize the d2 score, we would
            pass measures={'mse': 'min', 'd2': 'max'}. If None, the default is to
            minimize the mean squared error and minimize the standard deviation of the
            mean squared error. This is the recommended setting, and is meant to balance
            model performance and stability.

        Returns:
        -------
        A pandas DataFrame containing the Pareto front.
        """

        # Ensure model tuning has been called
        if hasattr(self, f"{model_type.lower()}_result"):
            pass
        else:
            if model_type.lower() == "tweedie":
                m = TweedieRegressor()
            elif model_type.lower() == "randomforest":
                m = RandomForestRegressor()
            elif model_type.lower() == "xgboost":
                m = XGBRegressor()
            else:
                raise ValueError(
                    "Model type must be one of\
                                 'Tweedie', 'RandomForest', or 'XGBoost'."
                )
            self.TuneModel(m)

        # If no measures are passed, use the default
        if measures is None:
            measures = {"ave_mse_test": "min", "sd_mse_test": "min"}

        # Create a copy of the results dataframe
        results = getattr(self, f"{model_type.lower()}_result").copy()

        # Initialize a boolean Series to keep track of whether each model
        # is Pareto optimal
        is_pareto_optimal = pd.Series([True] * len(results), index=results.index)

        for i, model in results.iterrows():
            # Compare the current model to all other models
            for measure, direction in measures.items():
                if direction == "min":
                    is_pareto_optimal[i] = not any(
                        (results[measure] <= model[measure])
                        & (
                            results[list(measures.keys())]
                            != model[list(measures.keys())]
                        ).any(axis=1)
                    )
                elif direction == "max":
                    is_pareto_optimal[i] = not any(
                        (results[measure] >= model[measure])
                        & (
                            results[list(measures.keys())]
                            != model[list(measures.keys())]
                        ).any(axis=1)
                    )

        # Return only the Pareto optimal models
        optimal_results = results[is_pareto_optimal]
        setattr(self, f"{model_type.lower()}_optimal", optimal_results)
        return optimal_results

    def OptimalModel(self, model_type, measures=None, tie_criterion=None):
        """
        Returns the optimal model, which is defined to be the Pareto-optimal
        model, optimized on selected optimization criteria. If more than one model is
        Pareto-optimal, the model with the lowest mean squared error is returned.

        If the ModelParetoFront method has not been called, we will run it
        using the measures passed to this method.

        This method is intended to be the only method that the user needs to call
        to get the optimal model.

        Parameters:
        ----------
        model_type: str
            The type of the model. It should be one of the following:
            'Tweedie', 'RandomForest', 'XGBoost'.

        measures: dict, default=None
            A dictionary where keys are the names of measures to consider when computing
            the Pareto front, and the values are 'min' if we want to minimize that
            measure or 'max' if we want to maximize that measure.

        tie_criterion: str, default=None
            The criterion to use when selecting the best model among those with the same
            score on the selected measures. If None, the mean squared error is used.

        Returns:
        -------
        The optimal model.
        """
        # Ensure ModelParetoFront has been called
        if not hasattr(self, f"{model_type.lower()}_optimal"):
            self.ModelParetoFront(model_type, measures=measures)

        if tie_criterion is None:
            tie_criterion = self.tie_criterion

        # if there is more than one optimal model, return the one with the lowest MSE
        optimal_models = getattr(self, f"{model_type.lower()}_optimal")
        if optimal_models.shape[0] > 1:
            optimal_model = optimal_models.sort_values(tie_criterion).iloc[0]
        else:
            optimal_model = optimal_models.iloc[0]

        # print(f"Optimal model: {optimal_model}")

        # get the optimal set of hyperparameters from the Pareto front df
        def process_dataframe():
            # Create a new DataFrame from the split index
            df = getattr(self, f"{model_type.lower()}_optimal").copy()
            df_new = (
                df.index.to_series().str.split("-", expand=True).reset_index(drop=True)
            )
            # print(f"df_new1: {df_new}")

            # Create an empty DataFrame to store the results
            results = pd.DataFrame()

            # For each column in the new DataFrame
            for col in df_new.columns:
                # print(f"{col}: {df_new[col]}")
                # Split the column into a new DataFrame with two columns
                split_col = df_new[col].str.rsplit("_", n=1, expand=True)
                # print(f"split_col: {split_col}")

                # Use the first column as the column names in the results DataFrame
                # and the second column as the values
                results[split_col[0]] = split_col[1]
                # print(f"results: {results}")

            df_new = df_new.iloc[0].str.rsplit("_", n=1, expand=True).set_index(0).T

            for col in df_new.columns.tolist():
                if col == "bootstrap":
                    df_new[col] = df_new[col].astype(bool)
                elif col == "max_depth":
                    df_new.loc[df_new["max_depth"].eq("None"), "max_depth"] = None
                    df_new.loc[
                        df_new["max_depth"].ne("None"), "max_depth"
                    ] = df_new.loc[df_new["max_depth"].ne("None"), "max_depth"].astype(
                        int
                    )
                elif col in ["min_samples_leaf", "min_samples_split", "n_estimators"]:
                    print(f"col: {col}\n{df_new[col]}")
                    df_new[col] = df_new[col].astype(int)

            return df_new

        optimal_df = process_dataframe()

        # get the optimal set of hyperparameters from the respective grid
        grid = getattr(self, f"{model_type.lower()}_grid")
        hyperparameters = {param: optimal_df[param].values[0] for param in grid.keys()}

        # create and fit the model with the optimal hyperparameters, and return it
        if model_type.lower() == "tweedie":
            best_model = TweedieRegressor(**hyperparameters, link="log")
        elif model_type.lower() == "randomforest":
            best_model = RandomForestRegressor(**hyperparameters)
        elif model_type.lower() == "xgboost":
            best_model = XGBRegressor(**hyperparameters)
        else:
            raise ValueError(f"Invalid model_type {model_type}")

        best_model.fit(self.tri.get_X_base("train"), self.tri.get_y_base("train"))
        return best_model
