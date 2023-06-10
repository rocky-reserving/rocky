try:
    from .triangle import Triangle
except ModuleNotFoundError:
    from triangle import Triangle
except ImportError:
    from triangle import Triangle

import pandas as pd
import numpy as np

from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, d2_tweedie_score
from sklearn.exceptions import ConvergenceWarning
import warnings

import itertools
from tqdm.notebook import tqdm as tqdm


class TriangleTimeSeriesSplit:
    def __init__(
        self,
        triangle: Triangle = None,
        n_splits: int = 5,
        tie_criterion: str = "ave_mse_test",
        tweedie_grid: dict = None,
        **kwargs,
    ):
        self.tri = triangle
        self.n_splits_ = n_splits
        self.split = []
        if tweedie_grid is None:
            self.tweedie_grid = {
                "alpha": np.arange(0, 3.1, 0.1),
                "power": np.array([0]) + np.arange(1, 3.1, 0.1),
                "max_iter": 100000,
            }
        else:
            self.tweedie_grid = tweedie_grid

        # if kwargs for alpha, p and max_iter are provided, use those
        if "alpha" in kwargs:
            self.tweedie_grid["alpha"] = kwargs["alpha"]
        if "power" in kwargs:
            self.tweedie_grid["power"] = kwargs["power"]
        if "max_iter" in kwargs:
            self.tweedie_grid["max_iter"] = kwargs["max_iter"]

        # set tie criterion if there is more than one
        # pareto optimal model
        self.tie_criterion = tie_criterion

    def GetSplit(self):
        X_id = self.tri.get_X_id().reset_index(drop=True)

        # current calendar period
        current_cal = self.tri.getCurCalendarIndex()

        for i in range(1, self.n_splits_ + 1):
            # get the calendar period for the current split
            split_cal = current_cal - i

            # get the indices for training and validation set
            train_indices = X_id.cal[X_id.cal.lt(split_cal)].index.to_numpy()
            test_indices = X_id.cal[
                X_id.cal.ge(split_cal) & X_id.cal.le(current_cal)
            ].index.to_numpy()

            yield train_indices, test_indices

    def GridTweedie(self, alpha=None, power=None, max_iter=None):
        if alpha is not None:
            self.tweedie_grid["alpha"] = alpha

        if power is not None:
            self.tweedie_grid["power"] = power

        if max_iter is not None:
            self.tweedie_grid["max_iter"] = max_iter

    def TuneTweedie(self):
        """Trains a set of Tweedie models with different hyperparameters,
        and stores the result.

        This method will fit a Tweedie model for each combination of hyperparameters
        (alpha, power), and each split of the data. The MSE and d2 score will be
        computed for each model and stored.
        """
        first_ay = self.tri.ay.min()

        result_dict = {}

        parameters = list(
            itertools.product(self.tweedie_grid["alpha"], self.tweedie_grid["power"])
        )
        n_parameters = len(parameters)

        n_failed_to_converge = 0

        for alpha, power in parameters:
            result_dict[f"alpha_{np.round(alpha, 2)}_power_{np.round(power, 2)}"] = {}

        for train, val in self.GetSplit():
            X_train = self.tri.get_X_base().iloc[train]
            y_train = self.tri.get_y_base()[train]

            X_test = self.tri.get_X_base().iloc[val]
            y_test = self.tri.get_y_base()[val]
            X_id_test = self.tri.get_X_id().iloc[val]

            excl_cal = X_id_test.cal.min() + first_ay - 1

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
                        model = TweedieRegressor(
                            power=np.round(power, 2),
                            alpha=np.round(alpha, 2),
                            max_iter=self.tweedie_grid["max_iter"],
                        ).fit(X_train, y_train)
                except (ConvergenceWarning, RuntimeWarning):
                    # if n_failed_to_converge == 0:
                    #     tqdm.write("Failed to converge: ", end="")
                    n_failed_to_converge += 1
                    # tqdm.write(f"{n_failed_to_converge} ", end="")
                    continue

                # add mean squared error to result dictionary
                result_dict[f"alpha_{np.round(alpha, 2)}_power_{np.round(power, 2)}"][
                    f"cy_{excl_cal}_mse_train"
                ] = mean_squared_error(y_train, model.predict(X_train))
                result_dict[f"alpha_{np.round(alpha, 2)}_power_{np.round(power, 2)}"][
                    f"cy_{excl_cal}_mse_test"
                ] = mean_squared_error(y_test, model.predict(X_test))

                # add d2 score to result dictionary
                result_dict[f"alpha_{np.round(alpha, 2)}_power_{np.round(power, 2)}"][
                    f"cy_{excl_cal}_d2_train"
                ] = d2_tweedie_score(
                    y_train, model.predict(X_train), power=np.round(power, 2)
                )
                result_dict[f"alpha_{np.round(alpha, 2)}_power_{np.round(power, 2)}"][
                    f"cy_{excl_cal}_d2_test"
                ] = d2_tweedie_score(
                    y_test, model.predict(X_test), power=np.round(power, 2)
                )

                # add MAE to result dictionary
                result_dict[f"alpha_{np.round(alpha, 2)}_power_{np.round(power, 2)}"][
                    f"cy_{excl_cal}_mae_train"
                ] = mean_absolute_error(y_train, model.predict(X_train))
                result_dict[f"alpha_{np.round(alpha, 2)}_power_{np.round(power, 2)}"][
                    f"cy_{excl_cal}_mae_test"
                ] = mean_absolute_error(y_test, model.predict(X_test))

            print(f"{n_failed_to_converge} / {n_parameters} models failed to converge.")

            tweedie_result = pd.DataFrame(result_dict).T

            for m in "mse_train mse_test d2_train d2_test mae_train mae_test".split():
                tweedie_result[f"ave_{m}"] = tweedie_result.filter(like=m).mean(axis=1)
                tweedie_result[f"sd_{m}"] = tweedie_result.filter(like=m).std(axis=1)
                tweedie_result[f"cv_{m}"] = (
                    tweedie_result[f"sd_{m}"] / tweedie_result[f"ave_{m}"]
                )

            cols_to_drop = [c for c in tweedie_result.columns if "cy_" in c]

            tweedie_result = tweedie_result.drop(columns=cols_to_drop)

            tweedie_result = tweedie_result.T

            for c in tweedie_result.columns.tolist():
                _, a, _, power = c.split("_")
                tweedie_result.loc["alpha", c] = np.round(float(a), 2)
                tweedie_result.loc["power", c] = np.round(float(power), 2)

            self.tweedie_result = tweedie_result.T.sort_values("ave_mse_test")

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
        alpha = optimal_model["alpha"]
        power = optimal_model["power"]

        # Re-fit a model with the optimal hyperparameters, and return it
        best_model = TweedieRegressor(alpha=alpha, power=power, link="log")
        best_model.fit(self.tri.get_X_base("train"), self.tri.get_y_base("train"))
        return best_model
