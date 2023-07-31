from rocky.triangle import Triangle

import pandas as pd
import numpy as np

import sklearn

# regression models
import xgboost
from sklearn.linear_model import (
    TweedieRegressor,
    ElasticNet,
    Lasso,
    Ridge,
    LinearRegression,
)

# regression metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

# clustering algorithms
from sklearn.cluster import AgglomerativeClustering

# clustering metrics
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score
    )


from sklearn.model_selection import ParameterGrid

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import itertools
from tqdm import tqdm


class TriangleTimeSeriesSplit:
    """
    Class for splitting a triangle into training and validation sets, and
    tuning the hyperparameters of a model using a grid search with a cross
    validation approach (modified for the triangle structure).

    Does hyperparameter tuning for both regression and clustering methods.


    Methods
    -------
    - GetSplit()
    """

    def __init__(
        self,
        triangle: Triangle = None,
        n_splits: int = 5,
        tie_criterion: str = "ave_mse_test",
        model_type: str = "tweedie",
        log_transform: bool = False,
        grid: dict = None,
        loglinear_grid: dict = None,
        tweedie_grid: dict = None,
        randomforest_grid: dict = None,
        xgboost_grid: dict = None,
        model=None,

        regression_hyperparameters: bool = True,

        clustering_hyperparameters: bool = False,
        clustering_grid: dict = None,
        n_failed_to_converge: int = 0,
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
        self.n_failed_to_converge = n_failed_to_converge

        self.regression_hyperparameters = regression_hyperparameters
        self.clustering_hyperparameters = clustering_hyperparameters
        self.clustering_grid = clustering_grid

        # if no grids are provided, use the default grids
        ## tweedie
        if tweedie_grid is None:
            if model_type == "tweedie":
                self.tweedie_grid = {
                    "alpha": np.arange(0, 3.1, 0.1),
                    "power": np.array([0]) + np.arange(1, 3.1, 0.1),
                    "max_iter": [100000],
                }
        else:
            self.tweedie_grid = tweedie_grid

        ## loglinear
        if loglinear_grid is None:
            if model_type == "loglinear":
                self.loglinear_grid = {
                    "alpha": np.arange(0, 3.1, 0.1),
                    "l1ratio": np.arange(0, 1.05, 0.05),
                    "max_iter": [100000],
                }
        else:
            self.loglinear_grid = loglinear_grid

        ## randomforest
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

        ## xgboost
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

        ## clustering
        if clustering_grid is None:
            # calculate reasonable upper bound based on the number of development 
            # periods that are being clustered
            n_devs = self.tri.n_dev

            upper_bound = int(np.ceil(n_devs / 2))+1
            
            # test n_clusters from 2 to 

            self.clustering_grid = {
                "n_clusters": list(range(2, upper_bound))
            } 
        else:
            self.clustering_grid = clustering_grid

        # depending on the self.model_type, set the grid
        if self.model_type == "tweedie":
            self.grid = self.tweedie_grid
        elif self.model_type == "loglinear":
            self.grid = self.loglinear_grid
        elif self.model_type == "randomforest":
            self.grid = self.randomforest_grid
        elif self.model_type == "xgboost":
            self.grid = self.xgboost_grid
        else:
            raise ValueError(
                f"model_type must be one of 'tweedie', 'loglinear', 'randomforest', or 'xgboost'."
            )

        # if **kwargs are provided, use them to update the grid
        if "alpha" in kwargs:
            self.tweedie_grid["alpha"] = kwargs["alpha"]
            self.loglinear_grid["alpha"] = kwargs["alpha"]
            self.xgboost_grid["alpha"] = kwargs["alpha"]
        if "power" in kwargs:
            self.tweedie_grid["power"] = kwargs["power"]
        if "l1ratio" in kwargs:
            self.loglinear_grid["l1ratio"] = kwargs["l1ratio"]
        if "max_iter" in kwargs:
            self.tweedie_grid["max_iter"] = kwargs["max_iter"]
            self.loglinear_grid["max_iter"] = kwargs["max_iter"]
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
        if "n_clusters" in kwargs:
            self.clustering_grid["n_clusters"] = kwargs["n_clusters"]

        # set tie criterion if there is more than one
        # pareto optimal model
        self.tie_criterion = tie_criterion

    def __repr__(self):
        reg_str = f"TriangleHyperparameterTuner(n_splits={self.n_splits_}, \
model_type={self.model_type})"
        cluster_str = f"ClusterGroupTuner(n_splits={self.n_splits_}, \
model_type={self.model_type})"
        if self.regression_hyperparameters:
            return reg_str
        elif self.clustering_hyperparameters:
            return cluster_str
        else:
            raise ValueError(
                "Either regression_hyperparameters or clustering_hyperparameters must be True."
            )

    def GetSplit(self):
        """Yields the indices for the training and validation sets."""
        X_id = self.tri.get_X_id().reset_index(drop=True)

        # current calendar period
        current_cal = self.tri.getCurCalendarPeriod()

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

    def SetParameterGrid(self, model_type="tweedie", **kwargs):
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
        l1ratio : array-like, default=None
        max_iter : int, default=None
        model_type : str, default='tweedie'
            The model type to use. Currently can be 'tweedie' or 'loglinear'.
        """
        # read named hyperparameters from **kwargs (if included)
        if model_type in ["tweedie", "loglinear"]:
            if "alpha" in kwargs:
                self.grid["alpha"] = kwargs["alpha"]
            if "max_iter" in kwargs:
                self.grid["max_iter"] = kwargs["max_iter"]
        if model_type in ["tweedie"]:
            if "power" in kwargs:
                self.grid["power"] = kwargs["power"]
        if model_type in ["loglinear"]:
            if "l1ratio" in kwargs:
                self.grid["l1ratio"] = kwargs["l1ratio"]
        if "n_clusters" in kwargs:
            self.grid["n_clusters"] = kwargs["n_clusters"]

    def RunCrossValidation(self):
        # Initialize storage for results
        self.tuning_results = []
        self.tuning_years = []
        self.tuning_param = []
        self.tuning_mse = []
        self.tuning_mae = []

        # extra metrics for clustering
        if self.clustering_hyperparameters:
            self.tuning_silhouette = []
            self.tuning_calinski_harabasz = []
            
        # Perform cross-validation
        for train_indices, val_indices in self.GetSplit():
            # Extract training and validation data
            X_train = self.tri.get_X_base().iloc[train_indices]
            if self.log_transform:
                y_train = np.log(self.tri.get_y_base()[train_indices])
            else:
                y_train = self.tri.get_y_base()[train_indices]

            X_val = self.tri.get_X_base().iloc[val_indices]
            if self.log_transform:
                y_val = np.log(self.tri.get_y_base()[val_indices])
            else:
                y_val = self.tri.get_y_base()[val_indices]

            excluded_cal = (
                self.tri.get_X_id().iloc[train_indices].calendar_period.max() + 1
            )

            # Iterate over the grid of parameters
            for params in tqdm(
                ParameterGrid(self.grid),
                desc=f"Tuning on {excluded_cal} and earlier",
            ):
                mse_values = []


                if self.regression_hyperparameters:
                    # Create and fit the model
                    if self.model_type == "tweedie":
                        model = TweedieRegressor(**params)
                    elif self.model_type == "loglinear":
                        # if alpha is 0, then it is linear regression
                        if params["alpha"] == 0:
                            # only return the first element of the list
                            # the rest will be identical in this case
                            if params["l1ratio"] == 0:
                                model = LinearRegression()
                            else:
                                continue
                        else:
                            # if l1ratio is 1, then it is lasso
                            if params["l1ratio"] == 1:
                                model = Lasso(alpha=params["alpha"])
                            # if l1ratio is 0, then it is ridge
                            elif params["l1ratio"] == 0:
                                model = Ridge(alpha=params["alpha"])
                            # if l1ratio is between 0 and 1, then it is elastic net
                            else:
                                model = ElasticNet(
                                    alpha=params["alpha"], l1_ratio=params["l1ratio"]
                                )
                elif self.clustering_hyperparameters:
                    model = AgglomerativeClustering(n_clusters=params["n_clusters"],
                                                    linkage="ward",
                                                    affinity="euclidean",
                                                    memory="./cache")


                # no matter which one, fit the model
                model.fit(X_train, y_train)

                # Compute the predictions and MSE for the validation set
                y_val_pred = model.predict(X_val)
                mse_values = mean_squared_error(y_val, y_val_pred)
                mae_values = mean_absolute_error(y_val, y_val_pred)

                # Store the results
                self.tuning_years.append(excluded_cal)
                self.tuning_param.append(params)
                self.tuning_mse.append(mse_values)
                self.tuning_mae.append(mae_values)

        # split out parameters
        tuning_parameters = {}
        for p in self.tuning_param:
            for k, v in p.items():
                if k not in tuning_parameters:
                    tuning_parameters[k] = []
                tuning_parameters[k].append(v)

        # Create a dataframe with the results
        self.tuning_results = pd.DataFrame(tuning_parameters).round(2)
        param_cols = self.tuning_results.columns.tolist()
        self.tuning_results["tuning_years"] = self.tuning_years
        self.tuning_results["tuning_mse"] = self.tuning_mse
        self.tuning_results["tuning_mae"] = self.tuning_mae

        # group by the param_cols, and get the mean and sd of mse, mae, d2
        self.tuning_results = (
            self.tuning_results.groupby(param_cols).agg(
                {
                    "tuning_mse": ["mean", "std"],
                    "tuning_mae": ["mean", "std"],
                }
            )
            # reset the index
            .reset_index()
            # flatten the column names
            .pipe(lambda x: x.set_axis(["_".join(col) for col in x.columns], axis=1))
            # sort by mean mse then std mse
            .sort_values(by=["tuning_mse_mean", "tuning_mse_std"], ascending=True)
        )

        self.has_tuning_results = True

    def CalculateParameterParetoFront(self, measures=None):
        """
        Get the Pareto front from the results of RunCrossValidation.
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
            pass measures={'mse': 'min'}. If None, the default is to
            minimize the mean squared error and minimize the standard deviation of the
            mean squared error. This is the recommended setting, and is meant to balance
            model performance and stability.

        Returns:
        -------
        A pandas DataFrame containing the Pareto front.
        """

        # Ensure RunCrossValidation has been called
        if not hasattr(self, "tuning_results"):
            self.RunCrossValidation()

        # If no measures are passed, use the default
        if measures is None:
            measures = {"tuning_mse_mean": "min", "tuning_mse_std": "min"}

        # Create a copy of the results dataframe
        results = self.tuning_results.copy()

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
        self.pareto_optimal_parameters = results[is_pareto_optimal]
        return self.pareto_optimal_parameters

    def OptimalParameters(self, measures=None, tie_criterion=None):
        """
        Returns the optimal model, which is defined to be the Pareto-optimal
        model, optimized on selected optimization criteria. If more than one model is
        Pareto-optimal, the model with the lowest mean squared error is returned.

        If the CalculateParameterParetoFront method has not been called, we will run it
        using the measures passed to this method.

        This method is intended to be the only method that the user needs to call
        to get the optimal Tweedie model.
        """
        # Ensure CalculateParameterParetoFront has been called
        if not hasattr(self, "pareto_optimal_parameters"):
            self.CalculateParameterParetoFront(measures=measures)

        if tie_criterion is None:
            tie_criterion = self.tie_criterion

        # if there is more than one optimal model, return the one with the lowest MSE
        if self.pareto_optimal_parameters.shape[0] > 1:
            optimal_model = (self.pareto_optimal_parameters
                             .sort_values(tie_criterion)
                             .iloc[0])
        else:
            optimal_model = self.pareto_optimal_parameters.iloc[0]

        # get the optimal set of hyperparameters
        if self.model_type == "tweedie":
            alpha = optimal_model["alpha_"]
            power = optimal_model["power_"]
        elif self.model_type == "loglinear":
            alpha = optimal_model["alpha_"]
            l1ratio = optimal_model["l1ratio_"]

        # Re-fit a model with the optimal hyperparameters, and return it
        if self.model_type == "tweedie":
            best_model = TweedieRegressor(alpha=alpha, power=power, link="log")
        elif self.model_type == "loglinear":
            best_model = ElasticNet(alpha=alpha, l1_ratio=l1ratio)

        if self.log_transform:
            y = np.log(self.tri.get_y_base("train"))
        else:
            y = self.tri.get_y_base("train")
        best_model.fit(self.tri.get_X_base("train"), y)
        return best_model

    def _fit_ward_clusters(self, X, n_clusters):
        # fit the ward clustering algorithm
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", affinity="euclidean"
        )
        
        # get the cluster labels
        cluster_labels = clusterer.fit_predict(X)

        # get the silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)

        # get the calinski harabasz score
        calinski_harabasz_avg = calinski_harabasz_score(X, cluster_labels)