import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Any

@dataclass
class Scenario:
    id: str = None
    start: np.datetime64 = None
    end: np.datetime64 = None
    trend: float = None

    # when calling a particular `Scenario`, it will return the id
    def __repr__(self) -> str:
        return f"Scenario({self.id})"
    def __str__(self) -> str:
        return f"Scenario({self.id})"  

@dataclass
class ForecastScenarios:
    default: Scenario

    def add(self,
            id: str = "default",
            start: np.datetime64 = None,
            end: np.datetime64 = None,
            trend: float = 0.0
            ) -> None:
        """
        Add a forecast scenario to the model.
        """
        # check that the forecast scenario is not already in the model scenarios
        if id in self.__dict__.keys():
            raise ValueError('The forecast scenario is already in the model.')
        
        # add the forecast scenario to the model
        setattr(self, id, Scenario(id, start, end, trend))

@dataclass
class Forecast:
    # def __init__(self) -> None:
    """
    Forecast class for completing the loss triangle. This is an abstract class
    that is an attribute of the main ROCKY3 class. It is a set of assumptions that
    are used to complete the loss triangle. 
    
    Starts with no calendar period trend, but allows for trend scenarios to be
    added.
    """
    trend_scenarios: object = ForecastScenarios(Scenario('default', None, None, 0))
    scenario_loaded: bool = False
    
    def add(self,
            id: str = "default",
            start: np.datetime64 = None,
            end: np.datetime64 = None,
            trend: float = 0.0
            ) -> None:
        """
        Add a trend scenario to the forecast.

        Parameters:
        -----------
        id: str
            The id of the trend scenario.
        start: np.datetime64
            The start date of the trend scenario.
        end: np.datetime64
            The end date of the trend scenario.
        trend: float, default=0.0
            The trend of the trend scenario.

        Returns:
        --------
        `None`
        """
        # check that the trend scenario is not already in the model
        if id in self.trend_scenarios.__dict__.keys():
            raise ValueError('The trend scenario is already in the model.')
    
        # add the trend scenario to the dictionary
        fcst = Scenario(id, start, end, trend)
        self.trend_scenarios.add(fcst.id, fcst.start, fcst.end, fcst.trend)

        # set the scenario_loaded flag to True
        self.scenario_loaded = True

        

    

### NOT IMPLEMENTED YET ### 
    def predict_interval(self,
                         X_future : np.ndarray,
                         quantiles : tuple = (0.025, 0.975)
                         ):
        """
        Predict future loss intervals using the fitted model.

        Parameters:
        -----------
        X_future: array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples for future periods.
        quantiles: tuple of float, default=(0.025, 0.975)
            The quantiles for the prediction interval.

        Returns:
        --------
        interval: tuple of arrays of shape (n_samples,)
            The lower and upper bounds of the prediction interval.
        """
        
        raise NotImplementedError("Prediction intervals are not implemented for this estimator.")

    def complete_triangle(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          X_future: np.ndarray,
                          return_triangle: bool = False
                          ) -> np.ndarray:
        """
        Complete the loss triangle by combining the known losses and predicted future losses.

        Parameters:
        -----------
        X: array-like or pd.DataFrame of shape (n_samples_known, n_features)
            The input samples for known periods.
        y: array-like or pd.Series of shape (n_samples_known,)
            The known losses.
        X_future: array-like or pd.DataFrame of shape (n_samples_future, n_features)
            The input samples for future periods.
        return_triangle: bool, default=False
            Whether to return the completed loss triangle in a triangle format
            (with origin periods as rows, development periods as columns, amounts
            as values), or as a flat array (with origin, development, and amount
            columns)

        Returns:
        --------
        loss_triangle: array of shape (n_samples_known + n_samples_future,)
            The completed loss triangle.
        """
        y_future = self.predict(X_future)
        if return_triangle:
            origin = np.concatenate([np.arange(1, len(y) + 1), np.arange(1, len(y_future) + 1)])
            development = np.concatenate([np.arange(1, len(y) + 1), np.arange(1, len(y_future) + 1)])
            amount = np.concatenate([y, y_future])
            return np.array([origin, development, amount]).T
        else:
            return np.concatenate([X, X_future, np.concatenate([y, y_future])])

