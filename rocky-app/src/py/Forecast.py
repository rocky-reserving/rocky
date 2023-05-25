try:
    from .triangle import Triangle
except ImportError:
    from triangle import Triangle

from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class Forecast:
    tri: Triangle = None
    cal_periods: pd.Series = None

    # generator function to generate forecasted parameter values
    def forecast_parameters(self, gen: dict = None):
        """
        Generator function to generate forecasted parameter values.

        Parameters
        ----------
        gen : dict, optional
            Dictionary of parameters to generate forecasts for. The default is None.
        """
