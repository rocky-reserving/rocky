"""
Implements the ROCKYPlot class. This class calls the plotting routines in
a ROCKY3 model. It is used by the ROCKY3 class, and is not intended to be
used directly by the user.
"""

import pandas as pd
import plotly.express as px
from forecast import Forecast

class ROCKYPlot(Forecast):
    """
    This class calls the plotting routines in a ROCKY3 model. It is used by
    the ROCKY3 class, and is not intended to be used directly by the user.
    """
    def __init__(self) -> None:
        super().__init__()

    def residual_plot(self,
                      plotby: str = 'alpha',
                      **kwargs
                      ) -> px.scatter:
        """
        Plots the standardized 

        Parameters
        ----------
        plotby : str, optional
            The variable to plot by. The default is 'month'.

        Returns
        -------
        None.

        """
        # df =
        pass