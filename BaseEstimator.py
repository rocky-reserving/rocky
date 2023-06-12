# import pandas as pd
# import numpy as np

# import plotly.graph_objects as go

# # from plot import residual_plot

# from dataclasses import dataclass

# @dataclass 
# class Plot:
#     width : int = 1000
#     height : int = 800
#     margin : dict = None

#     def __post_init__(self) -> None:
#         if self.margin is None:
#             self.margin = dict(l=40, r=40, t=60, b=40)


# class BaseEstimator:
#     def __init__(self
#                  , width : int = 1000
#                  , height : int = 800
#                  , margin : dict = None
#                  ) -> None:
        
#         self.plot = Plot(width, height, margin)

#     # these are the methods that must be implemented by any estimator:
#     def Predict(self, X) -> pd.Series:
#         """
#         Takes a matrix of features and returns a vector of predictions.

#         Every estimator needs to be able to call this method in order for the residuals
#         to be calculated, for example.
#         """
#         raise NotImplementedError # pragma: no cover
    
#     def GetX(self) -> pd.DataFrame:
#         """
#         Returns the matrix of features used to train the estimator.
#         """
#         raise NotImplementedError # pragma: no cover
    
#     def GetY(self) -> pd.Series:
#         """
#         Returns the vector of targets used to train the estimator.
#         """
#         raise NotImplementedError # pragma: no cover
    
#     def HatMatrix(self) -> pd.DataFrame:
#         """
#         Returns a matrix of hat values.
#         """
#         raise NotImplementedError
    
#     def GetResiduals(self) -> pd.Series:
#         """
#         Returns a vector of residuals.
#         """
#         raise self.GetY() - self.Predict(self.GetX())
    

#     def PearsonResiduals(self) -> pd.Series:
#         """
#         Returns a vector of Pearson residuals.
#         """
#         return self.GetResiduals() / np.sqrt(self.Predict())
    

    
