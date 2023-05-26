# import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# import numpy as np


class Plot:
    def __init__(self, X_train=None, y_train=None, X_forecast=None, X_id=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_forecast = X_forecast
        self.X_id = X_id

    def SetXTrain(self, X):
        self.X_train = X

    def SetYTrain(self, y):
        self.y_train = y

    def SetXForecast(self, X):
        self.X_forecast = X

    def SetXLabels(self, X):
        self.X_id = X

    def residual(
        self,
        residuals,
        plot_by,
        scatterpoint_outline_color=(0, 0, 0),
        scatterpoint_outline_width=1,
        scatterpoint_fill_color=(0, 0, 0, 0.5),
        plot_title=None,
        y_axis_title="resid",
        x_axis_title="",
    ):
        outline_color = "rgba" + str(scatterpoint_outline_color)
        fill_color = "rgba" + str(scatterpoint_fill_color)
        point_marker = dict(
            color=fill_color,
            line=dict(width=scatterpoint_outline_width, color=outline_color),
            size=6,
            opacity=0.5,
        )

        if plot_title is None:
            plot_title = "Pearson Residuals vs. " + plot_by.name

        df = pd.DataFrame({plot_by.name: plot_by, "resid": residuals})

        fig = go.Figure()

        for category, df_category in df.groupby(plot_by.name):
            fig.add_trace(
                go.Violin(
                    x=df_category[plot_by.name],
                    y=df_category["resid"],
                    name=category,
                    points="all",
                    pointpos=0,
                    jitter=0.05,
                    marker=point_marker,
                    box_visible=False,
                    line_color="black",
                    meanline_visible=False,
                )
            )

        fig.update_layout(
            xaxis=dict(type="category", title=x_axis_title),
            yaxis=dict(title=y_axis_title),
        )

        fig.show()
