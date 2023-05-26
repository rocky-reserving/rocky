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

    def ResidualPlot(
        self,
        residuals,
        plot_by,
        boxpoints="all",
        jitter=0.5,
        pointpos=0,
        scatter_mode="markers",
        scatterpoint_outline_color=(0, 0, 0, 0),
        scatterpoint_outline_width=1,
        scatterpoint_fill_color=(0, 0, 0, 0.5),
        y_axis_title="resid",
        x_axis_title="",
    ):
        outline_color = "rgba" + str(scatterpoint_outline_color)
        fill_color = "rgba" + str(scatterpoint_fill_color)

        fig = go.Figure()

        by = plot_by.name
        df = pd.DataFrame({"resid": residuals})
        df[by] = plot_by

        categories = df[by].unique()

        for category in categories:
            df_category = df[df[by] == category]
            # Invisible box plot just for jitter effect
            fig.add_trace(
                go.Box(
                    x=df_category[by],
                    y=df_category["resid"],
                    name=category,
                    marker=dict(
                        color="rgba(0,0,0,0)",  # Invisible
                    ),
                    boxpoints=boxpoints,  # To show all points
                    jitter=jitter,  # Add jitter for better visibility
                    pointpos=pointpos,  # Position of points relative to box
                )
            )

            # Actual scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df_category[by],
                    y=df_category["resid"],
                    mode=scatter_mode,
                    name=category,
                    marker=dict(
                        color=fill_color,  # Partially transparent fill
                        line=dict(
                            width=scatterpoint_outline_width, color=outline_color
                        ),  # Black outline
                    ),
                    hoverinfo="skip",  # No hover info for this trace
                )
            )

        fig.update_layout(
            xaxis=dict(type="category", title=x_axis_title),
            yaxis=dict(title=y_axis_title),
            boxmode="group",  # Group together boxes for the same x location
        )

        fig.show()
