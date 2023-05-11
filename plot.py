"""
This module contains the Plot class, which is used to plot the results of
a rocky model. It is a wrapper around the plotly library, and is used to
produce interactive plots of the model results.
"""

from rockycore import ROCKY

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import mean_squared_error

from typing import Any, Dict, List, Optional, Tuple, Union

##### CFC COLORS
light_cfc = "#337ab7"
dark_cfc = "#2a6496"

dark_cfc2 = "rgb(0, 77, 113)"

custom_theme = {
    "title_font": {"family": "Arial", "size": 18},
    "axis_title_font": {"family": "Arial", "size": 14},
    "tick_font": {"family": "Arial", "size": 12},
    "bg_color": "rgba(240, 240, 240, 1)",
    "axis_color": "black",
}


def _plot_standardized_pearson_residuals_vs_fitted(
    df: pd.DataFrame,
    model: Any,
    title: str = "Standardized Pearson Residuals vs Fitted Values",
    return_: bool = False,
) -> None:
    """
    Plots the standardized Pearson residuals versus the fitted values for a given dataset and model.

    Args:
    - df: Pandas DataFrame containing the input and output data (y and y_pred).
    - model: Trained regression model object with a 'predict' method that can make predictions on input data.
    - title: String representing the title of the plot (default is 'Standardized Pearson Residuals vs Fitted Values').

    Returns:
    - None. Displays a scatter plot of the standardized Pearson residuals versus the fitted values.
    """

    # calculate standardized pearson residuals
    # df['residuals'] = (df['y'] - df['y_pred']) / np.sqrt(mean_squared_error(df['y'], df['y_pred']))

    # create message for hover template
    hover = "<b>Fitted Values:</b> %{x}<br>"
    hover += "<b>Standardized Pearson Residuals:</b> %{customdata[3]}<br>"
    hover += "<b>AY:</b> %{customdata[0]}<br>"
    hover += "<b>DY:</b> %{customdata[1]}<br>"
    hover += "<b>CY:</b> %{customdata[2]}<br>"
    hover += "<extra></extra>"

    # add a scatter plot of the standardized residuals vs the fitted values, and a horizontal line at 0
    fig = go.Figure(
        go.Scatter(
            x=df["y_pred"],
            y=df["residuals"],
            mode="markers",
            customdata=(
                df[["ay", "dev", "cal", "residuals"]].assign(
                    residuals=df.residuals.round(3)
                )
            ).values,
            marker=dict(
                color="red",
                colorscale="RdBu",
                line=dict(color="black", width=1),
                opacity=0.5,
            ),
            hovertemplate=hover,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Fitted Values",
        yaxis_title="Standardized Pearson Residuals",
        coloraxis_showscale=False,
    )

    # add a horizontal line at 0
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=df["y_pred"].max(),
        y1=0,
        line=dict(
            color="black",
            width=3,
            dash="dashdot",
        ),
    )

    # show the plot
    if return_:
        return {"trace": fig.data[0], "max_y_pred": df["y_pred"].max()}
    else:
        fig.show()


def _plot_standardized_residuals_by_column(
    df, model, x_col, title=None, return_=False
) -> None:
    if title is None:
        title = f"Standardized Pearson Residuals by {x_col}"

    # Group by x_col and calculate the mean and standard deviation of residuals for each group
    grouped = df.groupby(x_col)["residuals"].agg(["mean", "std"]).reset_index()

    fig = go.Figure()

    # Scatter plot of residuals
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df["residuals"],
            mode="markers",
            customdata=df[["ay", "dev", "cal"]].values,
            marker=dict(
                color=df["y"],
                colorscale="RdBu",
                line=dict(color="black", width=1),
                opacity=0.5,
            ),
            hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>Standardized Pearson Residuals:</b> %{{y}}<br><b>AY:</b> %{{customdata[0]}}<br><b>Dev:</b> %{{customdata[1]}}<br><b>Cal:</b> %{{customdata[2]}}<extra></extra>",
            showlegend=False,
        )
    )

    # Plot mean residuals
    fig.add_trace(
        go.Scatter(
            x=grouped[x_col],
            y=grouped["mean"],
            mode="lines",
            line=dict(color="darkgray", width=2),
            showlegend=False,
        )
    )

    # Plot standard deviation bounds and shading
    for level, fillcolor in zip(
        [1, -1], ["rgba(128, 128, 128, 0.1)", "rgba(128, 128, 128, 0.1)"]
    ):
        fig.add_trace(
            go.Scatter(
                x=grouped[x_col],
                y=grouped["mean"] + level * grouped["std"],
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                fill="tonexty",
                fillcolor=fillcolor,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title="Standardized Pearson Residuals",
        coloraxis_showscale=False,
    )

    if return_:
        return {"trace": fig.data[0], "traces_std_bounds": fig.data[1:]}
    else:
        fig.show()


def residual_plot(df, model, column=None, width=None, height=None, margin=None) -> None:
    if width is None:
        width = 1000
    if height is None:
        height = 800
    if margin is None:
        margin = dict(l=40, r=40, t=60, b=40)

    if column is None:
        # Create a 2x2 subplot
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Standardized Pearson Residuals by Dev",
                "Standardized Pearson Residuals by AY",
                "Standardized Pearson Residuals by Cal",
                "Standardized Pearson Residuals vs Fitted Values",
            ),
        )

        dev_plot = _plot_standardized_residuals_by_column(
            df, model, "dev", return_=True
        )
        ay_plot = _plot_standardized_residuals_by_column(df, model, "ay", return_=True)
        cal_plot = _plot_standardized_residuals_by_column(
            df, model, "cal", return_=True
        )
        fitted_plot = _plot_standardized_pearson_residuals_vs_fitted(
            df, model, return_=True
        )

        fig.add_trace(dev_plot["trace"], row=1, col=1)
        fig.add_trace(ay_plot["trace"], row=1, col=2)
        fig.add_trace(cal_plot["trace"], row=2, col=1)
        fig.add_trace(fitted_plot["trace"], row=2, col=2)

        # Add y=0 line for the fitted plot
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=fitted_plot["max_y_pred"],
            y1=0,
            line=dict(color="black", width=3, dash="dashdot"),
            row=2,
            col=2,
        )

        # Add shaded areas for heteroscedasticity checks
        for trace in dev_plot["traces_std_bounds"]:
            fig.add_trace(trace, row=1, col=1)
        for trace in ay_plot["traces_std_bounds"]:
            fig.add_trace(trace, row=1, col=2)
        for trace in cal_plot["traces_std_bounds"]:
            fig.add_trace(trace, row=2, col=1)

        # Update subplot layout
        fig.update_layout(showlegend=False, margin=margin)

        if width is not None and height is not None:
            fig.update_layout(width=width, height=height)

        fig.show()

    else:
        if column.lower() == "fitted":
            _plot_standardized_pearson_residuals_vs_fitted(df, model)

        else:
            _plot_standardized_residuals_by_column(df, model, column)
