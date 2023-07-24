import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# from _util.BaseEstimator import BaseEstimator

# column name mappings
column_name_map = {
    "acc": "Accident Period",
    "accident_period": "Accident Period",
    "ay": "Accident Period",
    "dev": "Development Period",
    "development_period": "Development Period",
    "cal": "Calendar Period",
    "cy": "Calendar Period",
    "calendar_period": "Calendar Period",
    "y": "Observed",
    "yhat": "Predicted",
    "yhat_lower": "Predicted Lower",
    "yhat_upper": "Predicted Upper",
    "resid": "Residual",
    "std_resid": "Standardized Residual",
}


def map_col(col):
    return column_name_map[col] if col in column_name_map else col


class Plot:
    def __init__(
        self,
        X_train=None,
        y_train=None,
        X_forecast=None,
        X_id=None,
        acc=None,
        dev=None,
        cal=None,
        fitted_model=None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_forecast = X_forecast
        self.X_id = X_id
        self.acc = acc
        self.dev = dev
        self.cal = cal
        self.fitted_model = fitted_model

    def SetXTrain(self, X):
        self.X_train = X

    def SetYTrain(self, y):
        self.y_train = y

    def SetXForecast(self, X):
        self.X_forecast = X

    def SetXLabels(self, X):
        self.X_id = X

    def obs_pred(
        self,
        y=None,
        yhat=None,
        color="acc",
        trendline="ols",
        opacity=0.5,
        hover_data=None,
        title="Observations vs. Predictions",
        log=False,
    ):
        if y is None:
            y = self.y_train

        if yhat is None:
            yhat = self.yhat

        if hover_data is None:
            hover_data = [map_col(i) for i in ["acc", "dev", "cal", "y", "yhat"]]

        color = map_col(color)

        obs_vs_pred = pd.DataFrame(
            {
                map_col("y"): y if not log else np.log(y),
                map_col("yhat"): yhat if not log else np.log(yhat),
            }
        )
        obs_vs_pred[map_col("acc")] = self.acc
        obs_vs_pred[map_col("dev")] = self.dev
        obs_vs_pred[map_col("cal")] = self.cal

        obs_vs_pred["color"] = obs_vs_pred[color]

        # Recode the color variable to be categorical if it's numeric
        if pd.api.types.is_numeric_dtype(obs_vs_pred["color"]):
            obs_vs_pred[color] = obs_vs_pred[color].astype(str)

        # Create the scatter plot
        fig = px.scatter(
            obs_vs_pred.rename(columns=column_name_map),
            x=map_col("y"),
            y=map_col("yhat"),
            color=color,
            opacity=opacity,
            hover_data=hover_data,
            title=f"{title}{'' if not log else ' (log scale)'}",
            # labels={"y": "Observed", "yhat": "Predicted"},
        )

        # Add a single trend line
        fig.add_trace(
            px.scatter(
                obs_vs_pred.rename(columns=column_name_map),
                x=map_col("y"),
                y=map_col("yhat"),
                color=color,
                hover_data=hover_data,
                opacity=opacity,
                title=f"{title}{'' if not log else ' (log scale)'}",
                # labels={"y": "Observed", "yhat": "Predicted"},
            ).data[0]
        )

        # Add a 45-degree black dashed line
        fig.add_shape(
            type="line",
            x0=obs_vs_pred[map_col("y")].min(),
            y0=obs_vs_pred[map_col("yhat")].min(),
            x1=obs_vs_pred[map_col("y")].max(),
            y1=obs_vs_pred[map_col("yhat")].max(),
            line=dict(color="black", dash="dash"),
        )

        # Update marker properties to include thin black outlines
        fig.update_traces(marker=dict(line=dict(color="black", width=0.75)))

        # Show the plot
        fig.show()

    def residual(
        self,
        plot_by=None,
        scatterpoint_outline_color=(0, 0, 0),
        scatterpoint_outline_width=1,
        scatterpoint_fill_color=(0, 0, 0, 0.5),
        plot_title=None,
        y_axis_title="Standardized Residuals",
        x_axis_title="",
        log=False,
    ):
        if plot_by is None:
            plot_by = "yhat"

        df = pd.DataFrame({"y": self.y_train, "yhat": self.yhat})
        df["y"] = np.log(df["y"]) if log else df["y"]
        df["yhat"] = np.log(df["yhat"]) if log else df["yhat"]
        df["resid"] = df["y"] - df["yhat"]
        df["std_resid"] = (df["resid"] - df["resid"].mean()) / df["resid"].std()
        df["acc"] = self.acc
        df["dev"] = self.dev
        df["cal"] = self.cal
        df[plot_by] = getattr(self, plot_by)

        outline_color = "rgba" + str(scatterpoint_outline_color)
        fill_color = "rgba" + str(scatterpoint_fill_color)
        point_marker = dict(
            color=fill_color,
            line=dict(width=scatterpoint_outline_width, color=outline_color),
            size=6,
            opacity=0.5,
        )

        if plot_title is None:
            plot_title = (
                "Pearson Residuals vs. " + plot_by + " (log scale)"
                if log
                else "Pearson Residuals vs. " + plot_by
            )

        if plot_by == "yhat":
            fig = px.scatter(
                df,
                x=map_col("plot_by"),
                y=map_col("Standardized Residuals"),
                color=map_col("Standardized Residuals"),
                hover_data=[
                    "Observed",
                    "Predicted",
                    "Accident Period",
                    "Development Period",
                    "Calendar Period",
                ],
                title=plot_title,
                labels={"yhat": "Predicted", "resid": "Residual"},
            )

            # Add a horizontal black dashed line
            fig.add_shape(
                type="line",
                x0=df["acc"].min(),
                # x0=df["acc"].min(),
                y0=0,
                x1=df["acc"].max(),
                y1=0,
                title=plot_title,
                line=dict(color="black", dash="dash"),
            )

        else:
            fig = go.Figure()

            for category, df_category in df.groupby(plot_by):
                fig.add_trace(
                    go.Violin(
                        x=df_category[plot_by],
                        y=df_category["resid"],
                        name=category,
                        points="all",
                        pointpos=0,
                        jitter=0.05,
                        marker=point_marker,
                        box_visible=False,
                        line_color="black",
                        meanline_visible=True,
                    )
                )

            fig.update_layout(
                xaxis=dict(type="category", title=x_axis_title),
                yaxis=dict(title=y_axis_title),
                title=plot_title,
            )

        fig.show()

    def residual_qq(self, log=False):
        df = pd.DataFrame({map_col("y"): self.y_train, map_col("yhat"): self.yhat})
        df[map_col("y")] = np.log(df[map_col("y")]) if log else df[map_col("y")]
        df[map_col("yhat")] = (
            np.log(df[map_col("yhat")]) if log else df[map_col("yhat")]
        )
        df[map_col("resid")] = df[map_col("y")] - df[map_col("yhat")]
        df[map_col("std_resid")] = (
            df[map_col("resid")] - df[map_col("resid")].mean()
        ) / df[map_col("resid")].std()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                df.rename(columns=column_name_map),
                x=("resid"),
                y=map_col("y"),
                mode="markers",
                marker=dict(color="black", size=3, opacity=0.5),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[-3, 3],
                y=[-3, 3],
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
            )
        )

        fig.update_layout(
            title="Normal Q-Q Plot",
            xaxis=dict(title="Standardized Residuals"),
            yaxis=dict(title="Observed"),
        )

        fig.show()

    def model(self, variable=None):
        if variable is None:
            variable = "acc"
        params = self.fitted_model.GetParameters(column=variable)
        fig = px.line(params, x="parameter", y="cumsum")
        fig.show()
