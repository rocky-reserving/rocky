import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


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
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_forecast = X_forecast
        self.X_id = X_id
        self.acc = acc
        self.dev = dev
        self.cal = cal

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
            hover_data = ["acc", "dev", "cal", "y", "yhat"]

        obs_vs_pred = pd.DataFrame(
            {
                "y": y if not log else np.log(y),
                "yhat": yhat if not log else np.log(yhat),
                "acc": self.acc,
                "dev": self.dev,
                "cal": self.cal,
            }
        )

        obs_vs_pred["color"] = obs_vs_pred[color]

        # Recode the color variable to be categorical if it's numeric
        if pd.api.types.is_numeric_dtype(obs_vs_pred["color"]):
            obs_vs_pred[color] = obs_vs_pred[color].astype(str)

        # Create the scatter plot
        fig = px.scatter(
            obs_vs_pred,
            x="y",
            y="yhat",
            color=color,
            opacity=opacity,
            hover_data=hover_data,
            title=f"{title}{'' if not log else ' (log scale)'}",
            labels={"y": "Observed", "yhat": "Predicted"},
        )

        # Add a single trend line
        fig.add_trace(
            px.scatter(
                obs_vs_pred,
                x="y",
                y="yhat",
                color=color,
                hover_data=hover_data,
                opacity=opacity,
                title=f"{title}{'' if not log else ' (log scale)'}",
                labels={"y": "Observed", "yhat": "Predicted"},
            ).data[0]
        )

        # Add a 45-degree black dashed line
        fig.add_shape(
            type="line",
            x0=obs_vs_pred["y"].min(),
            y0=obs_vs_pred["yhat"].min(),
            x1=obs_vs_pred["y"].max(),
            y1=obs_vs_pred["yhat"].max(),
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
        y_axis_title="resid",
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
                x=plot_by,
                y="std_resid",
                color="std_resid",
                hover_data=["y", "yhat", "acc", "dev", "cal"],
                title=plot_title,
                labels={"yhat": "Predicted", "resid": "Residual"},
            )

            # Add a horizontal black dashed line
            fig.add_shape(
                type="line",
                x0=df["acc"].min(),
                y0=0,
                x1=df["acc"].max(),
                y1=0,
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
            )

        fig.show()
