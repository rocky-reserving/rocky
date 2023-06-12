# GLM model
try:
    # import .GLM
    from .GLM import glm
except ImportError:
    import GLM
    from GLM import glm

try:
    from .TriangleTimeSeriesSplit import TriangleTimeSeriesSplit
except ImportError:
    from TriangleTimeSeriesSplit import TriangleTimeSeriesSplit

import itertools
import pandas as pd


class ModelInteractions:
    """
    Performs a feature importance analysis on the model based on fitting
    a random forest model to the data before and after adding all interactions
    of order `interactions_order` to the data.
    """

    def __init__(self, model: glm, interactions_order: int, **kwargs):
        self.glm = model
        self.tri = model.tri

        self.interactions_order = interactions_order
        self.interactions = None

        assert self.interactions_order in [
            1,
            2,
            3,
        ], "interactions_order must be 1, 2, or 3"

        # get the three types of parameters
        self.acc = self.tri.get_X(column_query="acc", split="train").set_index(
            "accident_period"
        )
        self.dev = self.tri.get_X(column_query="dev", split="train").set_index(
            "development_period"
        )
        self.cal = self.tri.get_X(
            column_query="cal", use_cal=True, split="train"
        ).set_index("cal")

    def list_interactions(self):
        """
        Returns a list of all interactions of order `interactions_order` from
        parameters in `acc`, `dev`, and `cal`.
        Each type of parameter may not interact with itself, but may interact
        with other types of parameters. Eg `acc` may interact with any of the
        `dev` or `cal` parameters, but not with any other `acc` parameters.
        """

        if self.interactions_order == 1:
            return []
        elif self.interactions_order == 2:
            return self._list_interactions_order_2()
        elif self.interactions_order == 3:
            return self._list_interactions_order_3() + self._list_interactions_order_2()

    def _list_interactions_order_2(self):
        interactions = []
        # get all interactions of order `interactions_order` between `acc` and `dev`
        interactions += list(itertools.product(self.acc, self.dev))
        # get all interactions of order `interactions_order` between `acc` and `cal`
        interactions += list(itertools.product(self.acc, self.cal))
        # get all interactions of order `interactions_order` between `dev` and `cal`
        interactions += list(itertools.product(self.dev, self.cal))
        return interactions

    def _list_interactions_order_3(self):
        interactions = []
        # get all interactions of order `interactions_order` between `acc` and `dev`
        interactions += list(itertools.product(self.acc, self.dev, self.cal))
        return interactions

    def _get_X(self):
        """
        Returns the X matrix for the given interactions.
        """

        # get the X matrix for the given interactions
        X = (
            self.tri.get_X(use_cal=True, split="train")
            .set_index(["accident_period", "development_period", "cal"])
            .fillna(0)
        )

        # get the list of interactions
        interactions = self.list_interactions()

        inter_list = []

        for a in self.acc:
            inter_list.append(pd.DataFrame({f"{a}": X[a]}))

        for d in self.dev:
            inter_list.append(pd.DataFrame({f"{d}": X[d]}))

        if self.glm.use_cal:
            for c in self.cal:
                inter_list.append(pd.DataFrame({f"{c}": X[c]}))

        # add the interactions to the X matrix
        for interaction in interactions:
            title = (
                " * ".join(interaction)
                .replace("accident_period", "acc")
                .replace("development_period", "dev")
                .replace("calendar_period", "cal")
            )
            inter_list.append(
                pd.DataFrame({f"{title}": X[interaction[0]] * X[interaction[1]]})
            )

        # combine the interactions with the X matrix
        interactions = pd.concat(inter_list, axis=1)
        self.interactions = interactions

        return interactions

    def tune_hyperparameters(self, **kwargs):
        """
        Tune the hyperparameters of the model.
        """

        # get the design matrix, before and after adding interactions
        dm_before = self.tri.get_X(use_cal=True, split="train")
        dm_after = self._get_X()

        # set up the cross validation
        ts = TriangleTimeSeriesSplit(triangle=self.tri, n_splits=5)
