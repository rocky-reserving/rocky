try:
    from .triangle import Triangle
except:
    from triangle import Triangle

import pandas as pd
import numpy as np


class TriangleTimeSeriesSplit:
    def __init__(self, triangle: Triangle = None, n_splits: int = 5):
        self.tri = triangle
        self.n_splits_ = n_splits
        self.split = []

    def GetSplit(self):
        X = self.tri.get_X_base().reset_index(drop=True)
        y = self.tri.get_y_base().reset_index(drop=True)
        X_id = self.tri.get_X_id().reset_index(drop=True)

        # current calendar period
        current_cal = self.tri.getCurCalendarIndex()

        for i in range(1, self.n_splits_ + 1):
            # get the calendar period for the current split
            split_cal = current_cal - i

            # get the indices for training and validation set
            train_indices = X_id.cal[X_id.cal.lt(split_cal)].index.to_numpy()
            test_indices = X_id.cal[
                X_id.cal.ge(split_cal) & X_id.cal.le(current_cal)
            ].index.to_numpy()

            yield train_indices, test_indices
