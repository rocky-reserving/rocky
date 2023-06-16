import sys
import os

curdir = os.path.abspath(os.path.dirname("."))
sys.path.append(curdir)

from util import count_rocky
from triangle import Triangle
from GLM import glm
from LogLinear import loglinear


from dataclasses import dataclass
from typing import Any

from warnings import filterwarnings

import pandas as pd

filterwarnings("ignore", category=UserWarning, module="openpyxl")

# add all implemented model types to this list
all_models = "tweedie glm".split()


@dataclass
class rockyObj:
    id: str = None
    obj: object = None


@dataclass
class rockyContainer:
    def __repr__(self) -> str:
        # only show comma-separated list of ids if there are any objects in
        # the container
        if len(self.__dict__) > 0:
            if len(self.__dict__) > 1:
                return "(" + ", ".join([f'"{k}"' for k in self.__dict__.keys()]) + ")"
            else:
                return ", ".join([f'"{k}"' for k in self.__dict__.keys()])
        else:
            return "()"

    def add(self, _obj, id=None) -> None:
        if id is None:
            setattr(self, _obj.id, _obj)
        else:
            setattr(self, id, _obj)


@dataclass
class rocky:
    """
    A rocky object is a container for triangles, models, forecasts, and plots. A rocky
    object is also the main interface for the rocky package.
    """

    # initialize the model attribute, triangle attribute, forecast attribute,
    # validation attribute, and plotting attribute
    id: str = None
    mod: Any = rockyContainer()  # models
    f: Any = rockyContainer()  # forecasts
    plot: Any = rockyContainer()  # plots
    t: Any = rockyContainer()  # triangles
    # rockylog: Any = None  # rockylog -- not implemented yet

    def __post_init__(self) -> None:
        if self.id is None:
            rockies = count_rocky()
            self.id = f"rocky{rockies}"

    def rename(self, id: str = None, obj: str = None) -> None:
        """
        Rename the rocky object or a rocky object attribute.

        Parameters
        ----------
        id : str, optional
            The new name for the rocky object.
            Default is None.
        obj : str, optional
            The name of the rocky object attribute to rename.
            Default is None.
        """
        if obj is None:
            obj = self
        if id is None:
            id = obj.id
        setattr(self, "id", id)

    def load_mack_1994(self, id='rpt_loss') -> None:
        """
        Load the Mack 1994 Sample Incurred Loss triangles.

        A reported loss triangle is created and added to the rocky object.
        """
        tri = Triangle.from_mack_1994()
        tri.base_linear_model()

        if id is None:
            id = "rpt_loss"
        self.t.add(tri, id)
        setattr(self, f"{id}", tri)

    def load_taylor_ashe(self, id=None) -> None:
        """
        Load the Taylor-Ashe triangle data set. This is the set of triangles
        used in the Mack paper from 1994.

        A paid loss triangle is created and added to the rocky object.
        """
        tri = Triangle.from_taylor_ashe()
        tri.base_linear_model()

        if id is None:
            id = "paid_loss"
        self.t.add(tri, id)
        setattr(self, f"{id}", tri)

    def load_dahms(self) -> None:
        """
        Load the Dahms triangle data set. This is the set of triangles used in
        the Paid-Incurred Chain method paper by Merz and Wüthrich.

        A reported loss triangle and a paid loss triangle are created and added
        to the rocky object.
        """
        d = {}
        d["rpt_loss"], d["paid_loss"] = Triangle.from_dahms()
        for id in d.keys():
            d[id].base_linear_model()
            self.t.add(d[id], f"{id}")
            setattr(self, f"{id}", d[id])

    def SampleTri(self, sample: str, id: str = None) -> None:
        """
        Load a sample triangle data set.

        Parameters
        ----------
        sample : str
            The name of the sample triangle data set to load.
            Currently, the only available sample triangle data set is
            "taylor_ashe".
        id : str, optional
            The name to assign to the triangle object.
            Default is None.
        """
        if sample.lower() == "taylor_ashe":
            if id is None:
                id = "paid_loss"
            self.load_taylor_ashe(id=id)
            getattr(self, f"{id}").base_linear_model()
        elif sample.lower()=="mack_1994":
            if id is None:
                id = "rpt_loss"
            self.load_mack_1994(id=id)
            getattr(self, f"{id}").base_linear_model()

    def FromClipboard(self, id: str = "rpt_loss") -> None:
        """
        Load a triangle from the clipboard. The triangle must include the accident
        years in the first column and the development years in the first row.

        This method will not work if either the origin periods or the development
        periods take up more than one column or row. If you have a triangle that
        does not meet these requirements, please try one of the following methods:
         - `.FromCSV()`
         - `.FromExcel()`
         - `.FromDF()`

        Parameters
        ----------
        id : str, optional
            The name to assign to the triangle object.
        """
        tri = Triangle.from_clipboard(id=id)
        tri.base_linear_model()
        self.t.add(tri, f"{id}")
        setattr(self, f"{id}", tri)

    def FromCSV(
        self, filename: str, origin_columns: int = 1, id: str = "rpt_loss"
    ) -> None:
        """
        Load a triangle from a CSV file. The triangle must include the origin periods
        starting in the first column, but allows for multiple columns to be used for
        the origin periods. The development periods must be in the first row.

        Parameters
        ----------
        filename : str
            The name of the CSV file to load.
        origin_columns : int, optional
            The number of columns used for the origin periods.
            Default is 1, and this is (by far) the most common.
        id : str, optional
            The name to assign to the triangle object.
            Default is "rpt_loss".
        """
        tri = Triangle.from_csv(filename=filename, origin_columns=origin_columns, id=id)
        tri.base_linear_model()
        self.t.add(tri, f"{id}")
        setattr(self, f"{id}", tri)

    def FromExcel(
        self,
        filename,
        origin_columns=1,
        id="rpt_loss",
        sheet_name=None,
        sheet_range=None,
    ) -> None:
        """
        Load a triangle from an Excel file. The triangle must include the origin periods
        starting in the first column, but allows for multiple columns to be used for
        the origin periods. The development periods must be in the first row.

        Parameters
        ----------
        filename : str
            The name of the Excel file to load.
        origin_columns : int, optional
            The number of columns used for the origin periods.
            Default is 1, and this is (by far) the most common.
        id : str, optional
            The name to assign to the triangle object.
            Default is "rpt_loss".
        sheet_name : str, optional
            The name of the sheet to load.
            Default is None, which will load the first sheet.
        sheet_range : str, optional
            The range of cells to load.
            Default is None, which will load all cells.
        """
        # raise NotImplementedError
        tri = Triangle.from_excel(
            filename=filename,
            origin_columns=origin_columns,
            id=id,
            sheet_name=sheet_name,
            sheet_range=sheet_range,
        )
        tri.base_linear_model()
        self.t.add(tri, f"{id}")
        setattr(self, f"{id}", tri)

    def FromDF(self, df, id="rpt_loss") -> None:
        """
        Load a triangle from a Pandas DataFrame. The triangle must include the origin
        periods in the first column and the development periods in the first row.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to load.
        id : str, optional
            The name to assign to the triangle object.
            Default is "rpt_loss".
        """

        tri = Triangle.from_df(df=df, id=id)
        tri.base_linear_model()
        self.t.add(tri, f"{id}")
        setattr(self, f"{id}", tri)

    def AddModel(
        self,
        id: str = None,
        model_class: str = "tweedie",
        tri: Triangle = None,
        cal=False,
        n_validation=5,
    ):
        """
        Add a model to the ROCKY object.

        Parameters
        ----------
        id : str, optional
            The name to assign to the model.
            Default is None, which will assign a default name.
        model_class : str, optional
            The model class to use.
            Default is "tweedie".
        tri : Triangle, optional
            The triangle object to use.
            Default is None, which will use the first triangle in the ROCKY object.
        cal : bool, optional
            Whether to use calendar periods as variables.
            Default is False.
        n_validation : int, optional
            The number of validation folds to use for hyperparameter tuning.
            Default is 5.

        Notes
        -----
        The following model classes are available:
            - "tweedie" (alias "glm") (starting with v0.0.1)

        """

        if id is None:
            if model_class.lower() in all_models:
                id = "PaidLossGLM" + ("_Cal" if cal else "")
            else:
                raise ValueError(
                    f"Model class {model_class} not recognized. Please choose from {all_models}"
                )
        if tri is None:
            raise ValueError("Triangle object must be provided")

        if type(tri) == str:
            if type(getattr(self, tri)) != Triangle:
                raise ValueError("Triangle object must be provided")
            else:
                tri = getattr(self, tri)

        # add the model to the model container
        if model_class.lower() in ['tweedie','glm']:
            self.mod.add(
                glm(
                    id=id,
                    model_class=model_class,
                    tri=tri,
                    use_cal=cal,
                    n_validation=n_validation,
                ),
                f"{id}",
            )
        elif model_class.lower() in ['loglinear']:
            self.mod.add(
                loglinear(
                    id=id,
                    model_class=model_class,
                    tri=tri,
                    use_cal=cal,
                    n_validation=n_validation,
                )
            )

        # add the model directly to the ROCKY object
        try:
            getattr(self, f"{id}")
            UserWarning(
                f"{id} already exists in ROCKY object. Please pass a different id."
            )
            pass
        except AttributeError:
            setattr(self, f"{id}", getattr(self.mod, f"{id}"))

    def ForecastScenario(
        self,
        id: str = None,
        model: glm = None,
        cal: bool = False
        #  , forecast:
    ) -> None:
        # must provide an id and model object
        if id is None:
            raise ValueError("id must be provided")
        if model is None:
            raise ValueError(
                "Fitted model object must be provided. Run `TweedieGLM` first."
            )

        # check that model is a GLM object
        if not isinstance(model, glm):
            raise ValueError(
                "Model object must be a GLM object. Run `TweedieGLM` or create a custom `GLM.glm`."
            )

        # extract the triangle from the model object
        tri = model.tri

        # get the current calendar index and calendar index from the triangle
        cur_cal = tri.getCurCalendarIndex()
        cal_idx = tri.getCalendarIndex()

        # calculate future calendar periods (that the forecast applies to)
        future_cal = cal_idx[cal_idx > cur_cal].values

        # drop na, flatten, and convert to list
        future_cal = future_cal[~pd.isna(future_cal)].flatten().tolist()
