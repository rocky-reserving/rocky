try:
    from .triangle import Triangle
except ImportError:
    from triangle import Triangle

try:
    from .TriangleTimeSeriesSplit import TriangleTimeSeriesSplit
except ImportError:
    from TriangleTimeSeriesSplit import TriangleTimeSeriesSplit

try:
    from .GLM import glm
except ImportError:
    from GLM import glm


from dataclasses import dataclass
from typing import Any, Optional

from warnings import filterwarnings

import numpy as np
import pandas as pd

filterwarnings("ignore", category=UserWarning, module="openpyxl")

# add all implemented model types to this list
all_models = "Poisson Gamma".split()


@dataclass
class rockyObj:
    id: str = None
    obj: object = None


@dataclass
class rockyContainer:
    def __repr__(self) -> str:
        # only show comma-separated list of ids if there are any objects in the container
        if len(self.__dict__) > 0:
            return ", ".join([f'"{k}"' for k in self.__dict__.keys()])
        else:
            return "()"

    def add(self, _obj, id=None) -> None:
        if id is None:
            setattr(self, _obj.id, _obj)
        else:
            setattr(self, id, _obj)


@dataclass
class ROCKY:
    # initialize the model attribute, triangle attribute, forecast attribute,
    # validation attribute, and plotting attribute
    id: str = None
    mod: Any = rockyContainer()  # models
    f: Any = rockyContainer()  # forecasts
    plot: Any = rockyContainer()  # plots
    t: Any = rockyContainer()  # triangles

    def load_taylor_ashe(self, id=None) -> None:
        tri = Triangle.from_taylor_ashe()
        tri.base_linear_model()

        if id is None:
            id = "paid_loss"
        self.t.add(tri, id)
        setattr(self, f"{id}", tri)

    def load_dahms(self, id="rpt_loss") -> None:
        d = {}
        d["rpt_loss"], d["paid_loss"] = Triangle.from_dahms()
        for id in d.keys():
            d[id].base_linear_model()
            self.t.add(d[id], f"{id}")
            setattr(self, f"{id}", d[id])

    def SampleTri(self, sample, id=None) -> None:
        if sample.lower() == "taylor_ashe":
            if id is None:
                id = "paid_loss"
            self.load_taylor_ashe(id=id)
            getattr(self, f"{id}").base_linear_model()
        # elif sample.lower() == "dahms":
        #     self.load_dahms(id=id)
        #     for id in ["rpt_loss", "paid_loss"]:
        #         getattr(self, f"{id}").base_linear_model()

        # self.t.add(tri, f"{id}")
        # setattr(self, f"{id}", tri)

    def FromCSV(self, filename, origin_columns=1, id="rpt_loss") -> None:
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
        tri = Triangle.from_df(df=df, id=id)
        tri.base_linear_model()
        self.t.add(tri, f"{id}")
        setattr(self, f"{id}", tri)

    def TweedieGLM(
        self,
        id: str = None,
        model_class: str = "tweedie",
        tri: Triangle = None,
        cal=False,
        n_validation=5,
    ):
        if id is None:
            id = "PaidLossGLM" + ("_Cal" if cal else "")
        if tri is None:
            raise ValueError("Triangle object must be provided")

        # add the model to the model container
        self.mod.add(
            glm(
                id=id,
                model_class=model_class,
                tri=tri,
                cal=cal,
                n_validation=n_validation,
            ),
            f"{id}",
        )

        # add the model directly to the ROCKY object
        if getattr(self, f"{id}") is None:
            setattr(self, f"{id}", getattr(self.mod, f"{id}"))
        else:
            raise ValueError(
                f"{id} already exists in ROCKY object. Please pass a different id."
            )

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
