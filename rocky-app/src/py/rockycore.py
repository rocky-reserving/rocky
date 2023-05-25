# from forecast import Forecast
try:
    from .triangle import Triangle
except ImportError:
    from triangle import Triangle

from dataclasses import dataclass
from typing import Any, Optional

from warnings import filterwarnings

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

    def tri_from_sample(self, sample, id=None) -> None:
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

    def tri_from_csv(self, filename, origin_columns=1, id="rpt_loss") -> None:
        tri = Triangle.from_csv(filename=filename, origin_columns=origin_columns, id=id)
        tri.base_linear_model()
        self.t.add(tri, f"{id}")
        setattr(self, f"{id}", tri)

    def tri_from_excel(
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

    def tri_from_df(self, df, id="rpt_loss") -> None:
        tri = Triangle.from_df(df=df, id=id)
        tri.base_linear_model()
        self.t.add(tri, f"{id}")
        setattr(self, f"{id}", tri)
