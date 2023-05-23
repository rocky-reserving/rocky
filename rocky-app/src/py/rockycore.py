# from forecast import Forecast
from .triangle import Triangle

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

    def load_taylor_ashe(self):
        tri = Triangle.from_taylor_ashe()
        # set the triangle as an attribute of the ROCKY object
        self.t.add(tri, "paid_loss")
        # setattr(self, tri.id, tri)

    def load_dahms(self, tri_type="rpt"):
        d = {}
        d["rpt"], d["paid"] = Triangle.from_dahms()
        self.t.add(d[tri_type], f"{tri_type}_loss")
