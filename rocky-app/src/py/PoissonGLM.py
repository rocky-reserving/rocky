try:
    from .GLM import glm
except:
    from GLM import glm

from dataclasses import dataclass


@dataclass
class PoissonGLM(glm):
    # call the parent class constructor
    def __init__(self, id=None, tri=None, n_validation=0):
        super().__init__(tri, n_validation)
