try:
  from .GLM import glm
except:
  from GLM import glm

from dataclasses import dataclass

@dataclass
class PoissonGLM(glm):
  # call the parent class constructor
  