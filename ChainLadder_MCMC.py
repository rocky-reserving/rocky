"""
This is a simplified version of the MegaModel for a single triangle. 
"""

import pymc
from dataclasses import dataclass
from triangle import Triangle
from typing import Tuple
import numpy as np
# import torch


@dataclass
class MegaData:
  """
  Dataclass for storing data for the MegaModel. The data is stored in triangles,
  which are numpy arrays with the first dimension being the accident period and
  the second dimension being the development period. The accident periods and
  development periods are also stored as numpy arrays.

  """
  triangle: Triangle = None

  def __post_init__(self):
    # Set the accident and development periods
    self.accident_periods = self.triangle.tri.index if self.triangle is not None else None
    self.development_periods = self.triangle.tri.columns if self.triangle is not None else None

    # Several aliases for the accident and development periods
    self.accident_period = self.accident_periods if self.accident_periods is not None else None
    self.ay = self.accident_periods if self.accident_periods is not None else None
    self.acc = self.accident_periods if self.accident_periods is not None else None
    self.origin = self.accident_periods if self.accident_periods is not None else None
    self.origin_periods = self.accident_periods if self.accident_periods is not None else None
    self.origin_period = self.accident_periods if self.accident_periods is not None else None
    
    self.development_period = self.development_periods if self.development_periods is not None else None
    self.dy = self.development_periods if self.development_periods is not None else None
    self.dev = self.development_periods if self.development_periods is not None else None
    self.dev_months = self.development_periods if self.development_periods is not None else None
    self.dev_period = self.development_periods if self.development_periods is not None else None
  

class ChainLadderMCMC:
  def __init__(self,
               data: MegaData):
    """
    Read in the data and store it in the MegaModel class. The data is stored in
    triangles that have been converted to pytorch tensors. 
    """
    self.data = data
    self.acc = data.accident_periods.year
    self.dev = data.development_periods.astype(int).values
    # self.acc = torch.as_tensor(data.accident_periods.year, dtype=torch.int16)
    # self.dev = torch.as_tensor(data.development_periods.astype(int).values, dtype=torch.int16)

    # loss triangles
    self.tri = data.triangle.tri.values

  # @pymc.dtrm(trace=True)
  def prior_ultimate_distributions(self
                                 , mu: float = 10
                                 , tau: float = 0.2):
    """
    Define informative prior distributions for the latent variables. The prior distributions are
    defined as pymc deterministic variables. Doing is this way should capture the relationship
    between the different ultimate values - this relationship is fixed and not a random variable.
    """
    # latent variables for ultimate
    _ultimate = pymc.Normal('latent_ultimate', mu=mu, tau=tau, shape=self.tri.shape[0])

    # deterministic functions for the prior estimates of ultimates
    # it doesn't make sense to do this like this, but it mirrors the method in
    # the MegaModel class
    alpha = pymc.Deterministic('alpha', _ultimate)
    
    return alpha

  # @pymc.stoch(trace=True)
  def prior_development_distributions(self) -> Tuple:
    """
    Define noninformative prior distributions for the development factors. The prior distributions are
    defined as pymc stochastic variables, and are assumed to be almost between 0 and 1. The development
    parameters are denoted as beta, and are the percent of total for the development period. 

    Use normal distributions for the development parameters.
    """
    # prior distributions for development
    beta = pymc.Normal('beta', mu=0.5, tau=0.2, size=self.dev.shape[0])

    return beta

  # @pymc.stoch(trace=True)
  def prior_sigma_distributions(self) -> Tuple:
    """
    Define noninformative prior distributions for the standard deviations of the alpha parameters. The
    prior distributions half Cauchy distributions with a scale parameter of 2.5. The standard deviations
    vary by the type of triangle and the development period.
    """
    # variance of each of the triangles
    sigma = pymc.HalfCauchy('sigma', beta=2.5, size=self.dev.shape[0])
    
    return sigma

  # @pymc.dtrm(trace=True)
  def _E(self
       , alpha = None 
       , beta = None):
    """
    Helper function for defining the expected value of a cell in the triangle. The expected value
    is the product of the ultimate and the development factor.
    """
    print("Shape of alpha:", alpha.shape)
    print("Shape of beta:", beta.shape)

    alpha = alpha.reshape((-1, 1))  # Reshape alpha into a column vector with shape (m, 1)
    beta = beta.reshape((1, -1))  # Reshape beta into a row vector with shape (1, n)

    # alpha_np = alpha.to_numpy().reshape(-1, 1)  # Convert alpha to NumPy array and reshape it to a column vector
    # beta_np = beta.to_numpy().reshape(1, -1)  # Convert beta to NumPy array and reshape it to a row vector



    # shape = accident_periods x development_periods
    return np.multiply(alpha, beta)
    # return np.multiply(alpha_np, beta_np)
    # return torch.multiply(alpha, beta)

  def _gamma_alpha(self, E, sigma):
    """
    Helper function for calculating the alpha parameter for the gamma distribution. The alpha parameter
    is the square of the expected value divided by the variance.
    """
    return np.divide(np.power(E, 2), np.power(sigma, 2))
    # return torch.divide(torch.power(E, 2), torch.power(sigma, 2))

  def _gamma_beta(self, E, sigma):
    """
    Helper function for calculating the beta parameter for the gamma distribution. The beta parameter
    is the expected value divided by the variance.
    """
    return np.divide(E, np.power(sigma, 2))
    # return torch.divide(E, torch.power(sigma, 2))

  def _beta_alpha(self, E, sigma):
    """
    Helper function for calculating the alpha parameter for the beta distribution. The alpha parameter
    is the expected value times the variance plus 1.

    alpha = E * (E * (1 - E) / sigma**2 - 1)
    """
    return np.multiply(E, np.subtract(np.divide(np.multiply(E, np.subtract(1, E)), np.power(sigma, 2)), 1))
    # return torch.multiply(E, torch.subtract(torch.divide(torch.multiply(E, torch.subtract(1, E)), torch.power(sigma, 2)), 1))

  def _beta_beta(self, E, sigma):
    """
    Helper function for calculating the beta parameter for the beta distribution. The beta parameter
    is (1 - the expected value) times the variance plus 1.

    beta = (1 - E) * (E * (1 - E) / sigma**2 - 1)
    """
    return np.multiply(np.subtract(1, E), np.subtract(np.divide(np.multiply(E, np.subtract(1, E)), np.power(sigma, 2)), 1))
    # return torch.multiply(torch.subtract(1, E), torch.subtract(torch.divide(torch.multiply(E, torch.subtract(1, E)), torch.power(sigma, 2)), 1))


  def chain_ladder(self):
    with pymc.Model() as model:
      # prior distributions for the ultimate parameters
      alpha = self.prior_ultimate_distributions()
      
      # prior distributions for the development parameters
      beta = self.prior_development_distributions()

      # prior distributions for the standard deviations
      sigma = self.prior_sigma_distributions()

      # expected values for the triangles
      # shape = accident_periods x development_periods
      E = self._E(alpha, beta) # paid/reported count ratios

      print("Shape of E:", E.shape)
      print("Shape of sigma:", sigma.shape)
      print("Shape of gamma_alpha:", self._gamma_alpha(E, sigma).shape)
      print("Shape of gamma_beta:", self._gamma_beta(E, sigma).shape)


      # likelihood distributions for the observed data
      # shape = accident_periods x development_periods
      # loss distributions get gamma distributions
      loglik = pymc.Gamma('loglik'
                        , alpha=self._gamma_alpha(E, sigma)
                        , beta=self._gamma_beta(E, sigma)
                        , observed=True)

    self.model = model
    return model
