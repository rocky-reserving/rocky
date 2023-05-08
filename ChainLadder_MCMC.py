"""
This is a simplified version of the MegaModel for a single triangle. 
"""

import pymc
from dataclasses import dataclass
from triangle import Triangle
from typing import Tuple
import numpy as np
from scipy import stats


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
  

class MegaChainLadder:
  def __init__(self,
               data: MegaData,
               burnin: int = 1000,
               samples: int = 4000,
               chains: int = 4):
    """
    Read in the data and store it in the MegaModel class. The data is stored in
    triangles that have been converted to pytorch tensors. 
    """
    self.data = data
    self.acc = data.accident_periods.year.values
    self.dev = data.development_periods.astype(int).values

    # loss triangles
    self.tri = data.triangle.tri.values

    # need a triangle mask -- true if the value is not nan
    self.tri_mask = ~np.isnan(data.triangle.tri.values)

    # MCMC parameters
    self.burnin = burnin
    self.samples = samples
    self.chains = chains

    self.cl_ult = np.multiply(self.data.triangle.diag().values,
                           self.data.triangle.atu().sort_index(ascending=False).values)

  def prior_ultimate_distributions(self
                                 , name: str = 'alpha'
                                 , mu: float = None
                                 , sigma: float = None
                                 , standalone: bool = True
                                 ) -> Tuple:
    """
    Define informative prior distributions for the latent variables. The prior
    distributions are defined as pymc deterministic variables. Doing is this way
    should capture the relationship between the different ultimate values - this
    relationship is fixed and not a random variable.

    Use normal distributions for the latent variables.

    Parameters
    ----------
    name: str
      The name given to the parameters. This is how the parameters are referred
      to in some model diagnostics. Defaults to 'alpha'.
    mu: float
      The mu from the lognormal distribution for the latent variables. The default
      value is None, which means that mu is estimated from the data.
    sigma: float
      The sigma of the lognormal distribution for the latent variables.
      The default value is None, which means that sigma is estimated from
      the data.
    DEPRECIATED standalone: bool
      If True, then the latent variables are defined as a standalone variable (eg
      does not need to be passed inside a pymc.Model). If False, then the latent
      variables are defined as variables inside a pymc.Model, and thus cannot
      be used outside of a block.
    """
    # if you don't pass lognormal parameters, they will be estimated from the data
    if mu is None or sigma is None:
      
      # method of moments is used for a prior estimate
      m = np.sum(np.log(self.cl_ult)) / self.cl_ult.shape[0]
      s2 = np.sum(np.power(np.log(self.cl_ult) - m, 2)) / self.cl_ult.shape[0]
      s = np.sqrt(s2)

      # only replace priors that aren't passed
      if mu is None:
        mu = m
      if sigma is None:
        sigma = s

    # latent variables for ultimate
    if standalone:
      alpha = pymc.LogNormal.dist(mu=mu,
                                  sigma=sigma,
                                  shape=self.tri.shape[0])
    else:
      _ultimate = pymc.LogNormal('latent_ultimate',
                                  mu=mu,
                                  sigma=sigma,
                                  shape=(self.tri.shape[0],1))

      # deterministic functions for the prior estimates of ultimates
      # it doesn't make sense to do this like this, but it mirrors the method in
      # the MegaModel class
      alpha = pymc.Deterministic(name, _ultimate)
    
    return alpha

  def prior_development_distributions(self
                                    , name: str = 'beta'
                                    , mu: float = 0
                                    , sigma: float = 5
                                    , standalone: bool = True
                                    ) -> Tuple:
    """
    Define noninformative prior distributions for the development factors. The
    prior distributions are defined as pymc stochastic variables, and are assumed
    to be almost between 0 and 1. The development parameters are denoted as beta,
    and are the percent of total for the development period. 

    Use normal distributions for the development parameters.

    Parameters
    ----------
    name: str
      The name given to the parameters. This is how the parameters are referred
      to in some model diagnostics. Defaults to 'beta'.
    mu: float
      The mean of the normal distribution for the development parameters. The
      default value is 0.5.
    tau: float
      The precision of the normal distribution for the development parameters.
      Tau is the inverse of the standard deviation. The default value is 0.2,
      which corresponds to a standard deviation of 5.
    DEPRECIATED standalone: bool
      If True, then the development parameters are defined as a standalone
      variable (eg does not need to be passed inside a pymc.Model). If False,
      then the development parameters are defined as variables inside a pymc.Model,
      and thus cannot be used outside of a block.

    Returns
    -------
    beta: pymc.Normal
      The development parameters.
    """

    # prior distributions for development
    if standalone:
      beta = pymc.Normal.dist(mu=mu,
                              sigma=sigma,
                              size=self.dev.shape[0])
    else:
      beta = pymc.Normal(name,
                        mu=mu,
                        sigma=sigma,
                        size=(1,self.dev.shape[0]))

    return beta

  def prior_sigma_distributions(self
                                , name: str = 'sigma'
                                , beta: float = 2.5
                                , standalone: bool = True
                                ) -> Tuple:
    """
    Define noninformative prior distributions for the standard deviations of the
    alpha parameters. The prior distributions half Cauchy distributions with a
    scale parameter of 2.5. The standard deviations vary by the type of triangle
    and the development period.
    """
    # variance of each of the triangles
    if standalone:
      sigma = pymc.HalfCauchy.dist(beta=beta,
                                    size=self.dev.shape[0])
    else:
      sigma = pymc.HalfCauchy(name,
                              beta=beta,
                              size=self.dev.shape[0])
      
    return sigma

  def _E(self
       , alpha = None 
       , beta = None):
    """
    Helper function for defining the expected value of a cell in the triangle.
    The expected value is the product of the ultimate and the development factor.
    """
    assert alpha is not None, "`alpha` must be passed to _E"
    assert beta is not None, "`beta` must be passed to _E"
    return (np.matmul(alpha.eval().reshape(-1, 1),
                     beta.eval().reshape(1, -1)))

  def _gamma_alpha(self, E, sigma):
    """
    Helper function for calculating the alpha parameter for the gamma distribution.
    The alpha parameter is the square of the expected value divided by the variance.
    """
    assert E is not None, "`E` must be passed to _gamma_alpha"
    assert sigma is not None, "`sigma` must be passed to _gamma_alpha"
    return np.divide(np.power(E, 2), np.power(sigma.eval(), 2))

  def _gamma_beta(self, E, sigma):
    """
    Helper function for calculating the beta parameter for the gamma distribution.
    The beta parameter is the expected value divided by the variance.
    """
    assert E is not None, "`E` must be passed to _gamma_beta"
    assert sigma is not None, "`sigma` must be passed to _gamma_beta"
    return np.divide(E, np.power(sigma.eval(), 2))

  def _beta_alpha(self, E, sigma):
    """
    Helper function for calculating the alpha parameter for the beta distribution.
    The alpha parameter is the expected value times the variance plus 1.

    alpha = E * (E * (1 - E) / sigma**2 - 1)
    """
    return E * (E * (1 - E) / sigma**2 - 1)

  def _beta_beta(self, E, sigma):
    """
    Helper function for calculating the beta parameter for the beta distribution.
    The beta parameter is (1 - the expected value) times the variance plus 1.

    beta = (1 - E) * (E * (1 - E) / sigma**2 - 1)
    """
    return ((1 - E) * (E * (1 - E) / sigma**2 - 1))
    


  def chain_ladder_model(self):
    with pymc.Model() as model:
      # prior distributions for the ultimate parameters
      alpha = self.prior_ultimate_distributions(standalone=False)
      
      # prior distributions for the development parameters
      beta = self.prior_development_distributions(standalone=False)

      # prior distributions for the standard deviations
      sigma = self.prior_sigma_distributions(standalone=False)

      # expected values for the triangles
      # shape = accident_periods x development_periods
      E = self._E(alpha, beta) # paid/reported count ratios

      # print("Shape of E:", E.shape)
      # print("Shape of sigma:", sigma.shape)
      # print("Shape of gamma_alpha:", self._gamma_alpha(E, sigma).shape)
      # print("Shape of gamma_beta:", self._gamma_beta(E, sigma).shape)


      # likelihood distributions for the observed data
      # shape = accident_periods x development_periods
      # loss distributions get gamma distributions
      # loglik = pymc.Gamma('loglik'
      #                   , alpha=self._gamma_alpha(E, sigma)[self.tri_mask]
      #                   , beta=self._gamma_beta(E, sigma)[self.tri_mask]
      #                   , observed=self.tri[self.tri_mask])
      
      loglik = pymc.Normal('loglik'
                        , mu=E[self.tri_mask]
                        , sigma=np.array([sigma.eval() for _ in range(self.tri.shape[0])])[self.tri_mask]
                        , observed=self.tri[self.tri_mask])
      # loglik = pymc.Gamma('loglik'
      #                   , alpha=self._gamma_alpha(E, sigma)
      #                   , beta=self._gamma_beta(E, sigma)
      #                   , observed=self.tri)

    self.model = model
    return model
  
  def fit(self, samples=None, burnin=None, chains=None):
    if samples is None:
      samples = self.samples
    if burnin is None:
      burnin = self.burnin
    if chains is None:
      chains = self.chains

    if self.model is None:
      mod = self.chain_ladder_model()
    else:
      mod = self.model
    with mod:
    # with self.chain_ladder_model():
    # inference
      self.trace = pymc.sample(draws=samples,
                               tune=burnin,
                               chains=chains,
                               cores=None)

