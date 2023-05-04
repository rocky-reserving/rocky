import pymc as pm
from dataclasses import dataclass
from triangle import Triangle
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import torch


@dataclass
class MegaData:
  """
  Dataclass for storing data for the MegaModel. The data is stored in triangles,
  which are numpy arrays with the first dimension being the accident period and
  the second dimension being the development period. The accident periods and
  development periods are also stored as numpy arrays.

  """
  reported_loss_triangle: Triangle = None
  paid_loss_triangle: Triangle = None
  
  reported_count_triangle: Triangle = None
  paid_count_triangle: Triangle = None

  exposure: pd.Series = None

  def __post_init__(self):
    # Set the accident and development periods
    self.accident_periods = self.reported_loss_triangle.accident_periods if self.reported_loss_triangle is not None else None
    self.development_periods = self.reported_loss_triangle.development_periods if self.reported_loss_triangle is not None else None

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
  

class MegaModel:
  def __init__(self,
               data: MegaData):
    """
    Read in the data and store it in the MegaModel class. The data is stored in
    triangles that have been converted to pytorch tensors. 
    """
    self.data = data
    self.acc = torch.as_tensor(data.accident_periods)
    self.dev = torch.as_tensor(data.development_periods)

    # loss triangles
    self.rpt_loss = torch.as_tensor(data.reported_loss_triangle)
    self.paid_loss = torch.as_tensor(data.paid_loss_triangle)
    if self.paid_loss is None or self.rpt_loss is None:
      self.case_reserve = None
    else:
      self.case_reserve = torch.sub(self.rpt_loss, self.paid_loss)

    # count triangles
    self.rpt_count = torch.as_tensor(data.reported_count_triangle)
    self.paid_count = torch.as_tensor(data.paid_count_triangle)
    if self.paid_count is None or self.rpt_count is None:
      self.open_count = None
    else:
      self.open_count = torch.sub(self.rpt_count, self.paid_count)

    # average claim size triangles
    if self.rpt_count is None or self.rpt_loss is None:
      self.ave_rpt_loss = None
    else:
      self.ave_rpt_loss = torch.divide(self.rpt_loss, self.rpt_count)
    
    if self.paid_count is None or self.paid_loss is None:
      self.ave_paid_loss = None
    else:
      self.ave_paid_loss = torch.divide(self.paid_loss, self.paid_count)
    
    if self.case_reserve is None or self.open_count is None:
      self.ave_case_os = None
    else:
      self.ave_case_os = torch.divide(self.case_reserve, self.open_count)

    # paid/reported triangles
    if self.rpt_loss is None or self.paid_loss is None:
      self.paid_rpt_loss = None
    
    else:
      self.paid_rpt_loss = torch.divide(self.paid_loss, self.rpt_loss)
    
    if self.rpt_count is None or self.paid_count is None:
      self.paid_rpt_count = None
    else:
      self.paid_rpt_count = torch.divide(self.paid_count, self.rpt_count)

    # exposure
    self.exposure = data.exposure

  def prior_ultimate_distributions(self
                                 , loss_mu: float = 10
                                 , loss_tau: float = 0.2
                                 , count_mu: float = 8
                                 , count_tau: float = 0.2
                                 , claim_size_mu: float = 2
                                 , claim_size_tau: float = 0.2
                                 ):
    """
    Define informative prior distributions for the latent variables. The prior distributions are
    defined as pymc deterministic variables. Doing is this way should capture the relationship
    between the different ultimate values - this relationship is fixed and not a random variable.
    """
    # latent variables for ultimate
    _ultimate_loss = yield pymc.Normal('latent_ultimate_loss', mu=loss_mu, tau=loss_tau, self.rpt_loss.shape[0])
    _ultimate_count = yield pymc.Normal('latent_ultimate_count', mu=count_mu, tau=count_tau, self.rpt_count.shape[0])
    _ultimate_claim_size = yield pymc.Normal('latent_ultimate_claim_size', mu=claim_size_mu, tau=claim_size_tau, self.ave_rpt_loss.shape[0])

    # deterministic functions for the prior estimates of ultimates
    alpha_loss = yield pymc.Deterministic('alpha_loss', torch.multiply(_ultimate_count, _ultimate_claim_size))
    alpha_count = yield pymc.Deterministic('alpha_count', torch.divide(_ultimate_loss, _ultimate_claim_size))
    alpha_claim_size = yield pymc.Deterministic('alpha_claim_size', torch.divide(_ultimate_loss, _ultimate_count))

    return alpha_loss, alpha_count, alpha_claim_size

  def prior_development_distributions(self) -> Tuple:
    """
    Define noninformative prior distributions for the development factors. The prior distributions are
    defined as pymc stochastic variables, and are assumed to be almost between 0 and 1. The development
    parameters are denoted as beta, and are the percent of total for the development period. 

    Use normal distributions for the development parameters.
    """
    # prior distributions for development
    beta_rpt_loss = yield pymc.Normal('beta_rpt_loss', mu=0.5, tau=0.2, self.development_periods.shape[0])
    beta_paid_loss = yield pymc.Normal('beta_paid_loss', mu=0.5, tau=0.2, self.development_periods.shape[0])
    beta_rpt_count = yield pymc.Normal('beta_rpt_count', mu=0.5, tau=0.2, self.development_periods.shape[0])
    beta_paid_count = yield pymc.Normal('beta_paid_count', mu=0.5, tau=0.2, self.development_periods.shape[0])
    beta_avg_rpt_loss = yield pymc.Normal('beta_avg_rpt_loss', mu=0.5, tau=0.2, self.development_periods.shape[0])
    beta_avg_paid_loss = yield pymc.Normal('beta_avg_paid_loss', mu=0.5, tau=0.2, self.development_periods.shape[0])
    beta_paid_rpt_loss = yield pymc.Normal('beta_paid_rpt_loss', mu=0.5, tau=0.2, self.development_periods.shape[0])
    beta_paid_rpt_count = yield pymc.Normal('beta_paid_rpt_count', mu=0.5, tau=0.2, self.development_periods.shape[0])

    return beta_rpt_loss, beta_paid_loss, beta_rpt_count, beta_paid_count, beta_avg_rpt_loss, beta_avg_paid_loss, beta_paid_rpt_loss, beta_paid_rpt_count

  def prior_sigma_distributions(self) -> Tuple:
    """
    Define noninformative prior distributions for the standard deviations of the alpha parameters. The
    prior distributions half Cauchy distributions with a scale parameter of 2.5. The standard deviations
    vary by the type of triangle and the development period.
    """
    # variance of each of the triangles
    sigma_rpt_loss = yield pymc.HalfCauchy('sigma_rpt_loss', beta=2.5, size=self.development_periods.shape[0])
    sigma_paid_loss = yield pymc.HalfCauchy('sigma_paid_loss', beta=2.5, size=self.development_periods.shape[0])
    sigma_rpt_count = yield pymc.HalfCauchy('sigma_rpt_count', beta=2.5, size=self.development_periods.shape[0])
    sigma_paid_count = yield pymc.HalfCauchy('sigma_paid_count', beta=2.5, size=self.development_periods.shape[0])
    sigma_avg_rpt_loss = yield pymc.HalfCauchy('sigma_avg_rpt_loss', beta=2.5, size=self.development_periods.shape[0])
    sigma_avg_paid_loss = yield pymc.HalfCauchy('sigma_avg_paid_loss', beta=2.5, size=self.development_periods.shape[0])
    sigma_paid_rpt_loss = yield pymc.HalfCauchy('sigma_paid_rpt_loss', beta=2.5, size=self.development_periods.shape[0])
    sigma_paid_rpt_count = yield pymc.HalfCauchy('sigma_paid_rpt_count', beta=2.5, size=self.development_periods.shape[0])

    return sigma_rpt_loss, sigma_paid_loss, sigma_rpt_count, sigma_paid_count, sigma_avg_rpt_loss, sigma_avg_paid_loss, sigma_paid_rpt_loss, sigma_paid_rpt_count


  def _E(self
       , alpha: pymc.
       , beta: pymc stochastic tensor variable # shape = 1 x development_periods
       ) -> pymc stochastic:
    """
    Helper function for defining the expected value of a cell in the triangle. The expected value
    is the product of the ultimate and the development factor.
    """
    # shape = accident_periods x development_periods
    return torch.multiply(alpha, beta)

  def _gamma_alpha(self, E, sigma) -> pymc stochastic:
    """
    Helper function for calculating the alpha parameter for the gamma distribution. The alpha parameter
    is the square of the expected value divided by the variance.
    """
    return torch.divide(torch.power(E, 2), torch.power(sigma, 2))

  def _gamma_beta(self, E, sigma) -> pymc stochastic:
    """
    Helper function for calculating the beta parameter for the gamma distribution. The beta parameter
    is the expected value divided by the variance.
    """
    return torch.divide(E, torch.power(sigma, 2))

  def _beta_alpha(self, E, sigma) -> pymc stochastic:
    """
    Helper function for calculating the alpha parameter for the beta distribution. The alpha parameter
    is the expected value times the variance plus 1.

    alpha = E * (E * (1 - E) / sigma**2 - 1)
    """
    return torch.multiply(E, torch.subtract(torch.divide(torch.multiply(E, torch.subtract(1, E)), torch.power(sigma, 2)), 1))

  def _beta_beta(self, E, sigma) -> pymc stochastic:
    """
    Helper function for calculating the beta parameter for the beta distribution. The beta parameter
    is (1 - the expected value) times the variance plus 1.

    beta = (1 - E) * (E * (1 - E) / sigma**2 - 1)
    """
    return torch.multiply(torch.subtract(1, E), torch.subtract(torch.divide(torch.multiply(E, torch.subtract(1, E)), torch.power(sigma, 2)), 1))


  @pymc.model
  def chain_ladder(self)
    # prior distributions for the ultimate parameters
    alpha_loss
    , alpha_count
    , alpha_claim_size = self.prior_ultimate_distributions()
    
    # prior distributions for the development parameters
    beta_rpt_loss
    , beta_paid_loss
    , beta_rpt_count
    , beta_paid_count
    , beta_avg_rpt_loss
    , beta_avg_paid_loss
    , beta_paid_rpt_loss
    , beta_paid_rpt_count = self.prior_development_distributions()

    # prior distributions for the standard deviations
    sigma_rpt_loss
    , sigma_paid_loss
    , sigma_rpt_count
    , sigma_paid_count
    , sigma_avg_rpt_loss
    , sigma_avg_paid_loss
    , sigma_paid_rpt_loss
    , sigma_paid_rpt_count = self.prior_sigma_distributions()

    # expected values for the triangles
    # shape = accident_periods x development_periods
    E_rpt_loss = self._E(alpha_loss, beta_rpt_loss) # reported loss
    E_paid_loss = self._E(alpha_loss, beta_paid_loss) # paid loss
    E_rpt_count = self._E(alpha_count, beta_rpt_count) # reported count
    E_paid_count = self._E(alpha_count, beta_paid_count) # paid count
    E_avg_rpt_loss = self._E(alpha_claim_size, beta_avg_rpt_loss) # average reported loss
    E_avg_paid_loss = self._E(alpha_claim_size, beta_avg_paid_loss) # average paid loss
    E_paid_rpt_loss = self._E(1, beta_paid_rpt_loss) # paid/reported loss ratios
    E_paid_rpt_count = self._E(1, beta_paid_rpt_count) # paid/reported count ratios

    # likelihood distributions for the observed data
    # shape = accident_periods x development_periods
    # loss distributions get gamma distributions
    L_rpt_loss = yield pymc.Gamma('L_rpt_loss'
                                , alpha=self._gamma_alpha(E_rpt_loss, sigma_rpt_loss)
                                , beta=self._gamma_beta(E_rpt_loss, sigma_rpt_loss)
                                , observed=True
                                , value=self.rpt_loss)
    L_paid_loss = yield pymc.Gamma('L_paid_loss'
                                  , alpha=self._gamma_alpha(E_paid_loss, sigma_paid_loss)
                                  , beta=self._gamma_beta(E_paid_loss, sigma_paid_loss)
                                  , observed=True
                                  , value=self.paid_loss)

    # count distributions get Poisson distributions
    L_rpt_count = yield pymc.Poisson('L_rpt_count'
                                    , mu=E_rpt_count
                                    , observed=True
                                    , value=self.rpt_count)
    L_paid_count = yield pymc.Poisson('L_paid_count'
                                    , mu=E_paid_count
                                    , observed=True
                                    , value=self.paid_count)

    # average loss distributions get gamma distributions
    L_avg_rpt_loss = yield pymc.Gamma('L_avg_rpt_loss'
                                    , alpha=self._gamma_alpha(E_avg_rpt_loss, sigma_avg_rpt_loss)
                                    , beta=self._gamma_beta(E_avg_rpt_loss, sigma_avg_rpt_loss)
                                    , observed=True
                                    , value=self.avg_rpt_loss)
    L_avg_paid_loss = yield pymc.Gamma('L_avg_paid_loss'
                                      , alpha=self._gamma_alpha(E_avg_paid_loss, sigma_avg_paid_loss)
                                      , beta=self._gamma_beta(E_avg_paid_loss, sigma_avg_paid_loss)
                                      , observed=True
                                      , value=self.avg_paid_loss)

    # paid/reported loss ratio distributions get beta distributions (paid / reported must be between 0 and 1)
    L_paid_rpt_loss = yield pymc.Beta('L_paid_rpt_loss'
                                    , alpha=self._beta_alpha(E_paid_rpt_loss, sigma_paid_rpt_loss)
                                    , beta=self._beta_beta(E_paid_rpt_loss, sigma_paid_rpt_loss)
                                    , observed=True
                                    , value=self.paid_rpt_loss)
    
    # paid/reported count ratio distributions get beta distributions (paid / reported must be between 0 and 1)
    L_paid_rpt_count = yield pymc.Beta('L_paid_rpt_count'
                                     , alpha=self._beta_alpha(E_paid_rpt_count, sigma_paid_rpt_count)
                                     , beta=self._beta_beta(E_paid_rpt_count, sigma_paid_rpt_count)
                                     , observed=True
                                     , value=self.paid_rpt_count)
