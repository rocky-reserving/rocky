import pymc
from pymc import Normal, LogNormal, HalfCauchy

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
  rpt_loss_tri: Triangle = None
  paid_loss_tri: Triangle = None
  rpt_count_tri: Triangle = None
  paid_count_tri: Triangle = None

  def __post_init__(self):
    # Set the accident and development periods
    self.accident_periods = self.rpt_loss_tri.tri.index if self.rpt_loss_tri is not None else None
    self.development_periods = self.rpt_loss_tri.tri.columns if self.rpt_loss_tri is not None else None

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
               data: MegaData,

               burnin: int = 1000,
               samples: int = 4000,
               chains: int = 4):
    """
    Read in the data and store it in the MegaModel class. The data is stored in
    triangles that have been converted to pytorch tensors. 
    """
    self.rpt_loss_tri = data.rpt_loss_tri
    self.paid_loss_tri = data.paid_loss_tri
    self.rpt_count_tri = data.rpt_count_tri
    self.paid_count_tri = data.paid_count_tri

    self.acc = data.rpt_loss_tri.tri.index.year.values
    self.dev = data.rpt_loss_tri.tri.columns.astype(int).values

    # loss triangles
    self.rpt_loss = self.rpt_loss_tri.tri.values
    self.paid_loss = self.paid_loss_tri.tri.values
    self.rpt_count = self.rpt_count_tri.tri.values
    self.paid_count = self.paid_count_tri.tri.values

    self.ave_rpt_loss = Triangle.from_dataframe(df=self.rpt_loss_tri.tri / self.rpt_count_tri.tri, id='ave_rpt')
    self.ave_paid_loss = Triangle.from_dataframe(df=self.paid_loss_tri.tri / self.paid_count_tri.tri, id='ave_paid')

    self.paid_rpt_loss_ratio = Triangle.from_dataframe(df=self.paid_loss_tri.tri / self.rpt_loss_tri.tri, id='paid_rpt_ratio')
    self.paid_rpt_count_ratio = Triangle.from_dataframe(
        df=self.rpt_count_tri.tri / self.paid_count_tri.tri, id='paid_rpt_count_ratio')

    # need a triangle mask -- true if the value is not nan
    self.tri_mask = ~np.isnan(self.rpt_loss)

    # MCMC parameters
    self.burnin = burnin
    self.samples = samples
    self.chains = chains
    
    # Prior ultimate estimates
    self.loss_ult_prior = np.multiply(self.rpt_loss_tri.diag().values,
                                      self.rpt_loss_tri.atu().sort_index(ascending=False).values)
    self.count_ult_prior = np.multiply(self.rpt_count_tri.diag().values,
                                       self.rpt_count_tri.atu().sort_index(ascending=False).values)
    self.ave_loss_prior = np.multiply(self.ave_rpt_loss.diag().values,
                                      self.ave_rpt_loss.atu().sort_index(ascending=False).values)

  def prior_ultimate_distributions(self
                                 , name: str = 'alpha'
                                #  , mu: float = None
                                #  , sigma: float = None
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
    # if mu is None or sigma is None:
      
    # method of moments is used for a prior estimate
    m_l = np.sum(np.log(self.loss_ult_prior)) / self.loss_ult_prior.shape[0]
    s2_l = np.sum(np.power(np.log(self.loss_ult_prior) - m_l, 2)) / \
        self.loss_ult_prior.shape[0]
    s_l = np.sqrt(s2_l)

    # method of moments is used for a prior estimate
    m_c = np.sum(np.log(self.count_ult_prior)) / self.count_ult_prior.shape[0]
    s2_c = np.sum(np.power(np.log(self.count_ult_prior) - m_c, 2)) / \
        self.count_ult_prior.shape[0]
    s_c = np.sqrt(s2_c)

    # method of moments is used for a prior estimate
    m_ave = np.sum(np.log(self.ave_loss_prior)) / self.ave_loss_prior.shape[0]
    s2_ave = np.sum(np.power(np.log(self.ave_loss_prior) - m_ave, 2)) / \
        self.ave_loss_prior.shape[0]
    s_ave = np.sqrt(s2_ave)



    # latent variables for ultimate
    if standalone:
      alpha_loss = pymc.LogNormal.dist(mu=m_l,
                                       sigma=s_l,
                                       shape=self.rpt_loss_tri.shape[0])
      alpha_count = pymc.LogNormal.dist(mu=m_c,
                                       sigma=s_c,
                                       shape=self.rpt_count_tri.shape[0])
      alpha_ave_loss = pymc.LogNormal.dist(mu=m_ave,
                                      sigma=s_ave,
                                      shape=self.ave_rpt_loss.shape[0])
    else:
      _ult_loss = pymc.LogNormal.dist('latent_ult_loss',
                                      mu=m_l,
                                      sigma=s_l,
                                      shape=(self.rpt_loss_tri.shape[0],1))
      _ult_count = pymc.LogNormal.dist('latent_ult_counts',
                                       mu=m_c,
                                       sigma=s_c,
                                       shape=(self.rpt_count_tri.shape[0], 1))
      _ult_ave = pymc.LogNormal.dist('latent_ult_ave_loss',
                                     mu=m_ave,
                                     sigma=s_ave,
                                     shape=(self.ave_rpt_loss.shape[0], 1))

      # deterministic functions for the prior estimates of ultimates
      # it doesn't make sense to do this like this, but it mirrors the method in
      # the MegaModel class
      alpha_loss = pymc.Deterministic('alpha_loss', _ult_count * _ult_ave)
      alpha_count = pymc.Deterministic('alpha_count', _ult_loss / _ult_ave)
      alpha_ave_loss = pymc.Deterministic(
          'alpha_ave_loss', _ult_loss / _ult_count)
    
    return alpha_loss, alpha_count, alpha_ave_loss

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
      beta_rpt_loss = pymc.Normal.dist(mu=mu,
                                       sigma=sigma,
                                       size=self.dev.shape[0])
      beta_paid_loss = pymc.Normal.dist(mu=mu,
                                       sigma=sigma,
                                       size=self.dev.shape[0])
      beta_rpt_count = pymc.Normal.dist(mu=mu,
                                       sigma=sigma,
                                       size=self.dev.shape[0])
      beta_paid_count = pymc.Normal.dist(mu=mu,
                                       sigma=sigma,
                                       size=self.dev.shape[0])
      beta_ave_rpt_loss = pymc.Normal.dist(mu=mu,
                                       sigma=sigma,
                                       size=self.dev.shape[0])
      beta_ave_paid_loss = pymc.Normal.dist(mu=mu,
                                       sigma=sigma,
                                       size=self.dev.shape[0])
      beta_paid_rpt_ratio = pymc.Normal.dist(mu=mu,
                                       sigma=sigma,
                                       size=self.dev.shape[0])
      beta_paid_rpt_count = pymc.Normal.dist(mu=mu,
                                       sigma=sigma,
                                       size=self.dev.shape[0])
    else:
      beta_rpt_loss = pymc.Normal('beta_rpt_loss',
                                  mu=mu,
                                  sigma=sigma,
                                  size=self.dev.shape[0])
      beta_paid_loss = pymc.Normal('beta_paid_loss',
                                   mu=mu,
                                   sigma=sigma,
                                   size=self.dev.shape[0])
      beta_rpt_count = pymc.Normal('beta_rpt_count',
                                   mu=mu,
                                   sigma=sigma,
                                   size=self.dev.shape[0])
      beta_paid_count = pymc.Normal('beta_paid_count',
                                    mu=mu,
                                    sigma=sigma,
                                    size=self.dev.shape[0])
      beta_ave_rpt_loss = pymc.Normal('beta_ave_rpt_loss',
                                      mu=mu,
                                      sigma=sigma,
                                      size=self.dev.shape[0])
      beta_ave_paid_loss = pymc.Normal('beta_ave_paid_loss',
                                       mu=mu,
                                       sigma=sigma,
                                       size=self.dev.shape[0])
      beta_paid_rpt_ratio = pymc.Normal('beta_paid_rpt_ratio',
                                        mu=mu,
                                        sigma=sigma,
                                        size=self.dev.shape[0])
      beta_paid_rpt_count = pymc.Normal('beta_paid_rpt_count',
                                        mu=mu,
                                        sigma=sigma,
                                        size=self.dev.shape[0])

    return (beta_rpt_loss, beta_paid_loss,
            beta_rpt_count, beta_paid_count,
            beta_ave_rpt_loss, beta_ave_paid_loss,
            beta_paid_rpt_ratio, beta_paid_rpt_count)

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
      sigma_rpt_loss = HalfCauchy.dist(beta=beta,
                                       size=self.dev.shape[0])
      sigma_paid_loss = HalfCauchy.dist(beta=beta,
                                        size=self.dev.shape[0])
      sigma_rpt_count = HalfCauchy.dist(beta=beta,
                                        size=self.dev.shape[0])
      sigma_paid_count = HalfCauchy.dist(beta=beta,
                                         size=self.dev.shape[0])
      sigma_ave_rpt_loss = HalfCauchy.dist(beta=beta,
                                           size=self.dev.shape[0])
      sigma_ave_paid_loss = HalfCauchy.dist(beta=beta,
                                            size=self.dev.shape[0])
      sigma_paid_rpt_loss = HalfCauchy.dist(beta=beta,
                                            size=self.dev.shape[0])
      sigma_paid_rpt_count = HalfCauchy.dist(beta=beta,
                                             size=self.dev.shape[0])
    else:
      sigma_rpt_loss = HalfCauchy('sigma_rpt_loss',
                                  beta=beta,
                                  size=self.dev.shape[0])
      sigma_paid_loss = HalfCauchy('sigma_paid_loss',
                                   beta=beta,
                                   size=self.dev.shape[0])
      sigma_rpt_count = HalfCauchy('sigma_rpt_count',
                                   beta=beta,
                                   size=self.dev.shape[0])
      sigma_paid_count = HalfCauchy('sigma_paid_count',
                                    beta=beta,
                                    size=self.dev.shape[0])
      sigma_ave_rpt_loss = HalfCauchy('sigma_ave_rpt_loss',
                                      beta=beta,
                                      size=self.dev.shape[0])
      sigma_ave_paid_loss = HalfCauchy('sigma_ave_paid_loss',
                                       beta=beta,
                                       size=self.dev.shape[0])
      sigma_paid_rpt_loss = HalfCauchy('sigma_paid_rpt_loss',
                                       beta=beta,
                                       size=self.dev.shape[0])
      sigma_paid_rpt_count = HalfCauchy('sigma_paid_rpt_count',
                                        beta=beta,
                                        size=self.dev.shape[0])
      
    return (sigma_rpt_loss, sigma_paid_loss,
            sigma_rpt_count, sigma_paid_count,
            sigma_ave_rpt_loss, sigma_ave_paid_loss,
            sigma_paid_rpt_loss, sigma_paid_rpt_count)

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
      alpha_loss, alpha_count, alpha_ave_loss = self.prior_ultimate_distributions(
          standalone=False)
      
      # prior distributions for the development parameters
      (beta_rpt_loss, beta_paid_loss,
       beta_rpt_count, beta_paid_count,
       beta_ave_rpt_loss, beta_ave_paid_loss,
       beta_paid_rpt_ratio, beta_paid_rpt_count) = self.prior_development_distributions(standalone=False)

      # prior distributions for the standard deviations
      (sigma_rpt_loss, sigma_paid_loss,
       sigma_rpt_count, sigma_paid_count,
       sigma_ave_rpt_loss, sigma_ave_paid_loss,
       sigma_paid_rpt_loss, sigma_paid_rpt_count) = self.prior_sigma_distributions(standalone=False)

      # expected values for the triangles
      E_rpt_loss = self._E(alpha_loss, beta_rpt_loss)
      E_paid_loss = self._E(alpha_loss, beta_paid_loss)
      E_rpt_count = self._E(alpha_count, beta_rpt_count)
      E_paid_count = self._E(alpha_count, beta_paid_count)
      E_ave_rpt_loss = self._E(alpha_ave_loss, beta_ave_rpt_loss)
      E_ave_paid_loss = self._E(alpha_ave_loss, beta_ave_paid_loss)
      E_paid_rpt_loss = self._E(1, beta_paid_rpt_ratio)
      E_paid_rpt_count = self._E(1, beta_paid_rpt_count)
      
      # likelihood functions
      loglik_rpt_loss = pymc.Normal('loglik_rpt_loss',
                                    mu=E_rpt_loss[self.tri_mask],
                                    sigma=np.array([sigma_rpt_loss.eval() for _ in range(self.acc.shape)])[self.tri_mask],
                                    observed=self.rpt_loss[self.tri_mask])
      loglik_paid_loss = pymc.Normal('loglik_paid_loss',
                                    mu=E_paid_loss[self.tri_mask],
                                    sigma=np.array([sigma_paid_loss.eval() for _ in range(self.acc.shape)])[
                                        self.tri_mask],
                                    observed=self.paid_loss[self.tri_mask])
      loglik_rpt_count = pymc.Normal('loglik_rpt_count',
                                     mu=E_rpt_count[self.tri_mask],
                                     sigma=np.array([sigma_rpt_count.eval() for _ in range(self.acc.shape)])[
                                         self.tri_mask],
                                     observed=self.rpt_count[self.tri_mask])
      loglik_paid_count = pymc.Normal('loglik_paid_count',
                                     mu=E_paid_count[self.tri_mask],
                                     sigma=np.array([sigma_paid_count.eval() for _ in range(self.acc.shape)])[
                                         self.tri_mask],
                                     observed=self.paid_count[self.tri_mask])
      loglik_ave_rpt_loss = pymc.Normal('loglik_ave_rpt_loss',
                                        mu=E_ave_rpt_loss[self.tri_mask],
                                        sigma=np.array([sigma_ave_rpt_loss.eval() for _ in range(self.acc.shape)])[
                                        self.tri_mask],
                                        observed=self.ave_rpt_loss[self.tri_mask])
      loglik_ave_paid_loss = pymc.Normal('loglik_ave_paid_loss',
                                        mu=E_ave_paid_loss[self.tri_mask],
                                        sigma=np.array([sigma_ave_paid_loss.eval() for _ in range(self.acc.shape)])[
                                            self.tri_mask],
                                        observed=self.ave_paid_loss[self.tri_mask])
      loglik_paid_rpt_loss = pymc.Normal('loglik_paid_rpt_loss',
                                         mu=E_paid_rpt_loss[self.tri_mask],
                                         sigma=np.array([sigma_paid_rpt_loss.eval() for _ in range(self.acc.shape)])[
                                             self.tri_mask],
                                         observed=self.paid_rpt_loss_ratio[self.tri_mask])
      loglik_paid_rpt_count = pymc.Normal('loglik_paid_rpt_count',
                                          mu=E_paid_rpt_count[self.tri_mask],
                                         sigma=np.array([sigma_paid_rpt_count.eval() for _ in range(self.acc.shape)])[
                                             self.tri_mask],
                                         observed=self.paid_rpt_count_ratio[self.tri_mask])

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

