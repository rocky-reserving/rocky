import pymc
from pymc import HalfCauchy, Normal, LogNormal, Deterministic

from dataclasses import dataclass
from triangle import Triangle
from typing import Tuple
import numpy as np
import pandas as pd
from scipy import stats

import pickle


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

    def __post_init__(self):
        # Set the accident and development periods
        self.accident_periods = (
            self.rpt_loss_tri.tri.index if self.rpt_loss_tri is not None else None
        )
        self.development_periods = (
            self.rpt_loss_tri.tri.columns if self.rpt_loss_tri is not None else None
        )

        # Several aliases for the accident and development periods
        self.accident_period = (
            self.accident_periods if self.accident_periods is not None else None
        )
        self.ay = self.accident_periods if self.accident_periods is not None else None
        self.acc = self.accident_periods if self.accident_periods is not None else None
        self.origin = (
            self.accident_periods if self.accident_periods is not None else None
        )
        self.origin_periods = (
            self.accident_periods if self.accident_periods is not None else None
        )
        self.origin_period = (
            self.accident_periods if self.accident_periods is not None else None
        )

        self.development_period = (
            self.development_periods if self.development_periods is not None else None
        )
        self.dy = (
            self.development_periods if self.development_periods is not None else None
        )
        self.dev = (
            self.development_periods if self.development_periods is not None else None
        )
        self.dev_months = (
            self.development_periods if self.development_periods is not None else None
        )
        self.dev_period = (
            self.development_periods if self.development_periods is not None else None
        )


class HalfMegaModel:
    def __init__(
        self, data: MegaData, burnin: int = 1000, samples: int = 4000, chains: int = 4
    ):
        """
        Read in the data and store it in the MegaModel class. The data is stored in
        triangles that have been converted to pytorch tensors.
        """
        self.rpt_loss_tri = data.rpt_loss_tri
        self.paid_loss_tri = data.paid_loss_tri
        # self.rpt_count_tri = data.rpt_count_tri
        # self.paid_count_tri = data.paid_count_tri

        self.acc = data.rpt_loss_tri.tri.index.year.values
        self.dev = data.rpt_loss_tri.tri.columns.astype(int).values

        # loss triangles
        self.rpt_loss = self.rpt_loss_tri.tri.values
        self.paid_loss = self.paid_loss_tri.tri.values

        # need a triangle mask -- true if the value is not nan
        self.tri_mask = pd.DataFrame(
            ~np.isnan(self.rpt_loss),
            index=self.rpt_loss_tri.tri.index,
            columns=self.rpt_loss_tri.tri.columns,
        )

        # triangle stats
        self.n_rows = self.rpt_loss_tri.tri.shape[0]
        self.n_cols = self.rpt_loss_tri.tri.shape[1]

        # MCMC parameters
        self.burnin = burnin
        self.samples = samples
        self.chains = chains

        # Prior ultimate estimates
        self.loss_ult_prior = np.multiply(
            self.rpt_loss_tri.diag().values,
            self.rpt_loss_tri.atu().sort_index(ascending=False).values,
        )

        # Prior beta means
        self.prior_beta_mean = {}
        self.prior_beta_mean["rpt_loss"] = np.divide(1, self.rpt_loss_tri.atu())
        self.prior_beta_mean["paid_loss"] = np.divide(1, self.paid_loss_tri.atu())

    def prior_ultimate_distributions(
        self, name: str = "alpha", standalone: bool = True
    ) -> Tuple:
        """
        Define informative prior distributions for the latent variables. The prior
        distributions are defined as pymc deterministic variables. Doing is this way
        should capture the relationship between the different ultimate values - this
        relationship is fixed and not a random variable.

        Use normal distributions for the latent variables.
        """
        # if you don't pass lognormal parameters, they will be estimated from the data
        # if mu is None or sigma is None:

        # method of moments is used for a prior estimate
        m_l = np.sum(np.log(self.loss_ult_prior)) / self.loss_ult_prior.shape[0]
        s2_l = (
            np.sum(np.power(np.log(self.loss_ult_prior) - m_l, 2))
            / self.loss_ult_prior.shape[0]
        )
        s_l = np.sqrt(s2_l)

        # latent variables for ultimate
        if standalone:
            alpha_loss = pymc.LogNormal.dist(
                mu=m_l, sigma=s_l, shape=self.rpt_loss_tri.tri.shape[0]
            )
        else:
            _ult_loss = pymc.LogNormal(
                "latent-ult-loss",
                mu=m_l,
                sigma=s_l,
                shape=(self.rpt_loss_tri.tri.shape[0], 1),
            )

            # deterministic functions for the prior estimates of ultimates
            # it doesn't make sense to do this like this, but it mirrors the method in
            # the MegaModel class
            alpha_loss = pymc.Deterministic(f"{name}-loss", _ult_loss)

        return alpha_loss

    def prior_development_distributions(
        self,
        name: str = "beta",
        # mu: float = 0,
        # sigma: float = 5,
        standalone: bool = True,
    ) -> Tuple:
        """
        Define noninformative prior distributions for the development factors. The
        prior distributions are defined as pymc stochastic variables, and are assumed
        to be almost between 0 and 1. The development parameters are denoted as beta,
        and are the percent of total for the development period.

        Use normal distributions for the development parameters.
        """

        # Estimate prior hyperparameters from data

        # prior distributions for development
        if standalone:
            beta_rpt_loss = pymc.Normal.dist(
                mu=self.prior_beta_mean["rpt_loss"],
                sigma=np.power(self.prior_beta_mean["rpt_loss"], 1 / 3),
                size=self.dev.shape[0],
            )
            beta_paid_loss = pymc.Normal.dist(
                mu=self.prior_beta_mean["paid_loss"],
                sigma=np.power(self.prior_beta_mean["paid_loss"], 1 / 3),
                size=self.dev.shape[0],
            )
        else:
            beta_rpt_loss = pymc.Normal(
                "beta-rpt-loss",
                mu=self.prior_beta_mean["rpt_loss"],
                sigma=np.power(self.prior_beta_mean["rpt_loss"], 1 / 3),
                size=self.dev.shape[0],
            )
            beta_paid_loss = pymc.Normal(
                "beta-paid-loss",
                mu=self.prior_beta_mean["paid_loss"],
                sigma=np.power(self.prior_beta_mean["paid_loss"], 1 / 3),
                size=self.dev.shape[0],
            )

        return beta_rpt_loss, beta_paid_loss

    def prior_sigma_distributions(
        self, name: str = "sigma", beta: float = 2.5, standalone: bool = True
    ) -> Tuple:
        """
        Define noninformative prior distributions for the standard deviations of the
        alpha parameters. The prior distributions half Cauchy distributions with a
        scale parameter of 2.5. The standard deviations vary by the type of triangle
        and the development period.
        """
        # variance of each of the triangles
        if standalone:
            sigma_rpt_loss = HalfCauchy.dist(beta=beta, size=self.dev.shape[0])
            sigma_paid_loss = HalfCauchy.dist(beta=beta, size=self.dev.shape[0])
        else:
            sigma_rpt_loss = HalfCauchy(
                "sigma-rpt-loss", beta=beta, size=self.dev.shape[0]
            )
            sigma_paid_loss = HalfCauchy(
                "sigma-paid-loss", beta=beta, size=self.dev.shape[0]
            )

        return sigma_rpt_loss, sigma_paid_loss

    def _E(self, alpha=None, beta=None):
        """
        Helper function for defining the expected value of a cell in the triangle.
        The expected value is the product of the ultimate and the development factor.
        """
        assert alpha is not None, "`alpha` must be passed to _E"
        assert beta is not None, "`beta` must be passed to _E"
        return np.matmul(alpha.reshape(-1, 1), beta.reshape(1, -1))
        # return np.matmul(alpha.eval().reshape(-1, 1), beta.eval().reshape(1, -1))

    def chain_ladder_model(self):
        with pymc.Model() as model:
            # prior distributions for the ultimate parameters
            alpha_loss = self.prior_ultimate_distributions(standalone=False)

            # prior distributions for the development parameters
            beta_rpt_loss, beta_paid_loss = self.prior_development_distributions(
                standalone=False
            )

            # prior distributions for the standard deviations
            sigma_rpt_loss, sigma_paid_loss = self.prior_sigma_distributions(
                standalone=False
            )
            # sigma_rpt = np.array([sigma_rpt_loss.eval() for _ in range(self.n_rows)])[
            #     self.tri_mask
            # ]
            # sigma_paid = np.array([sigma_paid_loss.eval() for _ in range(self.n_rows)])[
            #     self.tri_mask
            # ]
            sigma_rpt = np.tile(sigma_rpt_loss, (self.n_rows, 1))[self.tri_mask]
            sigma_paid = np.tile(sigma_paid_loss, (self.n_rows, 1))[self.tri_mask]

            # expected values for the triangles
            E_rpt_loss = self._E(alpha_loss, beta_rpt_loss)[self.tri_mask]
            E_paid_loss = self._E(alpha_loss, beta_paid_loss)[self.tri_mask]

            # likelihood functions
            loglik_rpt_loss = Normal(
                "loglik-rpt-loss",
                mu=E_rpt_loss,
                sigma=sigma_rpt,
                observed=self.rpt_loss[self.tri_mask],
            )
            loglik_paid_loss = Normal(
                "loglik-paid-loss",
                mu=E_paid_loss,
                sigma=sigma_paid,
                observed=self.paid_loss[self.tri_mask],
            )
        self.model = model
        return model

    # def chain_ladder_model2(self):
    #     with pymc.Model() as model:
    #         # prior distributions for the ultimate parameters
    #         ult_ln_prior = stats.lognorm.fit(self.loss_ult_prior)
    #         alpha_loss = LogNormal(
    #             "alpha-loss", mu=np.log(ult_ln_prior[2]), sigma=ult_ln_prior[0]
    #         )

    #         # prior distributions for the development parameters
    #         beta_rpt_loss = Normal(
    #             "beta-rpt-loss",
    #             mu=self.prior_beta_mean["rpt_loss"],
    #             sigma=np.power(self.prior_beta_mean["rpt_loss"], 1 / 3),
    #             size=self.dev.shape[0],
    #         )
    #         beta_paid_loss = Normal(
    #             "beta-paid-loss",
    #             mu=self.prior_beta_mean["paid_loss"],
    #             sigma=np.power(self.prior_beta_mean["paid_loss"], 1 / 3),
    #             size=self.dev.shape[0],
    #         )

    #         # prior distributions for the standard deviations
    #         # sigma_rpt_loss, sigma_paid_loss = self.prior_sigma_distributions(
    #         #     standalone=False
    #         # )
    #         sigma_rpt_loss = HalfCauchy(
    #             "sigma-rpt-loss", beta=10, size=self.dev.shape[0]
    #         )
    #         sigma_paid_loss = HalfCauchy(
    #             "sigma-paid-loss", beta=10, size=self.dev.shape[0]
    #         )
    #         # sigma_rpt = np.array([sigma_rpt_loss.eval() for _ in range(self.n_rows)])[
    #         #     self.tri_mask
    #         # ]
    #         # sigma_paid = np.array([sigma_paid_loss.eval() for _ in range(self.n_rows)])[
    #         #     self.tri_mask
    #         # ]

    #         # expected values for the triangles

    #         E_rpt_loss, E_paid_loss = [], []
    #         for row in self.rpt_loss_tri.shape[0]:
    #             E_rpt_loss.append(
    #                 [
    #                     (
    #                         (alpha_loss[row] * beta)
    #                         if ~self.rpt_loss_tri.tri.iloc[row, i].isna()
    #                         else np.nan
    #                     )
    #                     for i, beta in enumerate(beta_rpt_loss)
    #                 ]
    #             )
    #             E_paid_loss.append(alpha_loss[row] * beta_paid_loss)

    #         # E_rpt_loss = self._E(alpha_loss, beta_rpt_loss)[self.tri_mask]
    #         # E_paid_loss = self._E(alpha_loss, beta_paid_loss)[self.tri_mask]

    #         # likelihood functions
    #         loglik_rpt_loss = Normal(
    #             "loglik-rpt-loss",
    #             mu=E_rpt_loss,
    #             sigma=sigma_rpt_loss,
    #             observed=self.rpt_loss[self.tri_mask],
    #         )
    #         loglik_paid_loss = Normal(
    #             "loglik-paid-loss",
    #             mu=E_paid_loss,
    #             sigma=sigma_paid_loss,
    #             observed=self.paid_loss[self.tri_mask],
    #         )
    #     self.model = model
    #     return model

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
            self.trace = pymc.sample(
                draws=samples, tune=burnin, chains=chains, cores=None
            )
