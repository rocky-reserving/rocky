# import ChainLadder_MCMC as clmc

import triangle as tr
import numpy as np

t = tr.Triangle.from_taylor_ashe()
data = MegaData(triangle=t)

cl = MegaChainLadder(data=data)

alpha = cl.prior_ultimate_distributions()
beta = cl.prior_development_distributions()
sigma = cl.prior_sigma_distributions()
E = cl._E(alpha, beta)

Ga = cl._gamma_alpha(E, sigma)
Gb = cl._gamma_beta(E, sigma)

cl.chain_ladder_model()

