import ChainLadder_MCMC as clmc
import triangle as tr

t = tr.Triangle.from_taylor_ashe()
data = clmc.MegaData(triangle=t)

cl = clmc.ChainLadderMCMC(data=data)
