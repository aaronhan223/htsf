import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from preprocess import r_snippet
import numpy as np


def get_hts(size):
    if size == 'small':
        Matrix = importr('Matrix')
        MASS = importr('MASS')
        Diagonal = Matrix.Diagonal
        mvrnorm = MASS.mvrnorm
        sim = r_snippet.sim()
        powerpack = SignatureTranslatedAnonymousPackage(sim, "powerpack")
        bts, A = powerpack.simulate_hts(500)[0], powerpack.simulate_hts(500)[1]
    else:
        Matrix = importr('Matrix')
        hts = importr('hts')
        mvtnorm = importr('mvtnorm')
        bdiag = Matrix.bdiag
        rmvnorm = mvtnorm.rmvnorm
        hts = hts.hts
        sim_large = r_snippet.sim_large()
        powerpack = SignatureTranslatedAnonymousPackage(sim_large, "powerpack")
        A, bts = powerpack.simulate_large_hts(500)[0], powerpack.simulate_large_hts(500)[1]

    bights = r_snippet.bight()
    powerpack = SignatureTranslatedAnonymousPackage(bights, "powerpack")
    my_bights = powerpack.bights(bts, A)
    total_ts, S, nbts = np.array(my_bights[0]), np.array(my_bights[2]), my_bights[3]
    return total_ts, S, nbts


def simulate(size):
    package_names = ['stats', 'curl', 'MASS', 'mvtnorm', 'hts', 'magic', 'matrixcalc', 'Matrix']
    if all(rpackages.isinstalled(x) for x in package_names):
        have_package = True
    else:
        have_package = False

    if not have_package:
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)

        package_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        print(package_to_install)
        if len(package_to_install) > 0:
            utils.install_packages(StrVector(package_to_install))
    total_ts, S, nbts = get_hts(size=size)

    return total_ts, S, nbts

