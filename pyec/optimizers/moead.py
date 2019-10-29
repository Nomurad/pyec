import numpy as np

from ..base.indiv import Individual
from ..base.population import Population
from ..operators.initializer import UniformInitializer

from ..operators.crossover import SimulatedBinaryCrossover
from ..operators.selection import TournamentSelection, TournamentSelectionStrict
from ..operators.selection import SelectionIterator
from ..operators.mating import MatingIterator


################################################################################
# スカラー化関数
################################################################################
def scalar_weighted_sum(indiv, weight, ref_point):
    return -np.sum(weight * np.abs(indiv.wvalue - ref_point))

def scalar_chebyshev(indiv, weight, ref_point):
    return -np.max(weight * np.abs(indiv.wvalue - ref_point))

def scalar_boundaryintersection(indiv, weight, ref_point):
    ''' norm(weight) == 1
    '''
    nweight = weight / np.linalg.norm(weight)

    bi_theta = 5.0
    d1 = np.abs(np.dot((indiv.wvalue - ref_point), nweight))
    d2 = np.linalg.norm(indiv.wvalue - (ref_point - d1 * nweight))
    return -(d1 + bi_theta * d2)

################################################################################

class MOEAD(object):
    """MOEA/D

    """

    def __init__(self, popsize, problem):
        self.popsize = 10
        self.weight = None

    def calc_fitness(self):
        scalar_chebyshev(self.weight)