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
def weight_vector_generator(nobj=None, divisions=None, coeff=1):
    import copy

    if not nobj:
        nobj = self.nobj
    if not divisions:
        divisions = self.popsize
    if coeff:
        divisions = divisions*coeff
    
    # if nobj == 2:
    #     weights = [[1,0],[0,1]]
    #     weights.extend([(i/(divisions-1.0), 1.0-i/(divisions-1.0)) 
    #                                     for i in range(1, divisions-1)])
    # else:
    weight_vectors = []
    # ele_candidate = np.array(list(range(popsize+1)))/popsize

    def weight_recursive(weight_vectors, weight, left, total, idx=0):

        if idx == nobj-1:
            weight[idx] = float(left)/float(total)
            weight_vectors.append(copy.copy(weight))
            # return weight_vectors
        else:
            for i in range(left+1):
                weight[idx] = float(i)/float(total)
                weight_recursive(weight_vectors, weight, left-i, total, idx+1)

    weight_recursive(weight_vectors, [0.0]*nobj, divisions, divisions)

    weight_vectors = np.array(weight_vectors)
    # np.savetxt("temp.txt", weight_vectors, fmt='%.2f', delimiter='\t')
    return weight_vectors

################################################################################

class MOEAD(object):
    """MOEA/D

    """

    def __init__(self, popsize, problem, ksize=3):
        self.popsize = 10
        self.ref_points = []
        self.weight_vec = None

    def calc_fitness(self, indiv:Individual):
        scalar_chebyshev(indiv, self.weight_vec, self.ref_points)

