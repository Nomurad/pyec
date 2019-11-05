import numpy as np
import random

from ..base.indiv import Individual
from ..base.population import Population
from ..operators.initializer import UniformInitializer

from ..operators.crossover import SimulatedBinaryCrossover
from ..operators.mutation import PolynomialMutation
from ..operators.selection import TournamentSelection, TournamentSelectionStrict
from ..operators.selection import SelectionIterator, Selector
from ..operators.mating import MatingIterator, Mating


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
def weight_vector_generator(nobj, divisions, coeff=1):
    import copy

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
class MOEADError(Exception):
    pass


class MOEAD(object):
    """MOEA/D

    """
    name = "moead"

    def __init__(self, popsize:int, nobj:int,
                    selection:Selector, mating:Mating, ksize=3 ):
        self.popsize = popsize
        self.nobj = nobj
        self.ksize = ksize
        self.ref_points = []
        self.selector = selection
        self.mating = mating
        self.scalar = scalar_chebyshev
        self.init_weight()

    def __call__(self):
        pass

    def init_weight(self):
        self.weight_vec = weight_vector_generator(self.nobj, self.popsize)
        self.neighbers = np.array([self.get_neighber(i) for i in range(self.popsize)])
        self.ref_points = np.full(self.nobj, 'inf', dtype=np.float64)


    def get_neighber(self, index):
        norms = np.zeros((self.weight_vec.shape[0], self.weight_vec.shape[1]+2))
        self.neighbers = np.zeros((self.weight_vec.shape[0], self.ksize))
        w1 = self.weight_vec[index]

        for i, w2 in enumerate(self.weight):
            norms[i,0] = np.linalg.norm(w1 - w2)
            norms[i,1] = i
            norms[i,2:] = w2

        norms_sort = norms[norms[:,0].argsort(),:]  #normの大きさでnormsをソート
        # print(norms)
        neighber_index = np.zeros((self.ksize), dtype="int")
        for i in range(self.ksize):
            neighber_index[i] = norms_sort[i,1]
        
        # print(neighber_index)
        return neighber_index

    def update_reference(self, indiv:Individual):
        try:
            self.ref_point = np.min([self.ref_point, np.array(indiv.wvalue)],axis=0)
            # print("update ref point = ", self.ref_point)
        except:
            print(self.ref_point.dtype)
            print(self.ref_point)
            print(np.array(indiv.wvalue).dtype)
            print(np.array(indiv.wvalue))
            print([self.ref_point, np.array(indiv.wvalue)])
            raise MOEADError()

    def get_offspring(self, index, population:Population):
        subpop = [population[i] for i in self.weight_vec]

        for i, indiv in enumerate(subpop):
            fit_value = self.scalar(indiv, self.weight_vec, self.ref_points)
            indiv.set_fitness(fit_value)
        
        parents = self.selector(subpop)
        child = random.choice(self.mating(parents))
        child.set_fitness(self.scalar(child, self.weight_vec, self.ref_points))

        return max(population[index], child)

    def calc_fitness(self, population):
        """population内全ての個体の適応度を計算
        """
        for indiv in population:
            self.calc_fitness_single(indiv)

    def calc_fitness_single(self, indiv:Individual):
        """1個体の適応度計算
        """
        fit = scalar_chebyshev(indiv, self.weight_vec, self.ref_points)
        indiv.set_fitness(fit)

