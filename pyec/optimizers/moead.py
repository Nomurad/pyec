import numpy as np
import random

from ..base.indiv import Individual
from ..base.population import Population
from ..base.environment import Pool, Normalizer
from ..operators.initializer import UniformInitializer

from ..operators.crossover import SimulatedBinaryCrossover
from ..operators.mutation import PolynomialMutation
from ..operators.selection import TournamentSelection, TournamentSelectionStrict
from ..operators.selection import SelectionIterator, Selector
from ..operators.mating import MatingIterator, Mating


################################################################################
# スカラー化関数
################################################################################
class ScalarError(Exception):
    pass

def scalar_weighted_sum(indiv, weight, ref_point):
    return -np.sum(weight * np.abs(indiv.wvalue - ref_point))

def scalar_chebyshev(indiv, weight, ref_point):
    if not indiv.evaluated():
        raise ScalarError("indiv not evaluated.")
    res = -np.max(weight * np.abs(indiv.wvalue - ref_point))
    return res

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
        
        self.alternation = "normal"
        self.normalize = False
        self.normalizer = None
        self.EP = []

    def __call__(self, index:int, population:Population, eval_func) -> Individual:
        return self.get_offspring(index, population, eval_func)

    def init_weight(self):
        self.weight_vec = weight_vector_generator(self.nobj, self.popsize-1)
        self.neighbers = np.array([self.get_neighber(i) for i in range(self.popsize)])
        self.ref_points = np.full(self.nobj, 'inf', dtype=np.float64)

        # print("weight vector shape:", self.weight_vec.shape)
        # print("ref point:", self.ref_points.shape)
        # print("neighber:", self.neighbers)


    def get_neighber(self, index):
        norms = np.zeros((self.weight_vec.shape[0], self.weight_vec.shape[1]+2))
        self.neighbers = np.zeros((self.weight_vec.shape[0], self.ksize))
        w1 = self.weight_vec[index]

        for i, w2 in enumerate(self.weight_vec):
            # print(i)
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
            if self.normalize is not None:
                self.ref_points = np.zeros(self.ref_points.shape, dtype=np.float)
            else:
                self.ref_points = np.min([self.ref_points, np.array(indiv.wvalue)],axis=0)
            # print("update ref point = ", self.ref_point)
        except:
            print("\n Error")
            print(self.ref_points.dtype)
            print(self.ref_points)
            print(np.array(indiv.wvalue).dtype)
            print(np.array(indiv.wvalue))
            print([self.ref_points, np.array(indiv.wvalue)])
            raise MOEADError()

    def get_offspring(self, index, population:Population, eval_func) -> Individual:
        # print(self.neighbers[index])
        subpop = [population[i] for i in self.neighbers[index]]

        for i, indiv in enumerate(subpop):
            fit_value = self.scalar(indiv, self.weight_vec[index], self.ref_points)
            indiv.set_fitness(fit_value)
        
        parents = self.selector(subpop)
        # print("len parents", len(parents))
        childs = self.mating(parents)
        child = random.choice(childs)
        idx = 0
        if all(child.genome == childs[idx]):
            idx = 1
        self.mating._pool.pop(idx)
        # print(child.evaluated(), child.value)
        child.evaluate(eval_func, (child.get_design_variable()))
        self.update_reference(child)
        if self.normalizer is not None:
            self.normalizer.normalizing(child)
            # print("ori, normalize:",child.value, child.wvalue)
            # input()
        child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

        if self.alternation is "all":
            return max(population[index], child)
        else:
            return max(*subpop, child)

    def calc_fitness(self, population):
        """population内全ての個体の適応度を計算
        """
        for indiv in population:
            self.update_reference(indiv)

        if self.normalize is True:
            values = list(map(self._get_indiv_value, population))
            values = np.array(values)
            upper = [np.max(values[:,0]), np.max(values[:,1])]
            lower = [np.min(values[:,0]), np.min(values[:,1])]
            # print("upper/lower: ",(upper), (lower))
            # input()
            if self.normalizer is None:
                self.normalizer = Normalizer(upper, lower)
            else:
                self.normalizer.ref_update(upper, lower)

            for indiv in population:
                self.normalizer.normalizing(indiv)

        for idx, indiv in enumerate(population):
            self.calc_fitness_single(indiv, idx)

    def calc_fitness_single(self, indiv:Individual, index):
        """1個体の適応度計算
        """
        fit = scalar_chebyshev(indiv, self.weight_vec[index], self.ref_points)
        indiv.set_fitness(fit)
        # print("fit:",fit)

    def _get_indiv_value(self, indiv:Individual):
        return indiv.value


class MOEAD_DE(MOEAD):
    name = "moead_de"

    def __init__(self, popsize:int, nobj:int, pool:Pool,
                    selection:Selector, mating:Mating, ksize=3):
        super().__init__(popsize, nobj, selection, mating, ksize=3)

        self.pool = pool
        self.CR = 0.9   #交叉率
        self.scaling_F = 0.5    #スケーリングファクタ--->( 0<=F<=1 )
        self.offspring_delta = 0.9 #get_offspringで交配対象にする親個体の範囲を近傍個体集団にする確率
        print(self.name)

        
    def get_offspring(self, index, population:Population, eval_func) -> Individual:
        rand = random.random()
        
        if rand >= self.offspring_delta:
            subpop = [population[i] for i in range(len(population))]
        else:
            subpop = [population[i] for i in self.neighbers[index]]

        for i, indiv in enumerate(subpop):
            fit_value = self.scalar(indiv, self.weight_vec[index], self.ref_points)
            indiv.set_fitness(fit_value)
        
        # parents = self.selector(subpop)
        parents = random.sample(subpop, 2)
        # child = Individual(np.random.rand(len(parents[0].genome)))
        child = self.pool.indiv_creator(np.random.rand(len(parents[0].genome)))

        rand = random.random()
        if rand < self.CR:
            lower, upper = parents[0].bounds
            de = self.scaling_F*(parents[0]-parents[1])
            child_dv = population[index] + de
            for i,dv in enumerate(child_dv):
                if dv < lower or dv > upper:
                    child_dv[i] = random.random()*(upper-lower)+lower

            # print("child_dv:", (child_dv))
            child.encode(child_dv)
        else:
            child = population[index]

        mutate_genome = self.mating._mutation(child.get_genome())
        child.set_genome(mutate_genome)

        child.evaluate(eval_func, (child.get_design_variable()))
        # print(child.evaluated(), child.value)
        if self.normalizer is not None:
            self.normalizer.normalizing(child)
        self.update_reference(child)
        child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

        if self.alternation is "all":
            return max(population[index], child)
        else:
            return max(*subpop, child)


class C_MOEAD_DE(MOEAD_DE):

    def __init__(self, popsize:int, nobj:int, pool:Pool,
                    selection:Selector, mating:Mating, ksize=3):
        super().__init__(popsize, nobj, pool, selection, mating, ksize=3)

    def get_offspring(self, index, population:Population, eval_func) -> Individual:
        rand = random.random()
        
        if rand >= self.offspring_delta:
            subpop = [population[i] for i in range(len(population))]
        else:
            subpop = [population[i] for i in self.neighbers[index]]

        for i, indiv in enumerate(subpop):
            fit_value = self.scalar(indiv, self.weight_vec[index], self.ref_points)
            indiv.set_fitness(fit_value)
        
        # parents = self.selector(subpop)
        parents = random.sample(subpop, 2)
        # child = Individual(np.random.rand(len(parents[0].genome)))
        child = self.pool.indiv_creator(np.random.rand(len(parents[0].genome)))

        rand = random.random()
        if rand < self.CR:
            lower, upper = parents[0].bounds
            de = self.scaling_F*(parents[0]-parents[1])
            child_dv = population[index] + de
            for i,dv in enumerate(child_dv):
                if dv < lower or dv > upper:
                    child_dv[i] = random.random()*(upper-lower)+lower

            # print("child_dv:", (child_dv))
            child.encode(child_dv)
        else:
            child = population[index]

        mutate_genome = self.mating._mutation(child.get_genome())
        child.set_genome(mutate_genome)

        child.evaluate(eval_func, (child.get_design_variable()))
        # print(child.evaluated(), child.value)
        if self.normalizer is not None:
            self.normalizer.normalizing(child)
        self.update_reference(child)
        child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

        if self.alternation is "all":
            return max(population[index], child)
        else:
            return max(*subpop, child)