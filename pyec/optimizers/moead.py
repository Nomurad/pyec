import random
import copy
import numpy as np

from ..base.indiv import Individual
from ..base.population import Population
from ..base.environment import Pool, Normalizer
from ..operators.initializer import UniformInitializer

from ..operators.crossover import SimulatedBinaryCrossover, DifferrentialEvolutonary_Crossover
from ..operators.mutation import PolynomialMutation
from ..operators.selection import TournamentSelection, TournamentSelectionStrict
from ..operators.selection import SelectionIterator, Selector
from ..operators.mating import MatingIterator, Mating
from ..operators.sorting import NonDominatedSort


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
        self.division = popsize
        self.popsize = popsize
        self.nobj = nobj
        self.ksize = ksize
        self.ref_points = []
        self.selector = selection
        self.mating = mating
        self.sorting = NonDominatedSort()
        self.scalar = scalar_chebyshev
        # self.scalar = scalar_weighted_sum
        print("scalar func is ", self.scalar)
        self.init_weight()
        
        self.alternation = "normal"
        self.normalize = False
        self.normalizer = None
        self.EP = []
        self.n_EPupdate = 0

    def __call__(self, index:int, population:Population, eval_func) -> Individual:
        child = self.get_offspring(index, population, eval_func)
        return child

    def init_weight(self):
        self.weight_vec = weight_vector_generator(self.nobj, self.popsize-1)
        self.popsize = len(self.weight_vec)
        print(f"popsize update -> {self.popsize}")
        # print([np.linalg.norm(n) for n in self.weight_vec])
        
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
        
        print(index, "neighbers_index", neighber_index)
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

    def get_offspring(self, index:int, population:Population, eval_func) -> Individual:
        # print(self.neighbers[index])
        subpop = [population[i] for i in self.neighbers[index]]

        for i, indiv in enumerate(subpop):
            self.calc_fitness_single(subpop[i], index)
            # fit_value = self.scalar(indiv, self.weight_vec[index], self.ref_points)
            # indiv.set_fitness(fit_value)
        
        parents = self.selector(subpop)
        # print("len parents", parents)
        # print("id_s", [p.get_id() for p in parents])
        childs = self.mating(parents)
        child = random.choice(childs)
        idx = 0
        if all(child.genome == childs[idx]):
            idx = 1
        self.mating.pool.pop(childs[idx].get_id())
        # print(child.evaluated(), child.value)
        child.evaluate(eval_func, (child.get_design_variable()))
        self.update_reference(child)
        if self.normalizer is not None:
            self.normalizer.normalizing(child)
            # print("ori, normalize:",child.value, child.wvalue)
            # input()
        child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

        # population[index] = res

        nr = int(len(subpop))
        res = self._alternate(child, nr, index, population, subpop)
        
        # if res == None:
        #     return population[index]
            
        # if res.get_id() == child.get_id():
        #     self.update_EP(res)
        #     self.n_EPupdate += 1
        
        return res

    def _alternate(self, child, nr, index, population, subpop):
        res = None

        neighber = self.neighbers[index]

        rands = list(range(len(subpop)))
        random.shuffle(rands)
        for c in range(nr):
            j2 = rands[c]%self.ksize
            j = int(neighber[j2])
            old_indiv = population[j]
            population[j] = max(old_indiv, child)
            if child.id == population[j].id:
                res = child
            # w = self.weight_vec[j]
            # fit1 = self.scalar(child, w, self.ref_points)
            # if fit1 > old_indiv.fitness.fitness:
            #     population[j] = child
            #     res = child

        # if self.alternation is "all":
        #     res = max(population[index], child)
        #     population[index] = res
        # else:
        #     res = max(*subpop, child)
        #     population[index] = res

        # if res == child:
        #     print("child ")
        if res == None:
            res = population[index]
        # print(res.fitness.fitness, " | ", child.fitness.fitness)

        return res

    def update_EP(self, indiv:Individual):
        # print("EP append")
        self.EP.append(indiv)
        if len(self.EP) > 2:
            # self.EP = self.sorting.sort(self.EP)[0]
            self.EP = self.sorting.output_pareto(self.EP)

    def calc_fitness(self, population):
        """population内全ての個体の適応度を計算
        """
        for indiv in population:
            self.update_reference(indiv)

        if self.normalize is True:
            values = list(map(self._get_indiv_value, population))
            values = np.array(values)
            # print(values.shape)
            # upper = [np.max(values[:,0]), np.max(values[:,1])]
            # lower = [np.min(values[:,0]), np.min(values[:,1])]
            upper = [np.max(values[:, idx]) for idx in range(values.shape[-1])]
            lower = [np.min(values[:, idx]) for idx in range(values.shape[-1])]
            print("upper/lower: ",(upper), (lower))
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
        fit = self.scalar(indiv, self.weight_vec[index], self.ref_points)
        indiv.set_fitness(fit)
        # print("fit:",fit)

    def _get_indiv_value(self, indiv:Individual):
        return indiv.value


class MOEAD_DE(MOEAD):
    name = "moead_de"

    def __init__(self, popsize:int, nobj:int,
                    selection:Selector, mating:Mating, ksize=3,
                    CR=0.9, F=0.7, eta=20
                ):
        super().__init__(popsize, nobj, selection, mating, ksize=ksize)

        self.pool = mating.pool
        self.CR = CR   #交叉率
        self.scaling_F = F    #スケーリングファクタ ->( 0<=F<=1 )
        self.pm = self.mating._mutation.rate
        # self.pm = 1.0/len(self.ref_points)
        self.eta = self.mating._mutation.eta
        self.offspring_delta = 0.9 #get_offspringで交配対象にする親個体の範囲を近傍個体集団にする確率
        self.crossover = \
            DifferrentialEvolutonary_Crossover(
                    self.CR,
                    self.scaling_F,
                    self.pm,
                    self.eta
                )

    def _set_parents(self):
        pass

    def get_offspring(self, index, population:Population, eval_func) -> Individual:
        rand = random.uniform(0.0, 1.0)
        
        if rand >= self.offspring_delta:
            # subpop = [population[i] for i in range(len(population))]
            subpop = list(population)
        else:
            subpop = [population[i] for i in self.neighbers[index]]

        for i, indiv in enumerate(subpop):
            fit_value = self.scalar(indiv, self.weight_vec[index], self.ref_points)
            subpop[i].set_fitness(fit_value)
        
        # parents = self.selector(subpop)
        parents = random.sample(subpop, 2)
        # child = Individual(np.random.rand(len(parents[0].genome)))
        child = self.mating.pool.indiv_creator(np.random.rand(len(parents[0].genome)))
        
        p1 = population[index].get_genome()
        p2 = parents[0].get_genome()
        p3 = parents[1].get_genome()
        # if (parents[0].dominate(parents[1])):
        #     p2 = parents[0].get_genome()
        #     p3 = parents[1].get_genome()
        # else:
        #     p2 = parents[1].get_genome()
        #     p3 = parents[0].get_genome()
            
        self.pm = 1.0/len(p1)
        child_dv = self.crossover([p1, p2, p3])

        # print("child_dv:", (child_dv))
        child.set_genome(child_dv)
        child.set_boundary(parents[0].bounds)
        child.set_weight(parents[0].weight)

        child.evaluate(eval_func, (child.get_design_variable()))
        # print(child.evaluated(), child.value)
        if self.normalizer is not None:
            self.normalizer.normalizing(child)
        self.update_reference(child)
        child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

        # print(population)
        nr = int(len(subpop)/2)
        nr = 2
        res = self._alternate(child, nr, index, population, subpop)

                # for i, indiv in enumerate(population):
                #     if old_indiv.get_id() == indiv.get_id():
                #         # print("replace")
                #         population[i] = child
                #         res.append(child)
        
        # if res == None:
        #     return population[index]
        
        # if res.get_id() == child.get_id():
        #     self.update_EP(res)
        #     self.n_EPupdate += 1

        return res


class C_MOEAD(MOEAD):
    name = "c_moead"

    def __init__(self, popsize:int, nobj:int, pool:Pool, n_constraint:int,
                    selection:Selector, mating:Mating, ksize=3):
        super().__init__(popsize, nobj, selection, mating, ksize=3)
        self.n_constraint = n_constraint
        self.CVsort = NonDominatedSort()    #CVsort:constraint violation sort

    def get_offspring(self, index, population:Population, eval_func) -> Individual:
        
        subpop = [population[i] for i in self.neighbers[index]]

        for i, indiv in enumerate(subpop):
            self.calc_fitness_single(subpop[i], index)

        # rand = random.random()
        # if rand >= self.offspring_delta:
        #     subpop = [population[i] for i in range(len(population))]
        # else:
        #     subpop = [population[i] for i in self.neighbers[index]]

        # for i, indiv in enumerate(subpop):
        #     fit_value = self.scalar(indiv, self.weight_vec[index], self.ref_points)
        #     indiv.set_fitness(fit_value, self)
        
        parents = self.selector(subpop)
        childs = self.mating(parents)
        child = random.choice(childs)
        child.evaluate(eval_func, (child.get_design_variable()))
        self.update_reference(child)
        
        # feasible child solutions
        c_feasibles = [s for s in childs in s.constraint_violation == 0.0]
        # c_infeasibles = [s for s in childs in s.constraint_violation != 0.0]
        
        if len(c_feasibles) > 0:
            child = random.choice(childs)
            # solutions = parents + [child]
            idx = 0
            if all(child.genome == childs[idx].genome):
                idx = 1 
            self.mating.pool.pop(childs[idx].id)
        
        elif len(c_feasibles) == 0:
            self.mating.pool.pop(childs[0].id)
            self.mating.pool.pop(childs[1].id)

        # child = self.pool.indiv_creator(np.random.rand(len(parents[0].genome)))
        child.evaluate(eval_func, (child.get_design_variable()))
        self.update_reference(child)
        if self.normalizer is not None:
            self.normalizer.normalizing(child)

        child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

        nr = int(len(subpop))
        res = self._alternate(child, nr, index, population, subpop)

        return res 