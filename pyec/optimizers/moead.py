import random
import copy
from itertools import chain
from typing import List, Tuple, Union, Optional
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
from ..operators.scalaring import *

from . import Optimizer, OptimizerError, Solution_archive
################################################################################


def weight_vector_generator(n_obj, divisions, coeff=1):

    if coeff: 
        divisions = divisions*coeff

    # if n_obj == 2: 
    #     weights = [[1,0],[0,1]]
    #     weights.extend([(i/(divisions-1.0), 1.0-i/(divisions-1.0)) 
    #                                     for i in range(1, divisions-1)])
    # else:
    weight_vectors = []

    def weight_recursive(weight_vectors, weight, left, total, idx=0):

        if idx == n_obj-1: 
            weight[idx] = float(left)/float(total)
            weight_vectors.append(copy.copy(weight))
            # return weight_vectors
        else:
            for i in range(left+1):
                weight[idx] = float(i)/float(total)
                weight_recursive(weight_vectors, weight, left-i, total, idx+1)

    weight_recursive(weight_vectors, [0.0]*n_obj, divisions, divisions)

    weight_vectors = np.array(weight_vectors)
    # np.savetxt("temp.txt", weight_vectors, fmt='%.2f', delimiter='\t')
    return weight_vectors

################################################################################


class MOEADError(OptimizerError):
    pass


class MOEAD(Optimizer):
    """MOEA/D

    """
    name = "moead"

    def __init__(self, popsize: int, n_obj: int,
                 selection: Selector, mating: Mating, ksize=3, **kwargs):
        super().__init__(popsize, n_obj)
        self.division = popsize
        # self.popsize = popsize
        # self.n_obj = n_obj
        self.ksize = ksize
        self.ref_points: List = []
        self.min_or_max = np.zeros(n_obj, dtype=int)
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
        self.normalize_option = None
        self.normalize_only_feasible = kwargs.get("feasible_only")
        self.sort = NonDominatedSort()
        self.EP: List = []
        self.EPappend = self.EP.append
        self.EPpop = self.EP.pop
        self.n_EPupdate = 0

    def __call__(self, index: int, population: Population, eval_func) -> Individual: 
        child = self.get_offspring(index, population, eval_func)
        return child

    def init_weight(self):
        self.weight_vec = weight_vector_generator(self.n_obj, self.popsize-1)
        self.popsize = len(self.weight_vec)
        print(f"popsize update -> {self.popsize}")
        # print([np.linalg.norm(n) for n in self.weight_vec])

        self.neighbers = np.array([self.get_neighber(i) for i in range(self.popsize)])
        self.ref_points = np.full(self.n_obj, 'inf', dtype=np.float64)

        # print("weight vector shape: ", self.weight_vec.shape)
        # print("ref point: ", self.ref_points.shape)
        # print("neighber: ", self.neighbers)

    def get_neighber(self, index):
        norms = np.zeros((self.weight_vec.shape[0], self.weight_vec.shape[1]+2))
        self.neighbers = np.zeros((self.weight_vec.shape[0], self.ksize))
        w1 = self.weight_vec[index]

        for i, w2 in enumerate(self.weight_vec):
            # print(i)
            norms[i, 0] = np.linalg.norm(w1 - w2)
            norms[i, 1] = i
            norms[i, 2:] = w2

        norms_sort = norms[norms[:, 0].argsort(), :]   # normの大きさでnormsをソート
        # print(norms)
        neighber_index = np.zeros((self.ksize), dtype="int")
        for i in range(self.ksize):
            neighber_index[i] = norms_sort[i, 1]

        # print(index, "neighbers_index", neighber_index)
        return neighber_index

    def init_normalize(self, is_normalize: bool, option="unhold"):
        self.normalize = is_normalize
        self.normalize_option = option

    def update_reference(self, indiv: Individual):

        if self.normalize is not False: 
            if self.min_or_max[0] == 0: 
                eps = 1e-16
                self.min_or_max = np.array([int(wval/abs(wval + eps)) for wval in indiv.wvalue], dtype=int)
                self.ref_points = np.zeros(self.ref_points.shape, dtype=np.float)
                # self.ref_points[self.ref_points > self.min_or_max] = 1.0
                # self.ref_points[self.ref_points <= self.min_or_max] = 0.0
            # for i in range(len(self.ref_points)):
            #     if self.min_or_max[i] < 0: 
            #         self.ref_points[i] = 1.0
        else:
            wvals = np.array(indiv.wvalue)
            self.ref_points = np.min([self.ref_points, wvals], axis=0)
        # print("update ref point = ", self.ref_point)

    def get_new_generation(self, population: Population, eval_func) -> Population:
        for i, indiv in enumerate(population):
            child = self.get_offspring(i, population, eval_func)

        self.calc_fitness(population)
        return population

    def get_offspring(self, index: int, population: Population, eval_func) -> Individual: 
        # print(self.neighbers[index])
        subpop = [population[i] for i in self.neighbers[index]]

        for i, _ in enumerate(subpop):
            self.calc_fitness_single(subpop[i], index)
            # fit_value = self.scalar(indiv, self.weight_vec[index], self.ref_points)
            # indiv.set_fitness(fit_value)

        parents = self.selector(subpop)
        # print("len parents", parents)
        # print("id_s", [p.get_id() for p in parents])
        childs = self.mating(parents)
        # child = random.choice(childs)
        child = childs[0]
        idx = 0
        if all(child.genome == childs[idx]):
            idx = 1
        self.mating.pool.pop(childs[idx].get_id())
        # print(child.evaluated(), child.value)
        child.evaluate(eval_func, (child.get_design_variable()))
        self.update_reference(child)
        if self.normalizer is not None: 
            self.normalizer.normalizing(child)
            # print("ori, normalize: ",child.value, child.wvalue)
            # input()
        child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

        # population[index] = res

        nr = int(len(subpop))
        nr = 2
        res = self._alternate(child, nr, index, population, subpop)

        # if res == None: 
        #     return population[index]

        if res.get_id() == child.get_id():
            self.update_EP(res)
            self.n_EPupdate += 1

        return res

    def _alternate(self, child, nr, index, population, subpop):
        res = None

        neighber = self.neighbers[index]

        rands = list(range(len(subpop)))
        random.shuffle(rands)
        for c in range(nr):
            j2 = rands[c] % self.ksize
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
        if res is None:
            res = population[index]
        # print(res.fitness.fitness, " | ", child.fitness.fitness)

        return res

    def update_allEP(self, population):
        tmppop = self.EP + list(population)
        front = self.sort.output_pareto([indiv for indiv in tmppop if indiv.is_feasible])
        self.EP = front

    def update_EP(self, indiv: Individual):
        # print("EP append")
        if not indiv.is_feasible():
            return

        if len(self.EP) > 2:
            # tmpEP = []
            # self.EP.sort()
            for i in reversed(range(len(self.EP))):
                if all(indiv.wvalue == self.EP[i].wvalue):
                    return
                
                if indiv.dominate(self.EP[i]):
                    self.EPpop(i)
                elif self.EP[i].dominate(indiv):
                    return
                # else:
                #     tmpEP = self.EP.pop(i)

            # self.EP.append(indiv)
            # tmpEP = copy.deepcopy(self.EP)
            # poplist = [-1]*len(self.EP)
            # for i in reversed(range(len(tmpEP))):
            #     # if indiv.id != tmpEP[i].id and indiv.dominate(tmpEP[i]):
            #     if indiv.dominate(tmpEP[i]):
            #         poplist[i] = (tmpEP[i].id)
            # # self.EP = self.sorting.sort(self.EP)[0]
            # # self.EP = self.sorting.output_pareto(self.EP)
            # for i in reversed(range(len(self.EP))):
            #     # print(i)
            #     if self.EP[i].id == poplist[i]: 
            #         self.EP.pop(i)

            self.EPappend(indiv)

        else:
            self.EPappend(indiv)

    def calc_fitness(self, population):
        """population内全ての個体の適応度を計算
        """
        for indiv in population: 
            self.update_reference(indiv)

        if self.normalize is True: 
            if self.normalize_only_feasible:
                subpop = [indiv for indiv in population if indiv.is_feasible()]
            else:
                subpop = population
            values = list(map(self._get_indiv_value, subpop))
            values = np.array(values)
            # print(values.shape)
            # upper = [np.max(values[: ,0]), np.max(values[: ,1])]
            # lower = [np.min(values[: ,0]), np.min(values[: ,1])]
            upper = [np.max(values[:, idx]) for idx in range(values.shape[-1])]
            lower = [np.min(values[:, idx]) for idx in range(values.shape[-1])]
            print("upper/lower: ", (upper), (lower))
            # input()
            if self.normalizer is None: 
                self.normalizer = Normalizer(upper, lower, self.normalize_option)
            else:
                self.normalizer.ref_update(upper, lower)

            for indiv in population: 
                self.normalizer.normalizing(indiv)

        for idx, indiv in enumerate(population):
            self.calc_fitness_single(indiv, idx)

    def calc_fitness_single(self, indiv: Individual, index):
        """1個体の適応度計算
        """
        fit = self.scalar(indiv, self.weight_vec[index], self.ref_points)
        indiv.set_fitness(fit)
        # print("fit: ",fit)

    def _get_indiv_value(self, indiv: Individual):
        return indiv.value


class MOEAD_DE(MOEAD):
    name = "moead_de"

    def __init__(self, popsize: int, n_obj: int,
                 selection: Selector, mating: Mating, ksize=3,
                 CR=0.9, F=0.7, eta=20
                 ):
        super().__init__(popsize, n_obj, selection, mating, ksize=ksize)

        self.pool = mating.pool
        self.CR = CR   # 交叉率
        self.scaling_F = F   # スケーリングファクタ ->( 0<=F<=1 )
        self.pm = self.mating._mutation.rate
        # self.pm = 1.0/len(self.ref_points)
        self.eta = self.mating._mutation.eta
        self.offspring_delta = 0.9  # get_offspringで交配対象にする親個体の範囲を近傍個体集団にする確率
        self.crossover = \
            DifferrentialEvolutonary_Crossover(
                self.CR,
                self.scaling_F,
                self.pm,
                self.eta
            )

    def _set_parents(self):
        pass

    def get_offspring(self, index, population: Population, eval_func) -> Individual: 
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

        # print("child_dv: ", (child_dv))
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

        if res.get_id() == child.get_id():
            self.update_EP(res)
            self.n_EPupdate += 1

        return res


class C_MOEAD(MOEAD):
    name = "c_moead"

    def __init__(self, popsize: int, n_obj: int, selection: Selector, mating: Mating, 
                 pool: Pool, n_constraint: int, ksize=3, **kwargs):
        super().__init__(popsize, n_obj, selection, mating, ksize=ksize, **kwargs)
        # self.scalar = scalar_chebyshev
        # self.scalar = scalar_chebyshev_for_maximize
        # self.scalar = scalar_weighted_sum
        self.n_constraint = n_constraint
        self.CVsort = NonDominatedSort()   # CVsort: constraint violation sort
        self.fesible_indivs: List = []

    def CVcheck(self, indiv: Individual) -> bool: 
        """ if indiv is feasible => True
            else => False
        """

        cv = indiv.constraint_violation
        if hasattr(cv, "__iter__"):
            cv = sum([max(_cv, 0) for _cv in cv])

        if cv <= 0.0: 
            return True
        else:
            return False

    def get_offspring(self, index, population: Population, eval_func) -> Individual: 

        subpop = [population[i] for i in self.neighbers[index]]

        for i, _ in enumerate(subpop):
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
        # child = random.choice(childs)
        child = childs[0]
        res, _ = child.evaluate(eval_func, child.get_design_variable(), self.n_constraint)

        self.update_reference(child)

        # feasible child solutions
        # p_feasibles = [p for p in parents if p.constraint_violation <= 0.0]
        # c_feasibles = [s for s in childs if s.constraint_violation <= 0.0]
        # c_infeasibles = [s for s in childs in s.constraint_violation != 0.0]

        parent = population[index]
        if child.is_feasible():
            if parent.is_feasible():
                self.update_reference(child)
                if self.normalizer is not None: 
                    self.normalizer.normalizing(child)

                child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

                # nr = int(len(subpop))
                nr = int(len(subpop)/2)
                res = self._alternate(child, nr, index, population, subpop)
            else:
                population[index] = child
                res = child
        else:
            if child.cv_sum < parent.cv_sum: 
                population[index] = child
                res = child 
            else:
                res = parent

        # child = self.pool.indiv_creator(np.random.rand(len(parents[0].genome)))
        # child.evaluate(eval_func, (child.get_design_variable()), self.n_constraint)

        if res.get_id() == child.get_id():
            self.update_EP(res)
            self.n_EPupdate += 1

        return res 


class C_MOEAD_DMA(C_MOEAD):
    name = "c_moead_dma"

    def __init__(self, popsize: int, n_obj: int, selection: Selector, mating: Mating, 
                 pool: Pool, n_constraint: int, ksize=3, alpha=4, **kwargs):
        """ alpha is archive size(int).
        """

        super().__init__(popsize, n_obj, selection, mating, 
                         pool, n_constraint, ksize, **kwargs)

        self.archive_size = alpha
        self.archives = Solution_archive(len(self.weight_vec), self.archive_size)
        # self.scalar = scalar_chebyshev_for_maximize

        # default cmoea/d-dma's crossover operator & mutation operator
        # if "cross_rate_dm" in kwargs: 
        #     self.cross_rate_dm = kwargs.get("cross_rate_dm", 1.0)
        print("in moead cross_rate_dm: ", kwargs)
        self.cross_rate_dm = kwargs.get("cross_rate_dm", 1.0)
        rate_cross = self.mating._crossover.rate
        rate_mutate = self.mating._mutation.rate
        self.mating._crossover = SimulatedBinaryCrossover(rate_cross, 20)
        self.mating._mutation = PolynomialMutation(rate_mutate, 20)
        # print("init ", C_MOEAD_DMA.name)

    def get_offspring(self, index: int, population: Population, eval_func) -> Individual: 
        subpop = [population[i] for i in self.neighbers[index]]

        # select x^i as a parent
        parents = [population[index]]

        archive_size = self.archives.get_archive_size(index)
        # print("archive size", archive_size)
        if(parents[0].is_feasible()) and (archive_size > 0) and \
          (random.random() <= self.cross_rate_dm):
            pb_idx = random.randint(0, archive_size-1)
            parents.append(self.archives[index][pb_idx])
        else:
            pb_idx = random.randint(1, len(self.neighbers[index])-1)
            # subpop2 = [p for p in subpop]
            # subpop2.pop(0)
            parents.append(subpop[pb_idx])

        # print("Pa and Pb: ",[i.id for i in parents])

        child = self._SBXmating(parents, eval_func, index)

        nr = len(subpop)
        # nr = 2
        res = self.update_archives_and_alternate(child, index, subpop, population, nr=nr)
        # if res.id != parents[0].id: 
        #     # print(res.constraint_violation)
        #     population[index] = res
        if res.get_id() == child.get_id():
            self.update_EP(res)
            self.n_EPupdate += 1

        return res

    def _SBXmating(self, parents, eval_func, index) -> Individual:
        childs = self.mating(parents, singlemode=True)
        child = childs[0]
        # idx = random.randint(0, 1)
        # child = childs[idx]
        # # child: Individual = random.choice(childs)
        # if idx == 0: 
        #     idx = 1
        # else:
        #     idx = 0
        # self.mating.pool.pop(childs[idx].get_id())

        child.evaluate(eval_func, child.get_design_variable(), self.n_constraint)

        child = self._child_normalize(child)
        self.update_reference(child)
        child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

        return child

    def _child_normalize(self, child) -> Individual: 
        if self.normalizer is not None: 
            self.normalizer.normalizing(child)
        return child

    def update_archives_and_alternate(self,
                                      child: Individual,
                                      index: int,
                                      subpop: List[Individual],
                                      population: Population,
                                      nr: Optional[int] = None
                                      ):
        """ update archives & solution alternating.

        """

        res = population[index]  # parent
        neighber = list(self.neighbers[index].copy())
        random.shuffle(neighber)
        # neighber = random.sample(self.neighbers[index], self.ksize)

        c_fit = child.fitness.fitness
        xj_fits = [0]*len(subpop)
        wvec = self.weight_vec[index]

        if nr is None:
            nr = 2

        for j, xj in enumerate(subpop):
            if j > nr: 
                break

            if len(subpop) < len(population):
                nei_idx = neighber[j]
            else:
                nei_idx = j

            xj = population[nei_idx]
            xj_fits[j] = self.scalar(xj, wvec, self.ref_points)

            # x^j is feasible & y is feasible.
            if xj.is_feasible() and child.is_feasible():
                xj_fit = xj_fits[j]
                if c_fit > xj_fit: 
                    population[nei_idx] = child
                    res = child
                    if len(self.archives[index]) > 0: 
                        now_archives = copy.deepcopy(self.archives[index])
                        # print(now_archives)
                        self.archives.clear(index)
                        for _, a in enumerate(now_archives):
                            if self.scalar(a, self.weight_vec[nei_idx], self.ref_points) > c_fit: 
                                self.archives.append(a, index)

            # x^j is feasible & y is infeasible.
            elif xj.is_feasible() and (not child.is_feasible()):
                if c_fit > xj_fits[j]: 
                    self.archives.append(child, index)

            # x^j is infeasible & y is feasible.
            elif (not xj.is_feasible()) and child.is_feasible():
                population[nei_idx] = child
                res = child

            # x^j is infeasible & y is infeasible.
            elif (not xj.is_feasible()) and (not child.is_feasible()):
                if child.constraint_violation_dominate(xj):
                    population[nei_idx] = child
                    res = child
                elif xj.constraint_violation_dominate(child):
                    pass 
                else:
                    if c_fit > xj_fits[j]:
                        population[nei_idx] = child
                        res = child

        return res


class C_MOEAD_DEDMA(C_MOEAD_DMA):
    name = "c_moead_dedma"

    def __init__(self, popsize: int, n_obj: int, selection: Selector, mating: Mating, 
                 pool: Pool, n_constraint: int, ksize=3, alpha=4,
                 CR=0.9, F=0.7, eta=20, **kwargs
                 ):
        # mros = C_MOEAD_DEDMA.mro()
        # print("mro", (mros))

        super().__init__(
            popsize, n_obj, selection, mating, pool, n_constraint, ksize, alpha, **kwargs
        )
        # print("name is ", super(mros[0], self).name)
        print("cross_rate_dm", self.cross_rate_dm)

        # DE settings
        self.CR = CR   # 交叉率
        self.scaling_F = F    # スケーリングファクタ ->( 0<=F<=1 )
        self.scaling_sigma = kwargs.get("sigma")
        self.pm = self.mating._mutation.rate
        # self.pm = 1.0/len(self.ref_points)
        self.eta = self.mating._mutation.eta
        self.offspring_delta = 0.9  # get_offspringで交配対象にする親個体の範囲を近傍個体集団にする確率
        self.crossover = \
            DifferrentialEvolutonary_Crossover(
                self.CR,
                self.scaling_F,
                self.pm,
                self.eta,
                self.scaling_sigma
            )

        # print(self.__dict__)
        # input()

    def get_offspring(self, index: int, population: Population, eval_func) -> Individual: 

        # subpop is neighboring individuals
        parents, subpop = self._parent_selector(population, index=index, n_parent=3)

        child = self._DEmating(parents, eval_func, index)

        nr = 2
        res = self.update_archives_and_alternate(child, index, subpop, population, nr=nr)
        if res.get_id() == child.get_id():
            self.update_EP(res)
            self.n_EPupdate += 1

        return res

    def _parent_selector(self, 
                         population: Population, 
                         index: int,
                         n_parent: int = 3,
                         DEselection_mode="WR") -> Tuple[List[Individual], List[Individual]]:

        if random.uniform(0.0, 1.0) < self.offspring_delta:
            subpop = [population[i] for i in self.neighbers[index]]
        else:
            subpop = population[:]

        archive_size = self.archives.get_archive_size(index)
        # parents = [subpop[random.randint(1, len(subpop)-1)]]

        if(population[index].is_feasible()) and (archive_size > 0) and \
          (random.random() <= self.cross_rate_dm):
            # p1 = population[index]
            parents = [population[index]]
            # p2 = archive indiv
            pb_idx = random.randint(0, archive_size-1)
            parents.append(self.archives[index][pb_idx])
            # p3 = neighber indiv
            parents.append(subpop[random.randint(1, len(subpop)-1)])

        else:
            subpop2 = subpop + self.archives[index]
            if DEselection_mode == "WOR":
                # parents = random.sample(subpop2[1:], 2)
                parents = self._WORselection(population[index], subpop2, n_parent)
            elif DEselection_mode == "WR":
                parents = self._WRselection(population[index], subpop2, n_parent)
            else:
                raise MOEADError("Invalid parent selection method.")

        # parents = [population[index]] + parents
        return parents, subpop

    def _WORselection(self, target_indiv, poplist, n_parent):
        # randidx = np.random.randint(0, len(poplist))
        random.shuffle(poplist)
        parents = [target_indiv] + poplist[0:n_parent-1]
        return parents

    def _WRselection(self, target_indiv, poplist, n_parent):
        parents = [target_indiv] + random.choices(poplist, k=n_parent-1)
        return parents

    def _DEmating(self, parents, eval_func, index) -> Individual: 
        p1 = parents[0].get_genome()
        p2 = parents[1].get_genome()
        p3 = parents[2].get_genome()
        self.pm = 1.0/len(p1)
        # generating child dv
        child_dv = self.crossover([p1, p2, p3])

        # make child indiv
        child = self.mating.pool.indiv_creator(np.random.rand(len(p1)))
        child.set_genome(child_dv)
        child.set_boundary(parents[0].bounds)
        child.set_weight(parents[0].weight)

        child.evaluate(eval_func, (child.get_design_variable()), self.n_constraint)

        child = self._child_normalize(child)
        self.update_reference(child)
        child.set_fitness(self.scalar(child, self.weight_vec[index], self.ref_points))

        return child


class C_MOEAD_HXDMA(C_MOEAD_DEDMA):
    """ 
        Hybrid crossover method(SBX & DE)
    """

    def __init__(self, popsize: int, n_obj: int, selection: Selector, mating: Mating, 
                 pool: Pool, n_constraint: int, ksize=3, alpha=4,
                 CR=0.9, F=0.7, eta=20, **kwargs
                 ):
        super().__init__(
            popsize, n_obj, selection, mating, pool,
            n_constraint, ksize, alpha, CR, F, eta, **kwargs
        )

    def get_offspring(self, index: int, population: Population, eval_func) -> Individual: 

        archive_size = self.archives.get_archive_size(index)

        subpop = [population[i] for i in self.neighbers[index]]
        # subpop2 = subpop + self.archives[index]

        if(population[index].is_feasible()) and (archive_size > 0) and \
          (random.random() <= self.cross_rate_dm):
            # direct mating
            parents = [subpop[random.randint(1, len(subpop)-1)], None]
            pb_idx = random.randint(0, archive_size-1)
            parents[-1] = self.archives[index][pb_idx]
            child = self._SBXmating(parents, eval_func, index)

        else:
            # neighberhood mating
            parents, _ = self._parent_selector(population, index, n_parent=3)
            # if random.uniform(0.0, 1.0) < self.offspring_delta: 
            #     parents = [population[index]] + random.sample(subpop[1:], 2)
            # else:
            #     parents = random.sample((population[:index]+population[index+1:]), 3) 
            child = self._DEmating(parents, eval_func, index)

        res = self.update_archives_and_alternate(child, index, subpop, population)
        if res.get_id() == child.get_id():
            self.update_EP(res)
            self.n_EPupdate += 1

        return res

    def _parent_selector(self, 
                         population: Population,
                         index: int,
                         n_parent: int = 3,
                         DEselection_mode="WOR") -> Tuple[List[Individual], List[Individual]]:

        if random.uniform(0.0, 1.0) < self.offspring_delta:
            subpop = [population[i] for i in self.neighbers[index]]
        else:
            subpop = population[:]

        # WOR selection
        if DEselection_mode == "WOR":
            subpop = subpop[1:]
            random.shuffle(subpop)        
            parents = [population[index]] + subpop[0:n_parent]

        # WR selection
        elif DEselection_mode == "WR":
            parents = [population[index]] + random.choices(subpop, k=n_parent-1)

        # WPR selection
        elif DEselection_mode == "WPR":
            random.shuffle(subpop)        
            parents = [population[index]] + subpop[:n_parent]

        else:
            raise MOEADError("Invalid parent selection method")

        return parents, subpop
