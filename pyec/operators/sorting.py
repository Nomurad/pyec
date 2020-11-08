import itertools

import numpy as np 

from ..base.indiv import Individual
from ..base.population import Population

class NonDominatedSortError(Exception):
    pass

class NonDominatedSort(object):

    def __init__(self):
        self.product = None
        # self.pop = pop
    
    def _mask_func(self, idxes):
        i, j = idxes
        dom = self.pop[j].dominate(self.pop[i])
        is_dominated = (i!=j) and dom 
        return is_dominated

    def sort(self, population:Population, return_rank=False):
        popsize = len(population)
        self.pop = population

        # is_dominated = np.empty((popsize, popsize), dtype=np.bool)
        num_dominated = np.zeros(popsize, dtype=np.int64)
        mask = np.empty(popsize, dtype=np.bool)
        rank = np.zeros(popsize, dtype=np.int64)

        # if self.product is None:
        self.product = itertools.product(range(popsize), range(popsize))
        # print(popsize)
        is_dominated = list(map(self._mask_func, self.product))
        is_dominated = np.array(is_dominated, dtype=bool).reshape(popsize, popsize)
        # for i in range(popsize):
        #     for j in range(popsize):
        #         # if i == j:
        #         #     continue
        #         #iがjに優越されている -> True
        #         dom = population[j].dominate(population[i])
        #         is_dominated[i,j] = (i!= j) and dom

        #iを優越する個体の数
        is_dominated.sum(axis=(1,), out=num_dominated)
        # print(num_dominated)

        fronts = []
        limit = popsize
        for r in range(popsize):
            front = []
            for i in range(popsize):
                is_rank_ditermined = not(rank[i] or num_dominated[i])
                mask[i] = is_rank_ditermined
                if is_rank_ditermined:
                    rank[i] = r + 1
                    front.append(population[i])
                
            fronts.append(front)
            limit -= len(front)

            if return_rank:
                if rank.all():
                    return rank 
            elif limit <= 0:
                return fronts

            # print(np.sum(mask & is_dominated))
            num_dominated -= np.sum(mask & is_dominated, axis=(1,))

        raise NonDominatedSortError("Error: reached the end of function")

    def output_pareto(self, population: Population):
        popsize = len(population)

        is_dominated = np.empty((popsize, popsize), dtype=np.bool)
        num_dominated = np.zeros(popsize, dtype=np.int64)
        mask = np.empty(popsize, dtype=np.bool)
        rank = np.zeros(popsize, dtype=np.int64)

        n_dim = len(population[0].value)
        valarr = np.empty((popsize, n_dim))
        for i, indiv in enumerate(population):
            valarr[i] = indiv.value
        n_points = valarr.shape[0]

        is_efficient = np.arange(n_points)
        next_point_index = 0
        while next_point_index < len(valarr):
            nondominated_point_mask = np.any(valarr < valarr[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # remove dominated
            valarr = valarr[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
            print("index: ", next_point_index, end="\r")

        front = [population[i] for i in is_efficient]

        # front = []
        # # for i in range(popsize):
        # #     for j in range(popsize):
        # for i, j in itertools.product(range(popsize), range(popsize)):
        #         if j == 0:
        #             print(f"i, j = {i} ,{j}", end="\r")
        #         if not population[j].is_feasible():
        #             continue
        #         # if i == j:
        #         #     continue
        #         #iがjに優越されている -> True
        #         dom = population[j].dominate(population[i])
        #         is_dominated[i, j] = (i != j) and dom
        # print()
        # #iを優越する個体の数
        # is_dominated.sum(axis=(1,), out=num_dominated)

        # for i in range(popsize):
        #     for j in range(popsize):
        for i, j in itertools.product(range(popsize), range(popsize)):
            if not population[j].is_feasible():
                continue
            # if i == j:
            #     continue
            # iがjに優越されている -> True
            dom = population[j].dominate(population[i])
            is_dominated[i, j] = (i != j) and dom

        # iを優越する個体の数
        is_dominated.sum(axis=(1,), out=num_dominated)

        for i in range(popsize):
            if num_dominated[i] == 0:
                front.append(population[i])

        # # for idx in num_dominated[num_dominated == 0]:
        # #     front.append(population[idx])

        return front

    def constraint_violation_sort(self, population:Population, return_rank=False):
        popsize = len(population)

        is_dominated = np.empty((popsize, popsize), dtype=np.bool)
        num_dominated = np.zeros(popsize, dtype=np.int64)
        mask = np.empty(popsize, dtype=np.bool)
        rank = np.zeros(popsize, dtype=np.int64)

        for i in range(popsize):
            for j in range(popsize):
                # if i == j:
                #     continue
                #iがjに優越されている -> True
                dom = population[j].feasible_dominate(population[i])
                is_dominated[i,j] = (i!= j) and dom

        #iを優越する個体の数
        is_dominated.sum(axis=(1,), out=num_dominated)
        # print(num_dominated)

        fronts = []
        limit = popsize
        for r in range(popsize):
            front = []
            for i in range(popsize):
                is_rank_ditermined = not(rank[i] or num_dominated[i])
                mask[i] = is_rank_ditermined
                if is_rank_ditermined:
                    rank[i] = r + 1
                    front.append(population[i])
                
            fronts.append(front)
            limit -= len(front)

            if return_rank:
                if rank.all():
                    return rank 
            elif limit <= 0:
                return fronts

            # print(np.sum(mask & is_dominated))
            num_dominated -= np.sum(mask & is_dominated, axis=(1,))

        raise NonDominatedSortError("Error: reached the end of function")

def non_dominate_sort(population:Population, return_rank=False):
    popsize = len(population)

    is_dominated = np.empty((popsize, popsize), dtype=np.bool)
    num_dominated = np.zeros(popsize, dtype=np.int64)
    mask = np.empty(popsize, dtype=np.bool)
    rank = np.zeros(popsize, dtype=np.int64)

    for i in range(popsize):
        for j in range(popsize):
            if not population[j].is_feasible():
                continue
            
            # if i == j:
            #     continue
            #iがjに優越されている -> True
            dom = population[j].dominate(population[i])
            is_dominated[i,j] = (i!= j) and dom

    #iを優越する個体の数
    is_dominated.sum(axis=(1,), out=num_dominated)
    # print(num_dominated)

    fronts = []
    limit = popsize
    for r in range(popsize):
        front = []
        for i in range(popsize):
            is_rank_ditermined = not(rank[i] or num_dominated[i])
            mask[i] = is_rank_ditermined
            if is_rank_ditermined:
                rank[i] = r + 1
                front.append(population[i])
            
        fronts.append(front)
        limit -= len(front)

        if return_rank:
            if rank.all():
                return rank 
        elif limit <= 0:
            return fronts

        # print(np.sum(mask & is_dominated))
        num_dominated -= np.sum(mask & is_dominated, axis=(1,))

    raise NonDominatedSortError("Error: reached the end of function")


def identity(x):
    return x
class CrowdingDistanceCalculator(object):
    ''' key=attrgetter('data')
    '''
    def __init__(self, key=identity):
        self.key = key

    def __call__(self, population: Population, normalization=False):
        popsize = len(population)
        if popsize == 0:
            return

        distances = np.zeros(popsize, dtype=np.float32)

        indivs = [x for x in population]
        index = list(range(popsize))

        # n_obj = len(values[0])
        n_obj = len(indivs[0].value)

        for i in range(n_obj):
            get_value = lambda idx: np.array(indivs[idx].value[i])

            index.sort(key=get_value)

            distances[index[0]] = float('inf')
            distances[index[-1]] = float('inf')

            vrange = get_value(-1) - get_value(0)

            if vrange <= 0:
                continue

            norm = n_obj * vrange

            for l, c, r in zip(index[:-2], index[1:-1], index[2:]):
                distances[c] += (get_value(r) - get_value(l)) / norm

        return distances