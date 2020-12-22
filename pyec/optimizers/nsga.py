import random 
import copy 
from itertools import chain 
from operator import attrgetter, itemgetter
from typing import List, Tuple, Union, Optional
from types import FunctionType

import numpy as np 

from ..base.indiv import Individual
from ..base.population import Population
from ..base.environment import Pool, Normalizer

from ..operators.initializer import UniformInitializer
from ..operators.crossover import SimulatedBinaryCrossover as SBX
from ..operators.mutation import PolynomialMutation as PM
from ..operators.selection import TournamentSelection, TournamentSelectionStrict
from ..operators.selection import SelectionIterator, Selector
from ..operators.mating import MatingIterator, Mating
from ..operators.sorting import NonDominatedSort, CrowdingDistanceCalculator

from . import Optimizer, OptimizerError


class NSGAError(OptimizerError):
    pass


class NSGA2(Optimizer):
    """ NSGA-II
    """
    name = "NSGA-II"

    def __init__(self, popsize: int, n_obj: int,
                 selection: Selector, mating: Mating):

        super().__init__(popsize, n_obj)
        self.popsize = popsize 
        self.n_obj = n_obj
        self.selector = selection
        self.mating = mating
        self.n_parents = 2
        self.n_cycle = 2
        self.alternation = "join"

        self.sort = NonDominatedSort()
        self.share_fn = CrowdingDistanceCalculator()
        # self.EP = []

    def get_new_generation(self, population: Population, eval_func: FunctionType):
        if not self.popsize:
            self.popsize = len(population)

        next_pop = self.advance(population, eval_func)
        next_pop = self._alternate(population, next_pop)
        for i, indiv in enumerate(next_pop):
            population[i] = next_pop[i]
        return next_pop

    def get_offspring(self, index: int, population: Population, eval_func) -> Individual:
        parents = self.selector(population)
        child = self.mating(parents, singlemode=True)[0]
        child_fit = child.evaluate(eval_func, child.get_design_variable())
        # child.fitness.set_fitness(child_fit)
        return child

    def advance(self, population: Population, eval_func) -> Population:
        next_pop = Population(capa=self.popsize)
        # selector = self.selector(population) 

        while not next_pop.filled():
            # parents = self.selector(population)
            i = 0
            child = self.get_offspring(i, population, eval_func)
            next_pop.append(child)

        return next_pop

    def _alternate(self, population: Population, next_pop: Population):
        if self.alternation == "join":
            joined = population + next_pop

        next_pop = self.calc_fitness(joined)
        return Population(indivs=next_pop, capa=self.popsize)

    def calc_fitness(self, population):
        ''' 各個体の集団内における適応度を計算する
        1. 比優越ソート
        2. 混雑度計算
        '''
        # lim = len(population) if n is None else n
        lim = self.popsize
        selected = []

        # for i, front in enumerate(self.sort_it(population)):
        for i, front in enumerate(self.sort.sort(population)):
            # print('g:', self.generation, 'i:', i, 'l:', len(front))
            rank = i + 1
            fit_value = -i  # TODO: 可変にする
            # if i == 0:
            #     print('len(i==0):', len(front), ' ')

            if self.share_fn:
                it = self.share_fn(front)
                for fit, crowding in zip(front, it):
                    fitness = fit_value, crowding
                    # print(fitness)
                    fit.set_fitness(fitness, rank)
                # except:
                #     print('Error')
                #     print(front)
                #     print(it)
                #     raise
            else:
                for fit in front:
                    fitness = fit_value,
                    fit.set_fitness(fitness, rank)

            lim -= len(front)  # 個体追加後の余裕
            if lim >= 0:
                selected.extend(front)
                if lim == 0:
                    return selected
            # elif i == 0:
            #     return front
            else:
                # front.sort(key=itemgetter(1), reverse=True) # 混雑度降順で並べ替え
                # print(front[0].fitness, end="\r")
                front.sort(key=lambda x: x.fitness[1], reverse=True)  # 混雑度降順で並べ替え
                # print([itemgetter(1)(fit) for fit in front])
                # exit()
                selected.extend(front[:lim])
                return selected

    def calc_rank(self, population, n=None):
        ''' 各個体の集団内におけるランクを計算して設定する
        外部から呼ぶ
        '''
        for i, front in enumerate(self.sort.sort(population)):
            rank = i + 1
            for fit in front:
                fit.rank = rank
        return population


class TNSDM(NSGA2):
    def __init__(self, popsize: int, n_obj: int,
                 selection: Selector, mating: Mating, **kwargs):

        super().__init__(popsize, n_obj, selection, mating)
        self.cross_rate_dm = kwargs.get("cross_rate_dm", 1.0)
