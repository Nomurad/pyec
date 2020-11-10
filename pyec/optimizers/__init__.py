from itertools import chain
from types import FunctionType

from ..base.indiv import Individual
from ..base.population import Population
from abc import ABC, ABCMeta, abstractmethod


class OptimizerError(Exception):
    pass


class Optimizer(ABC):
    def __init__(self, popsize: int, n_obj: int):
        self.popsize = popsize 
        self.n_obj = n_obj

    def get_new_generation(self, 
                           population: Population, 
                           eval_func: FunctionType) -> Population:
        pass

    @abstractmethod
    def get_offspring(self, 
                      index: int, 
                      population: Population, 
                      eval_func: FunctionType) -> Individual:
        pass 

    def calc_fitness(self, population):
        pass

    def init_normalize(self, is_normalize: bool, option="unhold"):
        self.normalize = is_normalize 
        self.normalize_option = option 


class Solution_archive(object):

    def __init__(self, n_wvec, size):
        self.limit_size = size
        self._archives = [[] for _ in range(n_wvec)]

    def __getitem__(self, key):
        return self._archives[key]

    def __len__(self):
        return len(self._archives)

    @property
    def archives(self):
        return chain.from_iterable(self._archives)

    def append(self, indiv: Individual, index: int):
        self._archives[index].append(indiv)
        if self.get_archive_size(index) > self.limit_size:
            self._archives[index].sort()
            self._archives[index].pop(0)

    def get_archive_size(self, index):
        return len(self._archives[index])

    def clear(self, index):
        self._archives[index].clear()
