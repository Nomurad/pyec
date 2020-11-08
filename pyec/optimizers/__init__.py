#from .moead import *
from ..base.indiv import Individual
from ..base.population import Population
from abc import ABC, ABCMeta, abstractmethod

class OptimizerError(Exception):
    pass

# class Optimizer(metaclass=ABCMeta):
class Optimizer(ABC):
    def __init__(self, popsize: int, n_obj: int):
        self.popsize = popsize 
        self.n_obj = n_obj

    def get_new_generation(self) -> Population:
        pass

    @abstractmethod
    def get_offspring(self) -> Individual:
        pass 

    def calc_fitness(self, population):
        pass

    def init_normalize(self, is_normalize: bool, option="unhold"):
        self.normalize = is_normalize 
        self.normalize_option = option 


