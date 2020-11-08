#from .moead import *
from abc import ABC, ABCMeta, abstractmethod

class OptimizerError(Exception):
    pass

# class Optimizer(metaclass=ABCMeta):
class Optimizer(ABC):
    def __init__(self, popsize: int, n_obj: int):
        self.popsize = popsize 
        self.n_obj = n_obj

    def get_new_offspring(self):
        pass

    @abstractmethod
    def get_offspring(self):
        pass 

