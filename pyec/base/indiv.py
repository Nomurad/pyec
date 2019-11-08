import numpy as np

class IndividualError(Exception):
    pass     

class Fitness(object):
    """適応度
    """
    def __init__(self):
        self.fitness = None #適応度
        self.optimizer = None #NSGA-II , MOEA/D, etc...
    
    def set_fitness(self, value, optimizer=None):
        self.fitness = value
        self.optimizer = None
        

class Individual(object):

    def __init__(self, genome, parents=None):
        self._id = None
        self.parent_id = []
        self.bounds = (0,1) # ((lower), (upper))
        self.weight = None
        self.n_obj = 1

        self.genome = genome #遺伝子
        self.value = None #評価値
        self.wvalue = None #重みづけ評価値
        self.feasble_value = None #制約違反量
        self.fitness = Fitness()

    def __str__(self):
        return f"indiv_id:{self._id}"

    def set_id(self, _id):
        self._id = _id
        return (self._id + 1)

    def get_id(self):
        return self._id 

    def set_parents_id(self, parents):
        for parent in parents:
            self.parent_id.append(parent.get_id())
        
    def set_weight(self, weight):
        self.weight = weight

    def get_genome(self):
        return self.genome

    def decode(self, genome):        
        lower, upper = self.bounds
        return (upper - lower)*genome + lower

    def get_design_variable(self):
        return self.decode(self.genome)

    def set_boundary(self, bounds):
        self.bounds = tuple(bounds)

    def set_fitness(self, fit):
        self.fitness.set_fitness(fit)

    def set_value(self, value):
        if self.value is None:
            self.value = value
            self.n_obj = len(value)
        else:
            if len(value) != self.n_obj:
                raise IndividualError("Invaild value dimension")
            else:
                self.value = value
        
        if self.weight is not None:
            self.wvalue = self.weight*self.value
        else:
            self.wvalue = self.value

    def evaluated(self):
        return self.value is not None 

    def evaluate(self, func, funcargs):
        # self.function = func 
        res = func(funcargs)
        # print("indiv eval", (res))
        self.set_value(res)
        return res
        
    def __eq__(self, other):     #equal "=="
        if not isinstance(other, Individual):
            return NotImplemented
        return self.fitness.fitness == other.fitness.fitness

    def __lt__(self, other):     #less than "<"
        if not isinstance(other, Individual):
            return NotImplemented
        return self.fitness.fitness < other.fitness.fitness
    
    def __ne__(self, other):     #not equal "!="
        return not self.__eq__(other)

    def __le__(self, other):     #less than or equal "<="
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):     #greater than ">"
        return not self.__le__(other)

    def __ge__(self, other):     #greater than or equal ">="
        return not self.__lt__(other)

if __name__ == "__main__":
    indiv = Individual(10)
    print(indiv)