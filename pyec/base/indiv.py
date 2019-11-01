import numpy as np

class IndividualError(Exception):
    pass     

class Fitness(object):
    """適応度
    """
    def __init__(self):
        self.fitness = None #適応度
    
    def set_fitness(self, value):
        self.fitness = value
        

class Individual(object):

    def __init__(self, genome):
        self._id = None
        self.parent_id = None
        self.bounds = (0,1)
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

    def get_id(self):
        return self._id 

    def get_genome(self):
        return self.genome

    def decode(self, genome):        
        lower, upper = self.bounds
        return (upper - lower)*genome + lower

    def get_design_variable(self):
        return self.decode(self.genome)

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

    def evaluate(self, func): 
        if not self.evaluated():
            self.function = func 
            self.value = func()

if __name__ == "__main__":
    indiv = Individual(10)
    print(indiv)