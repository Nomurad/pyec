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
        self.optimizer = optimizer
        

class Individual(object):

    def __init__(self, genome:np.ndarray, parents=None):
        self._id = None
        self.parent_id = []
        self.bounds = (0,1) # ((lower), (upper))
        self.weight = None
        self.n_obj = 1

        self.genome = genome #遺伝子
        self.value = None #評価値
        self.wvalue = None #重みづけ評価値
        self.feasible_value = None #制約違反量(負の値=制約違反なし)
        self.feasible_rank = None
        self.fitness = Fitness()

    def __str__(self):
        return f"indiv_id:{self._id}"

    def set_id(self, _id):
        self._id = _id
        return (self._id + 1)

    def get_id(self):
        return self._id 

    def set_parents_id(self, parents):
        # print("N_parents", len(parents))
        for parent in parents:
            self.parent_id.append(parent.get_id())
            # print("parent id", parent.get_id())
        
    def set_weight(self, weight):
        self.weight = np.array(weight)
        # self.wvalue = self.weight*self.value

    def get_genome(self):
        return self.genome

    def set_genome(self, genome:np.ndarray):
        self.genome = genome

    def decode(self, genome):        
        lower, upper = self.bounds
        lower = np.array(lower)
        upper = np.array(upper)
        return (upper - lower)*genome + lower

    def encode(self, dv_list):
        lower, upper = self.bounds
        g = []
        for dv in dv_list:
            g.append( (dv-lower)/(upper-lower) )
        
        self.genome = np.array(g)


    def get_design_variable(self):
        return self.decode(self.genome)

    def set_boundary(self, bounds):
        self.bounds = tuple(bounds)

    def set_fitness(self, fit, optimizer=None):
        self.fitness.set_fitness(fit, optimizer)

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

    def set_feasible(self, feasible):
        self.feasible_value = feasible

    def evaluated(self):
        return self.value is not None 

    def evaluate(self, func, funcargs, n_feasible=0):
        # print("n_feasible:",n_feasible)
        if n_feasible == 0:
            res = func(*funcargs)
            # print("indiv eval", (res))
            self.set_value(res)
            return res

        else:
            res, feasible = func(*funcargs)
            self.set_value(res) 
            self.set_feasible(feasible)
            return res, feasible

    def dominate(self, other) -> bool:
        """selfがotherを優越する場合 -> True
           その他の場合             -> False
        
        Arguments:
            other {Individual} -- [description]
        
        Returns:
            bool -- [description]
        """
        if not isinstance(other, Individual):
            return NotImplemented
        res = False

        try:
            # print(len(self.value), len(self.weight))
            self.wvalue = self.weight*self.value
            other.wvalue = other.weight*other.value
        except:
            raise


        if all( s <= o for s,o in zip(self.wvalue, other.wvalue)) and \
            any( s != o for s,o in zip(self.wvalue, other.wvalue)):
            res = True
        return res

    def feasible_dominate(self, other) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        res = False 

        if all( s <= o for s,o in zip(self.feasible_value, other.feasible_value)) and \
            any( s != o for s,o in zip(self.feasible_value, other.feasible_value)):
            res = True
        return res


    def __eq__(self, other):     #equal "=="
        if not isinstance(other, Individual):
            return NotImplemented
        return (self.fitness.fitness == other.fitness.fitness)

    def __lt__(self, other):     #less than "<"
        if not isinstance(other, Individual):
            return NotImplemented
        return (self.fitness.fitness < other.fitness.fitness)
    
    def __ne__(self, other):     #not equal "!="
        return not self.__eq__(other)

    def __le__(self, other):     #less than or equal "<="
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):     #greater than ">"
        return not self.__le__(other)

    def __ge__(self, other):     #greater than or equal ">="
        return not self.__lt__(other)

    def __add__(self, other):
        if isinstance(other, Individual):
            return (self.get_design_variable() + other.get_design_variable())
        elif isinstance(other, np.ndarray):
            return (self.get_design_variable() + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Individual):
            return (self.get_design_variable() - other.get_design_variable())
        elif isinstance(other, np.ndarray):
            return (self.get_design_variable() - other)
        else:
            return NotImplemented

if __name__ == "__main__":
    indiv = Individual(10)
    print(indiv)