from typing import Union, List, Optional
import numpy as np


class IndividualError(Exception):
    pass     


class Fitness(object):
    """適応度
    """
    def __init__(self):
        self._fitness = None  # 適応度
        self.optimizer = None  # NSGA-II , MOEA/D, etc...

    @property
    def fitness(self):
        return self._fitness

    def set_fitness(self, value, optimizer=None):
        self._fitness = value
        self.optimizer = optimizer


class Individual(object):

    def __init__(self, genome: np.ndarray, parents=None):
        self._id: int = -1
        self.parent_id: List[int] = []
        self.bounds = (0, 1)  # ((lower), (upper))
        self.weight = None
        self.n_obj = 1

        self.genome = genome  # 遺伝子
        self.value = None  # 評価値
        self.wvalue: Optional[list] = None  # 重みづけ評価値
        self.constraint_violation: Union[List, int, None] = None  # 制約違反量(負の値=制約違反なし)
        self.feasible_rank = None
        self.fitness = Fitness()

    def __str__(self):
        return f"indiv_id:{self._id}"

    def set_id(self, _id):
        self._id = _id
        return (self._id + 1)

    def get_id(self):
        return self._id 

    @property
    def id(self) -> int:
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

    def set_constraint_violation(self, constraint_violation):
        if hasattr(constraint_violation, "__len__"):
            self.constraint_violation = list(constraint_violation)
            # print(self.constraint_violation)
            self.cv_sum = sum(constraint_violation)
        else:
            self.constraint_violation = [constraint_violation]
            self.cv_sum = constraint_violation

    def is_feasible(self)-> bool:
        """ if all constraint violation value is under 0.0 => True
            else => False
        """
        if self.constraint_violation is None:
            return True

        if hasattr(self.constraint_violation, "__len__"):
            cv_s = np.array(self.constraint_violation)
            if all(cv_s <= 0.0):
                return True
        else:
            cv_s = self.cv_sum
            if cv_s <= 0.0:
                return True
        
        return False

    def evaluated(self):
        return self.value is not None 

    def evaluate(self, func, funcargs, n_constraint=0):
        # print("n_constraint:",n_constraint)

        if n_constraint == 0:
            res = func(funcargs)
            # print("indiv eval", (res))
            self.set_value(res)
            return res

        else:
            res, cv = func(funcargs)
            self.set_value(res) 
            self.set_constraint_violation(cv)
            return res, cv

    def dominate(self, other:"Individual") -> bool:
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

    def constraint_violation_dominate(self, other:"Individual") -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        res = False 

        if not hasattr(self.constraint_violation, "__len__"):
            if self.cv_sum < other.cv_sum:
                res = True
            return res

        if all( s <= o for s,o in zip(self.constraint_violation, other.constraint_violation)) and \
            any( s != o for s,o in zip(self.constraint_violation, other.constraint_violation)):
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

    def __add__(self, other) -> np.ndarray:
        if isinstance(other, Individual):
            return (self.get_design_variable() + other.get_design_variable())
        elif isinstance(other, np.ndarray):
            return (self.get_design_variable() + other)
        else:
            return NotImplemented

    def __sub__(self, other) -> np.ndarray:
        if isinstance(other, Individual):
            return (self.get_design_variable() - other.get_design_variable())
        elif isinstance(other, np.ndarray):
            return (self.get_design_variable() - other)
        else:
            return NotImplemented

if __name__ == "__main__":
    indiv = Individual(10)
    print(indiv)
