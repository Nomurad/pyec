from pyec.base.indiv import Individual, Fitness
from pyec.base.population import Population
from pyec.base.environment import Environment

from pyec.solver import Solver

class Problem():
    def __init__(self):
        pass

    def __call__(self, a):
        return a*10

problem = Problem()

args = {
    "popsize":100,
    "dv_size":10,
    "optimizer":1,
    "eval_func":problem,
    "dv_bounds":(0,1)
}

solver = Solver(**args)

print(solver)