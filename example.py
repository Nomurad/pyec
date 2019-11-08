from pyec.base.indiv import Individual, Fitness
from pyec.base.population import Population
from pyec.base.environment import Environment

from pyec.operators.crossover import SimulatedBinaryCrossover
from pyec.operators.selection import Selector, TournamentSelectionStrict
from pyec.operators.mutation import PolynomialMutation
from pyec.operators.mating import Mating

from pyec.optimizers.moead import MOEAD
from pyec.solver import Solver

from pyec.testfunctions import zdt1, zdt2, zdt3

import numpy as np
import matplotlib.pyplot as plt


class Problem():
    def __init__(self):
        pass

    def __call__(self, a):
        return a*10

problem = Problem()
problem = zdt1

args = {
    "popsize":10,
    "dv_size":10,
    "nobj":2,
    "selector":Selector,
    "mating":Mating,
    "optimizer":MOEAD,
    "eval_func":problem,
    "dv_bounds":(0,1),
    "weight":[1, 0.1]
}

solver = Solver(**args)
# solver.env.weight = [1, 0.1]

print(solver)

max_epoch = 10
solver.run(max_epoch)

result = solver.result()

for epoch, pop in enumerate(result):
    data = []
    for i, indiv in enumerate(pop):
        data.append(indiv.value)

    data = np.array(data)
    plt.scatter(data[:,0], data[:,1], c="blue", alpha=i/max_epoch)

pop = result[-1]
for indiv in pop:
    # print(indiv.fitness.fitness)
    print(indiv.value, indiv.wvalue, indiv.weight)

plt.show()