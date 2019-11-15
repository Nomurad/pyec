from pyec.base.indiv import Individual, Fitness
from pyec.base.population import Population
from pyec.base.environment import Environment

from pyec.operators.crossover import SimulatedBinaryCrossover
from pyec.operators.selection import Selector, TournamentSelectionStrict
from pyec.operators.mutation import PolynomialMutation
from pyec.operators.mating import Mating

from pyec.optimizers.moead import MOEAD, MOEAD_DE
from pyec.solver import Solver

from pyec.testfunctions import zdt1, zdt2, zdt3

import numpy as np
import matplotlib.pyplot as plt
import time


class Problem():
    def __init__(self):
        pass

    def __call__(self, a):
        return a*10

# problem = Problem()
problem = zdt1

optimizer = MOEAD
max_epoch = 100*3

args = {
    "popsize":51,
    "dv_size":10,
    "nobj":2,
    "selector":Selector(TournamentSelectionStrict),
    "mating":[SimulatedBinaryCrossover(), PolynomialMutation()],
    "optimizer":optimizer,
    "eval_func":problem,
    "dv_bounds":(0,1),
    "weight":[1, 1],
    "normalize": True
}

print(optimizer.name)

solver = Solver(**args)
print(solver.optimizer)
# solver.env.weight = [1, 0.1]
pop = solver.env.history[0]
# for indiv in pop:
#     # print(indiv.fitness.fitness)
#     print(indiv.value, indiv.wvalue, indiv.fitness.fitness)
# input()

st_time = time.time()
solver.run(max_epoch)
print("calc time: ", time.time()-st_time)

result = solver.result()

cm = plt.get_cmap("Blues")

# for vec in solver.optimizer.weight_vec:
#     plt.plot([0,vec[0]], [0,vec[1]])
# print(solver.optimizer.weight_vec)

data = []
for epoch, pop in enumerate(result):
    for i, indiv in enumerate(pop):
        data.append(list(indiv.value)+[epoch])

data = np.array(data)
plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=cm)
# plt.scatter(data[-1,0], data[-1,1])

# print(data)

print()
# pop = result[-1]
# for indiv in pop:
#     print(indiv.value, indiv.wvalue, indiv.fitness.fitness)

print(f"ref_points={solver.optimizer.ref_points}")
print(f"pool size={len(solver.env.pool)}")
plt.show()