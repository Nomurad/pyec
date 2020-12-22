from pyec.base.indiv import Individual, Fitness
from pyec.base.population import Population
from pyec.base.environment import Environment

from pyec.operators.crossover import SimulatedBinaryCrossover as SBX
from pyec.operators.selection import Selector, TournamentSelectionStrict
from pyec.operators.mutation import PolynomialMutation as PM
from pyec.operators.mating import Mating
from pyec.operators.sorting import NonDominatedSort, non_dominate_sort

from pyec.optimizers.moead import MOEAD, MOEAD_DE, C_MOEAD
from pyec.optimizers.nsga import NSGA2
from pyec.solver import Solver

from pyec.testfunctions import zdt1, zdt2, zdt3, Fonseca_and_Fleming_func 
from pyec.testfunctions import tnk, mCDTLZ, Knapsack

import numpy as np
import matplotlib.pyplot as plt
import time
from pprint import pprint

class Problem():
    def __init__(self):
        pass

    def __call__(self, x):
        """ 
        0 <= x1 <= 5
        0 <= x2 <= 3
        """
        return self.biases(x)

    def belegundu(self, vars):
        x = vars[0]
        y = vars[1]
        return [-2*x + y, 2*x + y], [-x + y - 1, x + y - 7]

    def biases(self, x):
        x = np.array(x)
        n = len(x)
        x1 = x[0]
        x2 = x[1]
        gamma = 0.1

        f1 = 1 - np.exp(-4*x1)*(np.sin(5*np.pi*x1))**6
        g = 1 + 10*(sum(x[1:])/(n-1))**0.25
        h = 1 - (f1/g)**2
        if f1 > g:
            h = 0
        f2 = g*h
        return [f1, f2]



max_epoch = 200
dvsize = 10
problem = Fonseca_and_Fleming_func(2, dvsize)
problem = Problem()
optimizer = MOEAD_DE
optimizer = NSGA2
n_const = 0
# problem = Knapsack(n_const=n_const ,phi=0.5)


args = {
    "popsize": 100,
    "dv_size": dvsize,
    "n_obj": 2,
    "selector": Selector(TournamentSelectionStrict),
    "mating": [SBX(), PM()],
    "optimizer": optimizer,
    "eval_func": problem,
    "ksize": 10,
    "dv_bounds": ([0]*dvsize, [1]*dvsize),  # (lowerbounds_list, upperbounds_list)
    "weight": [1, 1],
    "normalize": False,
    "n_constraint": n_const,
    "save": False
}
#args["dv_bounds"] = ([-4]*dvsize, [4]*dvsize)

print(optimizer.name)

solver = Solver(**args)
print(solver.optimizer)
pprint(solver.env.__dict__) # for debug
pprint(solver.optimizer.__dict__)
pop = solver.env.history[0]
data = []
for indiv in pop:
    data.append(list(indiv.value))
data = np.array(data)
# plt.scatter(data[:,0], data[:,1], c="Blue")
# for d in data:
#     if d[-2]>=0 and d[-1]>=0:
#         plt.scatter(data[:,0], data[:,1], c="Red")
# plt.show()

st_time = time.time()
solver.run(max_epoch)
print("calc time: ", time.time()-st_time)

result = solver.result(save=True)

###############################################################################

cm = plt.get_cmap("Blues")

data = []
for epoch, pop in enumerate(result):
    for i, indiv in enumerate(pop):
        data.append([epoch]+list(indiv.value)+list(indiv.wvalue))

data = np.array(data)
print(data)
# plt.scatter(data[-1,0], data[-1,1])
if hasattr(solver.optimizer, "ref_points"):
    print(f"ref_points={solver.optimizer.ref_points}")
print(f"pool size={len(solver.env.pool)}")

sort_func = NonDominatedSort()
pop = solver.env.history[-1]
for i in range(1, 11):
    pop = pop + solver.env.history[-i]

print("popsize",len(pop))
# fronts = non_dominate_sort(pop)
# pareto = sort_func.output_pareto(pop)
# pareto = solver.optimizer.EP
# print("pareto size", len(pareto), end="\n\n")
# print("pop:fronts=",len(pop), ":", sum([len(front) for front in fronts]))
# pareto_val = np.array([indiv.value for indiv in pareto])
# print(pareto_val)

# np.savetxt("temp_data.csv", data, delimiter=",")
# np.savetxt("temp_pareto.csv", pareto_val, delimiter=",")

plt.scatter(data[:,1], data[:,2], c=data[:,0], cmap=cm)
# plt.scatter(pareto_val[:,0], pareto_val[:,1], c="red")

np.savetxt("gen000_pop_objs_eval.txt", data[:, 0:3])
print("data shape",data.shape)

### 以下，制約条件ありで行う場合使用
# data = []
# for pop in solver.env.history:
#     for indiv in pop:
#         if all( val <= 0 for val in indiv.feasible_value):
#             data.append([epoch]+list(indiv.value)+list(indiv.feasible_value))

# data = np.array(data)
# print(data)
# plt.scatter(data[:,1], data[:,2], c="Red")

# for i, indiv in enumerate(pareto):
#     dom = 0
#     for j, other in enumerate(pareto):
#         if i == j:
#             continue
#         dom += (indiv.dominate(other))
#     if dom != 0:
#         print("dominate:",i, dom)

#plt.ylim([0.0, 1.0])
plt.show()
