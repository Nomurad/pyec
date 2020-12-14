import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os 
import json
from pprint import pprint

from pyec.base.indiv import Individual, Fitness
from pyec.base.population import Population
from pyec.base.environment import Environment

from pyec.operators.crossover import SimulatedBinaryCrossover
from pyec.operators.selection import Selector, TournamentSelectionStrict
from pyec.operators.mutation import PolynomialMutation
from pyec.operators.mating import Mating
from pyec.operators.sorting import NonDominatedSort, non_dominate_sort

from pyec.optimizers.moead import *
from pyec.solver import Solver

from pyec.testfunctions import mCDTLZ, Knapsack, Circle_problem, WaterProblem

MAXIMIZE = -1
MINIMIZE = 1

max_epoch = 100*2
n_obj = 2
dvsize = n_obj
alpha = 4
phi = 0.3

optimizer = C_MOEAD
optimizer = C_MOEAD_DMA
optimizer = C_MOEAD_DEDMA

optimizer = eval("C_MOEAD_DMA")

def problem_set(prob:str):
    global n_obj, dvsize, bmax, problem, weights, phi
    print("problem name is ", prob)
    if prob == "mCDTLZ":
        # dvsize = n_obj
        bmax = 1.0
        problem = mCDTLZ(n_obj=n_obj, n_const=n_obj)
        weights = [MINIMIZE]*n_obj

    elif prob == "Knapsack":
        # dvsize = 500
        bmax = 1.0
        problem = Knapsack(n_obj=n_obj, n_items=dvsize, phi=phi)
        weights = [MAXIMIZE]*n_obj

    elif prob == "Circle":
        dvsize = n_obj
        bmax = 2.0
        problem = Circle_problem()
        weights = [MAXIMIZE]*n_obj
    
    elif prob == "WaterProblem":
        dvsize = 3
        bmax = 0.45
        problem = WaterProblem()
        n_obj = problem.n_obj
        n_const = problem.n_const
        weights = [MINIMIZE]*n_obj
    print("problem is ", problem)


problem_set("mCDTLZ")
n_obj = problem.n_obj
n_const = problem.n_const

cross = SimulatedBinaryCrossover(rate=1.0, eta=15)
mutate = PolynomialMutation(rate=1/dvsize, eta=20)

args = {
    "popsize":100,
    "dv_size":dvsize,
    "n_obj":n_obj,
    "selector":Selector(TournamentSelectionStrict),
    "mating":[cross, mutate],
    "optimizer":optimizer,
    "eval_func":problem,
    "ksize":17,
    "alpha":alpha,
    "dv_bounds":([0.0]*dvsize, [bmax]*dvsize),   #(lowerbounds_list, upperbounds_list)
    "weight":weights,
    "normalize": False,
    "n_constraint":n_const,
    "CR":1.0,
    "F":0.8,
    "save":False,
    "savepath": "result"
}

inpfile = "calc_input.json"
if os.path.exists(inpfile):
    with open(inpfile, "r") as f:
        inpdict = json.load(f)
    for argskey in args:
        if argskey in inpdict:
            args[argskey] = inpdict.get(argskey)
        
    args["dv_size"] = n_obj*10
    dvsize = args["dv_size"]
    n_const = args["n_constraint"]
    max_epoch = inpdict["Genelation"]
    print("set total num of genelation is ", max_epoch)
    mutate.rate = 1/dvsize

    n_obj = inpdict["n_obj"]
    problem_set(inpdict["problem"])
    print((inpdict["problem"]))
    args["weight"] = weights
    args["eval_func"] = problem
    args["dv_bounds"] = ([0.0]*dvsize, [bmax]*dvsize)
    args["optimizer"] = eval(inpdict["optimizer"])
    args["cross_rate_dm"] = inpdict.get("cross_rate_dm", 1.0)

    if inpdict["problem"] == "WaterProblem":
        args["dv_bounds"] = ([0.01]*dvsize, [0.45,0.1,0.1])
        n_obj = problem.n_obj
        n_const = problem.n_const
        args["n_obj"] = 5
        args["n_constraint"] = 7
        args["dv_size"] = 3
        args["weight"] = weights
    # pprint(inpdict)
    # print()
    # pprint(args)
    # input()
os.makedirs("result", exist_ok=True)

print(optimizer.name)

solver = Solver(**args)
print(solver.optimizer)
# pprint(solver.env.__dict__) # for debug
# pprint(solver.optimizer.__dict__)

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
# solver.run(max_epoch, savepath="result")
solver.run(max_epoch)
print("calc time: ", time.time()-st_time)
print("num of feasible indivs: ", len(solver.env.feasible_indivs_id))
# print(solver.optimizer.mating.__repr__())
# for indiv in solver.env.feasible_indivs:
#     print(indiv.id)

result = solver.result(save=True)

with open("result/result_"+ solver.optimizer.name +".json", "w") as f:
#     json.dump(solver.optimizer.__dict__, f, indent=4)
    pprint(solver.optimizer.__dict__, stream=f)

###############################################################################

data = []
for epoch, pop in enumerate(result):
    for i, indiv in enumerate(pop):
        data.append([epoch]+list(indiv.value)+list(indiv.wvalue)+list(indiv.constraint_violation))

data = np.array(data)
print("data :", data)
# np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=5, suppress=True)
# plt.scatter(data[-1,0], data[-1,1])
print(f"ref_points={solver.optimizer.ref_points}")
print(f"pool size={len(solver.env.pool)}")

sort_func = NonDominatedSort()
pop = solver.env.history[-1]
for i in range(1, 100):
    pop = pop + solver.env.history[-i]

print("popsize",len(pop))
# fronts = non_dominate_sort(pop)
# pareto = sort_func.output_pareto(pop)
# print("pareto size", len(pareto), end="\n\n")
# print("pop:fronts=",len(pop), ":", sum([len(front) for front in fronts]))
pareto = solver.optimizer.EP
pareto_val = np.array([indiv.value for indiv in pareto])
# print(pareto_val)
# np.savetxt("temp_data.csv", data, delimiter=",")
np.savetxt("temp_pareto.csv", pareto_val, delimiter=",")

feasible_dat = data[data[:,-1] < 0]
infeasible_dat = data[data[:,-1] > 0]
fig = plt.figure(figsize=(10,7))


cm = plt.get_cmap("Blues")
sc = plt.scatter(feasible_dat[:,1], feasible_dat[:,2], c=feasible_dat[:,0], cmap=cm)
cm = plt.get_cmap("Reds")
plt.scatter(infeasible_dat[:,1], infeasible_dat[:,2], c=infeasible_dat[:,0], cmap=cm)
data0 = data[data[:,0] == 1]
data_end = data[data[:,0] == max_epoch]
# data_end = pareto_val
# plt.scatter(data0[:,1], data0[:,2], c="green")
# plt.scatter(data0[:,1], data0[:,2], c="yellow")
plt.scatter(pareto_val[:,0], pareto_val[:,1], c="green")

np.savetxt("gen000_pop_objs_eval.txt", data[:, 0:3])

headers = "epoch, value1, value2, wvalue1, wvalue2, CV"
fmts = "%5f"
# fmts = ["%5d","%.5f","%.5f","%.5f","%.5f","%.5f", "%.5f"]
np.savetxt("const_opt_result.csv", data, delimiter=",", fmt=fmts, header=headers)
print("data shape",data.shape)
print("solver\n")
print(solver.optimizer.name)

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

plt.colorbar(sc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.tight_layout()
plt.savefig("result/fig.png", dpi=1200)
# plt.savefig("result/fig.svg")
# plt.show()
