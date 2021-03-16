import time
import sys
import os 
import json
import yaml
import argparse
import subprocess
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

from pyec.base.indiv import Individual, Fitness
from pyec.base.population import Population
from pyec.base.environment import Environment

from pyec.operators.crossover import SimulatedBinaryCrossover as SBX
from pyec.operators.selection import Selector, TournamentSelectionStrict
from pyec.operators.mutation import PolynomialMutation as PM
from pyec.operators.mating import Mating
from pyec.operators.sorting import NonDominatedSort, non_dominate_sort

from pyec.optimizers.moead import *
from pyec.solver import Solver

from pyec.testfunctions import TestProblem, Constraint_TestProblem
from pyec.testfunctions import mCDTLZ, Circle_problem, OSY, Welded_beam, zdt1

MAXIMIZE = -1
MINIMIZE = 1

max_epoch = 100*2
n_obj = 2
dvsize = 2
alpha = 4

# cmd argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", type=str, default="./inputs/calc_input.yml")
args = parser.parse_args()

# load setting file (calc_input.yml or .json)
inpfile = args.input_file
extention = os.path.splitext(inpfile)[-1]
ic(inpfile)
if os.path.exists(inpfile):
    with open(inpfile, "r") as f:
        if extention == ".json":
            inpdict = json.load(f)
        elif extention == ".yml":
            inpdict = yaml.safe_load(f)
        ic(inpdict)
        ic((inpdict.get("dv_bounds")))
        ic(type(inpdict.get("dv_bounds")))
        # input()
else:
    print(f"input file ({inpfile}) is not exist...")
    exit(1)


# Define the optimize problem for yourself.
class my_problem(Constraint_TestProblem):
    """
        problem example
    """
    def __init__(self):
        super().__init__(n_obj=2, n_const=0)

    
    def __call__(self, x):
        """
            argument "x" is design variable vector.
        """

        f1 = x[0]*2
        f2 = x[0]/2
        
        # g1 = x - 1
        
        return f1, f2


# test problem setting function
def problem_set(prob:str):
    global n_obj, dvsize, bmax, bmin, problem, weights
    print("problem name is ", prob)
    if prob == "mCDTLZ":
        # dvsize = n_obj
        problem = mCDTLZ(n_obj=n_obj, n_const=n_obj)
        weights = [MINIMIZE]*n_obj
        bmin = problem.dv_bounds[0]
        bmax = problem.dv_bounds[1]

    elif prob == "Circle":
        dvsize = n_obj
        bmin = 0.0
        bmax = 2.0
        problem = Circle_problem()
        weights = [MAXIMIZE]*n_obj

    elif prob == "OSY":
        problem = OSY()  
        dvsize = problem.dvsize
        n_obj = problem.n_obj
        n_const = problem.n_const
        weights = [MINIMIZE]*n_obj
        bmin = problem.dv_bounds[0]
        bmax = problem.dv_bounds[1]

    elif prob == "WB":  # welded beem
        problem = Welded_beam()
        dvsize = problem.dvsize
        n_obj = problem.n_obj
        n_const = problem.n_const
        weights = [MINIMIZE]*n_obj
        bmin = problem.dv_bounds[0]
        bmax = problem.dv_bounds[1]
    
    elif prob == "zdt1":
        problem = zdt1
        n_obj = 2
        n_const = 0
        weights = [MINIMIZE]*dvsize
        bmin = 0.0
        bmax = 1.0
    
    else:
        return -1

    print("problem is ", problem)
    return 0


os.makedirs("result", exist_ok=True)

cross = SBX(rate=1.0, eta=15)
mutate = PM(rate=1/dvsize, eta=20)
### read from inpdict ver. ###
# cross = SBX(rate=inpdict.get("Pc", 1.0), eta=inpdict.get("eta_c", 15))
# mutate = PM(rate=inpdict.get("Pm", 1.0), eta=inpdict.get("eta_m", 15))

res = problem_set(inpdict.get("problem"))
if res < 0:
    problem = my_problem
    solverargs = dict(
        popsize=inpdict.pop("popsize"),
        dv_size=inpdict.pop("dv_size"),
        n_obj=inpdict.pop("n_obj"),
        selector=Selector(TournamentSelectionStrict),  # SBX 
        mating=[cross, mutate],
        optimizer=eval(inpdict.pop("optimizer")),
        eval_func=problem,
        ksize=inpdict.pop("ksize"),
        alpha=inpdict.pop("alpha"), 
        dv_bounds=tuple(inpdict.pop("dv_bounds")),
        weight=weights,
        normalize=inpdict.pop("normalize"),
        n_constraint=inpdict.pop("n_constraint"),
        save=inpdict.pop("save"),
        savepath=inpdict.pop("savepath"),
        old_env=None,
        old_pop=None,
        feasible_only=True,
        **inpdict
    )
else:
    inpdict.pop("n_obj")
    inpdict.pop("n_constraint")
    inpdict.pop("dv_bounds")
    # inpdict.pop("")
    
    solverargs = dict(
        popsize=inpdict.pop("popsize"),
        dv_size=inpdict.pop("dv_size"),
        n_obj=problem.n_obj,
        selector=Selector(TournamentSelectionStrict),  # SBX 
        mating=[cross, mutate],
        optimizer=eval(inpdict.pop("optimizer")),
        eval_func=problem,
        ksize=inpdict.pop("ksize"),
        alpha=inpdict.pop("alpha"), 
        dv_bounds=problem.dv_bounds,
        weight=weights,
        normalize=inpdict.pop("normalize"),
        n_constraint=problem.n_const,
        save=inpdict.pop("save"),
        savepath=inpdict.pop("savepath"),
        old_env=None,
        old_pop=None,
        feasible_only=True,
        **inpdict
    )
ic(solverargs)
# input()
solver = Solver(**solverargs)
print(solver.optimizer)

pop = solver.env.history[0]
data = []
for indiv in pop:
    data.append(list(indiv.value))
data = np.array(data)

### Start Optimizing ==========================================================
st_time = time.time()
max_epoch = inpdict.get("Genelation", max_epoch)
solver.run(max_epoch)
print("calc time: ", time.time()-st_time)
print("num of feasible indivs: ", len(solver.env.feasible_indivs_id))

### result saving =============================================================
result = solver.result(save=True)
solver.save_all_indiv()

with open("result/result_"+ solver.optimizer.name +".json", "w") as f:
    # json.dump(solver.optimizer.__dict__, f, indent=4)
    pprint(solver.optimizer.__dict__, stream=f)

data = solver.save_history_to_csv()
print("data :", data)


### ===========================================================================
###
### ===========================================================================
np.set_printoptions(precision=5, suppress=True)
# plt.scatter(data[-1,0], data[-1,1])
print(f"ref_points = {solver.optimizer.ref_points}")
print(f"pool size = {len(solver.env.pool)}")

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

### result plot ===============================================================
fig = plt.figure(figsize=(10,7))
cm = plt.get_cmap("Blues")
sc = plt.scatter(feasible_dat[:,1], feasible_dat[:,2], 
                c=feasible_dat[:,0], cmap=cm, zorder=10)
# if solver.env.n_constraint > 0:
cm = plt.get_cmap("Reds")
plt.scatter(infeasible_dat[:,1], infeasible_dat[:,2], c=infeasible_dat[:,0], cmap=cm)

data0 = data[data[:,0] == 1]
data_end = data[data[:,0] == max_epoch]
# data_end = pareto_val
# plt.scatter(data0[:,1], data0[:,2], c="green")
# plt.scatter(data0[:,1], data0[:,2], c="yellow")
# plt.scatter(pareto_val[:,0], pareto_val[:,1], c="green")

# np.savetxt("gen000_pop_objs_eval.txt", data[:, 0:3])
print("solver\n")
print(solver.optimizer.name)

plt.colorbar(sc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.tight_layout()
plt.savefig("result/fig.png", dpi=1200)
# plt.savefig("result/fig.svg")
plt.show()
