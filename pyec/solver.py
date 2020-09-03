import numpy as np
import pickle
import copy
import os 

from .base.indiv import Individual
from .base.population import Population
from .base.environment import Environment

from .operators.selection import Selector, TournamentSelection, TournamentSelectionStrict
from .operators.mutation import PolynomialMutation as PM
from .operators.crossover import SimulatedBinaryCrossover as SBX
from .operators.mating import Mating

from .optimizers.moead import MOEAD, MOEAD_DE, C_MOEAD, C_MOEAD_DMA, C_MOEAD_DEDMA, C_MOEAD_HXDMA

class Solver(object):
    """進化計算ソルバー    
    """

    def __init__(self,  popsize: int, #1世代あたりの個体数
                        dv_size: int, #設計変数の数
                        n_obj: int, #目的関数の数
                        selector,
                        mating,
                        optimizer,
                        eval_func, 
                        ksize: int= 0,
                        alpha: int= 0,
                        dv_bounds: tuple = (0,1), #設計変数の上下限値
                        weight = None,
                        normalize = False,
                        n_constraint = 0,
                        save=True,
                        savepath=None,
                        old_pop=None,
                        **kwargs
                        ):
        """ solver initializer
        
        Arguments:
            popsize {int} -- [num of individual]
            dv_size {int} -- [num of design variable]
            selector      -- [selector]
            mating        -- [crossover,mutation]
            optimizer     -- [EA method] 
            eval_func {[type]} -- [objective function(evaluating function)]
        
        Keyword Arguments:
            ksize {int}       -- [近傍サイズ] (default: None)
            dv_bounds {tuple} -- [設計変数の上下限値] (default: {(0,1)})
            weight {list or tuple} -- [目的関数の重み付け] (default: None)
            normalize {bool} -- [評価値の正規化] (default: False)
            n_constraint {int} -- [制約条件数] (dafault: 0)
            old_pop [Population] -- [last population, Restart時に使用]
        """
        self.flag_save = save

        self.n_epoch = 0
        self.restart = 0
        if old_pop is not None:
            self.restart = len(old_pop)
            print(f"start epoch is {self.restart}")
            # print("oldpop", old_pop[-1].__dict__)
            self.env = Environment(popsize, dv_size, n_obj, optimizer,
                            eval_func, dv_bounds, n_constraint, 
                            old_pop=old_pop[-1])
            self.env.history.extend(old_pop)
            # print("history", len(self.env.history))
        else:
            self.env = Environment(popsize, dv_size, n_obj, optimizer,
                            eval_func, dv_bounds, n_constraint)
        self.eval_func = eval_func

        self.n_obj = self.env.n_obj
        # self.n_obj = len(eval_func( dummy_indiv.get_design_variable() ))
        print("n_obj:",self.n_obj)
        self.selector = selector
        self.mating = Mating(mating[0], mating[1], self.env.pool)

        print("set optimizer:", optimizer.name)
        if optimizer.name is "moead":
            if ksize == 0:
                ksize = 3
            self.optimizer = optimizer((self.env.popsize), self.n_obj, 
                                    self.selector, self.mating, ksize=ksize)

        elif optimizer.name is "moead_de":
            if ksize == 0:
                ksize = 3
            self.optimizer = optimizer((self.env.popsize), self.n_obj,
                                    self.selector, self.mating, ksize=ksize,
                                    CR=0.75, F=0.75, eta=20)

        elif optimizer.name is "c_moead":
            if ksize == 0:
                ksize = 3
            self.optimizer = optimizer((self.env.popsize), self.n_obj, 
                                        self.selector, self.mating,
                                        self.env.pool, n_constraint, ksize=ksize)
        
        # elif optimizer.name is "c_moead_dma":
        elif optimizer is C_MOEAD_DMA:
            if ksize == 0:
                ksize = 3
            if alpha == 0:
                alpha = 4
            self.optimizer = optimizer((self.env.popsize), self.n_obj, 
                                        self.selector, self.mating,
                                        self.env.pool, n_constraint, ksize=ksize, alpha=alpha,
                                        **kwargs)
        
        elif optimizer is C_MOEAD_DEDMA:
            if ksize == 0:
                ksize = 3
            if alpha == 0:
                alpha = 4
            self.optimizer = optimizer((self.env.popsize), self.n_obj, 
                                        self.selector, self.mating,
                                        self.env.pool, n_constraint, ksize=ksize, alpha=alpha,
                                        **kwargs)
            # print(C_MOEAD_DEDMA.mro())
            # input()

        elif optimizer is C_MOEAD_HXDMA:
            if ksize == 0:
                ksize = 3
            if alpha == 0:
                alpha = 4
            self.optimizer = optimizer((self.env.popsize), self.n_obj, 
                                        self.selector, self.mating,
                                        self.env.pool, n_constraint, ksize=ksize, alpha=alpha,
                                        **kwargs)

        self.optimizer.normalize = normalize

        # When running MOEA/D, updated popsize, so modify self.env.popsize.
        self.env.popsize = self.optimizer.popsize 
        self.env.nowpop.capacity = self.env.popsize
        # print("opt popsize", self.optimizer.popsize)

        if weight is not None:
            self.env.weight = np.array(weight)
            
        # Creating initial individuals.
        self.initialize(savepath)

    def __call__(self, iter):
        self.run(iter)

    # Creating initial individuals.
    def initialize(self, savepath=None):
        if self.restart == 0:
            for _ in range(self.optimizer.popsize):
                indiv = self.env.creator()
                
                # indiv.set_id(self.env.current_id)
                self.env.current_id = indiv.get_id()
                # print(type(indiv))
                indiv.set_boundary(self.env.dv_bounds)
                indiv.set_weight(self.env.weight)
                
                self.env.nowpop.append(indiv)

            for indiv in self.env.nowpop:
                #目的関数値を計算
                # print("func:", self.eval_func.__dict__)
                res = self.env.evaluate(indiv)
                if self.env.n_constraint > 0:
                    if indiv.is_feasible():
                        self.env.feasible_indivs_id.append(indiv.id)
            # print("res", res)

            #適応度計算
            self.optimizer.calc_fitness(self.env.nowpop)
                
            #初期個体を世代履歴に保存
            self.env.alternate()
            self.save_current_generation(savepath)


    def run(self, iter, savepath=None, nextline=None):
        if (nextline is None) and iter > 10:
            nextline = int(iter/10)
        else:
            nextline = 1

        # n_epoch = self.n_epoch
        if self.restart != 0:
            self.n_epoch = self.restart

        for i in range(iter):
            self.n_epoch += 1
            if i%nextline == 0:
                print()
            print(f"iter:{self.n_epoch:>5d}", end="\r\n")
            # for indiv in self.env.nowpop:
            #     print(indiv.get_id(), end=" ")
            self.optimizing()
            # if self.restart > 0:
            #     self.n_epoch = i + self.restart
            # else:
            #     self.n_epoch = i + 1
                
            if self.flag_save == True:
                self.result(save=True, fname=f"opt_result_epoch{self.n_epoch}.pkl")
                self.result(delete=True, fname=f"opt_result_epoch{self.n_epoch-1}.pkl")
            # print(len(self.optimizer.EP))
            print(f"EPsize:{len(self.optimizer.EP)}, Num of update ", self.optimizer.n_EPupdate, ", feasibleIndivs :", len(self.env.feasible_indivs_id))
            self.optimizer.n_EPupdate = 0
            print("ref point:", self.optimizer.ref_points)
            
            if savepath is not None:
                self.save_current_generation(savepath)

        print()


    def optimizing(self):
        # next_pop = Population(capa=len(self.env.nowpop))
        nowpop = copy.deepcopy(self.env.nowpop)
        # nowpop = self.env.nowpop
        
        for i in range(len(self.env.nowpop)):
            child = self.optimizer.get_offspring(i, nowpop, self.eval_func)
            if self.env.n_constraint > 0:
                # vioration = child.constraint_violation
                if child.is_feasible():
                    self.env.feasible_indivs_id.append(child.id)
                    # print("feasible append")

        next_pop = nowpop
        self.optimizer.calc_fitness(next_pop)

        self.env.alternate(next_pop)

    def result(self, save=False, fname=None, delete=False):
        result = np.array(self.env.history)
        
        if fname is None:
            fname = "opt_result.pkl"

        if save is True:
            with open(fname, "wb") as f:
                env = copy.copy(self.env)
                env.func = "problem"
                savedata = {
                    "result": result,
                    "env": env, 
                    "optimizer": self.optimizer }
                pickle.dump(savedata, f)
        
        if delete is True:
            if os.path.exists(fname):
                os.remove(fname)

        return result

    def save_current_generation(self, path):
        if path is None:
            return

        nowpop = self.env.nowpop
        gene = len(self.env.history)-1
        if gene < 1:
            gene = 0
        fname = os.path.join(path, f"gen_{gene}.pkl")
        print("save name = ", fname)
        EP_id = [p.id for p in self.optimizer.EP]
        savedata = {
                "nowpop": nowpop,
                "EP_id": EP_id,
                "epoch": self.n_epoch
            }
        if self.n_epoch == 0:
            fname == os.path.join(path, f"gen_0.pkl")
            savedata["optimizer"] = self.optimizer
            savedata["env"] = (
                        self.env.initializer,
                        self.env.weight,
                        self.env.dv_bounds,
                        self.env.dv_size,
                        # self.env.feasible_indivs_id,
                        self.env.func
                    )

        with open(fname, "wb") as f:
            pickle.dump(savedata, f)

