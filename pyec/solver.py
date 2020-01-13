import numpy as np
import pickle
import copy

from .base.indiv import Individual
from .base.population import Population
from .base.environment import Environment

from .operators.selection import Selector, TournamentSelection, TournamentSelectionStrict
from .operators.mutation import PolynomialMutation as PM
from .operators.crossover import SimulatedBinaryCrossover as SBX
from .operators.mating import Mating

from .optimizers.moead import MOEAD, MOEAD_DE, C_MOEAD_DE

class Solver(object):
    """進化計算ソルバー    
    """

    def __init__(self,  popsize: int, #1世代あたりの個体数
                        dv_size: int, #設計変数の数
                        nobj: int, #目的関数の数
                        selector,
                        mating,
                        optimizer,
                        eval_func, 
                        ksize: int= None,
                        dv_bounds: tuple = (0,1), #設計変数の上下限値
                        weight = None,
                        normalize = False,
                        n_constraint = 0,
                        old_pop=None
                        ):
        """solver initializer
        
        Arguments:
            popsize {int} -- [個体数]
            dv_size {int} -- [設計変数の数]
            selector      -- [selector]
            mating        -- [crossover,mutation]
            optimizer     -- [進化計算手法] 
            eval_func {[type]} -- [目的関数(評価関数)]
        
        Keyword Arguments:
            ksize {int}       -- [近傍サイズ] (default: None)
            dv_bounds {tuple} -- [設計変数の上下限値] (default: {(0,1)})
            weight {list or tuple} -- [目的関数の重み付け] (default: None)
            normalize {bool} -- [評価値の正規化] (default: False)
            n_constraint {int} -- [制約条件数] (dafault: 0)
            old_pop [Population] -- [last population, Restart時に使用]
        """
        self.restart = 0
        if old_pop is not None:
            self.restart = len(old_pop)
            print(f"start epoch is {self.restart}")
            # print("oldpop", old_pop[-1].__dict__)
            self.env = Environment(popsize, dv_size, optimizer,
                            eval_func, dv_bounds, n_constraint, 
                            old_pop=old_pop[-1])
            self.env.history.extend(old_pop)
            # print("history", len(self.env.history))
        else:
            self.env = Environment(popsize, dv_size, optimizer,
                            eval_func, dv_bounds, n_constraint)
        self.eval_func = eval_func

        self.nobj = nobj
        # self.nobj = len(eval_func( dummy_indiv.get_design_variable() ))
        print("nobj:",self.nobj)
        self.selector = selector
        self.mating = Mating(mating[0], mating[1], self.env.pool)

        print("set optimizer:", optimizer.name)
        if optimizer.name is "moead":
            if ksize is None:
                ksize = 3
            self.optimizer = MOEAD((self.env.popsize), self.nobj, 
                                    self.selector, self.mating, ksize=ksize)
        elif optimizer.name is "moead_de":
            if ksize is None:
                ksize = 3
            self.optimizer = MOEAD_DE((self.env.popsize), self.nobj, self.env.pool,
                                    self.selector, self.mating, ksize=ksize)
        elif optimizer.name is "c_moead_de":
            if ksize is None:
                ksize = 3
            self.optimizer = C_MOEAD_DE((self.env.popsize), self.nobj, self.env.pool,
                            n_constraint, self.selector, self.mating, ksize=ksize)

        self.optimizer.normalize = normalize

        #moea/dの場合, popsizeが更新されるので，修正する
        self.env.popsize = self.optimizer.popsize 
        self.env.nowpop.capacity = self.env.popsize
        # print("opt popsize", self.optimizer.popsize)

        if weight is not None:
            # print("set weight", weight)
            self.env.weight = np.array(weight)
            
        #初期個体の生成
        # if self.restart == 0:
        #     for _ in range(self.optimizer.popsize):
        #         indiv = self.env.creator()
                
        #         indiv.set_id(self.env.current_id)
        #         # print(type(indiv))
        #         indiv.set_boundary(self.env.dv_bounds)
        #         indiv.set_weight(self.env.weight)
                
        #         self.env.nowpop.append(indiv)

        #     for indiv in self.env.nowpop:
        #         #目的関数値を計算
        #         # print("func:", self.eval_func.__dict__)
        #         res = self.env.evaluate(indiv)
        #     # print("res", res)

        #     #適応度計算
        #     self.optimizer.calc_fitness(self.env.nowpop)
                
        #     #初期個体を世代履歴に保存
        #     self.env.alternate()

    def __call__(self, iter):
        self.run(iter)

    #初期個体の生成
    def initialize(self):
        if self.restart == 0:
            for _ in range(self.optimizer.popsize):
                indiv = self.env.creator()
                
                indiv.set_id(self.env.current_id)
                # print(type(indiv))
                indiv.set_boundary(self.env.dv_bounds)
                indiv.set_weight(self.env.weight)
                
                self.env.nowpop.append(indiv)

            for indiv in self.env.nowpop:
                #目的関数値を計算
                # print("func:", self.eval_func.__dict__)
                res = self.env.evaluate(indiv)
            # print("res", res)

            #適応度計算
            self.optimizer.calc_fitness(self.env.nowpop)
                
            #初期個体を世代履歴に保存
            self.env.alternate()

    def run(self, iter, nextline=None):
        if nextline is None:
            nextline = int(iter/10)

        for i in range(iter):
            if i%nextline == 0:
                print()
            print(f"iter:{i+1:>5d}", end="\r")
            # for indiv in self.env.nowpop:
            #     print(indiv.get_id(), end=" ")
            self.optimizing()
            if self.restart > 0:
                n_epoch = i + self.restart
            else:
                n_epoch = i + 1
                
            self.result(save=True, fname=f"opt_result_epoch{n_epoch}.pkl")
        print()

    def optimizing(self):
        next_pop = Population(capa=len(self.env.nowpop))
        # print(len(self.env.history))

        for i in range(len(self.env.nowpop)):
            # print(i, len(next_pop), self.optimizer.neighbers[i])
            child = self.optimizer.get_offspring(i, self.env.nowpop, self.eval_func)
            res = self.env.evaluate(child)
            next_pop.append(child)
            # print(child.value, " | ", res)

        self.optimizer.calc_fitness(next_pop)

        self.env.alternate(next_pop)

    def result(self, save=False, fname=None):
        result = np.array(self.env.history)
        # print("result shape",result.shape)

        # for i, pop in enumerate(result):
        #     print()
        #     for indiv in pop:
        #         print(f"{i}, {indiv._id:>10} \t{indiv.value}")
        # np.savetxt(path, res, delimiter=",")
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
        
        return result

    