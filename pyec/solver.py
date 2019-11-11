import numpy as np

from .base.indiv import Individual
from .base.population import Population
from .base.environment import Environment

from .operators.selection import Selector, TournamentSelection, TournamentSelectionStrict
from .operators.mutation import PolynomialMutation as PM
from .operators.crossover import SimulatedBinaryCrossover as SBX
from .operators.mating import Mating

from .optimizers.moead import MOEAD

class Solver(object):
    """進化計算ソルバー    
    """

    def __init__(self,  popsize:int, #1世代あたりの個体数
                        dv_size:int, #設計変数の数
                        nobj:int, #目的関数の数
                        selector,
                        mating,
                        optimizer,
                        eval_func, 
                        ksize:int=None,
                        dv_bounds:tuple=(0,1), #設計変数の上下限値
                        weight=None
                        ):
        """solver initializer
        
        Arguments:
            popsize {int} -- [個体数]
            dv_size {int} -- [設計変数の数]
            selector      -- [selector]
            mating        -- [mating]
            optimizer     -- [進化計算手法] 
            eval_func {[type]} -- [目的関数(評価関数)]
        
        Keyword Arguments:
            ksize {int}       -- [近傍サイズ] (default: None)
            dv_bounds {tuple} -- [設計変数の上下限値] (default: {(0,1)})
            weight {list or tuple} -- [目的関数の重み付け] (default: None)
        """
        self.env = Environment(popsize, dv_size, optimizer,
                          eval_func, dv_bounds)
        self.eval_func = eval_func
        
        # self.nowpop = self.env.nowpop
        self.nobj = nobj
        # dummy_indiv = self.env.creator.dummy_make()
        # self.nobj = len(eval_func( dummy_indiv.get_design_variable() ))
        print("nobj:",self.nobj)
        self.selector = Selector(TournamentSelectionStrict, reset_cycle=2)
        self.mating = Mating(SBX(rate=1.0), PM(), self.env.pool)
        if optimizer.name is "moead":
            if ksize is None:
                ksize = 3
            self.optimizer = MOEAD((popsize), self.nobj, 
                                    self.selector, self.mating, ksize=ksize)

        #初期個体の生成
        for _ in range(popsize):
            indiv = self.env.creator()
            
            # indiv.set_id(self.env.current_id)
            # print(type(indiv))
            indiv.set_boundary(self.env.dv_bounds)
            if weight is not None:
                print("set weight", weight)
                self.env.weight = np.array(weight)
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

    def __call__(self, iter):
        self.run(iter)

    def run(self, iter):
        for i in range(iter):
            print(f"iter:{i+1:>5d}")
            # for indiv in self.env.nowpop:
            #     print(indiv.get_id(), end=" ")
            # print()
            self.optimizing()

    def optimizing(self):
        # TODO: optimizerの実行コードを入れる
        next_pop = Population(capa=len(self.env.nowpop))
        # print(len(self.env.history))

        for i in range(len(self.env.nowpop)):
            # print(i, len(next_pop), self.optimizer.neighbers[i])
            child = self.optimizer.get_offspring(i, self.env.nowpop, self.eval_func)
            self.env.evaluate(child)
            next_pop.append(child)

        self.optimizer.calc_fitness(next_pop)

        self.env.alternate(next_pop)

    def result(self):
        result = np.array(self.env.history)
        print("result shape",result.shape)

        # for i, pop in enumerate(result):
        #     print()
        #     for indiv in pop:
        #         print(f"{i}, {indiv._id:>10} \t{indiv.value}")
        # np.savetxt(path, res, delimiter=",")
        return result

    def advance(self):
        pass