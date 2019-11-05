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
                        selector,
                        mating,
                        optimizer,
                        eval_func, 
                        ksize:int=None
                        dv_bounds:tuple=(0,1) #設計変数の上下限値
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
        """
        self.env = Environment(popsize, dv_size, optimizer,
                          eval_func, dv_bounds)
        
        self.nowpop = self.env.nowpop
        self.nobj = len(eval_func)
        self.selector = Selector(TournamentSelectionStrict, reset_cycle=2)
        self.mating = Mating(SBX, PM, self.env.pool)
        if optimizer.name is "moead":
            if ksize is None:
                ksize = 3
            self.optimizer = MOEAD(len(self.nowpop), self.nobj, 
                                    self.selector, self.mating, ksize=ksize)

        #初期個体の生成
        for _ in range(popsize):
            indiv = self.env.creator()
            
            # indiv.set_id(self.env.current_id)
            indiv.bounds = self.env.dv_bounds
            
            self.nowpop.append(indiv)

        for indiv in self.nowpop:
            #目的関数値を計算
            self.env.evaluate(indiv)

        #適応度計算
        self.optimizer.calc_fitness(self.nowpop)
        
        #初期個体を世代履歴に保存
        self.env.alternate()

    def __call__(self, iter):
        self.run(iter)

    def run(self, iter):
        for i in range(iter):
            print(f"iter:{i:>5d}")
            self.optimizing()

    def optimizing(self):
        # TODO: optimizerの実行コードを入れる
        next_pop = Population(capa=len(self.nowpop))

        for i in range(len(self.nowpop)):
            child = self.optimizer.get_offspring(i, self.nowpop)
            self.env.evaluate(child)
            next_pop.append(child)

        self.optimizer.calc_fitness(next_pop)

        self.env.alternate(next_pop)

    def advance(self):
        pass