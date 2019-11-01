import numpy as np 

from .indiv import Individual, Fitness
from .population import Population
from ..operators.initializer import UniformInitializer

class EnvironmentError(Exception):
    pass 

class Pool(object):
    """
    """

    def __init__(self):
        self.cls = Individual
        self.current_id = 0
        self.data = [] #全個体リスト

    def __call__(self, genome:np.ndarray):
        """遺伝子情報から個体を生成，全個体リストに追加しておく
        
        Arguments:
            genome {np.ndarray} -- [遺伝子情報]
        """
        indiv = self.cls(genome)
        self.current_id = indiv.set_id(self.current_id) #set id & renew current_id
        self.append(indiv)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return self.current_id
    
    def append(self, indiv):
        self.data.append(indiv)

class Environment(object):
    """進化計算のパラメータなどを保存するクラス
    """
    
    def __init__(self,  popsize:int, #1世代あたりの個体数
                        dv_size:int, #設計変数の数
                        optimizer,
                        eval_func=None, 
                        dv_bounds:tuple=(0,1) #設計変数の上下限値
                        ):

        self.current_id = 0
        self.nowpop = Population(capa=popsize)
        self.pool = Pool()
        self.func = eval_func
        self.optimizer = optimizer()
        self.weight = None #重み(正=>最小化, 負=>最大化)

        #設計変数の上下限値 # None or (low, up) or ([low], [up])
        self.dv_bounds = dv_bounds

        #initializerの設定
        self.initializer = UniformInitializer(dv_size) 
        self.creator = Creator(self.initializer, dv_size)

        
    def evaluate(self, indiv:Individual):
        """目的関数値を計算
           適応度はoptimizerを使って設定
        
        Arguments:
            indiv {Individual} -- [個体情報]
        """
        res = indiv.evaluate(self.func)
        return res 

    def evaluated_all(self):
        flag_evaluated = True   
        for indiv in self.nowpop:
            flag_evaluated = indiv.evaluated()
            if flag_evaluated is False:
                return False
        
        return True



class Creator(object):
    """初期個体の生成器
    """
    
    def __init__(self, initializer, pool:Pool):
        self.initializer = initializer
        self._pool = pool
        
    def __call__(self):
        genome = np.array(self.initializer())
        indiv = Individual(genome)
        indiv = self._pool()
        return indiv