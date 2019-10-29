from .indiv import Individual, Genome, Fitness
from .population import Population
from ..operators.initializer import UniformInitializer

class EnvironmentError(Exception):
    pass 

class Pool(object):
    """
    すべてのPopulationが入る(世代ごと)
    """

    def __init__(self):
        self.current_epoch = 0
        self.data = []

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return self.current_epoch
    
    def append(self, pop):
        self.current_epoch += 1
        self.data.append(pop)

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
        genome = indiv.genome
        val = self.evaluate(genome)
        indiv.set_value(val)

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
    
    def __init__(self, initializer, dv_size:int):
        self.initializer = initializer
        self.dv_size = dv_size
        
    def __call__(self):
        genome = Genome(self.initializer())
        indiv = Individual(genome)
        return indiv