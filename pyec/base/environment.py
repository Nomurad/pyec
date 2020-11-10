from typing  import List, Any
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
        self.data = []  # 全個体リスト

    def __call__(self, genome: np.ndarray, parents: Individual = None):
        self.indiv_creator(genome, parents)

    def indiv_creator(self, genome: np.ndarray, parents: Individual = None):
        """遺伝子情報から個体を生成，全個体リストに追加しておく

        Arguments:
            genome {np.ndarray} -- [遺伝子情報]
            parents {Individual} -- [親]
        """
        indiv = self.cls(genome)
        self.current_id = indiv.set_id(self.current_id)  # set id & renew current_id
        if parents is not None:
            indiv.set_parents_id(parents)
        self.append(indiv)
        return indiv

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return self.current_id

    def append(self, indiv):
        self.data.append(indiv)

    def pop(self, index):
        self.current_id -= 1
        return self.data.pop(index)

    def clear_indivpool(self):
        self.data = [] 


class Environment(object):
    """進化計算のパラメータなどを保存するクラス
    """

    def __init__(self, popsize: int,  # 1世代あたりの個体数
                 dv_size: int,  # 設計変数の数
                 n_obj: int,
                 optimizer,
                 eval_func=None, 
                 dv_bounds: tuple = (0, 1),   # 設計変数の上下限値
                 n_constraint=0,  # 制約条件の数
                 normalize=False,
                 old_pop=None
                 ):

        self.current_id = 0
        self.popsize = popsize
        self.dv_size = dv_size
        self.n_obj = n_obj
        if old_pop is None:
            print("Start EA.")
            self.nowpop = Population(capa=popsize)
        else:
            print("Re Start EA.")
            print("oldpop dict", old_pop.__dict__)
            self.nowpop = old_pop
        self.history: List[Any] = []  # 過去世代のpopulationのリスト
        self.EP_history: List[Any] = []
        self.pool = Pool()
        self.func = eval_func
        self.optimizer = optimizer
        self.weight = None  # 重み(正=>最小化, 負=>最大化)

        # 設計変数の上下限値 # None or (low, up) or ([low], [up])
        self.dv_bounds = dv_bounds

        self.n_constraint = n_constraint
        self.feasible_indivs_id: List[int] = []

        # initializerの設定
        self.initializer = UniformInitializer(dv_size) 
        self.creator = Creator(self.initializer, self.pool)

    def alternate(self, population=None, indivs=None):
        """世代交代時に呼び出し
        """
        self.history.append(tuple(self.nowpop))

        if population is not None:
            self.nowpop = population
        elif indivs is not None:
            self.nowpop = Population(indivs=indivs)
        else:
            # self.nowpop = Population(capa=self.popsize)
            pass

    def evaluate(self, indiv: Individual):
        """目的関数値を計算
           適応度はoptimizerを使って設定

        Arguments:
            indiv {Individual} -- [個体情報]
        """
        res = indiv.evaluate(self.func, indiv.get_design_variable(),
                             n_constraint=self.n_constraint)
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

    def __init__(self, initializer, pool: Pool):
        self.initializer = initializer
        self._pool = pool

    def __call__(self):
        genome = np.array(self.initializer())
        # indiv = Individual(genome)
        indiv = self._pool.indiv_creator(genome)
        return indiv

    def dummy_make(self):
        genome = np.array(self.initializer())
        indiv = Individual(genome)
        return indiv


class Normalizer(object):
    """評価値のnormalizer
    """
    def __init__(self, upper: list, lower: list, option="unhold"):
        if len(upper) != len(lower):
            raise Exception("UpperList size != LowerList size")
        self.upper = np.array(upper)
        self.lower = np.array(lower)
        self.option = option
        self.eps = 1e-16

    def _modificator(self, coeff):
        if self.lower < 0:
            self.lower = self.lower * coeff
        else:
            self.lower = 0.0

    def normalizing(self, indiv: Individual):
        val = np.array(indiv.value)
        res = (val - self.lower)/(self.upper - self.lower + self.eps)
        indiv.wvalue = list(res) 
        return res

    def ref_update(self, upper, lower):
        if self.option != "unhold":
            return

        upper = np.array(upper)
        lower = np.array(lower)

        temp = np.vstack((self.upper, upper))
        self.upper = np.max(temp, axis=0)

        temp = np.vstack((self.lower, lower))
        self.lower = np.min(temp, axis=0)
        print(f"upper/lower ref_points: {self.upper},{self.lower}")
