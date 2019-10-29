from .base.environment import Environment


class Solver(object):
    """進化計算ソルバー    
    """

    def __init__(self,  popsize:int, #1世代あたりの個体数
                        dv_size:int, #設計変数の数
                        optimizer,
                        eval_func=None, 
                        dv_bounds:tuple=(0,1) #設計変数の上下限値
                        ):
        """solver initializer
        
        Arguments:
            popsize {int} -- [個体数]
            dv_size {int} -- [設計変数の数]
            optimizer     -- [進化計算手法]
        
        Keyword Arguments:
            eval_func {[type]} -- [目的関数(評価関数)] (default: {None})
            dv_bounds {tuple} -- [設計変数の上下限値] (default: {(0,1)})
        """
        
        self.env = Environment(popsize, dv_size, optimizer,
                          eval_func, dv_bounds)
        
        self.nowpop = self.env.nowpop
        self.optimizer = optimizer


        #初期個体の生成
        for _ in range(popsize):
            indiv = self.env.creator()
            
            indiv.id = self.env.current_id
            indiv.bounds = self.env.dv_bounds
            
            self.nowpop.append(indiv)
            self.env.current_id += 1

        for indiv in self.nowpop:
            self.env.evaluate(indiv)

        #適応度計算
        self.optimizer.calc_fitness(self.nowpop)