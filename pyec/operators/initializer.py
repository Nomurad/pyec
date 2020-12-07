import numpy as np

class UniformInitializer(object):
    ''' [0, 1]の範囲の一様乱数による実数配列を返す '''

    def __init__(self, dvsize, **kwargs):
        self._size = dvsize

    def __call__(self):
        return np.random.uniform(size=self._size)

class Latin_HyperCube_Sampling(object):

    def __init__(self, dvsize, popsize, n_obj=2, bounds=(0.0, 1.0)):
        self._size = dvsize 
        self.ini_indivs = []
        self.popsize = popsize
        self.n_obj = n_obj
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.count = 0
        self.calc_lhs()

    def calc_lhs(self):
        lb = self.lb 
        ub = self.ub
        n = self._size
        M = self.popsize

        rng = np.random.RandomState()
        f = lambda x: (ub-lb)*x + lb 
        g = lambda x: (x-rng.uniform())/M
        rnd_grid = np.array([rng.permutation(list(range(1, M + 1))) for _ in range(n)])
        lhs = [[f(g(rnd_grid[d][m])) for d in range(n)] for m in range(M)]
        self.lhs = np.array(lhs)

    def __call__(self):
        if self.count >= len(self.lhs):
            raise Exception
        else:
            print(self.count)

        res = self.lhs[self.count]
        self.count += 1
        
        return res