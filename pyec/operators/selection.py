import random 
from operator import itemgetter

###########################################################

def identity(x):
    return x

def separate_random_func(self, pop, k):
    selected = []
    rest = list(pop)
    size = len(pop)
    for i in range(k):
        index = random.randrange(size - i)
        selected.append(rest.pop(index))
    return selected, rest

###########################################################
class SelectorError(Exception):
    pass 

class Selector(object):
    
    def __init__(self, selection, reset_cycle):
        self._selection = selection()
        self._reset_cycle = reset_cycle
        self._population = None
        self._stored = []

    def __call__(self, population=None):
        if (population is None) and (self._population is None):
            raise SelectorError("You should set population.")
        elif (population is None) and (self._population is not None):
            population = self._population
        
        self._stored = []
        rest = []
        while(True):
            if not rest:
                rest = list(population)
            
            selected, rest = self._selection(rest)
            
            if selected is None:
                continue
            
            self._stored.append(selected)
            if len(self._stored) >= self._reset_cycle:
                break
        
        return self._stored

    def Set_population(self, population):
        self._population = population


class SelectionIterator(object):
    ''' 交配の親選択イテレータ
    個体集団から親個体を選択し，選択された親個体を解集団から削除する(削除方式はselection関数に依存)
    reset_cycleが与えられた場合は解をreset_cycle個生成するごとに解集団をpopulationで初期化する
    '''
    def __new__(cls, selection, population=None, reset_cycle=None):
        # if population is None:
        #     return PartialSelectionIterator(selection, pool)
        return super().__new__(cls)

    def __init__(self, selection, population, reset_cycle=None):
        self._selection = selection
        self._population = population
        self._reset_cycle = reset_cycle
        self._stored = []

    def __iter__(self):
        rest = []
        counter = 0
        i = 0
        while True:
            # print('iter:', i)
            if not rest or (self._reset_cycle and counter == self._reset_cycle):
                # print('reset:', i, counter)
                rest = list(self._population)
                counter = 0
            selected, rest = self._selection(rest)
            if selected is None:
                continue
            counter += 1
            self._stored.append(selected)
            i += 1
            yield selected

    def __getnewargs__(self):
        return self._selection, self._population, self._reset_cycle

###########################################################


class RandomSelection(object):
    def __init__(self):
        pass

    def __call__(self, population):
        s = len(population)
        if s == 0:
            return None, []
        pop = list(population)
        index = random.randrange(s)
        return pop.pop(index), pop


class RouletteSelection(object):
    def __init__(self, key=itemgetter(0)):
        # default key: fitst item of touple
        self.key = key

    def __call__(self, population):
        pop = list(population)
        fits = [self.key(x) for x in pop]
        wheel = sum(fits) * random.random() # fitness[0]
        for i, fit in enumerate(fits):
            if fit <= 0:
                continue
            wheel -= fit
            if wheel < 0:
                return pop.pop(i), pop
        raise RuntimeError('Error: in roulette')


class TournamentSelection(object):
    def __init__(self, key=identity, ksize=2):
        self.key = key
        self.ksize = ksize

    def __call__(self, population):
        s = len(population)
        # k = min(self.ksize, s)
        k = self.ksize
        if s < k:
            return None, []
        pop = list(population)
        indices = random.sample(range(s), k)

        # pop = random.sample(population, k)
        index = max(indices, key=pop.__getitem__)
        return pop.pop(index), pop


class TournamentSelectionStrict(object):
    separate_random = separate_random_func

    def __init__(self, key=identity, ksize=2):
        self.key = key
        self.ksize = ksize

    # def __len__(self):
    #     pass

    def __call__(self, population):
        s = len(population)
        # k = min(self.ksize, s)
        k = self.ksize
        if s < k:
            return None, []
        pop, rest = self.separate_random(population, k)
        return max(pop), rest


class TournamentSelectionDCD(object):
    separate_random = separate_random_func

    def __init__(self, key=identity):
        self.key = key
        # self.pat = [0, 0, 0, 0, 0]

    def __call__(self, population):
        s = len(population)
        # k = min(self.ksize, s)
        k = 2
        if s < k:
            return None, []
        pop, rest = self.separate_random(population, k)

        # pop = list(population)
        # indices = random.sample(range(s), k)

        # pop = random.sample(population, k)

        # def ret(i):
        #     # return pop.pop(indices[i]), pop
        #     return getpop(i), [x for i, x in enumerate(pop) if i not in indices]
        # def getpop(i):
        #     return pop[indices[i]]

        # 優越関係比較
        if pop[0].dominates(pop[1]):
            # self.pat[0] += 1
            return pop[0], rest
        elif pop[1].dominates(pop[0]):
            # self.pat[1] += 1
            return pop[1], rest

        if len(pop[0]) >= 2 and len(pop[1]) >= 2:
            # 混雑度比較
            if pop[0][1] < pop[1][1]:
                # self.pat[2] += 1
                return pop[1], rest
            elif pop[0][1] > pop[1][1]:
                # self.pat[3] += 1
                return pop[0], rest
        # self.pat[4] += 1

        if random.random() <= 0.5:
            return pop[0], rest
        return pop[1], rest