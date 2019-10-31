import numpy as np 
from ..base.indiv import Individual, Genome
from ..base.environment import Pool

from .selection import TournamentSelection, TournamentSelectionStrict
from .crossover import BlendCrossover, SimulatedBinaryCrossover
from .mutation import PolynomialMutation

class PartialMatingIterator(object):
    ''' MatingIteratorの部分適用オブジェクト
    '''
    def __init__(self, crossover, mutation, pool):
        self._crossover = crossover
        self._mutation = mutation
        self._pool = pool

    def __call__(self, parents):
        return MatingIterator(self._crossover, self._mutation, 
                              self._pool, parents)

class MatingIterator(object):

    def __new__(cls, crossover, mutation, pool, parents=None):
        if parents is None:
            return PartialMatingIterator(crossover, mutation, pool)
        return super().__new__(cls)

    def __init__(self, crossover, mutation, pool:Pool, parents):
        self._crossover = crossover
        self._mutation = mutation
        self._pool = pool
        self._parents = parents
        self._stored = []

    def __iter__(self):
        parent_genomes = [indiv.get_genome() for indiv in self._parents]
        child_genomes = self._crossover(parent_genomes)

        for child_genome in child_genomes:
            child_genome = self._mutation(child_genome) #一定確率で突然変異
            child_indiv = self._pool(child_genome)
            self._stored.append(child_indiv)
            yield child_indiv
        