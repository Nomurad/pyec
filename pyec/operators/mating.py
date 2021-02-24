from typing import Union, List
import random

import numpy as np 
from ..base.indiv import Individual
from ..base.environment import Pool

from .selection import TournamentSelection, TournamentSelectionStrict
from .crossover import AbstractCrossover
from .crossover import BlendCrossover, SimulatedBinaryCrossover
from .mutation import PolynomialMutation


class MatingError(Exception):
    pass 


class Mating(object):

    def __init__(self, crossover: AbstractCrossover, mutation: PolynomialMutation, pool: Pool):
        self._crossover = crossover
        self._mutation = mutation
        self._pool = pool
        self._parents: List[Individual] = []
        self._stored: List[Individual] = []
    
    def __repr__(self):
        return f"cross:{self._crossover}, mutate:{self._mutation}"

    @property
    def pool(self):
        return self._pool

    def clear_pool(self):
        print("indiv pool clear!")
        self._pool.clear_indivpool()
        return 

    def __call__(self, parents=None, singlemode=True) -> List[Individual]:
        if (parents is None) and (self._parents is None):
            raise MatingError("You should set parents.")
        elif (parents is None) and (self._parents is not None):
            parents = self._parents

        self._stored = []

        parent_genomes = [indiv.get_genome() for indiv in parents]
        # print("parent genomes:", parent_genomes)

        child_genomes = self._crossover(parent_genomes)

        if singlemode == True:
            child_genome = random.choice(child_genomes)
            child_genome = self._mutation(child_genome)   # 一定確率で突然変異
            child_indiv = self._pool.indiv_creator(child_genome, parents)
            child_indiv.set_weight(parents[0].weight)
            child_indiv.set_boundary(parents[0].bounds)
            self._stored.append(child_indiv)
            return self._stored

        for child_genome in child_genomes:
            child_genome = self._mutation(child_genome)   # 一定確率で突然変異
            child_indiv = self._pool.indiv_creator(child_genome, parents)
            # child_indiv.set_parents_id(parents)
            child_indiv.set_weight(parents[0].weight)
            child_indiv.set_boundary(parents[0].bounds)
            self._stored.append(child_indiv)

        return self._stored

    def set_parents(self, parents):
        self._parents = parents


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
        
