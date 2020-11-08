import random 
import copy 
from itertools import chain 
from typing import List, Tuple, Union, Optional

import numpy as np 

from ..base.indiv import Individual
from ..base.population import Population
from ..base.environment import Pool, Normalizer

from ..operators.initializer import UniformInitializer
from ..operators.crossover import SimulatedBinaryCrossover as SBX
from ..operators.mutation import PolynomialMutation as PM
from ..operators.selection import TournamentSelection, TournamentSelectionStrict
from ..operators.selection import SelectionIterator, Selector
from ..operators.mating import MatingIterator, Mating
from ..operators.sorting import NonDominatedSort

from . import Optimizer, OptimizerError

class NSGAError(OptimizerError):
    pass