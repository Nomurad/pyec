from typing import List, overload

from .indiv import Individual

class PopulationError(Exception):
    """
    population class 用例外クラス
    """


class Population(object):
    """
        Population class
    """

    def __init__(self, pop=None, indivs=None, capa=None):
        super().__init__()
        # if pop is Population => include pop data
        if isinstance(pop, Population):
            self.pop = pop 
            self.capacity = pop.capacity
            # self.current_id = 0
        
        else:
            if indivs is not None:
                self.pop:List[Individual] = indivs
            else:
                self.pop:List[Individual] = []

            if capa is None:
                raise Exception("You should set population capacity.")
            else:
                self.capacity = capa
                
    def get_inviv(self, key:int) -> Individual:
        return self.pop[key]

    @overload
    def __getitem__(self, key:int) -> Individual: ...

    @overload
    def __getitem__(self, s:slice) -> List[Individual]: ...

    def __getitem__(self, key):
        return self.pop[key]
    
    def __setitem__(self, key:int, indiv:Individual):
        self.pop[key] = indiv
    
    def __len__(self) -> int:
        return len(self.pop)

    def __add__(self, other) -> "Population":
        if not isinstance(other, Population):
            return NotImplemented
        pop = self.pop + other.pop
        return Population(indivs=pop, capa=len(pop))

    def append(self, indiv:Individual):
        if not isinstance(indiv, Individual):
            raise PopulationError("Only you can append indiv which class is Individual.")
        elif self.filled():
            raise PopulationError("This population is already filled")
        
        self.pop.append(indiv)

    def sort(self, *args, **kwargs):
        self.pop.sort(*args, **kwargs)

    def filled(self) -> bool:
        return (len(self.pop) >= self.capacity)

