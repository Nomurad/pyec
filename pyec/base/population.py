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
                self.pop = indivs
            else:
                self.pop = []

            if capa is None:
                raise Exception("You should set population capacity.")
            else:
                self.capacity = capa
                

    def __getitem__(self, key) -> Individual:
        return self.pop[key]
    
    def __setitem__(self, key, indiv):
        self.pop[key] = indiv
    
    def __len__(self):
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

