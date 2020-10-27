import random 
from abc import ABC, ABCMeta, abstractmethod
import numpy as np 

class CrossoverError(Exception):
    pass

class AbstractCrossover(metaclass=ABCMeta):
    def __init__(self, rate):
        self.rate = rate

    @abstractmethod
    def __call__(self, genomes):
        pass


class BlendCrossover(AbstractCrossover):
    """BLX-alpha
    """
    def __init__(self, rate=0.9, alpha=0.5, oneout=False):
        self.rate = rate
        self.alpha = alpha
        self.oneout = oneout

    def __call__(self, genomes):
        x1, x2 = genomes

        if random.random() > self.rate:
            return x1, x2
        gamma = (1 + 2 * self.alpha) * np.random.rand(x1.shape) - self.alpha

        if self.oneout:
            y = (1 - gamma) * x1 + gamma * x2
            return y
        else:
            y1 = (1 - gamma) * x1 + gamma * x2
            y2 = gamma * x1 + (1 - gamma) * x2
            return y1, y2


class SimulatedBinaryCrossover(AbstractCrossover):
    """Simulated Binary Crossover(SBX)
    """
    def __init__(self, rate=0.9, eta=20, oneout=False):
        self.rate = rate
        self.eta = eta
        self.oneout = oneout

    def __call__(self, genomes):
        y1, y2 = map(np.array, genomes)

        if random.random() > self.rate:
            return y1, y2

        size = min(len(y1), len(y2))

        xl, xu = 0.0, 1.0
        eta = self.eta

        for i in range(size):
            if random.random() <= 0.5:
                # This epsilon should probably be changed for 0 since
                # floating point arithmetic in Python is safer
                if abs(y1[i] - y2[i]) > 1e-14:
                    x1 = min(y1[i], y2[i])
                    x2 = max(y1[i], y2[i])
                    rand = random.random()

                    beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                    alpha = 2.0 - beta**-(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))

                    c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                    beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                    alpha = 2.0 - beta**-(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                    c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                    c1 = min(max(c1, xl), xu)
                    c2 = min(max(c2, xl), xu)

                    if random.random() <= 0.5:
                        y1[i] = c2
                        y2[i] = c1
                    else:
                        y1[i] = c1
                        y2[i] = c2

        if self.oneout:
            return y1
        else:
            return y1, y2


class DifferrentialEvolutonary_Crossover(object):
    """ Differential Evolution(DE) operator
        This operator contains Mutation operator.
    """

    def __init__(self, CR=1.0, F=0.5, pm=0.1, eta=20, sigma=None):
        self.CR = CR 
        self.set_scaling_F(F, sigma)
        # self.scaling_F = F
        # if type(F) is float:
        #     self.scaling_F = lambda x: F 
        # else:
        #     print("setting F by lambda")
        self.pm = pm
        self.eta = eta

        self._dv_modifier_initializer(1)

    def __call__(self, genomes):
        """ This method needs 3 genomes.
            1st genomes -> arbitrary indiv's genome,
            2nd & 3rd genomes -> random select indiv's genome from Population.
        """

        try:
            p1, p2, p3 = map(np.array, genomes)
        except:
            raise CrossoverError("you should set 3 parents")

        num_dv = len(p1)
        # Fs = [self.scaling_F() for _ in range(num_dv))]
        # vi_genome = p1 + Fs*(p2 - p3)
        Fs = self.scaling_F(2)
        # print("scaling F = ", Fs)
        vi_genome = p1 + Fs*(p2 - p3)
        j_rand = random.randint(0, num_dv)

        child_dv = np.zeros(num_dv)
        for i in range(num_dv):
            if random.uniform(0.0, 1.0) < self.CR or i == j_rand:
                child_dv[i] = vi_genome[i]
            else:
                child_dv[i] = p1[i]

        eta = self.eta
        rand = random.uniform(0.0, 1.0)
        if rand < 0.5:
            delta_k = (2*rand)**(1/(eta+1)) - 1
        else:
            delta_k = 1 - (2 - 2*rand)**(1/(eta+1))

        a_k = 0.0
        b_k = 1.0
        for i in range(num_dv):
            if random.uniform(0.0, 1.0) < self.pm:
                child_dv[i] = child_dv[i] + (delta_k*(b_k - a_k))

        self.modifier(child_dv)

        return child_dv

    def set_scaling_F(self, F, sigma=None):
        self._scaling_F = F
        self._scaling_sigma = sigma
        if sigma is None:
            self.scaling_F = self._constant_scaling_F
        elif sigma is not None:
            self.scaling_F = self._gauss_scaling_F
        else:
            raise CrossoverError("Invalid F.")
        print("scaling F is ", (self.scaling_F.__doc__))
        print(f"F, sigma = {self._scaling_F}, {self._scaling_sigma}")

    def _gauss_scaling_F(self):
        """ gaussian scaling factor
        """
        res = random.gauss(self._scaling_F, self._scaling_sigma)
        return res

    def _gauss_scaling_F_list(self):
        """ gaussian scaling factor
        """
        res = random.gauss(self._scaling_F, self._scaling_sigma)
        return res

    def _constant_scaling_F(self, optional=None):
        """ const scaling factor
        """
        if optional is not None:
            return self._scaling_F(optional)
        else:
            self._scaling_F = lambda x: self._scaling_F
        return self._scaling_F

    def _dv_modifier_initializer(self, mode):
        """ mode  | func
            -------------------
            0     | random 
            1     | minmax
        """

        if mode == 0:
            self.modifier = self._modifier_rand
        elif mode == 1:
            self.modifier = self._modifier_minmax
        else:
            raise CrossoverError("mode number are 0, 1.")

    def _modifier_minmax(self, dv):
        for i in range(len(dv)):
            if dv[i] < 0.0:
                dv[i] = 0.0
                continue

            if dv[i] > 1.0:
                dv[i] = 1.0
        return dv

    def _modifier_rand(self, dv):
        for i in range(len(dv)):
            if dv[i] < 0.0 or dv[i] > 1.0:
                dv[i] = random.uniform(0.0, 1.0)
        return dv


if __name__ == "__main__":
    g_size = 3
    genome1 = np.random.rand(g_size)
    genome2 = np.random.rand(g_size)

    cross = SimulatedBinaryCrossover(rate=1.0, eta=20)
    # cross = BlendCrossover(rate=1.0, alpha=0.5)

    out1, out2 = cross([genome1, genome2])

    print(genome1, genome2)
    print(out1, out2)
