#! /usr/bin/env python3

'''
Abstruct
'''

import math
import random
from abc import ABCMeta, abstractmethod

import numpy as np


A = 10
PI2 = 2 * math.pi
PI4 = 4 * math.pi
PI6 = 6 * math.pi
PI10 = 10 * math.pi

class TestProblem_Error(Exception):
    pass
class TestProblem(metaclass=ABCMeta):

    def __init__(self, n_obj):
        self.n_obj = n_obj

    @abstractmethod
    def __call__(self):
        pass 

################################################################################
# S.O.
################################################################################

def rastrigin(x):
    return A * len(x) + sum(map(lambda v: v ** 2 - A * math.cos(PI2 * v), x))

# rastrigin.range = [-5, 5]


def rosenbrock(x):
    return sum(map(lambda i: 100 * (x[i+1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2,
                   range(len(x) - 1)))

# rosenbrock.range = [-2, 2]


################################################################################
# M.O.
################################################################################

def zdt1(x, *args):
    if args:
        x = [x, *args]
    n = len(x)
    if n == 1:
        return x[0], 1 - math.sqrt(x[0])

    # g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f1 = x[0]
    g = 1 + 9/(n - 1) * sum(x[1:])
    h = 1.0 - math.sqrt(f1/g)
    f2 = g*h
    # return x[0], g * (1 - math.sqrt(x[0] / g))
    return f1, f2


def zdt2(x):
    if len(x) == 1:
        return x[0], 1 - x[0] ** 2

    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    return x[0], g * (1 - (x[0] / g) ** 2)


def zdt3(x):
    if len(x) == 1:
        return x[0], 1 - math.sqrt(x[0]) - x[0] * math.sin(PI10 * x[0])

    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    v = x[0] / g
    return x[0], g * (1 - math.sqrt(v) - v * math.sin(PI10 * x[0]))


def zdt4(x):
    if len(x) == 1:
        return x[0], 1 - math.sqrt(x[0])

    g = 1 + 10 * (len(x) - 1) \
        + sum(map(lambda v: v ** 2 - 10 * math.cos(PI4 * v), x[1:]))
    return x[0], g * (1 - math.sqrt(x[0] / g))


def zdt6(x):
    f = 1 - math.exp(-4 * x[0]) * math.sin(PI6 * x[0]) ** 6
    if len(x) == 1:
        return f, 1 - f ** 2

    g = 1 + 9 * (np.sum(x[1:]) / (len(x) - 1)) ** 0.25
    return f, g * (1 - (f / g) ** 2)


################################################################################
# M.O. w/ const
################################################################################

def osy(x):
    """ The Osyczka and Kundu(OSY) test problem is 
        a six-variable problem with 2-objectives 
        and 6 inquality constraints.

        where 
        0<=x1,x2,x6<=10,
        1<=x3,x5<=5,
        0<=x4<=4

    """
    if len(x) == 1:
        return x[0], 1 - math.sqrt(x[0])

    f1 = sum(map(lambda i, v: (-1 if i else -25) * (x[i] - v)**2, 
                range(5), ([2, 2, 1, 4, 1]) ))
    f2 = sum(map(lambda v: v ** 2, x))
    g1 = x[0] + x[1] - 2
    g2 = 6 - x[0] - x[1]
    g3 = 2 - x[1] + x[0]
    g4 = 2 - x[0] + 3 * x[1]
    g5 = 4 - (x[2] - 3) ** 2 - x[3]
    g6 = (x[4] - 3) ** 2 + x[5] - 4
    return (f1, f2), (g1, g2, g3, g4, g5, g6)

def tnk(x):
    """TNK -- n_obj=2, n_dv=2, n_constraint=2
    
    Arguments:
        x {[np.ndarray]} -- [ 0 <= x[0],x[1] <= PI ]
    """
    f1 = x[0]
    f2 = x[1]

    g1 = x[0]**2 + x[1]**2 -1 - 0.1*np.cos(16*np.arctan(x[0]/x[1])) #>=0
    g2 = -(x[0] - 0.5)**2 - (x[1] - 0.5)**2 + 0.5   #>=0

    return (f1, f2), (g1, g2)

class Constraint_TestProblem(TestProblem):
    def __init__(self, n_obj=2, n_const=2):
        super().__init__(n_obj)
        self.n_const = n_const
    
    @abstractmethod
    def __call__(self):
        pass


class mCDTLZ(Constraint_TestProblem):

    def __init__(self, n_obj=2, n_const=2, phi=0.1):
        super().__init__(n_obj, n_const)
        self.phi = phi

    def __call__(self, x):
        f = []
        for i in range(len(x)):
            f_i = (1/())

class Knapsack(Constraint_TestProblem):

    def __init__(self, n_obj=2, n_items=500, n_const=2, phi=0.8):
        """ 
            n_obj: num of objectives (m)
            n_items: num of items (l)
            n_const: num of knapsack (k)
            phi: feasibility ratio for each knapsack.
        """
        super().__init__(n_obj, n_const)

        class kp_item(object):
            def __init__(self, 
                            n_obj, 
                            n_const, 
                            rand_lower=10,
                            rand_upper=100
                            ):
                self.m = n_obj  # num of obj
                self.k = n_const    # num of knapsack(constraint)
                self.profits = [random.randint(rand_lower, rand_upper) for i in range(n_obj)]
                self.weights = [random.randint(rand_lower, rand_upper) for i in range(n_const)]

        # phi setting
        if not hasattr(phi, "__iter__"):
            self.phi = [phi]*self.n_const
        elif len(phi) == self.n_const:
            self.phi = phi
        else:
            raise TestProblem_Error("phi array size is invalid.")
        print(self.phi)
        
        # num of items
        self.n_items = n_items
        # init items
        self.items = [kp_item(self.n_obj, self.n_const) for i in range(self.n_items)]
        # knapsack's capacity
        self.capa = []
        for j in range(self.n_const):
            c_j = self.phi[j]*sum([self.items[l].weights[j] for l in range(self.n_items)])
            self.capa.append(c_j)
        
        # for debug
        print("init knapsack")
        print("items : ", self.items[0].__dict__)
        print("capa : ", self.capa)
    
    def __call__(self, x):
        # print("callable")

        if not hasattr(x, "__iter__"):
            raise TestProblem_Error("x must be iterable.")
        
        f = []
        for i in range(self.n_obj):
            _profits = [self.items[l].profits[i] for l in range(self.n_items)]
            _p_times_x = list(map(lambda p_li,x_l: p_li*x_l, _profits,x))
            f_i = sum( _p_times_x )
            f.append(f_i)

        cv = []
        for j in range(self.n_const):
            _weights = [self.items[l].weights[j] for l in range(self.n_items)]
            _w_times_x = list(map(lambda w_li,x_l: w_li*x_l, _weights,x))
            v_i = sum(_w_times_x) - self.capa[j]
            print(self.capa[j], v_i)
            if v_i <= 0:
                v_i = 0
            cv.append(v_i)

        return f, cv 



################################################################################

def __test__():
    x = [0, 1, 2, 3, 4, 5]
    print(*osy(x))

    x = [10, 11, 12, 13, 14, 15]
    knap = Knapsack(n_const=5, phi=0.1)
    print()
    res = knap(x)
    print(*res)


################################################################################


def main():
    '''
    docstring for main.
    '''

    __test__()


if __name__ == '__main__':
    main()
