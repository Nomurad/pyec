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


def DTLZ2(x):

    f1 = sum(math.pow(xi-0.5, 2.0) for xi in x)


################################################################################
# M.O. w/ const
################################################################################

class Constraint_TestProblem(TestProblem):
    def __init__(self, n_obj=2, n_const=2):
        super().__init__(n_obj)
        self.n_const = n_const

    @abstractmethod
    def __call__(self):
        pass


class OSY(Constraint_TestProblem):
    def __init__(self):
        super().__init__(2, 6)

    def __call__(self, x):
        return self.osy(x)

    def osy(self, x):
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
                 range(5), ([2, 2, 1, 4, 1])))
        f2 = sum(map(lambda v: v ** 2, x))
        g1 = x[0] + x[1] - 2
        g2 = 6 - x[0] - x[1]
        g3 = 2 - x[1] + x[0]
        g4 = 2 - x[0] + 3 * x[1]
        g5 = 4 - (x[2] - 3) ** 2 - x[3]
        g6 = (x[4] - 3) ** 2 + x[5] - 4
        return (f1, f2), (-g1, -g2, -g3, -g4, -g5, -g6)


def tnk(x):
    """TNK -- n_obj=2, n_dv=2, n_constraint=2

    Arguments:
        x {[np.ndarray]} -- [ 0 <= x[0],x[1] <= PI ]
    """
    f1 = x[0]
    f2 = x[1]

    g1 = x[0]**2 + x[1]**2 - 1 - 0.1*np.cos(16*np.arctan(x[0]/x[1]))  # >=0
    g2 = -(x[0] - 0.5)**2 - (x[1] - 0.5)**2 + 0.5   # >=0

    return (f1, f2), (g1, g2)


class Circle_problem(Constraint_TestProblem):
    def __init__(self):
        super().__init__(2, 1)

    def __call__(self, x):
        if len(x) != 2:
            raise TestProblem_Error("len(x) != 2.")

        value = list(x)
        violation = np.sqrt(x[0]**2 + x[1]**2) - 1

        return (value, violation)


class mCDTLZ(Constraint_TestProblem):

    def __init__(self, n_obj=2, n_const=2):
        super().__init__(n_obj, n_const)
        self.n_dv = None

    def __call__(self, x):
        if not hasattr(x, "__len__"):
            raise TestProblem_Error("dv size error.")

        if self.n_dv is None:
            self.n_dv = len(x)
            self.n_bar_m = int(self.n_dv/self.n_obj)
            print("n/m = ", self.n_bar_m)

        # n = self.n_dv  # num of design variable
        # m = self.n_obj #num of objective function

        f = [0]*self.n_obj
        g = [0]*self.n_obj

        # calc objective func 
        f = [self.calc_objfunc(x, i) for i in range(self.n_obj)]
        # for i in range(self.n_obj):
        #     f[i] = self.calc_objfunc(x, i)

        # calc constraint violation
        g = [self.calc_constfunc(x, f, i) for i in range(self.n_obj)]
        # for i in range(self.n_obj):
        #     g[i] = self.calc_constfunc(x, f, i)

        return f, g

    def calc_objfunc(self, x, i):
        n_bar_m = self.n_bar_m
        # st = i * n_bar_m
        # fin = (i + 1) * n_bar_m
        st = int(i * self.n_dv / self.n_obj)
        fin = int((i + 1) * self.n_dv / self.n_obj)

        lis = tuple(math.sqrt(xl) for xl in x[st:fin])
        # res = (1/n_bar_m)*sum(lis)
        res = sum(lis) / n_bar_m
        return res

    def calc_constfunc(self, x, f, i):
        # g_fin = self.n_obj-1
        # g_st = 0
        lis = [f[l]**2 for l in range(self.n_obj) if l != i]
        res = f[i]**2 + 4*(sum(lis)) - 1
        return -res


class WaterProblem(Constraint_TestProblem):
    def __init__(self, n_obj=5, n_const=7):
        super().__init__(5, 7)

    def __call__(self, x):
        if not hasattr(x, "__len__"):
            raise TestProblem_Error("dv size error.")

        x1 = x[0]; x2 = x[1]; x3 = x[2]

        f1 = 106780.37*(x2+x3)+61704.67
        f2 = 3000.0*x1
        f3 = 30570*0.022890*x2/(0.06*2289.0)**0.65
        f4 = 250.0*2289.0*math.exp(-39.75*x2+9.9*x3+2.74)
        f5 = 25.0*((1.39/(x1*x2)) + 4940.0*x3-80.0)
        g1 = -1 + (0.00139/(x1*x2)+4.94*x3-0.08)
        g2 = -1 + (0.000306/(x1*x2)+1.082*x3-0.0986)
        g3 = -50000 + (12.307/(x1*x2)+49408.24*x3+4051.02)
        g4 = -16000 + (2.098/(x1*x2)+8046.33*x3-696.71)
        g5 = -10000 + (2.138/(x1*x2)+7883.39*x3-705.04)
        g6 = -2000 + (0.417*(x1*x2)+1721.26*x3-136.54)
        g7 = -550 + (0.164/(x1*x2)+631.13*x3-54.48)

        return [f1, f2, f3, f4, f5], [g1, g2, g3, g4, g5, g6, g7]


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


class Knapsack(Constraint_TestProblem):

    def __init__(self, n_obj=2, n_items=500, n_const=2, phi=0.8):
        """ 
            n_obj: num of objectives (m),
            n_items: num of items (l),
            n_const: num of knapsack (k),
            phi: feasibility ratio for each knapsack.

            dv range = [0.0~0.1]
        """
        super().__init__(n_obj, n_const)

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

        f = [0]*(self.n_obj)
        x = [1 if xi > 0.5 else 0 for xi in x]

        for i in range(self.n_obj):
            _profits = np.array([self.items[l].profits[i] for l in range(self.n_items)])
            # _p_times_x = list(map(lambda p_li,x_l: p_li*x_l, _profits,x))
            _p_times_x = _profits*np.array(x)
            f_i = sum(_p_times_x)
            # f.append(f_i)
            f[i] = f_i

        cv = [0]*self.n_const
        for j in range(self.n_const):
            _weights = np.array([self.items[l].weights[j] for l in range(self.n_items)])
            # _w_times_x = list(map(lambda w_li,x_l: w_li*x_l, _weights,x))
            _w_times_x = _weights*np.array(x)
            v_i = sum(_w_times_x) - self.capa[j]
            # for debug
            # print(self.capa[j], v_i)

            # if v_i <= 0:
            #     v_i = 0
            # cv.append(v_i)
            cv[j] = v_i
            # print(f, cv)

        return f, cv 


################################################################################

def __test__():
    import matplotlib.pyplot as plt
    # x = [0, 1, 2, 3, 4, 5]
    # print(*osy(x))

    # x = [np.cos(np.pi/4.0), np.sin(np.pi/4.0)]
    # circle = Circle_problem()
    # res = circle(x)

    div = 21
    n_obj = 2
    dv_size = 2
    # random.seed(10)
    cdtlz = mCDTLZ(n_obj=n_obj, n_const=n_obj)
    wp = WaterProblem()
    x = [
        random.uniform(0.01, 0.45),
        random.uniform(0.01, 0.1),
        random.uniform(0.01, 0.1)
    ]
    res = wp(x)
    print(x)
    print(res)
    # res = cdtlz(x)

    # res2 = []
    # for i in range(300000):
    #     x = [random.uniform(0.0, 1.0) for _ in range(dv_size)]
    #     # x = [0]
    #     res2.append(cdtlz(x))

    # res2 = np.array(res2)

    # # response
    # print("\ndv: ", x)
    # # print("return: ", res2[:,:,:])

    # res3 = res2[res2[:,1,0]<=0]
    # res3 = res3[res3[:,1,1]<=0]

    # infeasible = res2[res2[:,1,0] > 0]
    # infeasible2 = res2[res2[:,1,1] > 0]

    # plt.scatter(res3[:,0,0], res3[:,0,1])
    # plt.scatter(infeasible[:,0,0], infeasible[:,0,1], c="red")
    # plt.scatter(infeasible2[:,0,0], infeasible2[:,0,1], c="red")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.show()


################################################################################


def main():
    '''
    docstring for main.
    '''

    __test__()


if __name__ == '__main__':
    main()
