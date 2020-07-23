import numpy as np

from ..base.indiv import Individual



################################################################################
# スカラー化関数
################################################################################
class ScalarError(Exception):
    pass

def scalar_weighted_sum(indiv, weight, ref_point):
    return -np.sum(weight * np.abs(indiv.wvalue - ref_point))

def scalar_chebyshev(indiv:Individual, weight, ref_point):
    # return scalar_chebyshev_for_minimize(indiv, weight, ref_point)
    return scalar_chebyshev_for_maximize(indiv, weight, ref_point)

def scalar_chebyshev_for_minimize(indiv, weight, ref_point):
    if not indiv.evaluated():
        raise ScalarError("indiv not evaluated.")
    res = -np.max(weight * np.abs(indiv.wvalue - ref_point))
    return res

def scalar_chebyshev_for_maximize(indiv, weight, ref_point):
    if not indiv.evaluated():
        raise ScalarError("indiv not evaluated.")
    res = np.min(weight * np.abs(indiv.wvalue - ref_point))
    return res

def scalar_boundaryintersection(indiv, weight, ref_point):
    ''' norm(weight) == 1
    '''
    nweight = weight / np.linalg.norm(weight)

    bi_theta = 5.0
    d1 = np.abs(np.dot((indiv.wvalue - ref_point), nweight))
    d2 = np.linalg.norm(indiv.wvalue - (ref_point - d1 * nweight))
    return -(d1 + bi_theta * d2)