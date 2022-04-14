import numpy as np

from njet.jet import factorials

from bnf.lieop.magnus import forests
from bnf.lieop.lie import combine, create_coords


def number_of_graphs(order):
    '''
    The number of trees of a given order.
    
    Notation in Iserles & Norsett: "On the solution of linear differential equations in Lie groups."
    '''
    facts = factorials(2*order)
    return [int(facts[2*k]/facts[k]/facts[k + 1]) for k in range(order)]

def nc(*args):
    '''
    Returns the nested commutator expression (sometimes called Chibrikov basis)
    x1@(x2@(x3@(x4@ .... )))
    '''
    out = args[-1]
    for x in args[len(args) - 2::-1]:
        out = x@out
    return out

def bch(x, y):
    '''
    Compute the explicit commutators of the Baker-Campbell-Hausdorff series for
    orders up and including 6.
    '''
    result = {}
    result[1] = x + y
    result[2] = 0.5*(x@y)
    result[3] = 1/12*(nc(x, x, y) + nc(y, y, x))
    result[4] = -1/24*nc(y, x, x, y)
    #result[5] = 1/180*nc(y, y, x, x, y) - 1/360*(x@y)@nc(y, x, y) - 1/180*nc(y, x, x, x, y) - 1/120*(x@y)@nc(x, x, y) + 1/720*nc(y, y, y, x, y) - 1/720*nc(x, x, x, x, y)
    result[5] = -1/720*(nc(y, y, y, y, x) + nc(x, x, x, x, y)) + \
                1/360*(nc(x, y, y, y, x) + nc(y, x, x, x, y)) + \
                1/120*(nc(y, x, y, x, y) + nc(x, y, x, y, x))
    # From Wiki (fails at symmetry test!)
    #result[6] = 1/240*nc(x, y, x, y, x, y) + \
    #            1/720*(nc(x, y, x, x, x, y) - nc(x, x, y, y, x, y)) + \
    #            1/1440*(nc(x, y, y, y, x, y) - nc(x, x, y, x, x, y))
    # Instead, we will use the results from the BCH routine of H. Hofstaetter (https://github.com/HaraldHofstaetter/BCH)
    result[6] = -1/1440*nc(y, x, x, x, y, x) + 1/720*nc(y, y, x, x, y, x) - 1/240*nc(y, x, y, x, y, x) + 1/1440*nc(y, y, y, x, y, x) - 1/720*nc(y, x, y, y, y, x)
    result[7] = -1/30240 * nc(x, x, x, x, x, y, x) + \
                 1/10080 * nc(y, x, x, x, x, y, x) - \
                 1/10080 * nc(x, y, x, x, x, y, x) - \
                 1/3360  * nc(y, y, x, x, x, y, x) - \
                 1/5040 *  nc(x, x, y, x, x, y, x) + \
                 1/1260 *  nc(y, x, y, x, x, y, x) + \
                 1/7560 *  nc(x, y, y, x, x, y, x) - \
                 1/7560 *  nc(y, y, y, x, x, y, x) + \
                 1/10080 * nc(x, x, y, y, x, y, x) - \
                 1/1008 *  nc(x, y, x, y, x, y, x) + \
                 1/3360 *  nc(y, y, x, y, x, y, x) + \
                 1/1680 *  nc(y, x, y, y, x, y, x) - \
                 1/3360 *  nc(x, y, y, y, x, y, x) - \
                 1/10080 * nc(y, y, y, y, x, y, x) - \
                 1/5040 *  nc(x, y, x, y, y, y, x) + \
                 1/2520 *  nc(y, y, x, y, y, y, x) - \
                 1/10080 * nc(x, y, y, y, y, y, x) + \
                 1/30240 * nc(y, y, y, y, y, y, x)
    return result

def bch_symmetry_c(x, y, order, tol=1e-10):
    '''
    Let Z(X, Y) be the result of the Baker-Campbell-Hausdorff series:
      exp(Z(X, Y)) = exp(X) exp(Y)
    Then it must hold [1]:
      Z(X, Y) = -Z(-Y, -X)  
    This routine will test this equation for the routine 'combine', up to a specific order.
    
    Reference(s):
    [1] B. Mielnik and J. Plebanski: Combinatorial approach to Baker-Campbell-Hausdorff exponents.
        Annales de l’I. H. P., section A, tome 12, no 3 (1970), p. 215-254
    '''
    z1, ham1 = combine(x, y, power=order, time=False)
    z2, ham2 = combine(-y, -x, power=order, time=False)
    assert z1.keys() == z2.keys()
    
    for key in z1.keys():
        for kk in z1[key].keys():
            A = z1[key][kk]
            B = 0
            if kk in z2[key].keys():
                B = z2[key][kk]
            C = A + B
            assert abs(C) < tol
            
    return z1

def bch_vs_reference(x, y, result, tol=1e-11):
    '''
    Check combine(x, y) vs. the prediction from the clasical Baker-Campbell-Hausdorff equation
    for orders up and including 7.
    '''
    reference = bch(x, y)
    for r in result.keys():
        if r == 0:
            continue
        if r > 7:
            break
        diff = reference[r] - result[r - 1]
        hh = np.abs(list(diff.values.values()))
        if len(hh) > 0:
            hh = max(hh)
        else:
            hh = 0
        assert hh < tol

#########
# Tests #
#########

def test_number_of_graphs(order=10):
    '''
    Test whether the number of trees from the forest routine agrees with the prediction.
    '''
    prediction = number_of_graphs(order)
    a, _ = forests(order)
    
    for k in range(order):
        assert prediction[k] == len(a[k])
        
def test_bch_symmetry_c(request, tol=1e-10):
    '''
    Test the symmetry of the BCH formula (see bch_symmetry_c routine) 
    and vs. the reference BCH.
    '''
    q, p = create_coords(1, cartesian=True)
    x = p
    y = p*q**2
    z = p*q**4
    
    r1 = bch_symmetry_c(order=8, x=x, y=y, tol=tol)
    r2 = bch_symmetry_c(order=7, x=x, y=z, tol=tol)
    
    bch_vs_reference(x, y, r1)
    bch_vs_reference(x, z, r2)
    

    
    