import shelve
import os
import pytest
import numpy as np

from lieops import lexp, poly
from lieops.core.forest import fnf
from lieops.core.tools import tpsa, taylor_map

# The following data has been produced using a realistic example of a section of a beamline
# with 80 elements and stored using shelve with the following commands:
#
# sf = shelve.open(filename)
# k = 0
# for e in part1.elements:
#     sf[str(k)] = e.operator.argument._values
#     k += 1
# Hereby 'part1' denotes a beamline object in the accphys package.
# We retreive the data:
filename = os.path.join(os.path.dirname(__file__), 'example1.shlf')
sh00 = shelve.open(filename)
operators = []
for k in range(len(sh00)):
    operators.append(lexp(poly(values=sh00[str(k)], max_power=10)))
sh00.close()

default_ordering = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 10, 6, 11, 11, 6, 12, 6, 9, 6, 13, 6, 9, 6, 12, 6, 11, 11, 6, 12, 6, 9, 6, 13, 6, 9, 6, 12, 6, 11, 11, 6, 12, 6, 9, 6, 13, 6, 9, 6, 12, 6, 11, 11, 6, 12, 6, 9, 6, 13, 6, 9, 6, 12, 6, 11, 11, 6, 10, 6, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

########################
# Function definitions #
########################

def normalform(operators, ordering, order, position, **kwargs):
    '''
    Compute the normal form of a series of operators at a specific position.
    '''
    dc = tpsa(*operators, ordering=ordering, order=order, position=position, **kwargs) # power should be sufficiently high here
    tm = taylor_map(*dc._evaluation, max_power=operators[0].argument.max_power) # note that max_power < inf is important here, otherwise the code becomes very (!) slow, if it stops at all...
    return fnf(*tm, order=order, **kwargs)

def cycle(operators, ordering, order, position, **kwargs):
    '''
    Compute the normal form of a series of operators at a specific position.
    '''
    dc = tpsa(*operators, ordering=ordering, order=order, mode='chain', **kwargs) # power should be sufficiently high here
    cyc = dc.cycle(*position, outf=0, **kwargs)
    cc = cyc.compose()
    tm = taylor_map(*cc, max_power=operators[0].argument.max_power) # note that max_power < inf is important here, otherwise the code becomes very (!) slow, if it stops at all...
    return fnf(*tm, order=order, **kwargs)

def above_hompart(p, tolerances):
    '''
    Helper function for checks; remove terms of a polynomial relative to given tolerances,
    for each homogeneous part. The reason is that higher-order terms tend to be much larger
    than lower-order terms, so tolerances in checks can be finetuned for each order.
    '''
    assert len(tolerances) == p.maxdeg()
    p_new = 0
    for k in range(len(tolerances)):
        tol = tolerances[k]
        p_new += p.homogeneous_part(k).above(tol)
    return p_new

#########
# Tests #
#########

@pytest.mark.parametrize("order, position, power, ordering", [(4, (0, 0), 30, default_ordering)])
def test_normalform1(order, position, power, ordering, tol1=1e-8, tol2=1e-7):
    '''
    Test if the normal form operations actually yield a result in which the non-action terms vanish.
    '''
    nf_dict = normalform(operators=operators, ordering=ordering, order=order, 
                         position=position, power=power)

    # short reference check regarding the (complex) symplectic matrix S:
    Sinv0 = np.array([[1.76625646, 0],
                      [0, 5.66169197e-01]])
    
    Sinv = nf_dict['bnfout']['nfdict']['Sinv']
    assert np.linalg.norm(Sinv - Sinv0) < tol1
    
    nf = sum(nf_dict['normalform']).above(tol2)
    
    # check if normal form is canceling the non-action variables:
    assert list(nf.keys()) == [(j, j) for j in range(1, order//2 + 1)]
    
@pytest.mark.parametrize("ordering, position, power", [(default_ordering, (0, 0), 30)])
def test_normalform2(ordering, position, power, order=4, tol1=1e-8, tol2=5e-7, tol3=1e-3, tol4=1e-13, **kwargs):
    '''
    Test if cycling a shifted series of operators yield the same results as the original one.
    
    Note: This may not work with offsets, because with offsets only interior parts are normalized. 
          So we only test this at the (fixed) point (0, 0).
    '''
    nf_dict = cycle(operators=operators, ordering=ordering, order=order, position=position,
                    power=power, **kwargs)
    
    nf = sum(nf_dict['normalform'])
    
    # the coefficients in normal form should all be equal
    assert max(abs(nf[1, 1] - 1.6508026167756547)) < tol1
    assert max(abs(nf[2, 2] + 3923.6611704910206)) < tol2
    
    # Now move the chain of functions by one, and recalculate the normal form. Then compare with
    # the original results (shifted by one accordingly):
    first_operator = operators[ordering[0]]
    new_ordering = ordering[1:] + [ordering[0]]

    nf_dict2 = cycle(operators=operators, order=order, ordering=new_ordering, 
                position=first_operator(*position, power=power, **kwargs), 
                power=power, **kwargs)
    
    # We compare the maps to normal form:
    c1, c2 = nf_dict['chi'], nf_dict2['chi']
    assert len(c1) == len(c2)

    for k in range(len(c1)):
        c1k, c2k = c1[k], c2[k]
        assert c1k.keys() == c2k.keys()

        for key in c1k.keys():
            ar1 = c1k[key]
            ar2 = c2k[key]

            ar0 = np.array(list(ar1[1:]) + [ar1[0]]) # cyclic shift of array for comparison

            diff = abs(ar0 - ar2)
            mval = abs(ar2)
            valid_indices = mval > tol4 # only compare relative errors if the values of 'ar2' are above a certain threshold, given by the tol4 parameter.
            rel_errors = diff[valid_indices]/mval[valid_indices]
            
            assert max(rel_errors) < tol3

def test_normalform3():
    '''
    Test the multi-dimensional capabilities of the routines involved in the normal form calculation.
    
    First, a run with a single value is performed. Then, two multi-dimensional arrays are processed. 
    The entries in the results are compared with each other.
    
    Test includes non-zero kicks.
    '''
    order = 6
    power = 30
    
    position1 = [0.0002, 0.001]
    # The other positions are arrays in which position1 will appear somewhere:
    position2 = [np.array([0, position1[0], 0.0006]), np.array([0, position1[1], -0.0005])]
    position3 = [np.array([[0, 0.0003, 0.0006], [position1[0], -0.00042, 0.0001]]), 
                 np.array([[0, 0.001, -0.0005], [position1[1], -0.00008, 0.00015]])]
    
    nf_dict1 = normalform(operators=operators, ordering=default_ordering, order=order, position=position1, power=power)
    nf_dict2 = normalform(operators=operators, ordering=default_ordering, order=order, position=position2, power=power)
    nf_dict3 = normalform(operators=operators, ordering=default_ordering, order=order, position=position3, power=power)
    
    n1 = sum(nf_dict1['normalform'])
    n2 = sum(nf_dict2['normalform'])
    n3 = sum(nf_dict3['normalform'])
    
    # Check 0: shape consistency
    assert np.array(position1).shape[1:] == ()
    assert np.array(position2).shape[1:] == (3,)
    assert np.array(position3).shape[1:] == (2, 3)
    assert n1[1, 1].shape == ()
    assert n2[1, 1].shape == (3,)
    assert n3[1, 1].shape == (2, 3)
        
    # Check 1: Normalization successfull?
    tolerances = [1e-14, 1e-14, 2e-14, 1e-13, 5e-10, 5e-8, 5e-5]
    n1_0 = above_hompart(n1, tolerances)
    n2_0 = above_hompart(n2, tolerances)
    n3_0 = above_hompart(n3, tolerances)
    assert list(n1_0.keys()) == [(1, 1), (2, 2), (3, 3)]
    assert list(n2_0.keys()) == [(1, 1), (2, 2), (3, 3)]
    assert list(n3_0.keys()) == [(1, 1), (2, 2), (3, 3)]
    
    # Check 2: Values agree in each result
    tolerances1 = [5e-15, 5e-10, 1e-1]
    j = 0
    for key in [(1, 1), (2, 2), (3, 3)]:
        assert abs(n1_0[key] - n2_0[key][1]) < tolerances1[j]
        assert abs(n1_0[key] - n3_0[key][1, 0]) < tolerances1[j]
        j += 1
        
    zero_threshold = 1e-12 # threshold beyond which we shall consider relative errors rather than absolute errors
    tolerance3 = 2e-9
    c1, c2, c3 = nf_dict1['chi'], nf_dict2['chi'], nf_dict3['chi']
    assert len(c1) == len(c2)
    assert len(c1) == len(c3)

    for k in range(len(c1)):
        c1k, c2k, c3k = c1[k], c2[k], c3[k]
        assert c1k.keys() == c2k.keys()
        assert c1k.keys() == c3k.keys()

        for key in c1k.keys():
            ar1 = c1k[key]
            ar2 = c2k[key]
            ar3 = c3k[key]

            rel_err12 = abs(ar1 - ar2[1])
            rel_err13 = abs(ar1 - ar3[1, 0])
            if abs(ar1) > zero_threshold:
                rel_err12 = rel_err12/abs(ar1)
                rel_err13 = rel_err13/abs(ar1)

            assert rel_err12 < tolerance3
            assert rel_err13 < tolerance3
            
def test_normalform4():
    '''
    Test the multi-dimensional capabilities of the routines involved in the cycling -- in conjunction with
    normal form procedures.
    
    First, a run with a single value is performed. Then, a multi-dimensional array is processed. 
    The entries in the results are compared with each other.
    
    Test includes non-zero kicks.
    '''
    order = 6
    power = 30
    
    position1 = [0.0002, 0.001]
    position2 = [np.array([[0, 0.0003, 0.0006], [position1[0], -0.00042, 0.0001]]), 
                 np.array([[0, 0.001, -0.0005], [position1[1], -0.00008, 0.00015]])]
    
    nf_dict1 = cycle(operators=operators, ordering=default_ordering, order=order, position=position1, power=power)
    nf_dict2 = cycle(operators=operators, ordering=default_ordering, order=order, position=position2, power=power)
    
    na = sum(nf_dict1['normalform'])
    nb = sum(nf_dict2['normalform'])
    
    # Check 0: Output shapes correct?
    assert na[1, 1].shape == (80,) # len(default_ordering) == 80
    assert nb[1, 1].shape == (80, 2, 3)
    
    # Check 1: Normalization successfull?
    tolerances = [1e-14, 1e-14, 2e-14, 1e-13, 3e-10, 1e-8, 5e-6]
    na_0 = above_hompart(na, tolerances)
    nb_0 = above_hompart(nb, tolerances)
    assert list(na_0.keys()) == [(1, 1), (2, 2), (3, 3)]
    assert list(nb_0.keys()) == [(1, 1), (2, 2), (3, 3)]
    
    # Check 2: Values agree in each result
    tolerances1 = 5e-15
    for key in [(1, 1), (2, 2), (3, 3)]:
        assert max(abs(na_0[key] - nb_0[key][..., 1, 0])) < tolerances1
        
    c1, c2 = nf_dict1['chi'], nf_dict2['chi']
    assert len(c1) == len(c2)

    for k in range(len(c1)):
        c1k, c2k = c1[k], c2[k]
        assert c1k.keys() == c2k.keys()

        for key in c1k.keys():
            ar1 = c1k[key]
            ar2 = c2k[key]

            err12 = max(abs(ar1 - ar2[..., 1, 0]))
            assert err12 < tolerances1
