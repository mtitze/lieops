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

    
    