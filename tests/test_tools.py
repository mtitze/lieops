import pytest

import numpy as np
from lieops.ops.tools import poly2ad, ad2poly, poly3ad, ad3poly, get_2flow
from lieops import create_coords

def make_random_cmplx(n):
    '''
    Create a random complex number with real and imaginary parts between -1 and 1.
    '''
    return [complex(e) for e in 1 - np.random.rand(n)*2 + (1 - np.random.rand(n)*2)*1j] # We change to default complex numbers to prevent numpy operator overloading if multiplying polynomials with those quantities from left.

def get_max(delta):
    '''
    Return the maximal value of the absolute values of a polynomial.
    '''
    if len(delta) > 0:
        diff = max(np.abs(list(delta.values())))
    else:
        diff = 0
    return diff

def test_poly2ad(n=4, tol=1e-14):
    '''
    Test to check whether the adjoint representation of a 2nd order homogeneous polynomial properly reflects
    the commutator relations.
    '''
    n_const = 10
    q1, q2, p1, p2 = create_coords(2, real=True)
    for k in range(n):
        f = make_random_cmplx(n_const)
        g = make_random_cmplx(n_const)

        h1 = q1**2*f[0] + q2**2*f[1] + p1**2*f[2] + p2**2*f[3] + \
             q1*q2*f[4] + q1*p1*f[5] + q1*p2*f[6] + q2*p1*f[7] + \
             q2*p2*f[8] + p1*p2*f[9]

        h2 = q1**2*g[0] + q2**2*g[1] + p1**2*g[2] + p2**2*g[3] + \
             q1*q2*g[4] + q1*p1*g[5] + q1*p2*g[6] + q2*p1*g[7] + \
             q2*p2*g[8] + p1*p2*g[9]
        
        h12 = h1@h2

        h1mat = poly2ad(h1)
        h2mat = poly2ad(h2)
        h12mat = h1mat@h2mat - h2mat@h1mat
        h12_back = ad2poly(h12mat, tol=tol)

        # check if back and forth transformation are inverses to each other:
        assert (poly2ad(h12_back) - h12mat < tol).all()
        diff = np.array(list((ad2poly(poly2ad(h12), tol=tol) - h12).values()))
        assert (diff < tol).all()

        # check if the commutator is satisfied:
        diff_c = np.array(list((h12_back - h12).values()))
        assert (diff_c < tol).all()

def test_poly3ad(n=4, tol=1e-14):
    '''
    Test to check whether the adjoint representation of a 2nd order polynomial properly reflects
    the commutator relations.
    '''
    n_const = 15
    q1, q2, p1, p2 = create_coords(2, real=True)
    for k in range(n):
        f = make_random_cmplx(n_const)
        g = make_random_cmplx(n_const)

        h1 = q1**2*f[0] + q2**2*f[1] + p1**2*f[2] + p2**2*f[3] + \
             q1*q2*f[4] + q1*p1*f[5] + q1*p2*f[6] + q2*p1*f[7] + \
             q2*p2*f[8] + p1*p2*f[9] + \
             q1*f[10] + q2*f[11] + p1*f[12] + p2*f[13]

        h2 = q1**2*g[0] + q2**2*g[1] + p1**2*g[2] + p2**2*g[3] + \
             q1*q2*g[4] + q1*p1*g[5] + q1*p2*g[6] + q2*p1*g[7] + \
             q2*p2*g[8] + p1*p2*g[9] + \
             q1*g[10] + q2*g[11] + p1*g[12] + p2*g[13]
        
        h12 = h1@h2
        h12_no_const = h12.pop((0, 0, 0, 0), None)
        
        h1mat = poly3ad(h1)
        h2mat = poly3ad(h2)
        h12mat = h1mat@h2mat - h2mat@h1mat
        h12_back = ad3poly(h12mat, tol=tol)

        # check if back and forth transformation are inverses to each other:
        assert (poly3ad(h12_back) - h12mat < tol).all()
        diff = np.array(list((ad3poly(poly3ad(h12_no_const), tol=tol) - h12_no_const).values()))
        assert (diff < tol).all()

        # check if the commutator is satisfied: poly3ad(h1@h2) = [poly3ad(h1), poly3ad(h2)]
        diff_c = np.array(list((h12_back - h12_no_const).values()))
        assert (diff_c < tol).all()

def test_2flow(tol=1e-12, tol2=5e-8, tol3=5e-7): # tol=5e-12, tol2=1e-7, tol3=1e-5): (values if using np.linalg.eig)
    '''
    Various tests for the exact flow exp(:H:), for H a 2nd order polynomial.
    '''
    hh = make_random_cmplx(10)
    xi1, xi2, eta1, eta2 = create_coords(2)
    testham = hh[0]*xi1**2 + hh[1]*xi2**2 + hh[2]*eta1**2 + hh[3]*eta2**2 + \
              hh[4]*xi1*xi2 + hh[5]*xi1*eta1 + hh[6]*xi1*eta2 + hh[7]*xi2*eta1 + \
              hh[8]*xi2*eta2 + hh[9]*eta1*eta2
    
    Hflow = get_2flow(testham) # exact flow
    testop = testham.lexp(power=40) # brute-force flow with high power as reference
    
    f = make_random_cmplx(15)
    g = make_random_cmplx(15)
    testpl1 = f[10] + f[11]*xi1 + f[12]*xi2 + f[13]*eta1 + f[14]*eta2 + \
              f[0]*xi1**2 + f[1]*xi2**2 + f[2]*eta1**2 + f[3]*eta2**2 + \
              f[4]*xi1*xi2 + f[5]*xi1*eta1 + f[6]*xi1*eta2 + f[7]*xi2*eta1 + \
              f[8]*xi2*eta2 + f[9]*eta1*eta2
    
    testpl2 = g[10] + g[11]*xi1 + g[12]*xi2 + g[13]*eta1 + g[14]*eta2 + \
              g[0]*xi1**2 + g[1]*xi2**2 + g[2]*eta1**2 + g[3]*eta2**2 + \
              g[4]*xi1*xi2 + g[5]*xi1*eta1 + g[6]*xi1*eta2 + g[7]*xi2*eta1 + \
              g[8]*xi2*eta2 + g[9]*eta1*eta2
    
    h1 = make_random_cmplx(5)
    h2 = make_random_cmplx(5)
    testpl3 = h1[0] + h1[1]*xi1 + h1[2]*xi2 + h1[3]*eta1 + h1[4]*eta2
    testpl4 = h2[0] + h2[1]*xi1 + h2[2]*xi2 + h2[3]*eta1 + h2[4]*eta2
    
    # check flow identity for t=0
    check0 = get_max(Hflow(testpl2, t=0) - testpl2)
    assert check0 < tol, f'{check0} >= {tol}' 
    
    # check exact flow vs. brute-force reference
    t1 = 1
    check1 = get_max(Hflow(testpl1, t=t1) - testop(testpl1, t=t1))
    assert check1 < tol, f'{check1} >= {tol}' 
    
    # check linearity and multiplicity
    check2 = get_max(Hflow(testpl1) - Hflow(f[10]) - f[11]*Hflow(xi1) - f[12]*Hflow(xi2) - f[13]*Hflow(eta1) - f[14]*Hflow(eta2) - \
              f[0]*Hflow(xi1)**2 - f[1]*Hflow(xi2)**2 - f[2]*Hflow(eta1)**2 - f[3]*Hflow(eta2)**2 - \
              f[4]*Hflow(xi1)*Hflow(xi2) - f[5]*Hflow(xi1)*Hflow(eta1) - f[6]*Hflow(xi1)*Hflow(eta2) - f[7]*Hflow(xi2)*Hflow(eta1) - \
              f[8]*Hflow(xi2)*Hflow(eta2) - f[9]*Hflow(eta1)*Hflow(eta2))
    assert check2 < tol, f'{check2} >= {tol}'
    
    # check interoperability with commutator;
    # first check:
    t3 = 3.6
    check3 = get_max(Hflow(testpl3, t=t3)@Hflow(testpl4, t=t3) - Hflow(testpl3@testpl4, t=t3))
    assert check3 < tol2, f'{check3} >= {tol2}' 

    # second check:
    t4 = 3.6
    check4 = get_max(Hflow(testpl3, t=t4)*Hflow(testpl4, t=t4) - Hflow(testpl3*testpl4, t=t4))
    assert check4 < tol3, f'{check4} >= {tol3}' 
    
    # third check:
    t5 = 0.6
    check5 = get_max(Hflow(testpl1, t=t5)@Hflow(testpl2, t=t5) - Hflow(testpl1@testpl2, t=t5))
    assert check5 < tol, f'{check5} >= {tol}' 
    
    # forth check (against ref):
    t6 = t5
    check6 = get_max(Hflow(testpl1, t=t6)@Hflow(testpl2, t=t6) - testop(testpl1@testpl2, t=t6))
    assert check6 < tol, f'{check6} >= {tol}'
    