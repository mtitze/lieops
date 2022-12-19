from lieops import create_coords
from lieops.solver import get_2flow

from .common import make_random_cmplx, get_max

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
    
