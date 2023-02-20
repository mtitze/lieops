import pytest
import mpmath as mp
from njet import derive

from lieops.core import create_coords, lexp, poly
from lieops.core.tools import get_taylor_map, symcheck
from lieops.solver.channell import get_monomial_flow

@pytest.mark.parametrize("xi0, eta0", [(0, 0), (0.0287, -0.014)])
def test_symplecticity1(xi0, eta0, order=10, tol=2e-21):
    '''
    Compare high-order degrees after application of the solver,
    using mpmath.
    '''
    xi, eta = create_coords(1)
    ham = -0.06*xi*eta + 0.04*eta**2 + 0.04*xi**2 - 3.3*xi**3 - 10.2*xi**2*eta - 3.4*eta**3
    mp.mp.dps = 32
    ham = ham.apply(mp.mpc)
    op1 = lexp(ham)
    op1.calcFlow(method='channell')
    dop1 = derive(op1, order=order, n_args=2)
    op1map = get_taylor_map(*dop1.eval(xi0, eta0), max_power=order)

    # Let N be the order by which we want to compare the product xi_f@eta_f.
    # Then N = k + l - 2 where k and l are the orders of xi_f and eta_f. 
    # The smallest non-trivial power has k = 1, leading to 
    # N = l - 1. So the maximal order N of the check can not exceed l - 1, 
    # where l is the maximal order of the map which we have determined. 
    assert max(abs(op1map[0]@op1map[1] + 1j).truncate(order - 1)) < tol
    # second check (symcheck test):
    assert len(symcheck(op1map, tol=tol)) == 0
    
def test_symplecticity2(dps=64, order=6, tol=1e-51):
    '''
    Test symplecticity of Channell's solver in conjuction with mpmath.
    '''
    mp.mp.dps = dps
    testham = poly(values={(6, 1): -9.4 - 5524653677213*1j})
    testham = testham.apply(mp.mpc)
    testmap = get_monomial_flow(testham)
    dmap = derive(testmap, order=order, n_args=2)
    evaluation = dmap.eval(0, 0)
    tm = get_taylor_map(*evaluation, max_power=order)
    assert len(symcheck(tm, tol=tol)) == 0

def test_2d_hamiltonian(n_slices=100, tol=1e-2, tol2=3e-1):
    '''
    Test the outcome of applying a Lie operator of a 2D Hamiltonian.
    '''
    xi0, eta0 = 0.888 + 0.625*1j, 0.125 - 124*1j
    q, p = create_coords(1, real=True)
    
    ham = 0.423*q*p + 0.24*q - 1.002*p**2 + (0.032 - 0.64*1j)*q**2
    op = lexp(ham)
    
    reference1 = op(xi0, eta0, power=10)
    reference2 = op(xi0, eta0, power=10, t=-1)
    reference3 = op(xi0, eta0, method='2flow', t=1)
    reference4 = op(xi0, eta0, method='2flow', t=-1)
    
    ch1 = op(xi0, eta0, method='channell', n_slices=n_slices, t=1)
    ch2 = op(xi0, eta0, method='channell', n_slices=n_slices, t=-1)
    assert all([abs(reference1[k] - ch1[k]) < tol for k in range(2)])
    assert all([abs(reference2[k] - ch2[k]) < tol for k in range(2)])
    assert all([abs(reference3[k] - ch1[k]) < tol2 for k in range(2)])
    assert all([abs(reference4[k] - ch2[k]) < tol2 for k in range(2)])

def test_4d_monomial(n_slices=1000, tol=5e-9):
    '''
    Test the outcome of applying a Lie operator of a single 4D monomial to coordinate functions.
    '''
    xi0, xi1, eta0, eta1 = 0.223 + 0.2*1j, 0.637 - 0.53*1j, 0.32 + 0.63*1j, 0.553 - 0.155*1j
    xieta0 = [xi0, xi1, eta0, eta1]
    q1, q2, p1, p2 = create_coords(2, real=True)
    mon = 0.423*q1*q2**2*p1*p2
    op = lexp(mon)
    
    reference1 = op(*xieta0, power=8)
    reference2 = op(*xieta0, power=8, t=-1)
    
    ch1 = op(*xieta0, method='channell', n_slices=n_slices, t=1)
    ch2 = op(*xieta0, method='channell', n_slices=n_slices, t=-1)
    
    assert all([abs(reference1[k] - ch1[k]) < tol for k in range(4)])
    assert all([abs(reference2[k] - ch2[k]) < tol for k in range(4)])
    
def test_4d_hamiltonian(n_slices=100, tol=2e-3):
    '''
    Test the outcome of applying a Lie operator of a Hamiltonian consisting
    of three terms.
    '''
    xi0, xi1, eta0, eta1 = 0.223 + 0.2*1j, 0.637 - 0.53*1j, 0.32 + 0.63*1j, 0.553 - 0.155*1j
    xieta0 = [xi0, xi1, eta0, eta1]
    q1, q2, p1, p2 = create_coords(2, real=True)
    
    mon = 0.423*q1*q2**2*p1*p2
    ham2 = mon + 0.553*q1*q2 - 1j*p2
    op = lexp(ham2)
    
    reference1 = op(*xieta0, power=10)
    reference2 = op(*xieta0, power=10, t=-1)
    
    ch1 = op(*xieta0, method='channell', n_slices=n_slices, t=1)
    ch2 = op(*xieta0, method='channell', n_slices=n_slices, t=-1)
        
    assert all([abs(reference1[k] - ch1[k]) < tol for k in range(4)])
    assert all([abs(reference2[k] - ch2[k]) < tol for k in range(4)])
    
    
def test_poly_actions(tol=1e-16):
    '''
    Test if exp(-alpha :f:)q_j = ... , where the right-hand side is the prediction
    by Channell's equations and f a specific monomial.
    '''
    
    q1, q2, p1, p2 = create_coords(2, real=True, max_power=10)

    alpha = 0.4242

    # Ensure k*n1 = (k - 1)*m1 so that m1/(m1 - n1) = k in this example, so
    # we can computed the power of the respective Lie polynomial without having to
    # rely on some generating function modeling a general power.
    n1 = 3
    n2 = 1
    m1 = 4
    m2 = 2

    mon = -alpha*q1**n1*q2**n2*p1**m1*p2**m2
    op = lexp(mon)
    op.calcFlow(method='channell')    
    
    out_q1 = op@q1
    A1 = q2**n2*p2**m2
    ref_q1 = q1*(1 + alpha*(m1 - n1)*q1**(n1 - 1)*p1**(m1 - 1)*A1)**(int(m1/(m1 - n1)))
    
    assert (out_q1 - ref_q1).above(tol) == 0