import numpy as np
import mpmath as mp
from sympy import Symbol
import pytest

from njet.functions import cos, sin, exp
from njet import derive

from lieops import __version__
from lieops.core import poly, create_coords, construct, lexp

from lieops.linalg.matrix import expandingSum, create_J
from lieops.linalg.congruence.takagi import symplectic_takagi_old
from lieops.linalg.nf import first_order_nf_expansion

def check_2nd_orders(hdict, dim, tol=1e-14): 
    '''
    Check if the 2nd order coefficients of a poly-dictionary correspond to a 'diagonal'
    Hamiltonian, i.e. a Hamiltonian of the form H = mu_1*xi_1*eta_1 + mu_2*xi_2*eta_2 + ...
    '''
    # Attention: The tolerance is quite strong. tol=1e-15 may cause this test to fail.
    he2 = {k: v for k, v in hdict.items() if sum(k) == 2}
    for k in range(dim):
        ek2 = [0 if j != k and j != k + 2 else 1 for j in range(dim*2)]
        value = he2.pop(tuple(ek2))
        if value.imag >= tol: # the diagonal elements must be real, since the Hamiltonian is real
            return False
    # the remaining values must be smaller than the given tolerance
    return all([abs(v) < tol for v in he2.values()])

def exp_ad1(mu=-0.2371, power=18, tol=1e-14, **kwargs):
    # Test the exponential operator on Lie maps for the case of a 2nd order Hamiltonian (rotation) and
    # the linear map K to (first-order) normal form.
    
    H2 = lambda x, px: 0.5*(x**2 + px**2)
    expansion, nfdict = first_order_nf_expansion(H2, check=True, code='numpy', **kwargs)
    HLie = poly(values=expansion)
    Hop = lexp(HLie, power=power)
    Kinv = nfdict['Kinv'] # K(p, q) = (xi, eta)
    xieta = create_coords(1)

    # first apply K, then exp_ad:
    xy_mapped = [xieta[0]*Kinv[0, 0] + xieta[1]*Kinv[0, 1], xieta[0]*Kinv[1, 0] + xieta[1]*Kinv[1, 1]]    
    xy_final_mapped = Hop(*xy_mapped, t=mu)
    
    # first apply exp_ad, then K:
    xy_fin = Hop(*xieta, t=mu)
    xy_final = [xy_fin[0]*Kinv[0, 0] + xy_fin[1]*Kinv[0, 1], xy_fin[0]*Kinv[1, 0] + xy_fin[1]*Kinv[1, 1]]
    
    # Both results must be equal.
    for k in range(len(xy_final)):
        d1 = xy_final[k]
        d2 = xy_final_mapped[k]
        for key, v1 in d1.items():
            v2 = d2[key]
            assert abs(v1 - v2) < tol
            
    # check if the result also agrees with the analytical expectation
    K = nfdict['K']
    zz = [Symbol('x'), Symbol('px')]
    xf = np.cos(mu)*zz[0] - np.sin(mu)*zz[1]
    pxf = np.cos(mu)*zz[1] + np.sin(mu)*zz[0]
    expectation = [xf, pxf]
    for k in range(len(xy_final_mapped)):
        lie_k = xy_final_mapped[k] # lie_k = exp(:HLie:)((Kinv*xieta)[k])
        diff = expectation[k] - (lie_k( *sum([zz[l]*K[:, l] for l in range(len(zz))]) ) ).expand()
        assert abs(diff.coeff(zz[0])) < tol and abs(diff.coeff(zz[1])) < tol
        
def fonfe(tol=1e-14, code='numpy', **kwargs):
    # Test of the first-order normal form expansion and the result of reordering the canonical coordinates. Both results against each other.
    
    # compute expansion up to 3rd order:
    H = lambda x, y, px, py: 0.24*(x**2 + px**2) + 0.86*(y**2 + py**2) + x**3 - y**3 - sin(x + 9*x**4*y + 0.2*px*y + 0.411*x*y - 0.378*px*x - 0.039*x*py + 1.2*y*py + 0.13*py*px) + x
    he, he_dict = first_order_nf_expansion(H, order=3, warn=True, code=code, check=True, **kwargs)
    
    # compute expansion of the same Hamiltonian, but with respect to an alternative symplectic structure, up to third order:
    T = expandingSum(2)
    if code == 'mpmath':
        T = mp.matrix(T)
    HT = lambda x, px, y, py: H(x, y, px, py)
    heT, heT_dict = first_order_nf_expansion(HT, order=3, warn=True, T=T, code=code, check=True, **kwargs)
    
    assert check_2nd_orders(he, dim=2, tol=tol)
    assert check_2nd_orders(heT, dim=2, tol=tol)
    
    # compare results against each other
    assert he == heT
    
def exp_ad2(mu=0.6491, power=40, tol=1e-14, max_power=10, code='mpmath', dps=32, **kwargs):
    # Test the exponential operator on Lie maps for the case of a 5th order Hamiltonian and
    # a non-linear map, making use of K, the linear map to (first-order) normal form.
    
    # Attention: This test appears to be suceptible against round-off errors for higher powers (power and max_power),
    # and therefore requires mpmath to have sufficient precision (and sufficiently high power in exp).
    
    H2 = lambda x, px: mu*0.5*(x**2 + px**2) + x**3 + x*px**4
    expansion, nfdict = first_order_nf_expansion(H2, order=5, warn=True, code=code, dps=dps, check=True, **kwargs)
    HLie = poly(values=expansion, max_power=max_power)
    K = nfdict['K']
    xieta = create_coords(1, max_power=max_power)
    
    # first apply function, then exp_ad:
    if code == 'numpy':
        xy_mapped = (K@np.array([xieta]).transpose()).tolist()
    elif code == 'mpmath':
        xy_mapped = [[sum([xieta[k]*K[j, k] for k in range(len(xieta))])] for j in range(len(K))]
    xy_mapped = [xy_mapped[k][0]**3 + 0.753 for k in range(len(xy_mapped))] # apply an additional non-linear operation
    Hop = lexp(HLie, power=power)
    xy_final_mapped = Hop(*xy_mapped) 
    
    # first apply exp_ad, then function:
    xy_fin = Hop(*xieta)
    if code == 'numpy':
        xy_final = (K@np.array([xy_fin]).transpose()).tolist()
    elif code == 'mpmath':
        xy_final = [[sum([xy_fin[k]*K[j, k] for k in range(len(xieta))])] for j in range(len(K))]
    xy_final = [xy_final[k][0]**3 + 0.753 for k in range(len(xy_final))] # apply an additional non-linear operation
    
    # Both results must be relatively close (and every entry non-zero).
    for k in range(len(xy_final)):
        d1 = xy_final[k]
        d2 = xy_final_mapped[k]
        for key, v1 in d1.items():
            v2 = d2[key]
            assert abs(v1 - v2)/min([abs(v1), abs(v2)]) < tol
    
def stf_with_zeros(tol1=1e-18, tol2=1e-10, code='numpy', dps=32):
    # Test symplectic Takagi factorization with a diagonalizable GJ having some zeros on its main diagonal.
    
    A = [[0.53 - 2.647*1j, -0.1 + 2*1j, 0.1 + 0.93*1j, -1.1 - 0.24*1j],
         [4.24 -0.35*1j, 0.553 + 0.3*1j, -0.95 + 23.15*1j, 0.56 - 7.35*1j],
         [-5.37 + 1.44*1j, 0.331 + 0.352*1j, 0.925 - 2.743*1j, 3.3 - 9.1*1j],
         [0.252 + 0.673*1j, 6.8832 - 9.21*1j, 0.975 + 9.356*1j, 0.4242 - 0.773*1j]]
    
    if code == 'numpy':
        A = np.array(A)
        EE = np.eye(4)
        J = create_J(2)
    if code == 'mpmath':
        A = mp.matrix(A)
        EE = mp.eye(4)
        J = mp.matrix(create_J(2))        
                
    EE[2, 2] = 0
    EE[3, 3] = 0

    G = A.transpose()@J@EE@J@A
    G = G + G.transpose()
    
    S0, D0, _ = symplectic_takagi_old(G, tol=tol1, dps=dps)
    
    symplecticity = S0.transpose()@J@S0 - J
    factorization = S0@D0@S0.transpose() - G
    
    dim2 = len(G)
    assert max([max([abs(symplecticity[j, k]) for j in range(dim2)]) for k in range(dim2)]) < tol2
    assert max([max([abs(factorization[j, k]) for j in range(dim2)]) for k in range(dim2)]) < tol2
    
    
#########
# Tests #
#########

def test_version():
    assert __version__ == '0.1.8'
    
def test_jacobi():
    # Test the Jacobi-identity for the poly class
    
    p1 = poly(a=[0, 1, 0], b=[1, 1, 1])
    p2 = poly(a=[2, 0, 1], b=[1, 1, 0])
    p3 = poly(a=[9, 1, 4], b=[2, 5, 3])
    
    assert {} == (p3@p3)._values
    # check Jacobi-Identity
    assert {} == (p1@(p2@p3) - (p1@p2)@p3 - p2@(p1@p3))._values
    
    
def test_poisson(tol=1e-15):
    # Test the property {f, gh} = {f, g}h + {f, h}g
    
    xieta = create_coords(2)
    p = (xieta[0] + 0.31*xieta[1])**2
    q = (0.7*xieta[0] - 8.61052*xieta[2] + 2.32321*xieta[1] - 0.93343*xieta[3]**3)**2
    h = -0.24*xieta[2]**5 + 7.321*xieta[3]
    
    # check 1
    p1 = p@(q*h)
    p2 = (p@q)*h + (p@h)*q
    assert p1.keys() == p2.keys()
    for key in p1.keys():
        v1, v2 = p1[key], p2[key]
        assert abs(v1 - v2)/(min([abs(v1), abs(v2)])) < tol
    
    # check 2
    p1 = (q**3)@p
    p2 = 3*q**2*(q@p)
    assert p1.keys() == p2.keys()
    for key in p1.keys():
        v1, v2 = p1[key], p2[key]
        assert abs(v1 - v2)/(min([abs(v1), abs(v2)])) < tol
        
        
def test_transform(tol=1e-15):
    '''
    Test some transformations from complex to real representation and reverse.
    '''
    
    ham1 = poly(values={(1, 1): (0.5035851182583087+0j), (0, 2): (-0.24820744087084556+0j), 
                        (2, 0): (-0.24820744087084556+0j), (2, 1): (0.014968964264246843+0j), 
                        (1, 2): (0.014968964264246843+0j), (0, 3): (-0.014968964264246843+0j), 
                        (3, 0): (-0.014968964264246843+0j)})
    
    ham1_r = ham1.realBasis()
    assert max(abs(ham1_r.complexBasis() - ham1).values()) < tol
    
    # some consistency checks
    q, p = create_coords(1, real=True)
    assert max(abs(q@p - 1).values()) < tol
    qq, pp = q.realBasis(), p.realBasis()
    assert max(abs(qq@pp - 1).values()) < tol
    qc, pc = q.complexBasis(), p.complexBasis()
    assert max(abs(qc@pc - 1j).values()) < tol # -1j*{qc, pc} = {q, p} = 1, see eq. (19) in "M. Titze - PhD thesis."
        
    
def test_shift():
    # Test if the derivative of a given Hamiltonian at a specific point equals the derivative of the shifted Hamiltonian at zero.
    
    H = lambda x, y, px, py: 0.24*(x**2 + px**2) + 0.86*(y**2 + py**2) + x**3 - y**3 - cos(x) + 9*x**4*y
    dH = derive(H, order=3)

    z = [1.1, 0.14, 1.26, 0.42]
    z0 = [0, 0, 0, 0]

    H_shift = lambda x, y, px, py: H(x + z[0], y + z[1], px + z[2], py + z[3])
    dH_shift = derive(H_shift, order=3)
    
    assert dH.hess(*z, mult_drv=False) == dH_shift.hess(*z0, mult_drv=False)
    assert dH.hess(*z, mult_drv=True) == dH_shift.hess(*z0, mult_drv=True)
    assert dH_shift(*z0) == dH(*z)
    
    
@pytest.mark.parametrize("tol1, tol2, code", [(1e-18, 1e-10, 'numpy'), (1e-35, 1e-28, 'mpmath')])
def test_stf_with_zeros(tol1, tol2, code, dps=32):
    stf_with_zeros(tol1=tol1, tol2=tol2, code=code, dps=dps)
    
    
@pytest.mark.parametrize("code, mode", [('numpy', 'default'), ('numpy', 'classic'), ('mpmath', 'default'), ('mpmath', 'classic')])
def test_fonfe(code, mode):
    fonfe(code=code, mode=mode)
        
        
@pytest.mark.parametrize("mode", ('default', 'classic'))
def test_exp_ad1(mode):
    exp_ad1(mode=mode)
    
        
@pytest.mark.parametrize("mode", ('default', 'classic'))
def test_exp_ad2(mode):
    exp_ad2(mode=mode)
    
            
def test_flow1(mu0=0.43, z=0.046, a=1.23, b=2.07, power=40, tol=1e-15, **kwargs):
    # Test if the flow for the sum of two parameters equals the chain of flows applied for each parameter individually.
    coeff = 1j*mu0/np.sqrt(2)**3
    H_accu = poly(values={(1, 1): -mu0,
                          (3, 0): -coeff/(1 - np.exp(3*1j*mu0)),
                          (2, 1): -coeff/(1 - np.exp(1j*mu0)),
                          (1, 2): coeff/(1 - np.exp(-1j*mu0)),
                          (0, 3): coeff/(1 - np.exp(-3*1j*mu0))}, **kwargs)
    
    Hlo = H_accu.lexp(power=power, **kwargs)

    v1 = Hlo(*Hlo(z, t=a), t=b)
    v2 = Hlo(z, t=a + b)
    assert all([abs(v1[k] - v2[k]) < tol for k in range(1)])
    
    
def test_flow2(mu0=0.43, power=40, tol=1e-15, max_power=30, **kwargs):
    # Test if the flow of a Lie operator is a symplectic map, i.e. test if
    # exp(:H:){x, y} = {exp(:H:)x, exp(:H:)y}
    # holds.
    coeff = 1j*mu0/np.sqrt(2)**3
    H_accu = poly(values={(1, 1): -mu0,
                          (3, 0): -coeff/(1 - np.exp(3*1j*mu0)),
                          (2, 1): -coeff/(1 - np.exp(1j*mu0)),
                          (1, 2): coeff/(1 - np.exp(-1j*mu0)),
                          (0, 3): coeff/(1 - np.exp(-3*1j*mu0))}, max_power=max_power, **kwargs)
    
    xi, eta = create_coords(1)
    lp1 = 0.24*xi**2 + 0.824*eta
    lp2 = -0.66*eta**2
    
    Hop = lexp(H_accu, power=power)
    
    term1 = Hop(lp1@lp2)
    term2 = Hop(lp1)@Hop(lp2)
    
    sk1 = set(term1.keys())
    sk2 = set(term2.keys())

    k1m2 = list(sk1.difference(sk2))
    k2m1 = list(sk2.difference(sk1))
    for key in k1m2:
        assert abs(term1[key]) < tol
    for key in k2m1:
        assert abs(term2[key]) < tol
        
    common_keys = list(sk1.intersection(sk2))
    assert all([abs(term1[k] - term2[k]) < tol for k in common_keys])
    
    
def test_flow3(Q=0.252, p=0.232, max_power=30, order=10, power=50, tol=1e-12):
    # Test if the flow map of a Lie operator is a symplectic map: Test if
    # M := Jacobi_x(phi(t, x)) it holds M.transpose()@Jc@M - Jc = 0, where Jc is the complex symplectic structure given
    # by the (xi, eta)-coordinates.
    mu0 = 2*np.pi*Q
    w = -1
    coeff = w*1j*mu0/np.sqrt(2)**3

    H_accu = poly(values={(1, 1): -mu0,
                          (3, 0): -coeff/(1 - np.exp(3*1j*mu0)),
                          (2, 1): -coeff/(1 - np.exp(1j*mu0)),
                          (1, 2): coeff/(1 - np.exp(-1j*mu0)),
                          (0, 3): coeff/(1 - np.exp(-3*1j*mu0))})
    
    H_accu_f = lambda *z: H_accu((z[0] + 1j*z[1])/np.sqrt(2), (z[0] - 1j*z[1])/np.sqrt(2))

    xieta = create_coords(1)
    
    t_ref = 1
    L1 = lexp(H_accu_f, order=order, components=xieta, t=t_ref, power=power, n_args=2, max_power=max_power)
    # check Symplecticity of the flow of L1 at position p:
    dL1flow = derive(lambda *x: L1(*x, t=t_ref), order=1, n_args=2)
    ep, epc = dL1flow.eval(p, p.conjugate())
    jacobi = [[ep.get_taylor_coefficients(n_args=2)[(1, 0)], ep.get_taylor_coefficients(n_args=2)[(0, 1)]],
              [epc.get_taylor_coefficients(n_args=2)[(1, 0)], epc.get_taylor_coefficients(n_args=2)[(0, 1)]]]
    jacobi = np.array(jacobi)
    Jc = -1j*create_J(1)
    check = jacobi.transpose()@Jc@jacobi - Jc
    assert all([all([abs(check[i, j]) < tol for i in range(2)]) for j in range(2)])
    

def test_construct(a: int=3, b: int=5, k: int=7, tol=1e-14):
    # test if :sin(xi*eta):, :cos(xi*eta):, :exp(xi*eta): applied to xi**a*eta**b gives
    # values as expected.
    
    xi, eta = create_coords(1)
    eps = xi*eta
    eps_ab = xi**a*eta**b
    
    assert eps@eps_ab == -1j*(b - a)*eps_ab
    assert eps**k@eps_ab == -1j*k*eps**(k - 1)*(b - a)*eps_ab
    
    sin_eps = construct(sin, eps, power=30)
    cos_eps = construct(cos, eps, power=30)
    
    diff1 = sin_eps@eps_ab + 1j*cos_eps*(b - a)*eps_ab
    assert max([abs(v) for v in diff1.values()]) < tol
    
    exp_eps = construct(exp, 1j*eps/(b - a), power=30)
    
    diff2 = exp_eps@eps_ab - exp_eps*eps_ab
    assert max([abs(v) for v in diff2.values()]) < tol
    diff3 = exp_eps**k@eps_ab - k*exp_eps**k*eps_ab
    assert max([abs(v) for v in diff3.values()]) < tol
    
    
def test_lexp_flow_consistency(z=[0.2, 0.2], Q=0.252, order=20, power=30):
    # Check if the flow function of a Lie-polynomial gives the same result as the
    # flow of the Lie-operator, having this polynomial as argument (exponent).
    
    mu0 = 2*np.pi*Q
    w = -1
    coeff = w*1j*mu0/np.sqrt(2)**3
    H_accu = poly(values={(1, 1): -mu0,
                          (3, 0): -coeff/(1 - np.exp(3*1j*mu0)),
                          (2, 1): -coeff/(1 - np.exp(1j*mu0)),
                          (1, 2): coeff/(1 - np.exp(-1j*mu0)),
                          (0, 3): coeff/(1 - np.exp(-3*1j*mu0))})
            
    H_accu_f = lambda *z: H_accu((z[0] + 1j*z[1])/np.sqrt(2), (z[0] - 1j*z[1])/np.sqrt(2))
    
    L1 = lexp(H_accu_f, order=order, power=power, n_args=2)
    argflow = L1.argument.lexp(power=L1.power)
    return argflow(*z) == L1(*z)


def test_stold_consistency(n=101, dim=6, tol=5e-15):
    '''
    Test the consistency of "symplectic_takagi_old" routine:
    If given a diagonal matrix, it must return the input diagonal again.
    '''
    assert dim%2 == 0, 'Dimension must be even.'

    for k in range(n):
        coeffs = np.random.rand(dim)*2 - 1 # create 'dim' random values between -1 and 1
        D1 = [coeffs[k] + coeffs[k + dim//2]*1j for k in range(dim//2)]
        D = np.diag(D1 + D1)
        S, _, _ = symplectic_takagi_old(D, orientation=D1)
        Dout = S@D@S.transpose()
        diff = Dout - D
        assert max(np.absolute(diff.flatten())) < tol
        
    
def test_hadamard(mu0=0.206, lo_power=30, max_power=10, tol=5e-16):
    '''
    Test of Hadamard's lemma. Let H0 denote a basic rotation and Hs a sextupole. Then it must hold:
      exp(:H0/2:) exp(:Hs:) exp(:H0/2:) = exp(:H0:) exp(:exp(:-H0/2:)Hs:)
    '''

    mu0 = mu0*2*np.pi
    
    xi, eta = create_coords(1, max_power=max_power)
    X, _ = create_coords(1, real=True, max_power=max_power)

    H0 = -mu0*xi*eta
    w = 1
    Hs = w/3*X**3
    
    o0 = lexp(H0, power=lo_power)
    ohalf = lexp(H0/2, power=lo_power)
    os = lexp(Hs, power=lo_power)
    
    hadamard1 = ohalf(os(ohalf(xi)))
    hadamard2 = o0(lexp(lexp(-H0/2, power=lo_power)(Hs), power=lo_power)(xi))
    
    base = np.linspace(0, 0.5, 30)
    phi = -0.1
    q_insert = np.cos(phi)*base
    p_insert = np.sin(phi)*base
    xi_insert = (q_insert + p_insert*1j)/np.sqrt(2)
        
    assert max(np.abs(hadamard2(xi_insert) - hadamard1(xi_insert))) < tol
    
    