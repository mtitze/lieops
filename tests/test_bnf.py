import numpy as np
import mpmath as mp
import time
from sympy import Symbol
import pytest

from njet.functions import cos, sin, exp
from njet import derive

from bnf import __version__
from bnf.lieop import liepoly, exp_ad, create_coords, construct, bnf, first_order_nf_expansion, lexp
from bnf.lieop.nf import homological_eq
from bnf.linalg.matrix import qpqp2qp, column_matrix_2_code, create_J
from bnf.linalg.nf import symplectic_takagi

def referencebnf(H, order: int, z=[], tol=1e-14, **kwargs):
    '''
    !!! this may be a slow version of the BNF routine !!!!
    
    Compute the Birkhoff normal form of a given Hamiltonian up to a specific order.
    
    Attention: Constants and any gradients of H at z will be ignored. If there is 
    a non-zero gradient, a warning is issued by default.
    
    Parameters
    ----------
    H: callable or dict
        Defines the Hamiltonian to be normalized. If dict, then it must be of the
        form (e.g. for phase space dimension 4): {(i, j, k, l): value}, where the tuple (i, j, k, l)
        denotes the exponents in xi1, xi2, eta1, eta2.
                
    order: int
        The order up to which we build the normal form. Results up to this order will provide the exact
        derivatives.
    
    z: list, optional
        List of length according to the signature of H. The point around which we are going to 
        build the map to normal coordinates. H will be expanded around this point. If nothing specified,
        then the expansion will take place around zero.
        
    tol: float, optional
        Tolerance below which we consider a value as zero and ignore it from calculations. This may
        improve performance.
        
    **kwargs
        Keyword arguments are passed to 'first_order_nf_expansion' routine.
    '''
    
    max_power = order # !!! TMP; need to set this very carefully
    exp_power = order # !!! TMP; need to set this very carefully
    
    if type(H) != dict:
        # obtain an expansion of H in terms of complex first-order normal form coordinates
        taylor_coeffs, nfdict = first_order_nf_expansion(H, z=z, order=order, **kwargs)
    else:
        taylor_coeffs = H
        nfdict = {}
        
    # get the dimension (by looking at one key in the dict)
    dim2 = len(next(iter(taylor_coeffs)))
    dim = dim2//2
        
    # define mu and H0. For H0 we skip any (small) off-diagonal elements as they must be zero by construction.
    H0 = {}
    mu = []
    for j in range(dim): # add the second-order coefficients (tunes)
        tpl = tuple([0 if k != j and k != j + dim else 1 for k in range(dim2)])
        muj = taylor_coeffs[tpl]
        assert muj.imag < tol
        muj = muj.real
        H0[tpl] = muj
        mu.append(muj)
    H0 = liepoly(values=H0, dim=dim, max_power=max_power)
    
    # For H, we take the values of H0 and add only higher-order terms (so we skip any gradients (and constants). 
    # Note that the skipping of gradients leads to an artificial normal form which may not have anything relation
    # to the original problem. By default, the user will be informed if there is a non-zero gradient 
    # in 'first_order_nf_expansion' routine.
    H_values = {k: v for k, v in H0.values.items()}
    H_values.update({k: v for k, v in taylor_coeffs.items() if sum(k) > 2})
    H = liepoly(values=H_values, dim=dim, max_power=max_power)
    
    # Indution start (k = 2); get P_3 and R_4. Z_2 is set to zero.
    Zk = liepoly(dim=dim, max_power=max_power) # Z_2
    Pk = H.homogeneous_part(3) # P_3
    Hk = H.copy() # H_2 = H
        
    chi_all, Hk_all = [], [H]
    Zk_all, Qk_all = [], []
    for k in range(3, order + 1):
        chi, Q = homological_eq(mu=mu, Z=Pk, max_power=max_power) 
        if len(chi.values) == 0:
            # in this case the canonical transformation will be the identity and so the algorithm stops.
            break
        Hk = exp_ad(-chi, Hk, power=exp_power)
        # Hk = exp_ad(-chi, Hk, power=k + 1) # faster but likely inaccurate; need tests
        Pk = Hk.homogeneous_part(k + 1)
        Zk += Q 
        
        chi_all.append(chi)
        Hk_all.append(Hk)
        Zk_all.append(Zk)
        Qk_all.append(Q)

    # assemble output
    out = {}
    out['nfdict'] = nfdict
    out['H'] = H
    out['H0'] = H0
    out['mu'] = mu    
    out['chi'] = chi_all
    out['Hk'] = Hk_all
    out['Zk'] = Zk_all
    out['Qk'] = Qk_all
        
    return out

def check_2nd_orders(hdict, dim, tol=1e-14): 
    '''
    Check if the 2nd order coefficients of a liepoly-dictionary correspond to a 'diagonal'
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
    expansion, nfdict = first_order_nf_expansion(H2, warn=True, code='numpy', **kwargs)
    HLie = liepoly(values=expansion)
    Kinv = nfdict['Kinv'] # K(p, q) = (xi, eta)
    xieta = create_coords(1)

    # first apply K, then exp_ad:
    xy_mapped = [xieta[0]*Kinv[0, 0] + xieta[1]*Kinv[0, 1], xieta[0]*Kinv[1, 0] + xieta[1]*Kinv[1, 1]]
    xy_final_mapped = exp_ad(HLie, xy_mapped, power=power, t=mu) # (x, y) final in terms of xi and eta 
    
    # first apply exp_ad, then K:
    xy_fin = exp_ad(HLie, xieta, power=power, t=mu)
    xy_final = [xy_fin[0]*Kinv[0, 0] + xy_fin[1]*Kinv[0, 1], xy_fin[0]*Kinv[1, 0] + xy_fin[1]*Kinv[1, 1]]
    
    # Both results must be equal.
    for k in range(len(xy_final)):
        d1 = xy_final[k].values
        d2 = xy_final_mapped[k].values
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
        diff = expectation[k] - (lie_k( sum([zz[l]*K[:, l] for l in range(len(zz))]) ) ).expand()
        assert abs(diff.coeff(zz[0])) < tol and abs(diff.coeff(zz[1])) < tol
        
def fonfe(tol=1e-14, code='numpy', **kwargs):
    # Test of the first-order normal form expansion and the result of reordering the canonical coordinates. Both results against each other.
    
    # compute expansion up to 3rd order:
    H = lambda x, y, px, py: 0.24*(x**2 + px**2) + 0.86*(y**2 + py**2) + x**3 - y**3 - sin(x + 9*x**4*y + 0.2*px*y + 0.411*x*y - 0.378*px*x - 0.039*x*py + 1.2*y*py + 0.13*py*px) + x
    he, he_dict = first_order_nf_expansion(H, order=3, warn=True, code=code, **kwargs)
    
    # compute expansion of the same Hamiltonian, but with respect to an alternative symplectic structure, up to third order:
    T = qpqp2qp(2)
    if code == 'mpmath':
        T = mp.matrix(T)
    HT = lambda x, px, y, py: H(x, y, px, py)
    heT, heT_dict = first_order_nf_expansion(HT, order=3, warn=True, T=T, code=code, **kwargs)
    
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
    expansion, nfdict = first_order_nf_expansion(H2, order=5, warn=True, code=code, dps=dps, **kwargs)
    HLie = liepoly(values=expansion, max_power=max_power)
    K = nfdict['K']
    xieta = create_coords(1, max_power=max_power)
    
    # first apply function, then exp_ad:
    if code == 'numpy':
        xy_mapped = (K@np.array([xieta]).transpose()).tolist()
    elif code == 'mpmath':
        xy_mapped = [[sum([xieta[k]*K[j, k] for k in range(len(xieta))])] for j in range(len(K))]
    xy_mapped = [xy_mapped[k][0]**3 + 0.753 for k in range(len(xy_mapped))] # apply an additional non-linear operation
    xy_final_mapped = exp_ad(HLie, xy_mapped, power)    
    
    # first apply exp_ad, then function:
    xy_fin = exp_ad(HLie, xieta, power)
    if code == 'numpy':
        xy_final = (K@np.array([xy_fin]).transpose()).tolist()
    elif code == 'mpmath':
        xy_final = [[sum([xy_fin[k]*K[j, k] for k in range(len(xieta))])] for j in range(len(K))]
    xy_final = [xy_final[k][0]**3 + 0.753 for k in range(len(xy_final))] # apply an additional non-linear operation
    
    # Both results must be relatively close (and every entry non-zero).
    for k in range(len(xy_final)):
        d1 = xy_final[k].values
        d2 = xy_final_mapped[k].values
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
        J = np.matrix(create_J(2)).transpose()
    if code == 'mpmath':
        A = mp.matrix(A)
        EE = mp.eye(4)
        J = mp.matrix(create_J(2)).transpose()
                
    EE[2, 2] = 0
    EE[3, 3] = 0

    G = A.transpose()@J@EE@J@A
    G = G + G.transpose()
    
    S0, D0 = symplectic_takagi(G, tol=tol1, dps=dps)
    
    symplecticity = S0.transpose()@J@S0 - J
    factorization = S0@D0@S0.transpose() - G
    
    dim2 = len(G)
    assert max([max([abs(symplecticity[j, k]) for j in range(dim2)]) for k in range(dim2)]) < tol2
    assert max([max([abs(factorization[j, k]) for j in range(dim2)]) for k in range(dim2)]) < tol2
    
    
#########
# Tests #
#########

def test_version():
    assert __version__ == '0.1.0'
    
def test_jacobi():
    # Test the Jacobi-identity for the liepoly class
    
    p1 = liepoly(a=[0, 1, 0], b=[1, 1, 1])
    p2 = liepoly(a=[2, 0, 1], b=[1, 1, 0])
    p3 = liepoly(a=[9, 1, 4], b=[2, 5, 3])
    
    assert {} == (p3@p3).values
    # check Jacobi-Identity
    assert {} == (p1@(p2@p3) - (p1@p2)@p3 - p2@(p1@p3)).values
    
    
def test_poisson(tol=1e-15):
    # Test the property {f, gh} = {f, g}h + {f, h}g
    
    xieta = create_coords(2)
    p = (xieta[0] + 0.31*xieta[1])**2
    q = (0.7*xieta[0] - 8.61052*xieta[2] + 2.32321*xieta[1] - 0.93343*xieta[3]**3)**2
    h = -0.24*xieta[2]**5 + 7.321*xieta[3]
    
    # check 1
    p1 = p@(q*h)
    p2 = (p@q)*h + (p@h)*q
    assert p1.values.keys() == p2.values.keys()
    for key in p1.values.keys():
        v1, v2 = p1.values[key], p2.values[key]
        assert abs(v1 - v2)/(min([abs(v1), abs(v2)])) < tol
    
    # check 2
    p1 = (q**3)@p
    p2 = 3*q**2*(q@p)
    assert p1.values.keys() == p2.values.keys()
    for key in p1.values.keys():
        v1, v2 = p1.values[key], p2.values[key]
        assert abs(v1 - v2)/(min([abs(v1), abs(v2)])) < tol
        
    
def test_shift():
    # Test if the derivative of a given Hamiltonian at a specific point equals the derivative of the shifted Hamiltonian at zero.
    
    H = lambda x, y, px, py: 0.24*(x**2 + px**2) + 0.86*(y**2 + py**2) + x**3 - y**3 - cos(x) + 9*x**4*y
    dH = derive(H, order=3)

    z = [1.1, 0.14, 1.26, 0.42]
    z0 = [0, 0, 0, 0]

    H_shift = lambda x, y, px, py: H(x + z[0], y + z[1], px + z[2], py + z[3])
    dH_shift = derive(H_shift, order=3)
    
    assert dH.hess(z, mult_drv=False) == dH_shift.hess(z0, mult_drv=False)
    assert dH.hess(z, mult_drv=True) == dH_shift.hess(z0, mult_drv=True)
    assert dH_shift.get_taylor_coefficients(dH_shift.eval(z0)) == dH.get_taylor_coefficients(dH.eval(z))
    
    
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
    
            
def test_flow1(mu0=0.43, z=[0.046], a=1.23, b=2.07, power=40, tol=1e-15, **kwargs):
    # Test if the flow for the sum of two parameters equals the chain of flows applied for each parameter individually.
    coeff = 1j*mu0/np.sqrt(2)**3
    H_accu = liepoly(values={(1, 1): -mu0,
                             (3, 0): -coeff/(1 - np.exp(3*1j*mu0)),
                             (2, 1): -coeff/(1 - np.exp(1j*mu0)),
                             (1, 2): coeff/(1 - np.exp(-1j*mu0)),
                             (0, 3): coeff/(1 - np.exp(-3*1j*mu0))}, **kwargs)
    
    Hflow = H_accu.flow(power=power, **kwargs)

    v1 = Hflow(Hflow(z, t=a), t=b)
    v2 = Hflow(z, t=a + b)
    assert all([abs(v1[k] - v2[k]) < tol for k in range(1)])
    
    
def test_flow2(mu0=0.43, power=40, tol=1e-15, max_power=30, **kwargs):
    # Test if the flow of a Lie operator is a symplectic map, i.e. test if
    # exp(:H:){x, y} = {exp(:H:)x, exp(:H:)y}
    # holds.
    coeff = 1j*mu0/np.sqrt(2)**3
    H_accu = liepoly(values={(1, 1): -mu0,
                             (3, 0): -coeff/(1 - np.exp(3*1j*mu0)),
                             (2, 1): -coeff/(1 - np.exp(1j*mu0)),
                             (1, 2): coeff/(1 - np.exp(-1j*mu0)),
                             (0, 3): coeff/(1 - np.exp(-3*1j*mu0))}, max_power=max_power, **kwargs)
    
    xi, eta = create_coords(1)
    lp1 = 0.24*xi**2 + 0.824*eta
    lp2 = -0.66*eta**2
    
    term1 = exp_ad(H_accu, lp1@lp2, power=power) 
    term2 = exp_ad(H_accu, lp1, power=power)@exp_ad(H_accu, lp2, power=power)
    
    sk1 = set(term1.values.keys())
    sk2 = set(term2.values.keys())

    k1m2 = list(sk1.difference(sk2))
    k2m1 = list(sk2.difference(sk1))
    for key in k1m2:
        assert abs(term1.values[key]) < tol
    for key in k2m1:
        assert abs(term2.values[key]) < tol
        
    common_keys = list(sk1.intersection(sk2))
    assert all([abs(term1.values[k] - term2.values[k]) < tol for k in common_keys])
    
    
def test_flow3(Q=0.252, p=[0.232], max_power=30, order=10, power=50, tol=1e-12):
    # Test if the flow map of a Lie operator is a symplectic map: Test if
    # M := Jacobi_x(phi(t, x)) it holds M.transpose()@Jc@M - Jc = 0, where Jc is the complex symplectic structure given
    # by the (xi, eta)-coordinates.
    mu0 = 2*np.pi*Q
    w = -1
    coeff = w*1j*mu0/np.sqrt(2)**3

    H_accu = liepoly(values={(1, 1): -mu0,
                             (3, 0): -coeff/(1 - np.exp(3*1j*mu0)),
                             (2, 1): -coeff/(1 - np.exp(1j*mu0)),
                             (1, 2): coeff/(1 - np.exp(-1j*mu0)),
                             (0, 3): coeff/(1 - np.exp(-3*1j*mu0))})
    
    H_accu_f = lambda z: H_accu([(z[0] + 1j*z[1])/np.sqrt(2),
                                 (z[0] - 1j*z[1])/np.sqrt(2)])

    xieta = create_coords(1)
    
    t_ref = 1
    L1 = lexp(H_accu_f, order=order, components=xieta, t=t_ref, power=power, n_args=2, max_power=max_power)
    # check Symplecticity of the flow of L1 at position p:
    dL1flow = derive(lambda x: L1.flowFunc(t_ref, x), order=1, n_args=2)
    ep, epc = dL1flow.eval(p + p) # N.B. ep contains x0-terms, while ecp contains x1-terms
    jacobi = [[dL1flow.get_taylor_coefficients(ep)[(1, 0)], dL1flow.get_taylor_coefficients(ep)[(0, 1)]],
              [dL1flow.get_taylor_coefficients(epc)[(1, 0)], dL1flow.get_taylor_coefficients(epc)[(0, 1)]]]
    jacobi = np.array(jacobi)
    Jc = -1j*column_matrix_2_code(create_J(1), code='numpy')
    check = jacobi.transpose()@Jc@jacobi - Jc
    assert all([all([abs(check[i, j]) < tol for i in range(2)]) for j in range(2)])
    

def test_construct(a: int=3, b: int=5, k: int=7):
    # test if :sin(xi*eta):, :cos(xi*eta):, :exp(xi*eta): applied to xi**a*eta**b gives
    # values as expected.
    
    xi, eta = create_coords(1)
    eps = xi*eta
    eps_ab = xi**a*eta**b
    
    assert eps@eps_ab == -1j*(b - a)*eps_ab
    assert eps**k@eps_ab == -1j*k*eps**(k - 1)*(b - a)*eps_ab
    
    sin_eps = construct(eps, sin, power=20)
    cos_eps = construct(eps, cos, power=20)
    assert sin_eps@eps_ab == -1j*cos_eps*(b - a)*eps_ab
    
    exp_eps = construct(1j*eps/(b - a), exp, power=10)
    assert exp_eps@eps_ab == exp_eps*eps_ab
    assert exp_eps**k@eps_ab == k*exp_eps**k*eps_ab
    
    
def test_lexp_flow_consistency(z=[0.2, 0.2], Q=0.252, order=20, power=30):
    # Check if the flow function of a Lie-polynomial gives the same result as the
    # flow of the Lie-operator, having this polynomial as argument (exponent).
    
    mu0 = 2*np.pi*Q
    w = -1
    coeff = w*1j*mu0/np.sqrt(2)**3
    H_accu = liepoly(values={(1, 1): -mu0,
                             (3, 0): -coeff/(1 - np.exp(3*1j*mu0)),
                             (2, 1): -coeff/(1 - np.exp(1j*mu0)),
                             (1, 2): coeff/(1 - np.exp(-1j*mu0)),
                             (0, 3): coeff/(1 - np.exp(-3*1j*mu0))})
            
    H_accu_f = lambda z: H_accu([(z[0] + 1j*z[1])/np.sqrt(2),
                                 (z[0] - 1j*z[1])/np.sqrt(2)])
    
    L1 = lexp(H_accu_f, t=1, order=order, power=power, n_args=2)
    argflow = L1.argument.flow(power=L1.power) 
    assert argflow(z) == L1(z)
    
def test_bnf_performance(threshold=1.1, tol=1e-15):
    # Test if any modification of the bnf main routine will be slower than the reference bnf routine (defined in this script).
    
    H = lambda x, y, z, px, py, pz: 0.24*(x**2 + px**2) + 0.86*(y**2 + py**2) + x**3 + y**3 + z**2 + pz**2 + 0.4*pz*y
    # z0 = 6*[0]
    z1 = [np.pi/2, 0.55, 0.32, 7.13, -0.6311, 2.525]

    time1 = time.time()
    tcref = referencebnf(H, z=z1, order=10, warn=True)
    time_ref = time.time() - time1
    
    time2 = time.time()
    tc = bnf(H, z=z1, order=10, warn=True)
    time_bnf = time.time() - time2
    
    # performance check
    assert time_bnf*threshold >= time_ref, 'Error: new time*threshold = {} < {} (reference time)'.format(time_bnf*threshold, time_ref)

    # check on equality
    chifinal_ref = tcref['chi'][-1]
    chifinal = tc['chi'][-1]
    assert chifinal_ref.values.keys() == chifinal.values.keys()
    for key in chifinal_ref.values.keys():
        v1, v2 = chifinal_ref.values[key], chifinal.values[key]
        assert abs(v1 - v2)/(min([abs(v1), abs(v2)])) < tol
    
def test_hadamard(mu0=0.206, lo_power=30, max_power=10, tol=1e-16):
    '''
    Test of Hadamard's lemma. Let H0 denote a basic rotation and Hs a sextupole. Then it must hold:
      exp(:H0/2:) exp(:Hs:) exp(:H0/2:) = exp(:H0:) exp(:exp(:-H0/2:)Hs:)
    '''

    mu0 = mu0*2*np.pi
    
    xi, eta = create_coords(1, max_power=max_power)
    X, _ = create_coords(1, cartesian=True, max_power=max_power)

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
        
    assert max(np.abs(hadamard2([xi_insert]) - hadamard1([xi_insert]))) < tol
    
    