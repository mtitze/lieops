import time
from bnf import __version__
from bnf.lie import liepoly, exp_ad, exp_ad_par, create_coordinates
from bnf.nf import first_order_nf_expansion, homological_eq, bnf
import numpy as np
import mpmath as mp
from njet.functions import cos, sin
from njet import derive
from sympy import Symbol

def referencebnf(H, order: int, z=[], tol=1e-14, **kwargs):
    '''
    !!! this is a slow version of the BNF routine !!!!
    
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
        assert muj.imag == 0
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
        Hk = sum(exp_ad(-chi, Hk, power=exp_power))
        # Hk = sum(exp_ad(-chi, Hk, power=k + 1)) # faster but likely inaccurate; need tests
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


def is_pos_def(A, code='mpmath'):
    '''Check if a given matrix A is positive definite.
    
    For a real matrix $A$, we have $x^TAx = \frac{1}{2}(x^T(A+A^T)x)$, 
    and $A + A^T$ is a symmetric real matrix. So $A$ is positive definite 
    iff $A + A^T$ is positive definite, iff all the eigenvalues of $A + A^T$ are positive.
    
    Parameters
    ----------
    A: matrix
        The matrix to be checked.
        
    code: str, optional
        The code to be used for the check. Either 'mpmath' or 'numpy' (default).
        
    Returns
    -------
    boolean
        True if matrix is positive definite.
    '''
    if code == 'mpmath':
        result = np.all([mp.eig(A + A.transpose())[0][k] > 0 for k in range(len(A))])
    elif code == 'numpy':
        result = np.all(np.linalg.eigvals(A + A.transpose()) > 0)
    
    return result


def williamson_check(A, S, J, code='numpy', tol=1e-14):
    '''
    Check if a given matrix A and the matrix S diagonalizting A according to the theorem of Williamson
    by S.transpose()*A*S = D actually are satisfying all required conditions of the theorem.
    
    If any condition is violated, the routine raises an AssertionError.
    
    Parameters
    ----------
    A: matrix
        The matrix to be diagonalized.
        
    S: matrix
        The symplectic matrix obtained by Williamson's theorem.
        
    code: str, optional
        The code by which the check should be performed. Either 'mpmath' or 'numpy' (default).
        
    tol: float, optional
        A tolerance by which certain properties (like matrix entries) are considered to be zero (default 1e-14).
    '''

    if code == 'numpy':
        isreal = np.isreal(A).all()
        isposdef = is_pos_def(A, code=code)
        issymmetric = np.all(A - A.transpose()) == 0
        isevendim = len(A)%2 == 0
        symplecticity = np.linalg.norm(S.transpose()*J*S - J)
        issymplectic = symplecticity < tol
        
        diag = J@S@J@A@J@S.transpose()@J
        offdiag = np.array([[diag[k, l] if k != l else 0 for k in range(len(diag))] for l in range(len(diag))])
        isdiagonal = np.all(np.abs(offdiag) < tol)

    elif code == 'mpmath':
        isreal = mp.norm(A - A.conjugate()) == 0
        isposdef = is_pos_def(A, code=code)
        issymmetric = all([[(A - A.transpose())[i, j] == 0 for i in range(len(A))] for j in range(len(A))])
        isevendim = len(A)%2 == 0
        symplecticity = mp.norm(S.transpose()@J@S - J)
        issymplectic = symplecticity < tol
        
        diag = J@S@J@A@J@S.transpose()@J
        absoffdiag = np.array([[abs(complex(diag[k, l])) if k != l else 0 for k in range(len(diag))] for l in range(len(diag))])
        isdiagonal = np.all(absoffdiag < tol)
        
    assert isreal, 'Input matrix A not real.'
    assert isposdef, 'Input matrix A not positive definite.'
    assert issymmetric,  'Input matrix A not symmetric.'
    assert isevendim, 'Dimension not even.'
    assert issymplectic, f'Symplecticity not ensured: |S^(tr)@J@S - J| = {symplecticity} >= {tol} (tol)'
    assert isdiagonal, f'Matrix D = S^(-tr)@M@S^(-1) =\n{diag}\nappears not to be diagonal (one entry having abs > {tol} (tol)).'


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


def qpqp2qp(n):
    '''Compute a transformation matrix T by which we can transform a given
    (2n)x(2n) matrix M, represented in (q1, p1, q2, p2, ..., qn, pn)-coordinates, into
    a (q1, q2, ..., qn, p1, p2, ..., pn)-representation via
    M' = T^(-1)*M*T. T will be orthogonal, i.e. T^(-1) = T.transpose().
    
    Parameters
    ----------
    n: int
        number of involved coordinates (i.e. 2*n dimension of phase space)
        
    Returns
    -------
    np.matrix
        Numpy matrix T defining the aforementioned transformation.
    '''
    columns_q, columns_p = [], []
    for k in range(n):
        # The qk are mapped to the positions zj via k->j as follows
        # 1 -> 1, 2 -> 3, 3 -> 5, ..., k -> 2*k - 1. The pk are mapped to the 
        # positions zj via k->j as follows
        # 1 -> 2, 2 -> 4, 3 -> 6, ..., k -> 2*k. The additional "-1" is because
        # in Python the indices are one less than the mathematical notation.
        column_k = np.zeros(2*n)
        column_k[2*(k + 1) - 1 - 1] = 1
        columns_q.append(column_k)

        column_kpn = np.zeros(2*n)
        column_kpn[2*(k + 1) - 1] = 1
        columns_p.append(column_kpn)
        
    q = np.array(columns_q).transpose()
    p = np.array(columns_p).transpose()
    return np.bmat([[q, p]])

    
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
    
    xieta = create_coordinates(2)
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
    
    assert dH.hess(z, mult=False) == dH_shift.hess(z0, mult=False)
    assert dH.hess(z, mult=True) == dH_shift.hess(z0, mult=True)
    assert dH_shift.get_taylor_coefficients(dH_shift.eval(z0)) == dH.get_taylor_coefficients(dH.eval(z))
    
    
def test_fonfe(tol=1e-14, code='numpy'):
    # Test of the first-order normal form expansion and the result of reordering the canonical coordinates. Both results against each other.
    
    # compute expansion up to 3rd order:
    H = lambda x, y, px, py: 0.24*(x**2 + px**2) + 0.86*(y**2 + py**2) + x**3 - y**3 - sin(x + 9*x**4*y + 0.2*px*y + 0.411*x*y - 0.378*px*x - 0.039*x*py + 1.2*y*py + 0.13*py*px) + x
    he, he_dict = first_order_nf_expansion(H, order=3, warn=True, code=code)
    
    # compute expansion of the same Hamiltonian, but with respect to an alternative symplectic structure, up to third order:
    T = qpqp2qp(2)
    if code == 'mpmath':
        T = mp.matrix(T)
    HT = lambda x, px, y, py: H(x, y, px, py)
    heT, heT_dict = first_order_nf_expansion(HT, order=3, warn=True, T=T, code=code)
    
    assert check_2nd_orders(he, dim=2, tol=tol)
    assert check_2nd_orders(heT, dim=2, tol=tol)
    
    # compare results against each other
    assert he == heT
    
    
def test_exp_ad1(mu=-0.2371, power=18, tol=1e-15):
    # Test the exponential operator on Lie maps for the case of a 2nd order Hamiltonian (rotation) and
    # the linear map K to (first-order) normal form.
    
    H2 = lambda x, px: 0.5*(x**2 + px**2)
    expansion, nfdict = first_order_nf_expansion(H2, warn=True, code='numpy')
    HLie = liepoly(values=expansion)
    K = nfdict['K']
    xieta = create_coordinates(1)

    # first apply K, then exp_ad:
    xy_mapped = K@np.array([xieta]).transpose()
    xy_fin_series_mapped = [exp_ad(HLie, xy_mapped[k, 0], power) for k in range(len(xy_mapped))]
    xy_final_mapped = [sum(exp_ad_par(e, mu)) for e in xy_fin_series_mapped] # (x, y) final in terms of xi and eta 
    
    # first apply exp_ad, then K:
    xy_fin_series = [exp_ad(HLie, xieta[k], power) for k in range(len(xy_mapped))]
    xy_fin = [sum(exp_ad_par(e, mu)) for e in xy_fin_series]
    xy_final = K@np.array([xy_fin]).transpose() # (x, y) final in terms of xi and eta
    
    # Both results must be equal.
    for k in range(len(xy_final)):
        d1 = xy_final[k][0].values
        d2 = xy_final_mapped[k].values
        for key, v1 in d1.items():
            v2 = d2[key]
            assert abs(v1 - v2) < tol
            
    # check if the result also agrees with the analytical expectation
    Kinv = nfdict['Kinv'] # (x, y) = K*(xi, eta)
    zz = [Symbol('x'), Symbol('px')]
    xf = np.cos(mu)*zz[0] - np.sin(mu)*zz[1]
    pxf = np.cos(mu)*zz[1] + np.sin(mu)*zz[0]
    expectation = [xf, pxf]
    for k in range(len(xy_final_mapped)):
        lie_k = xy_final_mapped[k]
        diff = expectation[k] - (lie_k( sum([Kinv[:, l]*zz[l] for l in range(len(zz))]) ) ).expand()
        assert abs(diff.coeff(zz[0])) < tol and abs(diff.coeff(zz[1])) < tol
    
    
def test_exp_ad2(mu=0.6491, power=40, tol=1e-14, max_power=10, code='mpmath', **kwargs):
    # Test the exponential operator on Lie maps for the case of a 5th order Hamiltonian and
    # a non-linear map, making use of K, the linear map to (first-order) normal form.
    
    # Attention: This test appears to be suceptible against round-off errors for higher powers (power and max_power),
    # and therefore requires mpmath to have sufficient precision (and sufficiently high power in exp).
    
    H2 = lambda x, px: mu*0.5*(x**2 + px**2) + x**3 + x*px**4
    expansion, nfdict = first_order_nf_expansion(H2, order=5, warn=True, code=code, **kwargs)
    HLie = liepoly(values=expansion, max_power=max_power)
    K = nfdict['K']
    xieta = create_xieta(1, max_power=max_power)
    
    # first apply function, then exp_ad:
    if code == 'numpy':
        xy_mapped = (K@np.array([xieta]).transpose()).tolist()
    elif code == 'mpmath':
        xy_mapped = [[sum([K[j, k]*xieta[k] for k in range(len(xieta))])] for j in range(len(K))]
    xy_mapped = [xy_mapped[k][0]**3 + 0.753 for k in range(len(xy_mapped))] # apply an additional non-linear operation
    
    xy_fin_series_mapped = [exp_ad(HLie, xy_mapped[k], power) for k in range(len(xy_mapped))]    
    xy_final_mapped = [sum(e) for e in xy_fin_series_mapped]
    
    # first apply exp_ad, then function:
    xy_fin_series = [exp_ad(HLie, xieta[k], power) for k in range(len(xy_mapped))]
    xy_fin = [sum(e) for e in xy_fin_series]
    if code == 'numpy':
        xy_final = (K@np.array([xy_fin]).transpose()).tolist()
    elif code == 'mpmath':
        xy_final = [[sum([K[j, k]*xy_fin[k] for k in range(len(xieta))])] for j in range(len(K))]
    xy_final = [xy_final[k][0]**3 + 0.753 for k in range(len(xy_final))] # apply an additional non-linear operation
    
    # Both results must be relatively close (and every entry non-zero).
    for k in range(len(xy_final)):
        d1 = xy_final[k].values
        d2 = xy_final_mapped[k].values
        for key, v1 in d1.items():
            v2 = d2[key]
            assert abs(v1 - v2)/min([abs(v1), abs(v2)]) < tol
            
    
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

    
if __name__ == '__main__':
    test_version()
    test_jacobi()
    test_poisson()
    test_shift()
    test_fonfe(code='numpy')
    test_fonfe(code='mpmath')
    test_exp_ad1()
    test_exp_ad2()
    test_bnf_performance()
    