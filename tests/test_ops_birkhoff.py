import pytest
import numpy as np
import time
from itertools import combinations, chain, permutations
from operator import sub

from scipy.linalg import expm
from njet import derive

from lieops.ops import create_coords, poly, lexp
from lieops.ops.birkhoff import homological_eq, bnf
from lieops.linalg.nf import first_order_nf_expansion

# Some helper routines to generate arbitrary Lie-polynomials; these routines are not optimal 
# but they should be fine for the test(s) below.

def sum_to_n(n):
    'Generate the series of +ve integer lists which sum to a +ve integer, n.'
    # taken from
    # https://stackoverflow.com/questions/2065553/get-all-numbers-that-add-up-to-a-number
    b, mid, e = [0], list(range(1, n)), [n]
    splits = (d for i in range(n) for d in combinations(mid, i)) 
    return (list(map(sub, chain(s, e), chain(b, s))) for s in splits)

def get_all_powers(dim, order):
    all_powers = []
    for p in sum_to_n(order):
        if len(p) > dim:
            continue
        pp = (p + [0]*(dim - len(p)))
        for c in set(permutations(pp)):
            all_powers.append(c)
    return sorted(set(all_powers)) # set() necessary by construction; code may be improved

def create_random_poly(dim, order, scaling=1, **kwargs):
    '''
    create a random n-th order polynomial
    '''
    dim2 = dim*2
    r = (1 - 2*np.random.rand(1)) + (1 - 2*np.random.rand(1))*1j
    v = {tuple([0]*dim2): complex(r[0])*scaling}
    xieta = create_coords(dim, **kwargs)
    for power in range(order + 1):
        for powers in get_all_powers(dim2, power):
            r = (1 - 2*np.random.rand(1)) + (1 - 2*np.random.rand(1))*1j
            v[tuple(powers)] = complex(r[0])*scaling
    return poly(values=v, **kwargs)

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
        The order up to which we build the normal form. Here order = k means that we compute
        k homogeneous Lie-polynomials, where the smallest one will have power k + 2 and the 
        succeeding ones will have increased powers by 1.
    
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
    power = order + 2
    max_power = power # !!! TMP; need to set this very carefully
    exp_power = power # !!! TMP; need to set this very carefully
    
    if type(H) != dict:
        # obtain an expansion of H in terms of complex first-order normal form coordinates
        taylor_coeffs, nfdict = first_order_nf_expansion(H, z=z, power=power, **kwargs)
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
    H0 = poly(values=H0, dim=dim, max_power=max_power)
    
    # For H, we take the values of H0 and add only higher-order terms (so we skip any gradients (and constants). 
    # Note that the skipping of gradients leads to an artificial normal form which may not have anything relation
    # to the original problem. By default, the user will be informed if there is a non-zero gradient 
    # in 'first_order_nf_expansion' routine.
    H = H0.update({k: v for k, v in taylor_coeffs.items() if sum(k) > 2})
    
    # Indution start (k = 2); get P_3 and R_4. Z_2 is set to zero.
    Zk = poly(dim=dim, max_power=max_power) # Z_2
    Pk = H.homogeneous_part(3) # P_3
    Hk = H.copy() # H_2 = H
        
    chi_all, Hk_all = [], [H]
    Zk_all, Qk_all = [], []
    for k in range(3, power + 1):
        chi, Q = homological_eq(mu=mu, Z=Pk, max_power=max_power) 
        if len(chi) == 0:
            # in this case the canonical transformation will be the identity and so the algorithm stops.
            break
        Hk = lexp(-chi, power=exp_power)(Hk)
        # Hk = lexp(-chi, power=k + 1)(Hk) # faster but likely inaccurate; need tests
        Pk = Hk.homogeneous_part(k + 1)
        Zk += Q 
        
        chi_all.append(-chi)
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


#########
# Tests #
#########

@pytest.mark.parametrize("dim, power, check", [(1, 5, True)])
def test_bnf(dim, power, check, tol=1e-14, tol2=1e-4, symlogs_tol2=1e-12):
    '''
    Test of the Birkhoff normal form routine, applied to a random polynomial.
    
    check: Parameter to be passed to bnf routine; for example, it will check if G@J can be diagonalized, where
           G is the Hesse-matrix of the normal form at the point of interest. Since sympy is used,
           this works only for dim = 1 or dim = 2.
    '''
    # Attention: This test currently gets slow for dim > 1; Jordan normal form needs to be implemented
    # for dim >= 3 or check needs to be removed (TODO)
    r1 = create_random_poly(dim, power, max_power=10)
    H1 = r1.extract(key_cond=lambda k: sum(k) > 1) # drop constants and gradients

    bnfdict1 = H1.bnf(order=power - 1, symlogs_tol2=symlogs_tol2, check=check)
    A = bnfdict1['nfdict']['A']
    C1, C2 = bnfdict1['nfdict']['C1'], bnfdict1['nfdict']['C2']
    assert (np.abs(A - expm(C1)@expm(C2)) < tol).all()
    
    # check if A properly diagonalizes H1:
    dim2 = dim*2
    Hcheck = lambda *z: H1(*[sum([z[j]*A[i, j] for j in range(dim2)]) for i in range(dim2)])
    dHcheck = derive(Hcheck, n_args=dim2, order=2)
    dHcheck_at_0 = dHcheck(*[0]*dim2)
    non_zero_keys = [tuple([1 if j == k or j == k + dim else 0 for j in range(dim2)]) for k in range(dim)]
    # non_zero_keys consist of those tuples for which the first-order normal form must be non-zero. For example,
    # if dim = 2 then we have (1, 0, 1, 0) and (0, 1, 0, 1), corresponding to the polynomials xi_1*eta_1 and xi_2*eta_2.
    assert all([abs(v) < tol for k, v in dHcheck_at_0.items() if k not in non_zero_keys])
    for k in non_zero_keys:
        assert abs(dHcheck_at_0[k] - bnfdict1['H0'][k]) < tol
    
    
def test_bnf_diagonalization(tol=5e-15, **kwargs):
    '''
    Test if the first_order_nf_expansion routine will give the same result for a 1D-Hamiltonian (i.e.
    a Hamiltonian depending only on one set of coordinates), as it is already in a first-order
    NF (there is only one (1, 1)-term).
    '''
    heff = poly(values={(1, 2): (0.030281601477850428+0.022880847131515537j),
                (2, 1): (0.030281601477850428-0.022880847131515537j),
                (1, 1): (-1.2943361732789946+0j),
                (0, 3): (-0.0067437692097512825+0.022880847131512467j),
                (3, 0): (-0.0067437692097512825-0.022880847131512467j),
                (2, 2): (-0.0020696721592390653+0j),
                (0, 4): (0.0002573905636475989-0.0009855310036223186j),
                (3, 1): (-0.0007774455159719339+0.001600740049462997j),
                (1, 3): (-0.0007774455159719339-0.001600740049462997j),
                (4, 0): (0.0002573905636475989+0.0009855310036223186j),
                (0, 5): (9.871836039193e-06+2.1828185573764483e-05j),
                (3, 2): (9.871836039193003e-05-4.365637114752898e-05j),
                (1, 4): (4.935918019596501e-05+6.548455672129347e-05j),
                (4, 1): (4.935918019596501e-05-6.548455672129347e-05j),
                (2, 3): (9.871836039193003e-05+4.365637114752898e-05j),
                (5, 0): (9.871836039193e-06-2.1828185573764483e-05j)}, max_power=10)
    bnfdict = heff.bnf(order=3, check=True, **kwargs)
    assert all([abs(v) for v in (heff - bnfdict['H']).values()])
    
    
def test_bnf_performance(order=8, threshold=1.1, tol=1e-15):
    # Test if any modification of the bnf main routine will be slower than the reference bnf routine (defined in this script).
    
    H = lambda x, y, z, px, py, pz: 0.24*(x**2 + px**2) + 0.86*(y**2 + py**2) + x**3 + y**3 + z**2 + pz**2 + 0.4*pz*y
    # z0 = 6*[0]
    z1 = [np.pi/2, 0.55, 0.32, 7.13, -0.6311, 2.525]

    time1 = time.time()
    tcref = referencebnf(H, z=z1, order=order, check=True)
    time_ref = time.time() - time1
    
    time2 = time.time()
    tc = bnf(H, z=z1, order=order, check=True)
    time_bnf = time.time() - time2
    
    # performance check
    assert time_bnf <= time_ref*threshold, 'Error: new time = {} > {} threshold*(reference time)'.format(time_bnf, threshold*time_ref)

    # check on equality
    chifinal_ref = tcref['chi'][-1]
    chifinal = tc['chi'][-1]
    assert chifinal_ref.keys() == chifinal.keys()
    for key in chifinal_ref.keys():
        v1, v2 = chifinal_ref[key], chifinal[key]
        assert abs(v1 - v2)/(min([abs(v1), abs(v2)])) < tol
        
    