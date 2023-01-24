import warnings
import numpy as np

from lieops.core import lexp, create_coords, poly
from lieops.core.tools import ad2poly, poly2ad
from lieops.core.combine import bch
from lieops.linalg.bch import bch_2x2
from lieops.core.dragt import dragtfinn


twopi = float(np.pi*2)

def _rot_kernel(fk, mu):
    '''
    In case that R in Eq. (4.34), Ref. [1], consist of a rotation, we can
    apply Eq. (4.37) to obtain the normalizing polynomial.
    '''
    dim = len(mu)
    a = {}
    for powers, value in fk.items():
        m1 = powers[:dim]
        m2 = powers[dim:]
        z = sum([(m1[k] - m2[k])*mu[k] for k in range(dim)])*1j
        if z.real != 0 or (z.real == 0 and (z.imag/twopi)%1 != 0): # I.e. exp(z) != 1
            a[powers] = value/(1 - np.exp(z))
    return poly(values=a, dim=dim, max_power=fk.max_power)


def fnf(*p, bch_order=6, **kwargs):
    '''
    Obtain maps to Normal form for a given chain of Lie-operators. The concept
    is outlined in Sec. 4.4 in Ref. [1].
    
    Attention:
    Currently any first-order Lie-polynomials in the resulting Dragt/Finn expansion will be ignored.
    This means that if the given Taylor map represents an expansion of the symplectic map
    
    M = exp(:f1:) o exp(:f2:) o ... o exp(:fM:) o exp(:h1:) ,
    
    then we will return a chain of polynomials gk so that the
    part
    
    exp(:f2:) o ... o exp(:fM:)
    
    will be normalized. The chain f1, ... fM, h1 will be returned as well, so the user
    may handle the first-order polynomials, which represent the initial offset and final position
    of the point around which the normal form is considered.

    Parameters
    ----------
    p: list
        A list of poly objects, representing the Taylor expansion of a
        symplectic map (for example, the result of a TPSA calculation through a 
        chain of Lie-operators).
        
    nf_order: int, optional
        The order of the normalization proceduere.
        
    tol: float, optional
        A small number to identify polynomials which should be zero. Required in dragtfinn to
        identify if the two 2nd order polynomials can be combined.
        
    bch_order: int, optional
        If the given Taylor map requires two non-trivial Lie-polynomials, then attempt
        to combine them together using the Baker-Campbell-Hausdorff equation up
        to 'bch_order'. In this case a warning will be issued.

    Reference(s)
    ------------
    [1] E. Forest: "Beam Dynamics - A New Attitude and Framework" (1998).
    '''
    tol = kwargs.get('tol', 0)
    order = kwargs.setdefault('order', 1)
    
    # 1) Compute the Dragt/Finn factorization up to a specific order
    kwargs['pos2'] = 'left'
    kwargs['comb2'] = True
    df = dragtfinn(*p, **kwargs)

    # 2) First-order normalization (if applicable)
    nterms_1 = [f for f in df if f.maxdeg() > 1]
    df_orders = [f.maxdeg() for f in nterms_1]
    if 2 in df_orders:
        # 2) Find the parts in the factorization which belong to 2nd and higher-order; try to combine
        #    the two 2nd order polynomials using the BCH equation.
        #    TODO (later): Do not rely on the BCH Theorem, but instead determine the kernel of R - 1 (Eq. (4.34) in Ref. [1]).
        i1 = df_orders.index(2)
        i2 = len(df_orders) - 1 - df_orders[::-1].index(2)
        if i1 < i2:
            # then we shall apply the BCH theorem
            A1 = nterms_1[i1]
            A2 = nterms_1[i2]
            assert A1.dim == A2.dim
            dim = A1.dim
            if dim == 1: # Here we can use an exact equation
                C = ad2poly(bch_2x2(poly2ad(A1), poly2ad(A2), tol=tol), max_power=max([A1.max_power, A2.max_power]))
            else: # No exact equation for dim > 1 known to my knowledge
                warnings.warn(f"Two 2nd-order polynomials have been found and dim = {dim} (tol: {tol}). Attempting BCH with order {bch_order}.")            
                bchd = bch(A1, A2, order=bch_order)
                C = sum(bchd.values())
        else:
            C = nterms_1[i1]
            
        nl_part = nterms_1[i2 + 1:]

        # First-order step:
        nfdict = C.bnf(order=1)
        tunes = nfdict['mu']
        chi0s = nfdict['chi0']

        nterms_1 = [C.copy()] + [f.copy() for f in nl_part]    
        for chi0 in chi0s: # (chi0s may be of length 0, 1 or 2)
            nterms_1 = [lexp(chi0)(h, method='2flow') for h in nterms_1]

        if kwargs.get('tol_checks', 0) > 0:
            dim = len(tunes)
            dim2 = dim*2
            for k in range(dim):
                ek = [0]*dim
                ek[k] = 1
                assert abs(nterms_1[0][tuple(ek*2)] - tunes[k]) < tol
                
    # nterms_1 has been determined.
                
    
    if 2 not in df_orders: # tunes required below
        raise NotImplementedError('No 2nd order terms found. Case currently not implemented.')
    
    # 3) Perform higher-order normalization
    xieta = create_coords(dim=len(tunes), **kwargs)
    
    xietaf, xietaf2 = xieta, xieta
    for chi0 in chi0s:
        xietaf = lexp(chi0)(*xietaf, method='2flow')
    for chi0 in chi0s[::-1]:
        xietaf2 = lexp(-chi0)(*xietaf2, method='2flow')
        
    nmap = [ww(*[coord(*xietaf) for coord in p]) for ww in xietaf2] # the Taylor-map of the first-order normalized map 
    all_nterms = [nterms_1] # to collect the successive Dragt/Finn polynomials at each iteration step
    chi = chi0s # to collect the maps to normal form
    nterms_k = nterms_1 # Running D/F-factorization
    assert order == len(nterms_1)
    for k in range(1, order):
        fk = nterms_k[k]
        ak = _rot_kernel(fk, tunes)
        nterms_k = [lexp(-ak)(h.copy(), **kwargs) for h in nterms_k]
        chi.append(-ak)

        xietaf = lexp(-ak)(*xieta, **kwargs)
        xietaf2 = lexp(ak)(*xieta, **kwargs)
        nmap = [ww(*[coord(*xietaf) for coord in nmap]) for ww in xietaf2]
        
        nterms_k = dragtfinn(*[-h for h in nmap], **kwargs)
        all_nterms.append(nterms_k)
            
    out = {}
    out['dragtfinn'] = df
    out['bnfdict'] = nfdict
    out['nterms'] = all_nterms
    out['chi'] = chi
    return out
        
    
