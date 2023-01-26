import warnings
import numpy as np
from tqdm import tqdm

from lieops.core import lexp, create_coords, poly
from lieops.core.tools import ad2poly, poly2ad, tpsa
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
            a[powers] = value/(1 - np.exp(-z)) # for the -1 sign in front of z see my notes regarding Normal form (Lie_techniques.pdf)
    return poly(values=a, dim=dim, max_power=fk.max_power)

def fnf(*p, order: int=1, bch_order=6, mode='quick', **kwargs):
    '''
    Obtain maps to Normal form for a given chain of Lie-operators. The concept
    is outlined in Sec. 4.4 in Ref. [1].
    
    Attention:
    First-order Lie-polynomials in the resulting Dragt/Finn expansion will be dropped.
    This means that if the given Taylor map represents an expansion of the symplectic map
    
    M = exp(:f1:) o exp(:f2:) o ... o exp(:fM:) o exp(:h1:) ,
    
    then the routine will normalize the chain of Lie operators
    
    exp(:f2:) o ... o exp(:fM:)
    
    The chain f1, ... fM, h1 will be returned as well, so the user may handle the 
    first-order polynomials externally. The interior represent a normalized version
    with respect to the initial and final position.
    
    Parameters
    ----------
    p: list
        A list of poly objects, representing the Taylor expansion of a
        symplectic map (for example, the result of a TPSA calculation through a 
        chain of Lie-operators).
        
    order: int, optional
        The order of the normalization proceduere.
        
    tol: float, optional
        A small number to identify polynomials which should be zero. Required in dragtfinn to
        identify if the two 2nd order polynomials can be combined.
        
    mode: str, optional
        Control the way of how the successive Taylor maps are computed:
        'quick': The Taylor map of the next step (for the map M' = exp(:ak:) M exp(-:ak:)) is computed
                 from the previous Taylor map by pull-back using the maps exp(:ak:) and exp(-:ak:).
                 Note that if first-order elements in the Dragt/Finn factorization are detected,
                 one TPSA calculation will be done for this interior part.
        'tpsa': The Taylor map of the next step is computed by using TPSA on M'.
        
    bch_order: int, optional
        If the given Taylor map requires two non-trivial Lie-polynomials, then attempt
        to combine them together using the Baker-Campbell-Hausdorff equation up
        to 'bch_order'. In this case a warning will be issued.

    Reference(s)
    ------------
    [1] E. Forest: "Beam Dynamics - A New Attitude and Framework", harwood academic publishers (1998).
    '''
    # 0) Progress user input
    tol = kwargs.get('tol', 0)
    disable_tqdm = kwargs.get('disable_tqdm', False) # show progress bar (if requested) in the loop below
    kwargs['disable_tqdm'] = True # never show progress bars within the loop(s) itself
    
    # I) Compute the Dragt/Finn factorization up to a specific order
    kwargs['pos2'] = 'left'
    kwargs['comb2'] = True
    df = dragtfinn(*p, order=order, **kwargs)

    nterms_1 = [f for f in df if f.maxdeg() > 1]
    df_orders = [f.maxdeg() for f in nterms_1]
    
    # R^(-1) != 1, otherwise the normal form will be identical to the input
    if 2 not in df_orders:
        warnings.warn('No 2nd-order terms found => Normal form identical to input.')
        return {'dragtfinn': df, 'nterms': [nterms_1], 'chi': [], 'nmaps': [p]}
    
    # If first-order elements in the Dragt/Finn factorization have been found, and the mode was 'quick',
    # then we need to re-calculate the tpsa map p for this inner part (& raise a warning):
    if len(nterms_1) < len(df):
        warnings.warn('Non-zero kicks detected. Will normalize only the interior.')
        if mode == 'quick':
            warnings.warn("mode == 'quick' with non-zero kicks. Performing TPSA for the interior ...")
            tpsa_out = tpsa(*[lexp(a) for a in nterms_1], order=order, **kwargs)
            p = tpsa_out['taylor_map']

    # II) First-order normalization (if applicable)
    # Find the parts in the factorization which belong to 2nd and higher-order; try to combine
    # the two 2nd order polynomials using the BCH equation.
    # TODO (later): Do not rely on the BCH Theorem, but instead determine the kernel of R - 1 (Eq. (4.34) in Ref. [1]).
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
    nfdict = C.bnf(order=1, **kwargs)
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
                
    # nterms_1 has been determined. It is a list consisting of a normalized
    # first-order element + some higher-order non-normalized elements
    
    # III) Perform higher-order normalization
    #
    # Compute the Taylor map 'nmap' of the first-order normalized map. 
    # Instead of calling TPSA again for the new map M' = lexp(chi0) o M o lexp(-chi0), 
    # we modify the original Taylor map,
    # using the fact that the lexp-operators act on the coordinates by pullback. This will save calculation
    # time. Note that we might have two 2nd-order polynomials chi0s[0] and chi0s[1] in this first step. Thus:
    xieta = create_coords(dim=len(tunes), **kwargs)
    xietaf, xietaf2 = xieta, xieta
    for chi0 in chi0s:
        xietaf = lexp(chi0)(*xietaf, method='2flow')
    for chi0 in chi0s[::-1]:
        xietaf2 = lexp(-chi0)(*xietaf2, method='2flow')
    nmap = [ww(*[coord(*xietaf) for coord in p]) for ww in xietaf2]
    
    # Loop over the requested order
    all_nmaps = [nmap]
    all_nterms = [nterms_1] # to collect the successive Dragt/Finn polynomials at each iteration step
    chi = [w.copy() for w in chi0s] # to collect the maps to normal form
    nterms_k = nterms_1 # 'Running' D/F-factorization
    for k in tqdm(range(3, order + 2), disable=disable_tqdm): # k will run up and including order + 1, because 'dragtfinn' for a specific order refers to the order of the Taylor map, and therefore produces a chain of Hamiltonians up and including order + 1.
        
        # find the term of order k in the current Dragt/Finn factorization
        orders_k = [f.maxdeg() for f in nterms_k]
        if k in orders_k:
            fk = nterms_k[orders_k.index(k)]
        else:
            # Then fk = 0, so ak = 0 as well and we just continue (lexp(ak) = 1).
            continue
        
        ak = _rot_kernel(fk, tunes)
        chi.append(ak)

        if mode == 'quick':
            xietaf = lexp(ak)(*xieta, **kwargs)
            xietaf2 = lexp(-ak)(*xieta, **kwargs)
            nmap = [ww(*[coord(*xietaf) for coord in nmap]) for ww in xietaf2]
        elif mode == 'tpsa':
            # This mode is experimental and may require the removal of small non-zero operators at each step to reduce errors.
            operators = [lexp(ak)] + [lexp(f) for f in nterms_k] + [lexp(-ak)] # or [lexp(lexp(ak)(f, **kwargs)) for f in nterms_k], but first checks indicated that this may increase numerical errors           
            tpsa_out = tpsa(*operators, order=order, **kwargs)
            nmap = tpsa_out['taylor_map']
            
        all_nmaps.append(nmap)

        nterms_k = dragtfinn(*nmap, order=order, **kwargs)
        all_nterms.append(nterms_k)
        
    out = {}
    out['dragtfinn'] = df
    out['bnfdict'] = nfdict
    out['nterms'] = all_nterms
    out['chi'] = chi
    out['nmaps'] = all_nmaps
    return out
        
    
