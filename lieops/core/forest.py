import warnings
import numpy as np
from tqdm import tqdm

from lieops.core import lexp, create_coords, poly
from lieops.core.tools import tpsa, taylor_map
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
        
        cond = np.logical_or(z.real != 0, np.logical_and(z.real == 0, (z.imag/twopi)%1 != 0)) # I.e. exp(z) != 1
        a[powers] = np.divide(value, 1 - np.exp(-z), out=np.zeros_like(value), where=cond)
        # Remarks:
        # 1) for the -1 sign in front of z see my notes regarding Normal form (Lie_techniques.pdf); this -1 is due to the fact that R^(-1) must be considered.
        # 2) The poly class will check the values "a" above by zero. Only non-zero terms will be stored.
    return poly(values=a, dim=dim, max_power=fk.max_power)

    
def fnf(*p, order: int, mode='conj', **kwargs):
    '''
    Obtain maps to Normal form for a given chain of Lie-operators. The concept
    is outlined in Sec. 4.4 in Ref. [1].
    
    Attention:
    I)
    First-order Lie-polynomials in the resulting Dragt/Finn expansion will be dropped.
    This means that if the given Taylor map represents an expansion of the symplectic map
    
    M = exp(:f1:) o exp(:f2:) o ... o exp(:fM:) o exp(:h1:) ,
    
    then the routine will normalize the chain of Lie operators
    
    exp(:f2:) o ... o exp(:fM:)
    
    The chain f1, ... fM, h1 will be returned as well, so the user may handle the 
    first-order polynomials externally. The interior represent a normalized version
    with respect to the initial and final position.
    
    II)
    Cases in which two 2nd-order terms are present have not been implemented yet.
    
    Parameters
    ----------
    p: list
        A list of poly objects, representing the Taylor expansion of a
        symplectic map (for example, the result of a TPSA calculation through a 
        chain of Lie-operators).
        
    order: int
        The order of the normalization proceduere.
        
    tol_drop: float, optional
        A small number to identify polynomials which should be zero. Required in dragtfinn to
        identify if the two 2nd order polynomials can be combined.
        
    mode: str, optional
        Control the way of how the successive Taylor maps are computed:
        'conj': The Taylor map of the next step (for the map M' = exp(:ak:) M exp(-:ak:)) is computed
                from the previous Taylor map by conjugation with the map exp(:ak:).
                Note that if first-order elements in the Dragt/Finn factorization are detected,
                one TPSA calculation will be done for this interior part.
        'tpsa': The Taylor map of the next step is computed by using TPSA on M'.
        
    Returns
    -------
    dict
        A dictionary containing the following keys:
         'dragtfinn': The output of the Dragt/Finn factorization of the given Taylor map
            'bnfout': The output of the Birkhoff normal form routine for order 1, used in the first step
                      in the normalization.
            'nf_all': A list X of lists, where each element E of X contains the Lie-polynomials n_j
                      which represent one step in the normalization procedure according to Ref. [1], 
                      Eq. (4.31). The last elements of X therefore yield the entire normalization.
        'normalform': The last element of 'nf_all'.
               'chi': The normalizing Lie-polynomials in ascending order. This means that
                      "N = chi[::-1] o M o chi[::-1]**(-1)"
                      is the normalized form of M (the lowest orders needs to be applied on M first).
            'tm_all': A list of Taylor-maps which were used in each step in the normalization procedure,
                      see the discussion in Ref. [1] around Eq. (4.31).
        'taylor_map': The last element in 'tm_all'.               
        
    Reference(s)
    ------------
    [1] E. Forest: "Beam Dynamics - A New Attitude and Framework", harwood academic publishers (1998).
    '''
    #####################
    # Progress user input
    #####################
    tol_drop = kwargs.pop('tol_drop', 0)
    disable_tqdm = kwargs.get('disable_tqdm', False) # show progress bar (if requested) in the loop below
    kwargs['disable_tqdm'] = True # never show progress bars within the loop(s) itself
    
    # Compute the Dragt/Finn factorization up to a specific order
    kwargs['pos2'] = 'left' # ensure that the 2nd order term is neighbour to the third-order term
    kwargs['comb2'] = True
    df = dragtfinn(*p, order=order, tol_drop=tol_drop, **kwargs)

    nterms_1 = [f for f in df if f.maxdeg() > 1]
    df_orders = [f.maxdeg() for f in nterms_1]
    
    # R^(-1) != 1, otherwise the normal form will be identical to the input
    if 2 not in df_orders:
        warnings.warn('No 2nd-order terms found => Normal form identical to input.')
        return {'dragtfinn': df, 'nf_all': [nterms_1], 'chi': [], 'tm_all': [p], 'normalform': nterms_1}
    
    # If first-order elements in the Dragt/Finn factorization have been found, and the mode is 'conj',
    # then we will re-calculate the tpsa map p for this inner part (& raise a warning):
    default_max_power = kwargs.setdefault('max_power', min([f.max_power for f in df]))
    if len(nterms_1) < len(df):
        warnings.warn('Non-zero kicks detected. Will normalize only the interior.')
        if mode == 'conj':
            warnings.warn(f"mode == '{mode}' with non-zero kicks. Performing TPSA for the interior ...")
            position = [0]*df[0].dim*2
            tpsa_out = tpsa(*[lexp(a) for a in nterms_1], order=order, position=position, **kwargs)
            p = taylor_map(*tpsa_out._evaluation, max_power=default_max_power)

    ###############################
    # II) First-order normalization
    ###############################
    i1 = df_orders.index(2)
    i2 = len(df_orders) - 1 - df_orders[::-1].index(2)
    if i1 < i2:
        raise NotImplementedError(f"Two 2nd-order polynomials have been found.")
    assert i1 == i2
    nterms_1 = nterms_1[i1:]
    
    C = nterms_1[0]
    # Compute the first-order normalization of the 2nd-order polynomials
    nfdict = C.bnf(order=1, **kwargs)
    tunes = nfdict['mu']
    chi0s = nfdict['chi0'] # U@Sinv@U^* = exp(C1)@exp(C2) for nfdict['chi0'] = [C1, C2], see lieops.linalg.nf.normal_form; Sinv contains the linear optics functions.
    for chi0 in chi0s: # (chi0s may be of length 0, 1 or 2)
        nterms_1 = [lexp(chi0)(h, method='2flow') for h in nterms_1]

    tol_checks = kwargs.get('tol_checks', 0)
    if tol_checks > 0:
        # check if the first-order part has been properly normalized
        dim = len(tunes)
        dim2 = dim*2
        for k in range(dim):
            ek = [0]*dim
            ek[k] = 1
            zero_k = abs(nterms_1[0][tuple(ek*2)] - tunes[k])
            if hasattr(zero_k, '__iter__'):
                zero_k = max(zero_k)
            assert zero_k < tol_checks, f'First-order terms not normalized at {k}: {zero_k} >= {tol_checks} (tol_checks)'
    # nterms_1 has been determined. It is a list consisting of a normalized
    # first-order element + some higher-order non-normalized elements
    
    #################################
    # III) Higher-order normalization
    #################################
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
    final_coords = [coord(*xietaf) for coord in p]
    nmap = [ww(*final_coords, max_order=order + 1) for ww in xietaf2]
    
    # Loop over the requested order
    all_nmaps = [nmap]
    all_nterms = [nterms_1] # to collect the successive Dragt/Finn polynomials at each iteration step
    chi = [w.copy() for w in chi0s] # to collect the maps to normal form
    nterms_k = nterms_1 # 'Running' D/F-factorization
    for k in tqdm(range(3, order + 2), disable=disable_tqdm): # k will run up and including order + 1, because 'dragtfinn' for a specific order refers to the order of the Taylor map, and therefore produces a chain of Hamiltonians up and including order + 1.
        
        # find the term of order k in the current Dragt/Finn factorization
        fk_all = [f for f in nterms_k if f.maxdeg() == k]
        if len(fk_all) == 0:
            # Then fk = 0, so ak = 0 as well, and we just continue (lexp(ak) = 1).
            continue
        elif len(fk_all) == 1:
            fk = fk_all[0]
        else:
            raise RuntimeError(f'More than one term in the Dragt/Finn factorization at order {k} found.')
        
        ak = _rot_kernel(fk, tunes)
        chi.append(ak)

        if mode == 'conj':
            xietaf = lexp(ak)(*xieta, **kwargs)
            xietaf = [xe.extract(key_cond=lambda k: sum(k) <= order + 1) for xe in xietaf]
            xietaf2 = lexp(-ak)(*xieta, **kwargs)
            xietaf2 = [xe2.extract(key_cond=lambda k: sum(k) <= order + 1) for xe2 in xietaf2]
            final_coords = [coord(*xietaf, max_order=order + 1) for coord in nmap]
            nmap = [ww(*final_coords, max_order=order + 1) for ww in xietaf2]
        elif mode == 'tpsa':
            # This mode is experimental and may require the removal of small non-zero operators at each step to reduce errors.
            operators = [lexp(ak)] + [lexp(f) for f in nterms_k] + [lexp(-ak)] # or [lexp(lexp(ak)(f, **kwargs)) for f in nterms_k]
            position = [0]*df[0].dim*2
            tpsa_out = tpsa(*operators, order=order, position=position, **kwargs)
            nmap = taylor_map(*tpsa_out._evaluation, max_power=default_max_power)
            
        all_nmaps.append(nmap)
        nterms_k = dragtfinn(*nmap, order=order, tol_drop=tol_drop, **kwargs)
        all_nterms.append(nterms_k)
        
    out = {}
    out['dragtfinn'] = df
    out['bnfout'] = nfdict
    out['nf_all'] = all_nterms
    out['normalform'] = all_nterms[-1]
    out['chi'] = chi
    out['tm_all'] = all_nmaps
    out['taylor_map'] = all_nmaps[-1]
    return out
