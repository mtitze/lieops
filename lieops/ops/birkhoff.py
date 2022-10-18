import lieops.ops.lie
from lieops.solver.common import getRealHamiltonFunction
from lieops.linalg.nf import first_order_nf_expansion
from lieops.ops.tools import ad2poly

def homological_eq(mu, Z, **kwargs):
    '''
    Let e[k], k = 1, ..., len(mu) be actions, H0 := sum_k mu[k]*e[k] and Z a
    polynomial of degree n. Then this routine will solve 
    the homological equation 
    {H0, chi} + Z = Q with
    {H0, Q} = 0.

    Attention: No check whether Z is actually homogeneous or real, but if one of
    these properties hold, then also chi and Q will admit such properties.
    
    Parameters
    ----------
    mu: list
        list of floats (tunes).
        
    Z: poly
        Polynomial of degree n.
        
    **kwargs
        Arguments passed to poly initialization.
        
    Returns
    -------
    chi: poly
        Polynomial of degree n with the above property.
        
    Q: poly
        Polynomial of degree n with the above property.
    '''
    chi, Q = lieops.ops.lie.poly(values={}, dim=Z.dim, **kwargs), lieops.ops.lie.poly(values={}, dim=Z.dim, **kwargs)
    for powers, value in Z.items():
        om = sum([(powers[k] - powers[Z.dim + k])*mu[k] for k in range(len(mu))])
        if om != 0:
            chi[powers] = 1j/om*value
        else:
            Q[powers] = value
    return chi, Q

def bnf(H, order: int=1, tol=1e-12, cmplx=True, **kwargs):
    '''
    Compute the Birkhoff normal form of a given Hamiltonian up to a specific order.
    
    Attention: Constants and any gradients of H at z will be ignored. If there is 
    a non-zero gradient, a warning is issued by default.
    
    Parameters
    ----------
    H: callable or dict
        Defines the Hamiltonian to be normalized. 
        I) If H is callable, then a transformation to complex normal form is performed prior to
           applying the general algorithm.
        II) If H is a dictionary (e.g. from poly.items()), then its off-diagonal entries
            are ignored (so it is assumed that H has already been prepared to be in first-order
            normal form).
                
    order: int
        The order up to which we build the normal form. Here order = k means that we compute
        k homogeneous Lie-polynomials, where the smallest one will have power k + 2 and the 
        succeeding ones will have increased powers by 1.
    
    cmplx: boolean, optional
        If false, assume that the coefficients of the second-order terms of the Hamiltonian are real.
        In this case a check will be done and an error will be raised if that is not the case.
        
    tol: float, optional
        Tolerance below which values are assumed to be zero.
        
    **kwargs
        Keyword arguments are passed to .first_order_nf_expansion routine.
        
    Returns
    -------
    dict
        A dictionary with the following keys:
        nfdict : The output of hte first_order_nf_expansion routine.
        H      : Dictionary denoting the used Hamiltonian.
        H0     : Dictionary denoting the second-order coefficients of H.
        mu     : List of the tunes used (coefficients of H0).
        chi    : List of poly objects, denoting the Lie-polynomials of degree >= 3 which map to normal form.
        chi0   : List of the two Lie-polynomials of degree 2 which map the given Hamiltonian 
                 to its first-order normal form equivalent. This means that the list "chi0 + chi" performs the 
                 entire map up to the desired order (see also the example below). Not supported if mpmath has been used.                 
        Hk     : List of poly objects, corresponding to the transformed Hamiltonians.
        Zk     : List of poly objects, notation see Lem. 1.4.5. in Ref. [1]. 
        Qk     : List of poly objects, notation see Lem. 1.4.5. in Ref. [1].
        
    Example
    -------      
    nfdict = H1.bnf(order=4)
    w = H1.copy()
    for c in nfdict['chi0'] + nfdict['chi']:
        w = c.lexp(power=10)(w)
        print (w.above(1e-12)) # example

    References
    ----------
    [1]: M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019).
    [2]: B. Grebert. "Birkhoff normal form and Hamiltonian PDEs". (2006)
    '''
    power = order + 2 # the maximal power of the homogeneous polynomials chi mapping to normal form.
    max_power = kwargs.get('max_power', order + 2) # the maximal power to be taken into consideration when applying ad-operations between Lie-polynomials. TODO: check default & relation to 'power'
    lo_power = kwargs.get('power', order + 2) # The maximal power by which we want to expand exponential series when evaluating Lie operators. TODO: check default.
    
    #######################
    # STEP 1: Preparation
    #######################
    
    if hasattr(H, '__call__'):
        # assume that H is given as a function, depending on phase space coordinates.
        kwargs['power'] = power
        Hinp = H
        if isinstance(H, lieops.ops.lie.poly):
            # we have to transfom the call-routine of H: H depend on (xi, eta)-coordinates, but the nf routines later on assume (q, p)-coordinates.
            # In principle, one can change this transformation somewhere else, but this may cause the normal form routines
            # to either yield inconsistent output at a general point z -- or it may introduce additional complexity.
            Hinp = getRealHamiltonFunction(H, tol=tol, **kwargs)
        taylor_coeffs, nfdict = first_order_nf_expansion(Hinp, tol=tol, **kwargs)
        # N.B. while Hinp is given in terms of (q, p) variables, taylor_coeffs correspond to the Taylor-expansion
        # of the Hamiltonian around z=(q0, p0) with respect to the normal form (xi, eta) coordinates (see first_order_nf_expansion routine).
    else:
        # Attention: In this case we assume that H is already in complex normal form (CNF): Off-diagonal entries will be ignored (see code below).
        taylor_coeffs = H
        nfdict = {}
        
    # Note that if a vector z is given, the origin of the results correspond to z (since the normal form above
    # is constructed with respect to a Hamiltonian shifted by z.
        
    # get the dimension (by looking at one key in the dict)
    dim2 = len(next(iter(taylor_coeffs)))
    dim = dim2//2
            
    # define mu and H0. For H0 we skip any (small) off-diagonal elements as they must be zero by construction.
    H0_values = {}
    mu = []
    for j in range(dim): # add the second-order coefficients (tunes)
        tpl = tuple([0 if k != j and k != j + dim else 1 for k in range(dim2)])
        muj = taylor_coeffs[tpl]
        # remove tpl from taylor_coeffs, to verify that later on, all Taylor coefficients have no remaining 2nd-order coeff (see below).
        taylor_coeffs.pop(tpl)
        if not cmplx:
            # by default we assume that the coefficients in front of the 2nd order terms are real.
            assert muj.imag < tol, f'Imaginary part of entry {j} above tolerance: {muj.imag} >= {tol}. Check input or try cmplx=True option.'
            muj = muj.real
        H0_values[tpl] = muj
        mu.append(muj)
    H0 = lieops.ops.lie.poly(values=H0_values, dim=dim, max_power=max_power)
    assert len({k: v for k, v in taylor_coeffs.items() if sum(k) == 2 and abs(v) >= tol}) == 0 # All other 2nd order Taylor coefficients must be zero.

    # For H, we take the values of H0 and add only higher-order terms (so we skip any gradients (and constants). 
    # Note that the skipping of gradients leads to an artificial normal form which may not have any relation
    # to the original problem. By default, the user will be informed if there is a non-zero gradient 
    # in 'first_order_nf_expansion' routine.
    H = H0.update({k: v for k, v in taylor_coeffs.items() if sum(k) > 2})
        
    ########################
    # STEP 2: NF-Algorithm
    ########################
               
    # Induction start (k = 2); get P_3 and R_4. Z_2 is set to zero.
    Zk = lieops.ops.lie.poly(dim=dim, max_power=max_power) # Z_2
    Pk = H.homogeneous_part(3) # P_3
    Hk = H.copy() # H_2 = H
            
    chi_all, Hk_all = [], [H]
    Zk_all, Qk_all = [], []
    lchi_all = []
    for k in range(3, power + 1):
        chi, Q = homological_eq(mu=mu, Z=Pk, max_power=max_power)
        if len(chi) == 0:
            # in this case the canonical transformation will be the identity and so the algorithm stops.
            break
        lchi = lieops.ops.lie.lexp(-chi, power=lo_power)
        Hk = lchi(Hk)
        # Hk = lexp(-chi, power=k + 1)(Hk) # faster but likely inaccurate; need tests
        Pk = Hk.homogeneous_part(k + 1)
        Zk += Q
        
        lchi_all.append(lchi)
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
    out['lchi_inv'] = lchi_all
    out['chi'] = chi_all
    out['Hk'] = Hk_all
    out['Zk'] = Zk_all
    out['Qk'] = Qk_all
    out['order'] = order
    out['lo_power'] = lo_power
    out['max_power'] = max_power
    
    if 'C1' in nfdict.keys() and 'C2' in nfdict.keys():
        # In this case we can also compute the polynomials which provide the transformation to first-order normal form:
        C1, C2 = nfdict['C1'], nfdict['C2']
        Cp1, Cp2 = ad2poly(C1.transpose()), ad2poly(C2.transpose())
        out['chi0'] = [Cp1, Cp2]
        
    return out
