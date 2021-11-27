from njet.ad import standardize_function
from njet import derive
from .lie import liepoly, lieoperator, exp_ad
from .linalg import first_order_normal_form, matrix_from_dict

def Omega(mu, a, b):
    '''
    Compute the scalar product of mu and a - b.
    
    Parameters
    ----------
    mu: subscriptable
    a: subscriptable
    b: subscriptable
    
    Returns
    -------
    float
        The scalar product (mu, a - b).
    '''
    return sum([mu[k]*(a[k] - b[k]) for k in range(len(mu))])


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
        
    Z: liepoly
        Polynomial of degree n.
        
    **kwargs
        Arguments passed to liepoly initialization.
        
    Returns
    -------
    chi: liepoly
        Polynomial of degree n with the above property.
        
    Q: liepoly
        Polynomial of degree n with the above property.
    '''
    chi, Q = liepoly(values={}, dim=Z.dim, **kwargs), liepoly(values={}, dim=Z.dim, **kwargs)
    for powers, value in Z.values.items():
        om = Omega(mu, powers[:Z.dim], powers[Z.dim:])
        if om != 0:
            chi.values[powers] = 1j/om*value
        else:
            Q.values[powers] = value
    return chi, Q


def first_order_nf_expansion(H, order: int=2, z=[], warn: bool=True, n_args: int=0, tol: float=1e-14, **kwargs):
    '''
    Return the Taylor-expansion of a Hamiltonian H in terms of first-order complex normal form coordinates
    around an optional point of interest. For the notation see my thesis.
    
    Parameters
    ----------
    H: callable
        A real-valued function of 2*n parameters (Hamiltonian).
        
    order: int, optional
        The maximal order of expansion. Must be >= 2 (default: 2).
    
    z: subscriptable, optional
        A point of interest around which we want to expand.
        
    n_args: int, optional
        If H takes a single subscriptable as argument, define the number of arguments with this parameter.
        
    warn: boolean, optional
        Turn on some basic checks:
        a) Warn if the expansion of the Hamiltonian around z contains first-order terms larger than a specific value. 
        b) Verify that the 2nd order terms in the expansion of the Hamiltonian agree with those from the linear theory.
        Default: True.
        
    tol: float, optional
        An optional tolerance for checks. Default: 1e-14.
        
    **kwargs
        Arguments passed to linalg.first_order_normal_form
        
    Returns
    -------
    dict
        A dictionary of the Taylor coefficients of the Hamiltonian around z, where the first n
        entries denote powers of xi, while the last n entries denote powers of eta.
        
    dict
        The output of 'first_order_normal_form' routine, providing the linear map information at the requested point.
    '''
    assert order >= 2
    
    Hst, dim = standardize_function(H, n_args=n_args)
    
    # Step 1 (optional): Construct H locally around z (N.B. shifts are symplectic, as they drop out from derivatives.)
    # This step is required, because later on (at point (+)) we want to extract the Taylor coefficients, and
    # this works numerically only if we consider a function around zero.
    if len(z) > 0:
        H = lambda x: Hst([x[k] + z[k] for k in range(len(z))])
    else:
        H = Hst
    
    # Step 2: Obtain the Hesse-matrix of H.
    # N.B. we need to work with the Hesse-matrix here (and *not* with the Taylor-coefficients), because we want to get
    # a (linear) map K so that the Hesse-matrix of H o K is in CNF (complex normal form). This is guaranteed
    # if the Hesse-matrix of H is transformed to CNF.
    # Note that the Taylor-coefficients of H in 2nd-order are 1/2*Hesse_matrix. This means that at (++) (see below),
    # no factor of two is required.
    dH = derive(H, order=2, n_args=dim)
    z0 = dim*[0]
    Hesse_dict = dH.hess(z0)
    Hesse_matrix = matrix_from_dict(Hesse_dict, symmetry=1, **kwargs)
    
    # Optional: Raise a warning in case the shifted Hamiltonian still has first-order terms.
    if warn:
        gradient = dH.grad()
        if any([abs(gradient[k]) > tol for k in gradient.keys()]) > 0:
            print (f'Warning: H has a non-zero gradient around the requested point\n{z}\nfor given tolerance {tol}:')
            print ([gradient[k] for k in sorted(gradient.keys())])

    # Step 3: Compute the linear map to first-order complex normal form near z.
    nfdict = first_order_normal_form(Hesse_matrix, **kwargs)
    K = nfdict['K'] # K.transpose()*Hesse_matrix*K is in cnf
    
    # Step 4: Obtain the expansion of the Hamiltonian up to the requested order.
    Kmap = lambda zz: [sum([K[j, k]*zz[k] for k in range(len(zz))]) for j in range(len(zz))] # TODO: implement column matrix class 
    HK = lambda zz: H(Kmap(zz))
    dHK = derive(HK, order=order, n_args=dim)
    results = dHK(z0, mult=False) # mult=False ensures that we obtain the Taylor-coefficients of the new Hamiltonian. (+)
    
    if warn:
        # Check if the 2nd order Taylor coefficients of the derived shifted Hamiltonian agree in complex
        # normal form with the values predicted by linear theory.
        HK_hesse_dict = dHK.hess(Df=results)
        HK_hesse_dict = {k: v for k, v in HK_hesse_dict.items() if abs(v) > tol}
        for k in HK_hesse_dict.keys():
            diff = abs(HK_hesse_dict[k] - nfdict['cnf'][k[0], k[1]]) # (++)
            if diff > tol:
                raise RuntimeError(f'CNF entry {k} does not agree with Hamiltonian expansion: diff {diff} > {tol} (tol).')
        
    return results, nfdict


def bnf(H, order: int, z=[], tol=1e-14, **kwargs):
    '''
    Compute the Birkhoff normal form of a given Hamiltonian up to a specific order.
    
    Attention: Constants and any gradients of H at z will be ignored. If there is 
    a non-zero gradient, a warning is issued by default.
    
    Parameters
    ----------
    H: callable or dict
        Defines the Hamiltonian to be normalized. If H is of type dict, then it must be of the
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
    
    # Induction start (k = 2); get P_3 and R_4. Z_2 is set to zero.
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

