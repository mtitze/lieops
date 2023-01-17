import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
import warnings

from lieops.linalg.matrix import create_J
from lieops.linalg.nf import symlogs
from lieops.core.lie import create_coords, lexp
from lieops.core.tools import const2poly, poly2vec, ad2poly

def _integrate_k(p, k: int):
    '''
    Let p be a Lie polynomial. Then this routine will
    compute the integral of p*dz_k, where z_k denotes the k-th component.
    '''
    integral_values = {}
    dim = p.dim
    power_add = np.zeros(dim*2, dtype=int)
    power_add[k] = 1
    for powers, values in p.items():
        powers = np.array(powers)
        integral_values[tuple(powers + power_add)] = values/(sum(powers) + 1) # Basically Eq. (7.6.24) in Ref. [1] (reference in sympoincare)
    return p.__class__(values=integral_values, dim=dim, max_power=p.max_power)

def _integrate(*p):
    '''
    Let p_1, ..., p_n be some Lie polynomials.
    Then this routine will compute the (line) integral p_k*dz_k.
    '''
    dim = p[0].dim
    return sum([_integrate_k(p[k], k) for k in range(dim*2)])

def sympoincare(*g):
    '''
    Let g_1, ..., g_n be Lie polynomials and z_j the coordinates (in our setting complex xi/eta-coordinates),
    satisfying
    {g_i, z_j} = {g_j, z_i}.    (1)
    Because the Lie polynomials are analytic functions by construction, there must exist a 
    potential H (Hamiltonian) so that
    g_j = {H, z_j}     (2)
    holds. This is the 'symplectic' variant of the Poincare Lemma.
    Obviously, the degree of H is then given by max([deg(g_j) for j in 1, ..., n]) + 1.
    
    We shall follow the steps outlined in Ref. [1], Lemma 6.2 in Section 7 (Factorization Theorem).
    
    Parameters
    ----------
    g: poly
        One or more Lie polynomials having property (1) above (Warning: No check against this property here.)
        
    Returns
    -------
    poly
        A Lie polynomial H satisfying Eq. (2) above.
    
    References
    ----------
    [1] A. Dragt: "Lie Methods for Nonlinear Dynamics with Applications to Accelerator Physics", University of Maryland, 2020,
        http://www.physics.umd.edu/dsat/
    '''
    dim = g[0].dim
    dim2 = dim*2
    # assert all([e.dim == dim for e in g])
    assert len(g) == dim2
    pf = g[0]._poisson_factor
    # assert all([e._poisson_factor == pf for e in g])
    Jinv = create_J(dim).transpose()
    # Remarks to the next line:
    # 1) multiply from right to prevent operator overloading from numpy.
    # 2) The poisson factor pf is required, because we have to invert the poisson bracket using J and pf.
    # 3) The final minus sign is used to ensure that we have H on the left in Eq. (2)
    return -_integrate(*[sum([g[k]*Jinv[l, k] for k in range(dim2)]) for l in range(dim2)])/pf

def dragtfinn(*p, order='auto', offset=[], pos2='right', tol=1e-6, tol_checks=0, disable_tqdm=False, force_order=False, warn=True, **kwargs):
    '''
    Let p_1, ..., p_n be polynomials representing the Taylor expansions of
    the components of a symplectic map M. 
        
    Then this routine will find polynomials f_1, g_1, f2_a, f2_b, f3, f4, f5, ...,
    where f_k is a homogeneous polynomial of degree k, so that
    M ~ exp(:f1:) o exp(:f2_a:) o exp(:f2_b:) o exp(:f3:) o exp(:f4:) o ... o exp(:fn:) o exp(:g1:)
    holds. The position of the chain of 2nd order polynomials can hereby be chosen,
    by providing a 'pos2'-parameter.
    
    Parameters
    ----------
    p: poly
        Polynomials representing the components of the symplectic map M.
        
    order: int, optional
        The maximal degree of the approximated map (= maximal degree of the polynomials fk above - 1).
        
    offset: subscriptable, optional
        An optional point of reference around which the map should be represented.
        By default, this point is zero.
        
    pos2: str, optional
        Either 'left' or 'right' (default). The outcome will determine the representation of M:
        pos2 == 'right':
        M ~ exp(:f1:) o exp(:f3:) o exp(:f4:) o ... o exp(:fn:) o exp(:f2_a:) o exp(:f2_b:) o exp(:g1:)
        pos2 == 'left':
        M ~ exp(:f1:) o exp(:f2_a:) o exp(:f2_b:) o exp(:f3:) o exp(:f4:) o ... o exp(:fn:) o exp(:g1:)

    tol: float, optional
        Identify small fk and drop them if all their values are below this threshold.
        
    tol_checks: float, optional
        If > 0, perform certain consistency checks during the calculation.
                        
    disable_tqdm: boolean, optional
        If true, disable the tqdm status bar printout.
        
    force_order: boolean, optional
        If true, do not change the order by -1 in case 1) an offset is detected and 2) order >= degree of the Taylor map.
        Default: (false)
        
    warn: boolean, optional
        Supress or show warnings.
        
    **kwargs
        Further input parameters passed to lieops.core.lie.lexp flow calculation.
    
    Returns
    -------
    list
        A list of poly objects [f1, f2_a, f2_b, f3, f4, ...] in the order described above.
        
    References
    ----------
    [1] A. Dragt: "Lie Methods for Nonlinear Dynamics with Applications to Accelerator Physics", University of Maryland, 2020,
        http://www.physics.umd.edu/dsat/
    '''
    # check & update input
    dim = p[0].dim
    dim2 = dim*2
    assert all([e.dim == dim for e in p])
    pf = p[0]._poisson_factor
    assert all([e._poisson_factor == pf for e in p])
    assert len(p) == dim2, f'Polynomials received: {len(p)} Expected: {dim2}'
    max_power = max([e.max_power for e in p]) # Required for the input for ad2poly, otherwise ad2poly may produce polynomials with max_power = inf, even if input has < inf. This may result in slow code.
    max_deg = max([e.maxdeg() for e in p]) # The degree of the input Taylor map.
    if order == 'auto':
        order = max_power # see also (+++) below
    assert order < max_power + 1 and order < np.inf, f'Requested order of the Dragt-Finn series can not be >= {max_power + 1}.'
    if order > max_deg and warn:
        warnings.warn(f'Requested order {order} > {max_deg} (maximal degree of Taylor map).')
    if len(kwargs) == 0 and warn:
        warnings.warn("No flow parameters set.")
    
    # determine the start and end points of the map
    if len(offset) == 0:
        start = [0]*dim2
        final = [e.get((0,)*dim2, 0) for e in p]
    else:
        assert len(offset) == dim2, f'Reference point dimension: {len(offset)}, expected: {dim2}.'
        start = offset
        final = [e(*offset) for e in p]

    if order == 0:# or max_deg == 1: # In this case we can immediately return a first-order polynomial, which will provide the translation:
        diff = [final[k] - start[k] for k in range(dim2)]
        return [const2poly(*diff, poisson_factor=pf)]

    start_is_nonzero = any([e != 0 for e in start])
    if start_is_nonzero:
        # preparation step in case of translations, see Ref. [1], Eq. (7.7.17)
        h1 = const2poly(*start, poisson_factor=pf, max_power=max_power) # E.g.: lexp(h1)(xi) = xi + start[0] 
        p = lexp(h1)(*p, method='2flow')
        # the application of exp(:h1:), where h1 is a first-order polynomial, has produced a new Taylor map of max_deg - 1.
        # Therefore we should reduce the requested order by one if the order is larger or equal max_deg, to avoid an error with the final step: 
        # Due to truncation at the original map may not admit a symplectomorphism at the its final requested order.
        if order >= max_deg and not force_order:
            if warn:
                warnings.warn(f"Requested order {order} >= {max_deg} (maximal degree of Taylor map) & non-zero offset detected. Order reduced by 1.")
            order = order - 1
        
    # determine the linear part of the map
    R = np.array([poly2vec(e.homogeneous_part(1)).tolist() for e in p])
    A, B = symlogs(R.transpose(), tol2=tol_checks) # This means: exp(A) o exp(B) = R.transpose(). Explanation why we have to use transpose will follow at (++).
    SA = ad2poly(A, poisson_factor=pf, tol=tol_checks, max_power=max_power).above(tol)
    SB = ad2poly(B, poisson_factor=pf, tol=tol_checks, max_power=max_power).above(tol)
    # (++) 
    # Let us assume that we would have taken "symlogs(R) = A, B" (i.e. exp(A) o exp(B) = R) and consider a 1-dim case.
    # In the following the '~' symbol means that we identify the (1, 0)-key with xi and the (0, 1)-key with eta.
    # By definition of ad2poly:
    # SA@xi ~ A@[1, 0]
    # SA@eta ~ A@[0, 1]
    # SB@xi ~ B@[1, 0]
    # SB@eta ~ B@[0, 1]
    # and thus (e.g.):
    # SB@(SA@xi) ~ B@A@[1, 0] (Attention: The @-operator on the left side requires brackets: (SB@SA)@xi != SB@(SA@xi) )
    # hence:
    # lexp(SA)(xi) ~ expm(A)@[1, 0]
    # lexp(SA)(lexp(SB)(xi)) ~ expm(A)@expm(B)@[1, 0] = R@[1, 0]
    # So first SB needs to be executed, then SA, as expected from the relation R = exp(A) o exp(B)
    #
    # Let us translate the '~' relation back to an equality:
    # By the above consideration, lexp(SA)(lexp(SB)(xi)) applied to a vector (xi0, eta0) will yield the sum of *rows* of the first column of R:
    # lexp(SA)(lexp(SB)(xi))(xi0, eta0) = R[0, 0]*xi0 + R[1, 0]*eta0.
    # 
    # Therefore we have to construct SA and SB by R.transpose(), so we get for the corresponding SA' and SB':
    # lexp(SA')(lexp(SB')(xi))(xi0, eta0) = R[0, 0]*xi0 + R[0, 1]*eta0 = [R@[xi0, eta0]]_0  (notice that now we have an equality, not '~' as above)
    #
    # In general we thus have, by construction for the lie-polynomials xi/eta:
    # lexp(SA')(*lexp(SB')(*xieta)) = R        (1)
    # This is the reason why we had to use .transpose() in the construction of SA and SB above.
    # See also Sec. 8.3.5 "Dual role of the Phase-Space Coordinates" in Dragt's book [1]. In that
    # section, the transposition can be found as well.
    
    J = create_J(dim)    
    if tol_checks > 0: # Perform some consistency checks
        # symplecticity check:
        assert np.linalg.norm(R.transpose()@J@R - J) < tol_checks, f'Map not symplectic within a tolerance of {tol_checks}.'
        # check if symlogs gives correct results
        assert np.linalg.norm(expm(A)@expm(B) - R.transpose()) < tol_checks
        xieta = create_coords(dim, poisson_factor=pf, max_power=max_power) # for the two checks at (+) below
        # Now it holds with
        # op_result := lexp(SA)(*lexp(SB)(*xieta, power=30), power=30)
        # 1) p[k] == op_result[k]    (2)
        # 2) with
        # op_matrix = np.array([poly2vec(op) for op in op_result])
        # R == op_matrix
        #
        # Attention
        # ---------
        # let xieta0 be a set of start coordinates (the 0 should indicate that we consider complex numbers).
        # Then one may be tempting to consider Eq. (1) (see previous comment):
        # lexp(SA)(*lexp(SB)(*xieta0))   (1)
        # However, for *numbers* this will NOT agree with the outcome (3) of the original map p:
        # [e(*xieta0) for e in p]        (3)
        # Instead, due to the pull-back property of the lexp-operators, the correct way to obtain
        # the result (3) would be:
        # lexp(SB)(*lexp(SA)(*xieta0))   (4)
        # Note the difference of the order: Now SB and SA in Eq. (4) are reversed in comparison to Eq. (1).
        #
        #
        # (Further idea to run in this check: p[i]@p[k + dim] should be -1j*delta_{ik} etc. But this might often not be well satisfied in higher orders)
        
    # Ensure that the Poincare-Lemma is met for the first step; See Ref. [1], Eq. (7.6.17):
    Ri = -J@R.transpose()@J
    p_new = [sum([p[k]*Ri[l, k] for k in range(dim2)]) for l in range(dim2)] # multiply Ri from right to prevent operator overloading from numpy.
    if tol > 0:
        # not dropping small values may result in a slow-down of the code. Therefore:
        p_new = [e.above(tol) for e in p_new]
        
    # Construct & collect the chain of operators of the Dragt/Finn factorization.
    # We shall collect these operators in a list 'f_all' so that the first operator 
    # in f_all is exectued first, if applied to *numbers* (see the 'Attention' section above). Hence:
    f_all = []
    if SA != 0:
        f_all.append(SA)
    if SB != 0:
        f_all.append(SB)
    # now f_all = [SA, SB] according to Eq. (4) above; R = exp(A) o exp(B);
    f_nl = []
    for k in tqdm(range(2, order + 1), disable=disable_tqdm):
        gk = [e.homogeneous_part(k) for e in p_new]
        
        if len(gk) == 0 or len([ek for ek in gk if ek == 0]) == len(p_new):
            # skip this case if there is no homogeneous part of order k
            continue
                    
        if tol_checks > 0: # (+) check if prerequisits for application of the Poincare Lemma are satisfied
            for i in range(dim2):
                for j in range(i):
                    zero = xieta[j]@gk[i] + gk[j]@xieta[i]
                    assert zero.above(tol_checks) == 0, f'Poincare Lemma prerequisits not met for order {k} (tol: {tol_checks}):\n{zero}'
                            
        fk = sympoincare(*gk).above(tol) # deg(fk) = k + 1
        if fk == 0:
            # continue with the next k if the potential is zero within the given tolerance
            continue
        lk = lexp(-fk)
        p_new = lk(*p_new, **kwargs) # N.B.: order(:fk:p_new) = k + order(p_new) - 1, since deg(fk) = k + 1. The maximal order of p_new can theoretically be infinite, but technically it is limited by max_power. (+++)
        
        if tol_checks > 0: # (+) check if the Lie operators cancel the Taylor-map up to the current order
            # further idea: check if fk is the potential of the gk's
            for i in range(dim2):
                remainder = (p_new[i] - xieta[i]).above(tol_checks).extract(key_cond=lambda key: sum(key) >= 1)
                if remainder != 0:
                    assert remainder.mindeg() >= k + 1, f'Lie operator of order {k + 1} does not properly cancel the Taylor-map terms of order {k} (tol: {tol_checks}):\n{remainder.extract(key_cond=lambda key: sum(key) < k + 1)}'
                    
        f_nl.append(fk)
        
    # Now by construction we have (up to order k + 1) for the xi/eta Lie-polynomials xieta:
    # xieta + O(k + 1) = lexp(-f_k)(*lexp(-f_{k - 1})(...*lexp(-f_3)(*p_new)) ...)
    # or conversely:
    # lexp(f_3)(*lexp(f_4)( ...*lexp(f_k)(*xieta) ...)) = p_new.
    if pos2 == 'right':
        # Here p_new = Ri@p and so -- by linearity of the lexp-operators -- it follows:
        # lexp(f_3) o lexp(f_4) o ... o lexp(f_k)(R@xieta) = p,
        # hence (see Eq. (1) above: R = lexp(SA) o lexp(SB)):
        # lexp(f_3) o lexp(f_4) o ... o lexp(f_k) o lexp(SA) o lexp(SB) = p     (5)
        f_all = f_nl + f_all
        # note that on coordinates, f_3 needs to be executed first. So this definition is in 
        # line with (5) and our overall definition of 'f_all'.
    else:
        # We shall consider Eq. (5) and apply the Hadamard-Lemma on the non-linear parts to move
        # the SA and SB to the other side. The order of SA and SB is hereby given by the inverse of Eq. (1),
        # since we consider Lie-polynomials at this step (and not points/floats). 
        # It is also very important at this step that a symplectic integration routine is used. 
        # Fortunately there exist a straigthforward symplectic integrator for those 2nd-order 
        # Hamiltonians SA and SB:
        f_hdm = []
        if len(f_nl) > 0:
            f_hdm = lexp(-SB)(*lexp(-SA)(*f_nl, outl1=True, method='2flow'), outl1=True, method='2flow') # the outl1 parameters are True here to cope with special cases that SA or its input is zero. In this case the Lie-operators would return a Lie-polynomial. That would cause a problem with the use of '*' here and f_hdm may also not be a list ...
        f_all = f_all + [f.above(tol) for f in f_hdm if f.above(tol) != 0] # Every entry in f_nl is non-zero (wrt. tol) by construction. Since the lexp(-SB) and lexp(-SA) operators are invertible, the elements in f_hdm are therefore also non-zero. However, the lexp-operators may still generate many small terms which should be removed here.
        
    if start_is_nonzero:
        f_all.insert(0, -h1)
            
    if any([e != 0 for e in final]):
        g1 = const2poly(*final, poisson_factor=pf, max_power=max_power)
        if start_is_nonzero and len(f_all) == 1:
            # in this case only a single term (-h1) of order 1 is contained in f_all thus far. We shall combine this term with g1 (such a case may happen for a first-order polynomial + offset as input)
            f_all[0] += g1
        else:
            f_all.append(g1)
            
    return f_all
