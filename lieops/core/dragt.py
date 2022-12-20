import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

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
    Jinv = -create_J(dim)
    # Remarks to the next line:
    # 1) multiply from right to prevent operator overloading from numpy.
    # 2) The poisson factor pf is required, because we have to invert the poisson bracket using J and pf.
    # 3) The final minus sign is used to ensure that we have H on the left in Eq. (2)
    return -_integrate(*[sum([g[k]*Jinv[l, k] for k in range(dim2)]) for l in range(dim2)])/pf

def dragtfinn(*p, offset=[], tol=0, order=0, flinp={}, **kwargs):
    '''
    Let p_1, ..., p_n be polynomials representing the Taylor expansions of
    the components of a symplectic map M. 
        
    Then this routine will find polynomials f_1, f2_a, f2_b, f3, f4, f5, ...,
    where f_k is a homogeneous polynomial of degree k, so that
    M ~ exp(:f2_a:) o exp(:f2_b:) o exp(:f3:) o exp(:f4:) o ... o exp(:fn:) o exp(:f1:)
    holds.
    
    Parameters
    ----------
    p: poly
        Polynomials representing the components of the symplectic map M.
        
    tol: float, optional
        If > 0, perform certain consistency checks during the calculation.
        
    order: int, optional
        The maximal power of the polynomials f_k.
        
    offset: subscriptable, optional
        An optional point of reference around which the map should be represented.
        By default, this point is zero.
        
    flinp: dict, optional
        Specific input parameters passed to lieops.core.lie.lexp flow calculation.
        For example, if a flow input parameter needs an 'order' parameter, it can
        be specified here.
        
    **kwargs
        Further input parameters passed to lieops.core.lie.lexp flow calculation.
    
    Returns
    -------
    list
        A list of poly objects [f1, f2_a, f2_b, f3, f4, ...] as described above.
        Note that the rightmost Lie polynomial needs to be applied first.
        
    References
    ----------
    [1] A. Dragt: "Lie Methods for Nonlinear Dynamics with Applications to Accelerator Physics", University of Maryland, 2020,
        http://www.physics.umd.edu/dsat/
    '''
    # TODO: add option to reverse the order
    
    # check input consistency
    dim = p[0].dim
    dim2 = dim*2
    assert all([e.dim == dim for e in p])
    pf = p[0]._poisson_factor
    assert all([e._poisson_factor == pf for e in p])
    assert len(p) == dim2, f'Polynomials received: {len(p)} Expected: {dim2}'
    if order == 0:
        order = max([e.maxdeg() for e in p]) + 1
    assert order < np.inf, 'Requested order of the Dragt-Finn series infinite.'
    flinp.update(kwargs)
    
    # determine the start and end points of the map
    if len(offset) == 0:
        start = [0]*dim2
        final = [e.get((0,)*dim2, 0) for e in p]
    else:
        assert len(offset) == dim2, f'Reference point dimension: {len(offset)}, expected: {dim2}.'
        start = offset
        final = [e(*offset) for e in p]
    
    if order == 1: # return a first-order polynomial providing the translation:
        diff = [final[k] - start[k] for k in range(dim2)]
        return [const2poly(*diff, poisson_factor=pf)]
    
    # determine the linear map
    R = np.array([poly2vec(e.homogeneous_part(1)).tolist() for e in p])
    A, B = symlogs(R.transpose(), tol2=tol) # This means: exp(A) o exp(B) = R.transpose(). Explanation why we have to use transpose will follow at (++).
    SA, SB = ad2poly(A, poisson_factor=pf, tol=tol), ad2poly(B, poisson_factor=pf, tol=tol)
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
    # In general we thus have, by construction for the coordinates xi/eta:
    # lexp(SA')(*lexp(SB')(*xieta)) = R
    # This is the reason why we had to use .transpose() in the construction of SA and SB above.
    Ri = np.linalg.inv(R)
    
    if tol > 0: # Perform some consistency checks
        # symplecticity check:
        J = create_J(dim) 
        assert np.linalg.norm(R.transpose()@J@R - J) < tol, f'It appears that the given map is not symplectic within a tolerance of {tol}.'
        # check if symlogs gives correct results
        assert np.linalg.norm(expm(A)@expm(B) - R.transpose()) < tol
        xieta = create_coords(dim) # for checks at (+) below
        # Further idea: p[i]@p[k + dim] should be -1j*delta_{ik} etc. But this might often not be well satisfied in higher orders
    
    # invert the effect of the linear map R, see Ref. [1], Eq. (7.6.17):
    p1 = [e.extract(key_cond=lambda k: sum(k) >= 1) for e in p]
    p_new = [sum([p1[k]*Ri[l, k] for k in range(dim2)]) for l in range(dim2)] # multiply Ri from right to prevent operator overloading from numpy.
    
    f_all = [SA, SB] # R = exp(A) o exp(B)
    f_nl = []
    for k in tqdm(range(2, order), disable=kwargs.get('disable_tqdm', False)):
        gk = [e.homogeneous_part(k) for e in p_new]

        if tol > 0: # (+) check if prerequisits for application of the Poincare Lemma are satisfied
            for i in range(dim2):
                for j in range(i):
                    zero = xieta[j]@gk[i] + gk[j]@xieta[i]
                    assert zero.above(tol) == 0, f'It appears that the Poincare Lemma can not be applied for order {k} (tol: {tol}):\n{zero}'
        
        fk = sympoincare(*gk)
        lk = lexp(-fk)
        p_new = lk(*p_new, **flinp)
        
        if tol > 0: # (+) check if the Lie operators cancel the Taylor-map up to the current order
            # further idea: check if fk is the potential of the gk's
            for i in range(dim2):
                remainder = (p_new[i] - xieta[i]).above(tol)
                assert remainder.mindeg() >= k + 1, f'It appears that the Lie operators do not properly cancel the Taylor-map terms of order {k + 1} (tol: {tol}):\n{remainder}'

        f_all.append(fk)
        
    if any([e != 0 for e in start]):
        f_all.insert(0, -const2poly(*start, poisson_factor=pf))
    if any([e != 0 for e in final]):
        f_all.append(const2poly(*final, poisson_factor=pf))
    return f_all
