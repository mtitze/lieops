import numpy as np
from scipy.linalg import expm

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

def dragtfinn(*p, tol=0, **kwargs):
    '''
    Let p_1, ..., p_n be polynomials representing the Taylor expansions of
    the components of a symplectic map M. 
    
    TODO: 1) add constant 2) add option to reverse the order).
    
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
        
    **kwargs
        Optional keyworded arguments passed to lieops.ops.lie.lexp call (flow calculation).
    
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
    # check input consistency
    dim = p[0].dim
    dim2 = dim*2
    assert all([e.dim == dim for e in p])
    pf = p[0]._poisson_factor
    assert all([e._poisson_factor == pf for e in p])
    assert len(p) == dim2, f'Polynomials received: {len(p)} Expected: {dim2}'
    order = kwargs.pop('order', max([e.max_power for e in p]))
    assert order < np.inf
    
    # determine the polynomial determining the translation(s):
    g1 = const2poly(*[e.homogeneous_part(0).get((0,)*dim2, 0) for e in p], 
                    poisson_factor=pf)
    if order == 1:
        return [g1]
    
    # determine the linear map
    R = np.array([poly2vec(e.homogeneous_part(1)).tolist() for e in p])
    A, B = symlogs(R, tol2=tol)
    SA, SB = ad2poly(A, poisson_factor=pf), ad2poly(B, poisson_factor=pf)
    Ri = np.linalg.inv(R)
    
    if tol > 0:
        # symplecticity check:
        J = create_J(dim) 
        assert np.linalg.norm(R.transpose()@J@R - J) < tol, f'It appears that the given map is not symplectic within a tolerance of {tol}.'
        # check if symlogs gives correct results
        assert np.linalg.norm(expm(A)@expm(B) - R) < tol
        xieta = create_coords(dim) # for check at (+) below
    
    # invert the linear map, see Ref. [1], Eq. (7.6.17).
    p_new = [sum([p[k]*Ri[l, k] for l in range(dim2)]) for k in range(dim2)] # multiply Ri from right to prevent operator overloading from numpy
    
    f_all = [g1, SB, SA] # TODO: check (again) why g1 should be set at the first position here (compare 7.7.10 in Ref. [1], where it appears at the last position)
    for k in range(2, order):
        gk = [e.homogeneous_part(k) for e in p_new]

        if tol > 0: # (+)
            # check if prerequisits for application of the Poincare Lemma are satisfied
            for i in range(len(gk)):
                for j in range(i):
                    zero = xieta[j]@gk[i] + gk[j]@xieta[i]
                    assert zero.above(tol) == 0, f'It appears that the Poincare Lemma can not be applied, using tol: {tol})'
        
        fk = sympoincare(*gk)
        lk = lexp(-fk)
        p_new = [lk(e, **kwargs) for e in p_new]
        f_all.append(fk)

    return f_all