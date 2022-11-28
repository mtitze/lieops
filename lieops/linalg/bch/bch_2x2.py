import numpy as np
from scipy.linalg import schur
import warnings

'''
References
----------
[1] Foulis: "The algebra of complex 2 × 2 matrices and a 
             general closed Baker–Campbell–Hausdorff formula",
             J. Phys. A: Math. Theor. 50 305204 (2017).
'''

def Sinhc(x):
    # Note that this function has the property Sinhc(-x) = Sinhc(x) for every complex number x.
    if type(x) == np.ndarray:
        out = np.ones(len(x), dtype=np.complex128)
        non_zero_indices = (x != 0)
        x1 = x[non_zero_indices]
        v1 = np.sinh(x1)/x1
        out[non_zero_indices] = v1
    else:
        if x != 0:
            return np.sinh(x)/x
        else:
            return 1
    return out
    
def get_case(C):
    '''
    Determine the case outlined in Thm. 2, Ref. [1], for C.
    '''
    assert C[0, 0] + C[1, 1] == 0 # Tr(C) == 0
    detC = C[0, 0]*C[1, 1] - C[1, 0]*C[0, 1] # det(C)
    if detC != 0:
        # Case 1
        case = 1
    elif not (C == 0).all():
        # Case 2 (i)
        case = 2
    else:
        # C == 0; Case 2 (ii).
        case = 3
    return case
    
def get_params(A, B, tol=0):
    '''
    Compute the parameters in Ref. [1] requied to compute the Baker-Campbell-Hausdorff formula.
    
    Attention: For some isolated cases the BHC formula does not converge (These cases are
    related to the zeros of the Sinhc function in Ref. [1]).
    '''
    assert A.shape == (2, 2) and B.shape == (2, 2)
    #C = A@B - B@A
    #case = get_case(C)
    
    a = A[0, 0] + A[1, 1] # Tr(A)
    b = B[0, 0] + B[1, 1] # Tr(B)
    alpha = A[0, 0]*A[1, 1] - A[1, 0]*A[0, 1] # det(A)
    beta = B[0, 0]*B[1, 1] - B[1, 0]*B[0, 1] # det(B)
    
    AB = A@B
    omega = AB[0, 0] + AB[1, 1] # tr(AB)
    epsilon = omega - a*b/2
    
    sigma2 = a**2/4 - alpha
    tau2 = b**2/4 - beta
    # We do not have to bother with the sign of sigma and tau here, 
    # because cosh and Sinhc (see below) are both even functions; for every complex number
    # it holds cosh(-z) = cosh(z) and sinh(-z) = -sinh(z).
    sigma = np.sqrt(sigma2, dtype=np.complex128)
    tau = np.sqrt(tau2, dtype=np.complex128)
    Sinhc_sigma = Sinhc(sigma)
    Sinhc_tau = Sinhc(tau)
    cosh_sigma = np.cosh(sigma)
    cosh_tau = np.cosh(tau)
    
    cosh_chi = cosh_sigma*cosh_tau + epsilon/2*Sinhc_sigma*Sinhc_tau # Eq. (36) in Ref. [1]
    chi = np.arccosh(cosh_chi) # here one may add integer multiples of 2*pi. 
    # Attention: If chi is e.g. purely imaginary, Sinhc(chi) will cross zero periodically. 
    # This means that for those cases the BCH formula can not converge.
    # Due to the relationship between sinh and sin,
    # the zeros of Sinhc reside on the imaginary axis and look similar to the zeros of the sinc function,
    # see e.g. https://en.wikipedia.org/wiki/File:Sinc_cplot.svg.
    Sinhc_chi = Sinhc(chi)
    if tol > 0:
        # check against cases where the BHC equation may not converge
        if abs(Sinhc_chi) < tol:
            warnings.warn(f'|Sinhc(chi)| smaller than {tol} (tol) -- convergence not ensured.')

    p = Sinhc_sigma*cosh_tau/Sinhc_chi
    q = cosh_sigma*Sinhc_tau/Sinhc_chi
    r = Sinhc_sigma*Sinhc_tau/(2*Sinhc_chi)
    k = (a + b - a*p - b*q)/2
    
    out = {}
    out['C'] = A@B - B@A
    #out['case'] = case
    out['AB'] = AB
    out['I'] = np.eye(2)
    out['a'] = a
    out['b'] = b
    out['alpha'] = alpha
    out['beta'] = beta
    out['omega'] = omega
    out['epsilon'] = epsilon
    out['abs_sigma'] = sigma
    out['abs_tau'] = tau
    out['chi'] = chi
    out['p'] = p
    out['q'] = q
    out['r'] = r
    out['k'] = k
    return out

def bch_2x2(A, B, tol=0):
    '''
    Compute the Baker-Campbell-Hausdorff matrix C so that exp(C) = exp(A)@exp(B) holds,
    according to Ref. [1].
    '''
    params = get_params(A, B, tol=tol) # obtain the parameters introduced in Ref. [1].
    C = params['C']
    I = params['I']
    Z = params['k']*I + params['p']*A + params['q']*B + params['r']*C
    if tol > 0:
        # consistency check (see text in Ref. [1] below Eq. (37))
        chi2 = params['chi']**2
        ZZ = Z@Z
        zero = (ZZ[0, 0] + ZZ[1, 1])/2 - (Z[0, 0] + Z[1, 1])**2/4 - chi2 # chi2 == Tr(Z**2)/2 - (Tr(Z)**2)/4
        assert abs(zero) < tol, f'Consistency check "chi2 == Tr(Z**2)/2 - (Tr(Z)**2)/4" failed:\n{zero} >= {tol} (tol).'
    return Z

    